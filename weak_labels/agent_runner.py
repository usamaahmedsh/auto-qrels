"""
weak_labels/agent_runner.py

Config-driven agent runner (NO hidden knobs).

Design goals:
- All tunables come from config dict (later configs/base.yaml).
- CPU-only agent is supported (recommended when vLLM owns the GPU).
- Uses BM25 -> dense rerank -> LLM judge -> write qrels/triples.
- Checkpoint/resume.
- Optional parallelism across queries (ThreadPoolExecutor), controlled by config.

Expected config keys (aliases you will put in base.yaml later):

agent:
  # batching / parallelism
  batch_size: int
  concurrent_queries: int
  checkpoint_dir: str
  checkpoint_interval: int
  max_queries: null|int

  # retrieval
  global_top_k_bm25: int
  dense_top_k_from_bm25: int

  # LLM candidate selection / labeling
  llm_candidates_top_k: int
  llm_conf_threshold: float
  positives_max: int
  hard_negatives_per_query: int

  # filtering
  min_passage_tokens: int
  max_passages_per_page: int

output:
  qrels_path: str
  triples_path: str
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from .bm25_index import BM25Index
from .dense_encoder import DenseEncoder
from .llm_client import CachedLLMJudge


DEFAULT_AGENT_CFG: Dict[str, Any] = {
    "batch_size": 512,
    "concurrent_queries": 4,
    "checkpoint_dir": "data/checkpoints",
    "checkpoint_interval": 100,
    "max_queries": None,
    "global_top_k_bm25": 200,
    "dense_top_k_from_bm25": 100,
    "llm_candidates_top_k": 30,
    "llm_conf_threshold": 0.90,
    "positives_max": 7,
    "hard_negatives_per_query": 20,
    "min_passage_tokens": 50,
    "max_passages_per_page": 2,
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _count_tokens_rough(s: str) -> int:
    # Cheap approximation; real tokenizers are slower and not needed for filtering.
    return len((s or "").split())


def _page_id_from_doc_id(doc_id: str) -> str:
    # Matches your earlier convention "pageId_Pxx" if present.
    s = doc_id or ""
    if "_P" in s:
        return s.rsplit("_P", 1)[0]
    return s


class Agent:
    def __init__(
        self,
        bm25: BM25Index,
        dense_encoder: DenseEncoder,
        llm_judge: CachedLLMJudge,
        config: Dict[str, Any],
    ):
        self.bm25 = bm25
        self.dense_encoder = dense_encoder
        self.llm_judge = llm_judge

        self.cfg_raw = config
        self.agent_cfg = _deep_update(DEFAULT_AGENT_CFG, (config.get("agent") or {}))

        self.qrels_path = Path(config["output"]["qrels_path"])
        self.triples_path = Path(config["output"]["triples_path"])
        self.checkpoint_dir = Path(self.agent_cfg["checkpoint_dir"])
        self.checkpoint_file = self.checkpoint_dir / "progress.json"

        for p in [self.qrels_path, self.triples_path]:
            p.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Locks
        self._stats_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._processed_lock = threading.Lock()

        # Stats
        self.stats: Dict[str, Any] = {
            "queries_processed": 0,
            "passages_judged": 0,
            "positives_found": 0,
            "hard_negatives_found": 0,
            "total_time_s": 0.0,
            "avg_time_per_query_s": 0.0,
        }

        self.processed_queries = self._load_checkpoint()
        logger.info(f"✓ Agent initialized (checkpoint: {len(self.processed_queries):,} queries)")

        # Build a fast doc_id -> text mapping if BM25 exposes doc_ids/corpus.
        self._doc_text: Optional[Dict[str, str]] = None
        if hasattr(self.bm25, "doc_ids") and hasattr(self.bm25, "corpus"):
            try:
                doc_ids = list(self.bm25.doc_ids)
                corpus = list(self.bm25.corpus)
                if len(doc_ids) == len(corpus):
                    self._doc_text = dict(zip(doc_ids, corpus))
                    logger.info(f"✓ Built in-memory doc lookup: {len(self._doc_text):,} passages")
            except Exception:
                self._doc_text = None

    # -----------------------------
    # Checkpointing
    # -----------------------------
    def _load_checkpoint(self) -> set[str]:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                ids = set(data.get("processed_query_ids", []))
                prev_stats = data.get("stats")
                if isinstance(prev_stats, dict):
                    self.stats.update(prev_stats)
                return ids
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return set()

    def _save_checkpoint(self) -> None:
        with self._processed_lock, self._stats_lock:
            payload = {
                "processed_query_ids": list(self.processed_queries),
                "stats": self.stats,
            }
        tmp = str(self.checkpoint_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        Path(tmp).replace(self.checkpoint_file)

    # -----------------------------
    # Core steps
    # -----------------------------
    def _lookup_candidates_text(self, bm25_results: List[Tuple[str, float]]) -> Dict[str, str]:
        candidates: Dict[str, str] = {}
        if not bm25_results:
            return candidates

        if self._doc_text is not None:
            for doc_id, _ in bm25_results:
                t = self._doc_text.get(doc_id)
                if t is not None:
                    candidates[doc_id] = t
            return candidates

        # Fallback: if BM25Index has a method (optional)
        if hasattr(self.bm25, "get_text"):
            for doc_id, _ in bm25_results:
                try:
                    t = self.bm25.get_text(doc_id)
                    if t:
                        candidates[doc_id] = t
                except Exception:
                    pass
        return candidates

    def _filter_passages(self, passages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not passages:
            return []

        min_tokens = int(self.agent_cfg["min_passage_tokens"])
        max_per_page = int(self.agent_cfg["max_passages_per_page"])

        out: List[Dict[str, str]] = []
        per_page: Dict[str, int] = {}

        for p in passages:
            text = p.get("text") or ""
            if _count_tokens_rough(text) < min_tokens:
                continue

            page_id = _page_id_from_doc_id(p["doc_id"])
            used = per_page.get(page_id, 0)
            if used >= max_per_page:
                continue

            out.append(p)
            per_page[page_id] = used + 1

        return out

    def _select_llm_candidates(self, reranked: List[str], candidates_dict: Dict[str, str]) -> List[Dict[str, str]]:
        top_k = int(self.agent_cfg["llm_candidates_top_k"])
        selected_ids = reranked[:top_k]

        passages = [{"doc_id": did, "text": candidates_dict[did]} for did in selected_ids if did in candidates_dict]
        return self._filter_passages(passages)

    def _select_hard_negatives(
        self,
        judged: List[Dict[str, Any]],
        positives: List[Dict[str, Any]],
    ) -> List[str]:
        num_neg = int(self.agent_cfg["hard_negatives_per_query"])
        pos_ids = {p["doc_id"] for p in positives}

        # Hard negatives = judged NOs (or low relevance) closest to decision boundary.
        # In YES/NO mode, just take top judged that are not positives.
        negs: List[str] = []
        for j in judged:
            did = j["doc_id"]
            if did in pos_ids:
                continue
            negs.append(did)
            if len(negs) >= num_neg:
                break
        return negs

    def process_query(self, query_rec: Dict[str, Any], precomputed_bm25: Optional[List[Tuple[str, float]]] = None) -> Dict[str, Any]:
        qid = str(query_rec["query_id"])
        qtext = str(query_rec["text"])

        start = time.time()

        # BM25
        bm25_results = precomputed_bm25
        if bm25_results is None:
            bm25_results = self.bm25.search_with_scores(qtext, top_k=int(self.agent_cfg["global_top_k_bm25"]))

        if not bm25_results:
            return {"query_id": qid, "positives": 0, "hard_negatives": 0, "elapsed_s": time.time() - start}

        # candidate texts (doc_id -> passage text)
        candidates_dict = self._lookup_candidates_text(bm25_results)
        if not candidates_dict:
            return {"query_id": qid, "positives": 0, "hard_negatives": 0, "elapsed_s": time.time() - start}

        # Dense rerank
        dense_top_ids: List[str] = self.dense_encoder.rerank(
            qtext,
            candidates_dict,
            top_k=int(self.agent_cfg["dense_top_k_from_bm25"]),
        )

        # Choose LLM candidates
        llm_candidates = self._select_llm_candidates(dense_top_ids, candidates_dict)
        if not llm_candidates:
            return {"query_id": qid, "positives": 0, "hard_negatives": 0, "elapsed_s": time.time() - start}

        # LLM judge (sync wrapper)
        judged = self.llm_judge.judge_batch(qtext, llm_candidates)

        with self._stats_lock:
            self.stats["passages_judged"] += len(judged)

        # Positives
        conf_th = float(self.agent_cfg["llm_conf_threshold"])
        positives_max = int(self.agent_cfg["positives_max"])

        positives = [
            j for j in judged
            if bool(j.get("answerable", False)) and float(j.get("confidence", 0.0)) >= conf_th
        ][:positives_max]

        hard_neg_ids = self._select_hard_negatives(judged, positives)

        # Write outputs
        self._write_outputs(qid, qtext, positives, hard_neg_ids)

        elapsed = time.time() - start
        with self._stats_lock:
            self.stats["queries_processed"] += 1
            self.stats["positives_found"] += len(positives)
            self.stats["hard_negatives_found"] += len(hard_neg_ids)
            self.stats["total_time_s"] += elapsed
            self.stats["avg_time_per_query_s"] = self.stats["total_time_s"] / max(1, self.stats["queries_processed"])

        return {"query_id": qid, "positives": len(positives), "hard_negatives": len(hard_neg_ids), "elapsed_s": elapsed}

    def _write_outputs(
        self,
        query_id: str,
        query_text: str,
        positives: List[Dict[str, Any]],
        hard_neg_ids: List[str],
    ) -> None:
        if not positives:
            return

        triple = {
            "query_id": query_id,
            "query": query_text,
            "positive_doc_ids": [p["doc_id"] for p in positives],
            "positive_scores": [p.get("relevance_score", 3) for p in positives],
            "hard_negative_doc_ids": hard_neg_ids,
        }

        with self._write_lock:
            with open(self.qrels_path, "a") as f:
                for p in positives:
                    f.write(f"{query_id}\t0\t{p['doc_id']}\t{p.get('relevance_score', 3)}\n")

            with open(self.triples_path, "a") as f:
                f.write(json.dumps(triple) + "\n")

    # -----------------------------
    # Runner
    # -----------------------------
    def run(self, queries: List[Dict[str, Any]]) -> None:
        max_queries = self.agent_cfg.get("max_queries")
        checkpoint_interval = int(self.agent_cfg["checkpoint_interval"])
        batch_size = int(self.agent_cfg["batch_size"])
        concurrent_queries = int(self.agent_cfg["concurrent_queries"])

        remaining = [q for q in queries if str(q["query_id"]) not in self.processed_queries]
        if max_queries is not None:
            remaining = remaining[: int(max_queries)]

        total = len(queries)
        logger.info(f"Processing {len(remaining):,} queries ({len(self.processed_queries):,} already done)")

        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start : batch_start + batch_size]

            # Optional: batch BM25 if available
            bm25_batch: Optional[List[List[Tuple[str, float]]]] = None
            if hasattr(self.bm25, "batch_search_with_scores"):
                try:
                    bm25_batch = self.bm25.batch_search_with_scores(
                        [q["text"] for q in batch],
                        top_k=int(self.agent_cfg["global_top_k_bm25"]),
                    )
                except Exception as e:
                    logger.warning(f"BM25 batch_search_with_scores failed; falling back to per-query BM25: {e}")
                    bm25_batch = None

            if concurrent_queries <= 1:
                for i, q in enumerate(batch):
                    qid = str(q["query_id"])
                    res = self.process_query(q, precomputed_bm25=(bm25_batch[i] if bm25_batch else None))
                    with self._processed_lock:
                        self.processed_queries.add(qid)

                    if self.stats["queries_processed"] % checkpoint_interval == 0:
                        self._save_checkpoint()
                        self._log_stats(total)
            else:
                with ThreadPoolExecutor(max_workers=concurrent_queries) as ex:
                    futures = []
                    for i, q in enumerate(batch):
                        bm25_pre = (bm25_batch[i] if bm25_batch else None)
                        futures.append(ex.submit(self.process_query, q, bm25_pre))

                    for fut in as_completed(futures):
                        try:
                            res = fut.result()
                            with self._processed_lock:
                                self.processed_queries.add(str(res["query_id"]))

                            if self.stats["queries_processed"] % checkpoint_interval == 0:
                                self._save_checkpoint()
                                self._log_stats(total)
                        except Exception as e:
                            logger.error(f"Worker error: {e}")
                            continue

        self._save_checkpoint()
        self._log_stats(total)
        logger.info("✓ Agent run complete!")
        logger.info(f"  Qrels: {self.qrels_path}")
        logger.info(f"  Triples: {self.triples_path}")

    def _estimate_remaining(self, total_queries: int) -> str:
        avg = float(self.stats.get("avg_time_per_query_s", 0.0) or 0.0)
        if avg <= 0:
            return "calculating..."
        remaining = max(0, int(total_queries) - int(self.stats.get("queries_processed", 0)))
        seconds = remaining * avg
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"

    def _log_stats(self, total_queries: int) -> None:
        with self._stats_lock:
            qp = int(self.stats["queries_processed"])
            avg = float(self.stats["avg_time_per_query_s"])
            llm = int(self.stats["passages_judged"])
            pos = int(self.stats["positives_found"])
            neg = int(self.stats["hard_negatives_found"])
            remain = self._estimate_remaining(total_queries)

        logger.info(
            "\n"
            "================= PROGRESS =================\n"
            f"Queries processed: {qp:,}\n"
            f"Passages judged:   {llm:,}\n"
            f"Positives found:   {pos:,}\n"
            f"Hard negatives:    {neg:,}\n"
            f"Avg / query:       {avg:.3f}s\n"
            f"Est. remaining:    {remain}\n"
            "===========================================\n"
        )
