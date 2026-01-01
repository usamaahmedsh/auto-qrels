"""
weak_labels/agent_runner.py

Config-driven agent runner (NO hidden knobs).

Design goals:
- All tunables come from config dict (later configs/base.yaml).
- Multi-GPU support for embeddings, cross-encoder, and LLM.
- Uses BM25 -> dense rerank -> cross-encoder -> LLM judge -> write qrels/triples.
- Checkpoint/resume.
- Optional parallelism across queries (ThreadPoolExecutor), controlled by config.

Expected config keys:

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
  cross_encoder_top_k: int
  use_cross_encoder: bool

  # LLM candidate selection / labeling
  llm_candidates_top_k: int
  llm_conf_threshold: float
  positives_max: int
  hard_negatives_per_query: int

  # filtering
  min_passage_tokens: int
  max_passages_per_page: int

  # GPU acceleration
  device: str  # "cuda" or "cpu"
  dense_batch_size: int
  cross_encoder_batch_size: int

cross_encoder:
  model_name: str
  max_length: int
  device: str  # "cuda" or "cpu"
  batch_size: int

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
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import torch

from .bm25_index import BM25Index
from .dense_encoder import DenseEncoder
from .llm_client import CachedLLMJudge
from .cross_encoder import CrossEncoder


DEFAULT_AGENT_CFG: Dict[str, Any] = {
    "batch_size": 1024,
    "concurrent_queries": 8,
    "checkpoint_dir": "data/checkpoints",
    "checkpoint_interval": 50,
    "max_queries": None,
    "global_top_k_bm25": 500,
    "dense_top_k_from_bm25": 200,
    "cross_encoder_top_k": 50,
    "use_cross_encoder": True,
    "llm_candidates_top_k": 30,
    "llm_conf_threshold": 0.85,
    "positives_max": 10,
    "hard_negatives_per_query": 25,
    "min_passage_tokens": 30,
    "max_passages_per_page": 3,
    "device": "cuda",
    "dense_batch_size": 256,
    "cross_encoder_batch_size": 128,
}

DEFAULT_CROSS_ENCODER_CFG: Dict[str, Any] = {
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "max_length": 512,
    "device": "cuda",
    "batch_size": 128,
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
    return len((s or "").split())


def _page_id_from_doc_id(doc_id: str) -> str:
    s = doc_id or ""
    if "_P" in s:
        return s.rsplit("_P", 1)[0]
    return s


@dataclass
class RetrievalResult:
    query_id: str
    query_text: str
    candidates: List[Dict[str, Any]]  # [{doc_id, text, score, rank}, ...]


class Agent:
    def __init__(
        self,
        bm25: BM25Index,
        dense_encoder: DenseEncoder,
        llm_judge: Optional[CachedLLMJudge],
        config: Dict[str, Any],
        cross_encoder: Optional[CrossEncoder] = None,
    ):
        self.bm25 = bm25
        self.dense_encoder = dense_encoder
        self.llm_judge = llm_judge
        self.cross_encoder = cross_encoder

        self.cfg_raw = config
        self.agent_cfg = _deep_update(DEFAULT_AGENT_CFG, (config.get("agent") or {}))
        self.ce_cfg = _deep_update(DEFAULT_CROSS_ENCODER_CFG, (config.get("cross_encoder") or {}))

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
            "bm25_time_s": 0.0,
            "dense_time_s": 0.0,
            "cross_encoder_time_s": 0.0,
            "llm_time_s": 0.0,
            "total_time_s": 0.0,
            "avg_time_per_query_s": 0.0,
        }

        self.processed_queries = self._load_checkpoint()
        
        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✓ GPU: {gpu_count}x {gpu_name} ({gpu_mem:.1f}GB each)")
        
        logger.info(f"✓ Agent initialized (checkpoint: {len(self.processed_queries):,} queries)")

        # Build fast doc_id -> text mapping
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

        # Try to load precomputed dense index (for Phase 1)
        if self.dense_encoder.load_corpus_index():
            logger.info("✓ DenseEncoder: using precomputed corpus index for retrieval")
        else:
            logger.warning("DenseEncoder: no precomputed index loaded; falling back to per-query encoding")

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

        negs: List[str] = []
        for j in judged:
            did = j["doc_id"]
            if did in pos_ids:
                continue
            negs.append(did)
            if len(negs) >= num_neg:
                break
        return negs

    # -----------------------------
    # Retrieval-only helpers (Phase 1) – dense index aware
    # -----------------------------
    def retrieve_for_query(
        self,
        query_rec: Dict[str, Any],
        precomputed_bm25: Optional[List[Tuple[str, float]]] = None,
    ) -> RetrievalResult:
        """Phase-1 helper: BM25 -> dense index search (if available) -> optional CE."""
        qid = str(query_rec["query_id"])
        qtext = str(query_rec["text"])

        # BM25
        bm25_results = precomputed_bm25
        if bm25_results is None:
            bm25_results = self.bm25.search_with_scores(
                qtext, top_k=int(self.agent_cfg["global_top_k_bm25"])
            )
        if not bm25_results:
            return RetrievalResult(query_id=qid, query_text=qtext, candidates=[])

        # BM25 candidates + texts
        bm25_candidates = self._lookup_candidates_text(bm25_results)
        if not bm25_candidates:
            return RetrievalResult(query_id=qid, query_text=qtext, candidates=[])

        # Dense step:
        # If index loaded: full-corpus dense search, then intersect with BM25 IDs.
        # Else: fallback to per-query rerank on BM25 candidates.
        if self.dense_encoder.has_index():
            dense_top = self.dense_encoder.search(
                qtext,
                top_k=int(self.agent_cfg["dense_top_k_from_bm25"]),
            )
            dense_ids = [did for did, _ in dense_top]
            bm25_id_set = set(bm25_candidates.keys())
            dense_top_ids = [did for did in dense_ids if did in bm25_id_set]
        else:
            dense_top_ids = self.dense_encoder.rerank(
                qtext,
                bm25_candidates,
                top_k=int(self.agent_cfg["dense_top_k_from_bm25"]),
                batch_size=int(self.agent_cfg["dense_batch_size"]),
            )

        # Cross-encoder rerank (optional)
        if self.agent_cfg.get("use_cross_encoder") and self.cross_encoder is not None:
            ce_candidates = {
                did: bm25_candidates[did] for did in dense_top_ids if did in bm25_candidates
            }
            final_ranked = self.cross_encoder.rerank(
                qtext,
                ce_candidates,
                top_k=int(self.agent_cfg["cross_encoder_top_k"]),
            )
        else:
            final_ranked = dense_top_ids

        # Final filtered candidates (LLM input)
        llm_candidates = self._select_llm_candidates(final_ranked, bm25_candidates)

        ranked_candidates: List[Dict[str, Any]] = []
        for rank, p in enumerate(llm_candidates, start=1):
            did = p["doc_id"]
            ranked_candidates.append(
                {
                    "doc_id": did,
                    "text": p["text"],
                    "score": None,
                    "rank": rank,
                }
            )

        return RetrievalResult(
            query_id=qid,
            query_text=qtext,
            candidates=ranked_candidates,
        )

    def build_prompt(self, result: RetrievalResult) -> str:
        if not result.candidates:
            return f"Question: {result.query_text}\n\nNo candidate passages found."

        lines: List[str] = []
        lines.append(
            "You are a relevance judge. For each passage, decide if it is relevant to the query."
        )
        lines.append("")
        lines.append(f"Query: {result.query_text}")
        lines.append("")
        lines.append("Passages:")

        for c in result.candidates:
            doc_id = c["doc_id"]
            text = c["text"]
            lines.append(f"- [DOC_ID={doc_id}] {text}")

        lines.append("")
        lines.append(
            "For each passage, output a JSON array of objects, one per passage, "
            'with fields {"doc_id": "...", "label": "YES" or "NO"}.'
        )

        return "\n".join(lines)

    # -----------------------------
    # Full pipeline for a single query (Phase 1+2, legacy)
    # -----------------------------
    def process_query(self, query_rec: Dict[str, Any], precomputed_bm25: Optional[List[Tuple[str, float]]] = None) -> Dict[str, Any]:
        qid = str(query_rec["query_id"])
        qtext = str(query_rec["text"])

        start = time.time()
        stage_times = {}

        # BM25
        bm25_start = time.time()
        bm25_results = precomputed_bm25
        if bm25_results is None:
            bm25_results = self.bm25.search_with_scores(qtext, top_k=int(self.agent_cfg["global_top_k_bm25"]))
        stage_times["bm25"] = time.time() - bm25_start

        if not bm25_results:
            return {"query_id": qid, "positives": 0, "hard_negatives": 0, "elapsed_s": time.time() - start}

        # Candidate texts
        candidates_dict = self._lookup_candidates_text(bm25_results)
        if not candidates_dict:
            return {"query_id": qid, "positives": 0, "hard_negatives": 0, "elapsed_s": time.time() - start}

        # Dense rerank (legacy path still uses per-query rerank)
        dense_start = time.time()
        dense_top_ids: List[str] = self.dense_encoder.rerank(
            qtext,
            candidates_dict,
            top_k=int(self.agent_cfg["dense_top_k_from_bm25"]),
            batch_size=int(self.agent_cfg["dense_batch_size"]),
        )
        stage_times["dense"] = time.time() - dense_start

        # Cross-encoder rerank (optional)
        if self.agent_cfg.get("use_cross_encoder") and self.cross_encoder is not None:
            ce_start = time.time()
            ce_candidates = {did: candidates_dict[did] for did in dense_top_ids if did in candidates_dict}
            cross_encoder_top_ids = self.cross_encoder.rerank(
                qtext,
                ce_candidates,
                top_k=int(self.agent_cfg["cross_encoder_top_k"]),
            )
            stage_times["cross_encoder"] = time.time() - ce_start
            final_ranked = cross_encoder_top_ids
        else:
            stage_times["cross_encoder"] = 0.0
            final_ranked = dense_top_ids

        # Choose LLM candidates
        llm_candidates = self._select_llm_candidates(final_ranked, candidates_dict)
        if not llm_candidates or self.llm_judge is None:
            return {"query_id": qid, "positives": 0, "hard_negatives": 0, "elapsed_s": time.time() - start}

        # LLM judge
        llm_start = time.time()
        judged = self.llm_judge.judge_batch(qtext, llm_candidates)
        stage_times["llm"] = time.time() - llm_start

        with self._stats_lock:
            self.stats["passages_judged"] += len(judged)
            self.stats["bm25_time_s"] += stage_times["bm25"]
            self.stats["dense_time_s"] += stage_times["dense"]
            self.stats["cross_encoder_time_s"] += stage_times["cross_encoder"]
            self.stats["llm_time_s"] += stage_times["llm"]

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
        logger.info(f"Config: batch_size={batch_size}, concurrent={concurrent_queries}, device={self.agent_cfg['device']}")

        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start : batch_start + batch_size]

            # Batch BM25 if available
            bm25_batch: Optional[List[List[Tuple[str, float]]]] = None
            if hasattr(self.bm25, "batch_search_with_scores"):
                try:
                    bm25_batch = self.bm25.batch_search_with_scores(
                        [q["text"] for q in batch],
                        top_k=int(self.agent_cfg["global_top_k_bm25"]),
                    )
                except Exception as e:
                    logger.warning(f"BM25 batch_search_with_scores failed; falling back: {e}")
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
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"

    def _log_stats(self, total_queries: int) -> None:
        with self._stats_lock:
            qp = int(self.stats["queries_processed"])
            avg = float(self.stats["avg_time_per_query_s"])
            llm = int(self.stats["passages_judged"])
            pos = int(self.stats["positives_found"])
            neg = int(self.stats["hard_negatives_found"])
            remain = self._estimate_remaining(total_queries)
            
            bm25_time = self.stats.get("bm25_time_s", 0.0)
            dense_time = self.stats.get("dense_time_s", 0.0)
            ce_time = self.stats.get("cross_encoder_time_s", 0.0)
            llm_time = self.stats.get("llm_time_s", 0.0)
            
            avg_bm25 = bm25_time / max(1, qp)
            avg_dense = dense_time / max(1, qp)
            avg_ce = ce_time / max(1, qp)
            avg_llm = llm_time / max(1, qp)

        logger.info(
            "\n"
            "================= PROGRESS =================\n"
            f"Queries processed: {qp:,} / {total_queries:,}\n"
            f"Passages judged:   {llm:,}\n"
            f"Positives found:   {pos:,}\n"
            f"Hard negatives:    {neg:,}\n"
            f"Avg / query:       {avg:.3f}s\n"
            f"  - BM25:          {avg_bm25:.3f}s\n"
            f"  - Dense:         {avg_dense:.3f}s\n"
            f"  - Cross-Enc:     {avg_ce:.3f}s\n"
            f"  - LLM:           {avg_llm:.3f}s\n"
            f"Est. remaining:    {remain}\n"
            "===========================================\n"
        )
