#!/usr/bin/env python3
"""Main CLI entrypoint for Weak Labels agent (config-driven).

Everything configurable is read from configs/base.yaml via cfg.raw.

Modes:
  - phase0_build_dense_index: build precomputed dense passage embeddings (one-time)
  - phase1: retrieval + prompt logging (no vLLM calls)
  - phase2: LLM judging from saved prompts (no retrieval; populates SQLite cache)
  - phase3_export: materialize binary qrels.tsv + triples.jsonl (generic cross-encoder-ready) from cache
    + OPTIONAL: if cfg_raw["huggingface"]["auto_push"] is true, automatically push outputs to HF Hub.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger
from datasets import load_dataset
from tqdm import tqdm
import sqlite3
import hashlib
import re

from .config import load_config, validate_paths
from .chunker import PassageChunker
from .bm25_index import BM25Index
from .dense_encoder import DenseEncoder
from .cross_encoder import CrossEncoderReranker
from .llm_client import CachedLLMJudge
from .agent_runner import Agent, RetrievalResult

# NEW: HF push utility (you must place the updated script as weak_labels/push_to_hf.py)
from .push_to_hf import push_to_hub


def setup_logging(logging_cfg: Dict[str, Any]) -> None:
    """Configure loguru from config."""
    log_dir = Path(logging_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / logging_cfg.get("log_file", "agent.log")
    level = logging_cfg.get("level", "INFO")

    rotation = logging_cfg["rotation"]
    retention = logging_cfg["retention"]

    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(
        str(log_file),
        rotation=rotation,
        retention=retention,
        level="DEBUG",
        enqueue=True,
    )


def _as_queries_list(cfg_raw: Dict[str, Any], queries_ds) -> List[Dict[str, str]]:
    q_cfg = cfg_raw["dataset"]["queries"]
    id_field = q_cfg["id_field"]
    text_field = q_cfg["text_field"]

    if id_field in queries_ds.column_names and text_field in queries_ds.column_names:
        ids = queries_ds[id_field]
        texts = queries_ds[text_field]
        out = []
        for qid, qtext in zip(ids, texts):
            if qid is None or qtext is None:
                continue
            out.append({"query_id": str(qid), "text": str(qtext)})
        return out

    out = []
    for item in tqdm(queries_ds, desc="Converting queries"):
        qid = item.get(id_field)
        qtext = item.get(text_field)
        if qid is None or qtext is None:
            continue
        out.append({"query_id": str(qid), "text": str(qtext)})
    return out


def _load_datasets(cfg_raw: Dict[str, Any]):
    """Load corpus + queries and apply max_queries."""
    logger.info("Loading datasets...")

    hf_cache_dir = cfg_raw["paths"]["hf_cache_dir"]

    corpus_ds = load_dataset(
        cfg_raw["dataset"]["corpus"]["name"],
        split=cfg_raw["dataset"]["corpus"]["split"],
        cache_dir=hf_cache_dir,
    )

    queries_ds = load_dataset(
        cfg_raw["dataset"]["queries"]["name"],
        split=cfg_raw["dataset"]["queries"]["split"],
        cache_dir=hf_cache_dir,
    )

    logger.info(f"✓ Corpus: {len(corpus_ds):,} documents")
    logger.info(f"✓ Queries: {len(queries_ds):,} queries")

    max_queries = cfg_raw["agent"].get("max_queries")
    if max_queries is not None:
        max_q = int(max_queries)
        queries_ds = queries_ds.select(range(min(max_q, len(queries_ds))))
        logger.warning(f"TEST MODE: Limiting to {len(queries_ds):,} queries")

    logger.info("Preparing queries...")
    queries = _as_queries_list(cfg_raw, queries_ds)
    logger.info(f"✓ Prepared {len(queries):,} queries")

    return corpus_ds, queries


def _ensure_passages(cfg_raw: Dict[str, Any], corpus_ds):
    """Create or reuse corpus_passages.jsonl."""
    passages_dir = Path(cfg_raw["paths"]["passages_dir"])
    passages_dir.mkdir(parents=True, exist_ok=True)
    passages_path = passages_dir / "corpus_passages.jsonl"

    if not passages_path.exists():
        logger.info("Chunking corpus into passages...")

        corpus_cfg = cfg_raw.get("corpus", {}) or {}
        chunker = PassageChunker(
            chunk_size=int(
                corpus_cfg.get("passage_tokens", corpus_cfg.get("chunk_size", 160))
            ),
            stride=int(
                corpus_cfg.get("passage_stride", corpus_cfg.get("stride", 80))
            ),
            min_chunk_tokens=int(corpus_cfg.get("min_chunk_tokens", 30)),
            min_alpha_ratio=float(corpus_cfg.get("min_alpha_ratio", 0.5)),
            dedupe_hash_length=int(corpus_cfg.get("dedupe_hash_length", 200)),
            use_semantic_dedup=bool(corpus_cfg.get("use_semantic_dedup", False)),
            semantic_threshold=float(corpus_cfg.get("semantic_threshold", 0.95)),
            device=str(corpus_cfg.get("device", "cuda")),
        )

        passages = chunker.chunk_corpus(corpus_ds)

        logger.info(f"Writing passages: {passages_path}")
        with passages_path.open("w") as f:
            for p in tqdm(passages, desc="Writing passages"):
                f.write(json.dumps(p) + "\n")

        logger.info(f"✓ Saved {len(passages):,} passages")
    else:
        with passages_path.open("r") as f:
            passage_count = sum(1 for _ in f)
        logger.info(f"✓ Using existing passages: {passage_count:,}")

    return passages_path


# ---------------------------------------------------------------------
# Phase 0: build dense passage index (one-time)
# ---------------------------------------------------------------------
def run_phase0_build_dense_index(cfg_raw: Dict[str, Any]) -> None:
    logger.info("== Phase 0: build dense passage index ==")

    corpus_ds, _ = _load_datasets(cfg_raw)
    passages_path = _ensure_passages(cfg_raw, corpus_ds)

    logger.info(f"Reading passages from {passages_path}")
    doc_ids: List[str] = []
    texts: List[str] = []
    with passages_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading passages for dense index"):
            if not line.strip():
                continue
            rec = json.loads(line)
            doc_id = rec.get("doc_id")
            text = rec.get("text")
            if doc_id is None or text is None:
                continue
            doc_ids.append(str(doc_id))
            texts.append(str(text))

    logger.info(f"Building dense index for {len(doc_ids):,} passages")

    dense = DenseEncoder(dense_cfg=cfg_raw["dense"])
    dense.build_corpus_index(doc_ids, texts)

    logger.info("✓ Phase 0 finished successfully - dense index built")


# ---------------------------------------------------------------------
# Phase 1: retrieval + prompt logging (no vLLM calls)
# ---------------------------------------------------------------------
def run_phase1(cfg_raw: Dict[str, Any]) -> None:
    logger.info("== Phase 1: retrieval + prompt logging (no vLLM) ==")

    corpus_ds, queries = _load_datasets(cfg_raw)
    passages_path = _ensure_passages(cfg_raw, corpus_ds)

    logger.info("Initializing retrieval components (BM25, dense, CE)...")

    bm25 = BM25Index(
        passages_path,
        bm25_cfg=cfg_raw["bm25"],
        paths_cfg=cfg_raw["paths"],
    )

    dense = DenseEncoder(dense_cfg=cfg_raw["dense"])

    cross_encoder_cfg = cfg_raw.get("cross_encoder", {}) or {}
    cross_encoder = (
        CrossEncoderReranker(ce_cfg=cross_encoder_cfg)
        if cfg_raw["agent"].get("use_cross_encoder", False)
        else None
    )

    agent = Agent(
        bm25=bm25,
        dense_encoder=dense,
        llm_judge=None,  # Phase 1: no LLM
        config=cfg_raw,
        cross_encoder=cross_encoder,
    )

    prompts_path = Path(
        cfg_raw["paths"].get("phase1_prompts_path", "data/prepared/judge_prompts.jsonl")
    )
    prompts_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing Phase 1 prompts to {prompts_path}")

    with prompts_path.open("w", encoding="utf-8") as f_out:
        for q in tqdm(queries, desc="Phase 1: queries"):
            rr: RetrievalResult = agent.retrieve_for_query(q)
            if not rr.candidates:
                continue
            prompt = agent.build_prompt(rr)

            record = {
                "query_id": rr.query_id,
                "query_text": rr.query_text,
                "candidates": [
                    {
                        "doc_id": c["doc_id"],
                        "text": c["text"],
                        "score": c.get("score"),
                        "rank": c.get("rank"),
                    }
                    for c in rr.candidates
                ],
                "prompt": prompt,
                "prompt_version": cfg_raw["llm"].get("prompt_version", "v1"),
                "guided_mode": cfg_raw["llm"]["guided"]["mode"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("✓ Phase 1 finished successfully")


# ---------------------------------------------------------------------
# Phase 2: LLM judging from saved prompts (vLLM only; populates cache)
# ---------------------------------------------------------------------
def run_phase2(cfg_raw: Dict[str, Any]) -> None:
    logger.info("== Phase 2: LLM judging from prompts ==")

    prompts_path = Path(
        cfg_raw["paths"].get("phase1_prompts_path", "data/prepared/judge_prompts.jsonl")
    )
    if not prompts_path.exists():
        logger.error(f"Phase 1 prompts file not found: {prompts_path}")
        sys.exit(1)

    llm_judge = CachedLLMJudge(cfg_raw["llm"])

    logger.info(f"Reading prompts from {prompts_path}")
    with prompts_path.open("r", encoding="utf-8") as f_in:
        for line in tqdm(f_in, desc="Phase 2: prompts"):
            if not line.strip():
                continue
            rec = json.loads(line)

            query_text = str(rec.get("query_text") or rec.get("query") or "")
            candidates = rec.get("candidates") or []

            if not query_text or not candidates:
                continue

            _ = llm_judge.judge_batch(query_text, candidates)

    logger.info("✓ Phase 2 finished successfully")


def run_phase3_export(cfg_raw: Dict[str, Any]) -> None:
    """
    STRICT OFFLINE export (no vLLM calls).

    Phase-3-only quality improvements:
      - Rank-windowed negative sampling within judged-NO candidates (top_k=80)
      - Avoid very top ranks as negatives (reduce borderline false negatives)
      - Avoid tail junk by sampling from mid ranks
      - Per-positive uniqueness of negatives inside a query
      - Optional near-duplicate filtering (cheap) to avoid template/boilerplate duplicates
      - Optional negative text quality filtering (min chars / alpha ratio)
      - NEW: POSDOC-SIGNATURE + IDF-aware overlap gate (dynamic across topics)
    """
    logger.info("== Phase 3: export qrels + triples from cache (OFFLINE, bulk SQLite) ==")

    prompts_path = Path(
        cfg_raw["paths"].get("phase1_prompts_path", "data/prepared/judge_prompts.jsonl")
    )
    if not prompts_path.exists():
        logger.error(f"Prompts file not found: {prompts_path}")
        sys.exit(1)

    qrels_path = Path(cfg_raw["output"]["qrels_path"])
    triples_path = Path(cfg_raw["output"]["triples_path"])
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    triples_path.parent.mkdir(parents=True, exist_ok=True)

    agent_cfg = cfg_raw.get("agent") or {}

    qrels_positives_max = int(agent_cfg.get("positives_max", 10))
    triplet_positives_max = int(agent_cfg.get("triplet_positives_max", 2))
    negs_per_positive = int(agent_cfg.get("negs_per_positive", 8))
    hard_negs_cap = int(agent_cfg.get("hard_negatives_per_query", 40))

    add_semi_hard = bool(agent_cfg.get("add_semi_hard", True))
    semi_hard_per_positive = int(agent_cfg.get("semi_hard_per_positive", 0))
    listwise = bool(agent_cfg.get("export_listwise", False))

    # Rank-windowing (tuned for top_k=80)
    min_hard_rank = int(agent_cfg.get("min_hard_rank", 8))
    max_hard_rank = int(agent_cfg.get("max_hard_rank", 50))
    min_semi_rank = int(agent_cfg.get("min_semi_rank", 51))
    max_semi_rank = int(agent_cfg.get("max_semi_rank", 80))

    # Optional near-duplicate filter
    enable_near_dup_filter = bool(agent_cfg.get("enable_near_dup_filter", True))
    near_dup_ngrams = int(agent_cfg.get("near_dup_ngrams", 5))
    near_dup_threshold = float(agent_cfg.get("near_dup_threshold", 0.92))
    near_dup_max_chars = int(agent_cfg.get("near_dup_max_chars", 1200))

    # Optional negative text quality filters
    enable_neg_quality_filter = bool(agent_cfg.get("enable_neg_quality_filter", True))
    min_neg_chars = int(agent_cfg.get("min_neg_chars", 120))
    min_neg_alpha_ratio = float(agent_cfg.get("min_neg_alpha_ratio", 0.55))

    # --- NEW: IDF + posdoc signature negative gate (dynamic; no stopword list) ---
    enable_signature_gate = bool(agent_cfg.get("enable_signature_gate", True))
    idf_path = str(agent_cfg.get("idf_path", "artifacts/idf.json"))
    idf_default = float(agent_cfg.get("idf_default", 0.0))

    sig_top_k = int(agent_cfg.get("signature_top_k", 12))
    sig_token_min_len = int(agent_cfg.get("signature_token_min_len", 5))
    sig_idf_min = float(agent_cfg.get("signature_idf_min", 4.5))
    sig_min_terms = int(agent_cfg.get("signature_min_terms", 4))
    sig_fallback_idf_min = float(agent_cfg.get("signature_fallback_idf_min", 3.5))

    min_shared_signature_terms = int(agent_cfg.get("min_shared_signature_terms", 1))

    llm_cfg = cfg_raw.get("llm") or {}
    model = str(llm_cfg.get("model", ""))
    guided_mode = str((llm_cfg.get("guided") or {}).get("mode", "choice")).lower()
    PROMPT_VERSION = "judge_yesno_v5_guided_params_retry_http"  # from your llm_client.py

    cache_db_path = str(
        ((llm_cfg.get("cache") or {}).get("db_path")) or "data/prepared/llm_cache.db"
    )
    cache_db_path = str(Path(cache_db_path))

    _WORD_RE = re.compile(r"[A-Za-z0-9]+")

    def _tokenize(text: str) -> List[str]:
        return _WORD_RE.findall((text or "").lower())

    def _page_id(doc_id: str) -> str:
        s = doc_id or ""
        return s.rsplit("_P", 1)[0] if "_P" in s else s

    def _key_hash_for_passage(query: str, doc_id: str, text: str) -> str:
        key_str = f"{query}\nDOC_ID={doc_id}\nTEXT={text}"
        s = f"{model}|{PROMPT_VERSION}|{guided_mode}|{key_str}"
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

    def _alpha_ratio(text: str) -> float:
        if not text:
            return 0.0
        a = sum(ch.isalpha() for ch in text)
        return a / float(len(text))

    def _near_dup_score(a: str, b: str) -> float:
        """Cheap similarity proxy using hashed character ngrams (Jaccard)."""
        if not a or not b:
            return 0.0
        a = a[:near_dup_max_chars]
        b = b[:near_dup_max_chars]
        n = max(3, int(near_dup_ngrams))

        def grams(s: str) -> set[int]:
            s = re.sub(r"\s+", " ", s.strip().lower())
            if len(s) <= n:
                return {hash(s)}
            out = set()
            for i in range(0, len(s) - n + 1):
                out.add(hash(s[i : i + n]))
            return out

        ga = grams(a)
        gb = grams(b)
        if not ga or not gb:
            return 0.0
        inter = len(ga & gb)
        union = len(ga | gb)
        return inter / float(union) if union else 0.0

    def _load_idf_map(path: str) -> Dict[str, float]:
        p = Path(path)
        if not p.exists():
            logger.error(
                f"IDF map not found at {p}. "
                "Build artifacts/idf.json (token->idf float) or disable_signature_gate."
            )
            sys.exit(1)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    idf_map: Dict[str, float] = {}
    if enable_signature_gate:
        idf_map = _load_idf_map(idf_path)

    def _build_signature(postext: str) -> set[str]:
        toks = [t for t in _tokenize(postext) if len(t) >= sig_token_min_len]
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1

        def _score(tok: str) -> float:
            return float(tf.get(tok, 0)) * float(idf_map.get(tok, idf_default))

        def _make(threshold: float) -> List[str]:
            items = []
            for tok in tf.keys():
                if float(idf_map.get(tok, idf_default)) >= threshold:
                    items.append((tok, _score(tok)))
            items.sort(key=lambda x: x[1], reverse=True)
            return [tok for tok, _ in items[:sig_top_k]]

        sig = _make(sig_idf_min)
        if len(sig) < sig_min_terms:
            sig = _make(sig_fallback_idf_min)
        return set(sig)

    # ------------------------------------------------------------------
    # SQLite read-only + speed PRAGMAs
    # ------------------------------------------------------------------
    db_uri = f"file:{cache_db_path}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -524288")

    def _fetch_answerable_for_pairs(pairs: List[tuple[str, str]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        if not pairs:
            return out
        CHUNK = 400
        for i in range(0, len(pairs), CHUNK):
            chunk = pairs[i : i + CHUNK]
            where = " OR ".join(["(key_hash=? AND doc_id=?)"] * len(chunk))
            params: List[str] = []
            for kh, did in chunk:
                params.extend([kh, did])
            rows = conn.execute(
                f"SELECT key_hash, doc_id, answerable FROM judgments WHERE {where}",
                params,
            ).fetchall()
            for kh, did, ans in rows:
                out[f"{kh}|{did}"] = int(ans)
        return out

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    n_queries = 0
    n_qrels = 0
    n_triples = 0
    n_missing = 0
    n_no_hard_pool = 0
    n_dropped_near_dup = 0
    n_dropped_quality = 0
    n_dropped_sig = 0

    with qrels_path.open("w", encoding="utf-8") as fq, triples_path.open(
        "w", encoding="utf-8"
    ) as ft:
        with prompts_path.open("r", encoding="utf-8") as f_in:
            for line in tqdm(f_in, desc="Phase 3: export (offline)"):
                if not line.strip():
                    continue
                rec = json.loads(line)

                qid = str(rec.get("query_id") or "")
                qtext = str(rec.get("query_text") or "")
                candidates = rec.get("candidates") or []
                if not qid or not qtext or not candidates:
                    continue

                n_queries += 1

                cand_ids: List[str] = []
                text_map: Dict[str, str] = {}
                rank_map: Dict[str, int] = {}
                pairs: List[tuple[str, str]] = []

                for c in candidates:
                    did = str(c.get("doc_id") or "")
                    txt = str(c.get("text") or "")
                    if not did:
                        continue
                    text_map[did] = txt
                    rank_map[did] = int(c.get("rank") or 10**9)
                    kh = _key_hash_for_passage(qtext, did, txt)
                    pairs.append((kh, did))
                    cand_ids.append(did)

                ans_map = _fetch_answerable_for_pairs(pairs)

                pos_ids: List[str] = []
                neg_ids: List[str] = []
                for did in cand_ids:
                    kh = _key_hash_for_passage(qtext, did, text_map.get(did, ""))
                    key = f"{kh}|{did}"
                    if key not in ans_map:
                        n_missing += 1
                        continue
                    if ans_map[key] == 1:
                        pos_ids.append(did)
                    else:
                        neg_ids.append(did)

                pos_ids.sort(key=lambda d: rank_map.get(d, 10**9))
                neg_ids.sort(key=lambda d: rank_map.get(d, 10**9))

                # qrels export
                for pid in pos_ids[:qrels_positives_max]:
                    fq.write(f"{qid}\t0\t{pid}\t1\n")
                    n_qrels += 1

                if not pos_ids or not neg_ids:
                    continue

                triplet_pos = pos_ids[:triplet_positives_max]
                neg_ids = neg_ids[:hard_negs_cap]

                # Rank-windowed pools
                hard_pool = [
                    d
                    for d in neg_ids
                    if min_hard_rank <= rank_map.get(d, 10**9) <= max_hard_rank
                ]
                semi_pool = [
                    d
                    for d in neg_ids
                    if min_semi_rank <= rank_map.get(d, 10**9) <= max_semi_rank
                ]

                if not hard_pool:
                    hard_pool = [
                        d for d in neg_ids if rank_map.get(d, 10**9) >= min_hard_rank
                    ]
                    if not hard_pool:
                        n_no_hard_pool += 1
                        continue

                used_negs: set[str] = set()
                dropped_near_dup_local = 0
                dropped_quality_local = 0
                dropped_sig_local = 0

                def _eligible(
                    pool: List[str], pid_page: str, pos_text: str, signature: set[str]
                ) -> List[str]:
                    nonlocal dropped_near_dup_local, dropped_quality_local, dropped_sig_local
                    out: List[str] = []
                    for nid in pool:
                        if nid in used_negs:
                            continue
                        if _page_id(nid) == pid_page:
                            continue

                        neg_text = text_map.get(nid, "")

                        if enable_neg_quality_filter:
                            if len(neg_text) < min_neg_chars or _alpha_ratio(
                                neg_text
                            ) < min_neg_alpha_ratio:
                                dropped_quality_local += 1
                                continue

                        if enable_signature_gate and signature:
                            n_terms = set(_tokenize(neg_text))
                            if len(signature & n_terms) < min_shared_signature_terms:
                                dropped_sig_local += 1
                                continue

                        if enable_near_dup_filter and pos_text and neg_text:
                            if _near_dup_score(pos_text, neg_text) >= near_dup_threshold:
                                dropped_near_dup_local += 1
                                continue

                        out.append(nid)
                    return out

                for p_i, pid in enumerate(triplet_pos):
                    pid_page = _page_id(pid)
                    pos_text = text_map.get(pid, "")

                    signature: set[str] = set()
                    if enable_signature_gate:
                        signature = _build_signature(pos_text)

                    hard_elig = _eligible(hard_pool, pid_page, pos_text, signature)
                    if not hard_elig:
                        hard_elig = [nid for nid in hard_pool if _page_id(nid) != pid_page]
                        if not hard_elig:
                            continue

                    k = max(1, negs_per_positive)
                    start = (p_i * 7) % len(hard_elig)
                    rotated = hard_elig[start:] + hard_elig[:start]
                    picked_hard = rotated[:k]
                    for nid in picked_hard:
                        used_negs.add(nid)

                    picked_semi: List[str] = []
                    if add_semi_hard and semi_hard_per_positive > 0 and semi_pool:
                        semi_elig = _eligible(semi_pool, pid_page, pos_text, signature)
                        if semi_elig:
                            sk = min(semi_hard_per_positive, len(semi_elig))
                            sstart = (p_i * 13) % len(semi_elig)
                            srot = semi_elig[sstart:] + semi_elig[:sstart]
                            picked_semi = srot[:sk]
                            for nid in picked_semi:
                                used_negs.add(nid)

                    if listwise:
                        ft.write(
                            json.dumps(
                                {
                                    "query_id": qid,
                                    "query": qtext,
                                    "pos_doc_id": pid,
                                    "pos_text": pos_text,
                                    "negatives": (
                                        [
                                            {
                                                "doc_id": nid,
                                                "text": text_map.get(nid, ""),
                                                "kind": "hard",
                                            }
                                            for nid in picked_hard
                                        ]
                                        + [
                                            {
                                                "doc_id": nid,
                                                "text": text_map.get(nid, ""),
                                                "kind": "semi-hard",
                                            }
                                            for nid in picked_semi
                                        ]
                                    ),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n_triples += 1
                    else:
                        for nid in picked_hard:
                            ft.write(
                                json.dumps(
                                    {
                                        "query_id": qid,
                                        "query": qtext,
                                        "pos_doc_id": pid,
                                        "pos_text": pos_text,
                                        "neg_doc_id": nid,
                                        "neg_text": text_map.get(nid, ""),
                                        "neg_kind": "hard",
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            n_triples += 1

                        for nid in picked_semi:
                            ft.write(
                                json.dumps(
                                    {
                                        "query_id": qid,
                                        "query": qtext,
                                        "pos_doc_id": pid,
                                        "pos_text": pos_text,
                                        "neg_doc_id": nid,
                                        "neg_text": text_map.get(nid, ""),
                                        "neg_kind": "semi-hard",
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            n_triples += 1

                n_dropped_near_dup += dropped_near_dup_local
                n_dropped_quality += dropped_quality_local
                n_dropped_sig += dropped_sig_local

    conn.close()

    logger.info(f"✓ Exported {n_queries:,} queries")
    logger.info(f"✓ Wrote {n_qrels:,} qrels lines to {qrels_path}")
    logger.info(f"✓ Wrote triples/listwise records: {n_triples:,} to {triples_path}")
    if n_missing:
        logger.warning(
            f"⚠ Missing cached judgments for {n_missing:,} (cache-key mismatch or incomplete Phase 2)"
        )
    if n_no_hard_pool:
        logger.warning(
            f"⚠ Queries skipped due to empty hard pool after rank-windowing: {n_no_hard_pool:,}"
        )
    if n_dropped_near_dup:
        logger.info(f"Dropped near-duplicate negatives: {n_dropped_near_dup:,}")
    if n_dropped_quality:
        logger.info(f"Dropped low-quality negatives: {n_dropped_quality:,}")
    if n_dropped_sig:
        logger.info(f"Dropped signature-gate negatives: {n_dropped_sig:,}")


# ----------------------------
# NEW: Phase 3 -> HF push glue
# ----------------------------
def _phase3_outputs_ready(cfg_raw: Dict[str, Any]) -> tuple[Path, Path]:
    qrels_path = Path(cfg_raw["output"]["qrels_path"])
    triples_path = Path(cfg_raw["output"]["triples_path"])

    if (not qrels_path.exists()) or qrels_path.stat().st_size == 0:
        raise RuntimeError(f"Phase 3 incomplete: missing/empty qrels: {qrels_path}")
    if (not triples_path.exists()) or triples_path.stat().st_size == 0:
        raise RuntimeError(f"Phase 3 incomplete: missing/empty triples: {triples_path}")

    # Cheap format checks:
    with qrels_path.open("r", encoding="utf-8") as f:
        first = f.readline().rstrip("\n")
        parts = first.split("\t")
        if len(parts) != 4:
            raise RuntimeError(
                f"Bad qrels first line (expected 4 TSV cols): {first[:200]}"
            )

    with triples_path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first.strip():
            raise RuntimeError("Bad triples.jsonl: first line is blank")
        json.loads(first)  # validates first record is valid JSON

    return qrels_path, triples_path


def run_phase3_export_and_maybe_push(cfg_raw: Dict[str, Any]) -> None:
    run_phase3_export(cfg_raw)

    hf_cfg = cfg_raw.get("huggingface") or {}
    if not bool(hf_cfg.get("auto_push", False)):
        logger.info("HF auto_push disabled; skipping Hugging Face upload.")
        return

    repo_id = str(hf_cfg.get("repo_id", "")).strip()
    if not repo_id:
        raise RuntimeError("huggingface.auto_push is true but huggingface.repo_id is empty")

    private = bool(hf_cfg.get("private", False))
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var is not set; cannot push to Hugging Face")

    qrels_path, triples_path = _phase3_outputs_ready(cfg_raw)

    logger.info(f"Phase 3 done; pushing outputs to Hugging Face Hub: {repo_id}")

    # Recommended defaults for large exports: upload raw files only.
    push_to_hub(
        qrels_path=str(qrels_path),
        triples_path=str(triples_path),
        repo_id=repo_id,
        token=token,
        private=private,
        upload_raw=True,
        push_datasets=False,
        commit_message="Phase 3 export: qrels.tsv + triples.jsonl",
    )

    logger.info("✓ HF push completed")


def main() -> None:
    load_dotenv()

    mode = "phase1" if len(sys.argv) < 2 else sys.argv[1]
    if mode not in {"phase0_build_dense_index", "phase1", "phase2", "phase3_export"}:
        print(
            "Usage: python -m weak_labels.cli "
            "[phase0_build_dense_index|phase1|phase2|phase3_export]"
        )
        sys.exit(1)

    cfg = load_config()
    cfg_raw = cfg.raw

    setup_logging(cfg_raw["logging"])

    logger.info("=" * 60)
    logger.info(f"Weak Labels Agent - mode={mode}")
    logger.info("=" * 60)

    if not validate_paths(cfg):
        logger.error("Path validation failed!")
        sys.exit(1)

    if mode == "phase0_build_dense_index":
        run_phase0_build_dense_index(cfg_raw)
    elif mode == "phase1":
        run_phase1(cfg_raw)
    elif mode == "phase2":
        run_phase2(cfg_raw)
    else:
        # phase3_export
        run_phase3_export_and_maybe_push(cfg_raw)


if __name__ == "__main__":
    main()
