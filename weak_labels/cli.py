#!/usr/bin/env python3
"""Main CLI entrypoint for Weak Labels agent (config-driven).

Everything configurable is read from configs/base.yaml via cfg.raw.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger
from datasets import load_dataset
from tqdm import tqdm

from .config import load_config, validate_paths
from .chunker import PassageChunker
from .bm25_index import BM25Index
from .dense_encoder import DenseEncoder
from .llm_client import CachedLLMJudge
from .agent_runner import Agent


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


def main() -> None:
    load_dotenv()

    cfg = load_config()
    cfg_raw = cfg.raw

    setup_logging(cfg_raw["logging"])

    logger.info("=" * 60)
    logger.info("Weak Labels Agent")
    logger.info("=" * 60)

    if not validate_paths(cfg):
        logger.error("Path validation failed!")
        sys.exit(1)

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

    # ---------------------------
    # Passages file (chunk once)
    # ---------------------------
    passages_dir = Path(cfg_raw["paths"]["passages_dir"])
    passages_dir.mkdir(parents=True, exist_ok=True)
    passages_path = passages_dir / "corpus_passages.jsonl"

    if not passages_path.exists():
        logger.info("Chunking corpus into passages...")
        chunker = PassageChunker(
            chunk_size=int(cfg_raw["corpus"]["passage_tokens"]),
            stride=int(cfg_raw["corpus"]["passage_stride"]),
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

    # ---------------------------
    # Initialize components
    # ---------------------------
    logger.info("Initializing retrieval + judge components...")

    bm25 = BM25Index(
        passages_path,
        bm25_cfg=cfg_raw["bm25"],
        paths_cfg=cfg_raw["paths"],
    )

    dense = DenseEncoder(dense_cfg=cfg_raw["dense"])

    llm_judge = CachedLLMJudge(cfg_raw["llm"])

    agent = Agent(
        bm25=bm25,
        dense_encoder=dense,
        llm_judge=llm_judge,
        config=cfg_raw,
    )

    logger.info("Starting query processing...")
    agent.run(queries)

    logger.info("✓ Agent finished successfully")

    hf_cfg = cfg_raw.get("huggingface", {}) or {}
    if bool(hf_cfg.get("auto_push", False)):
        logger.info("Pushing outputs to Hugging Face Hub...")
        try:
            from .push_to_hf import push_to_hub

            push_to_hub(
                qrels_path=cfg_raw["output"]["qrels_path"],
                triples_path=cfg_raw["output"]["triples_path"],
                repo_id=hf_cfg["repo_id"],
                token=hf_cfg.get("token"),
                private=bool(hf_cfg.get("private", False)),
            )
        except Exception as e:
            logger.error(f"Failed to push to HuggingFace: {e}")
            logger.warning("Continuing anyway...")


if __name__ == "__main__":
    main()
