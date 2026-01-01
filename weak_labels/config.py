"""
weak_labels/config.py

Config loader + validation for Weak Labels.

Policy:
- No logging configuration at import time (avoids double handlers + hardcoded paths).
- Validation enforces sections and the specific keys used by the rewritten modules.
- YAML parsing uses yaml.safe_load().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger


@dataclass
class Config:
    raw: Dict[str, Any]
    path: Path

    # Convenience accessors
    dataset: Dict[str, Any]
    paths: Dict[str, Any]
    corpus: Dict[str, Any]
    bm25: Dict[str, Any]
    dense: Dict[str, Any]
    llm: Dict[str, Any]
    agent: Dict[str, Any]
    output: Dict[str, Any]
    logging: Dict[str, Any]
    huggingface: Dict[str, Any]


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key: {ctx}.{key}")
    return d[key]


def _require_str(d: Dict[str, Any], key: str, ctx: str) -> str:
    v = _require(d, key, ctx)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Invalid {ctx}.{key}: expected non-empty string")
    return v


def _require_num(d: Dict[str, Any], key: str, ctx: str) -> float:
    v = _require(d, key, ctx)
    if not isinstance(v, (int, float)):
        raise ValueError(f"Invalid {ctx}.{key}: expected number")
    return float(v)


def _require_int(d: Dict[str, Any], key: str, ctx: str) -> int:
    v = _require(d, key, ctx)
    if not isinstance(v, int):
        raise ValueError(f"Invalid {ctx}.{key}: expected int")
    return int(v)


def load_config(config_path: str = "configs/base.yaml") -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping/dict")

    _validate(raw)

    return Config(
        raw=raw,
        path=path,
        dataset=raw["dataset"],
        paths=raw["paths"],
        corpus=raw["corpus"],
        bm25=raw["bm25"],
        dense=raw["dense"],
        llm=raw["llm"],
        agent=raw["agent"],
        output=raw["output"],
        logging=raw["logging"],
        huggingface=raw.get("huggingface", {}) or {},
    )


def _validate(raw: Dict[str, Any]) -> None:
    # Top-level sections required by rewritten code
    required_sections = [
        "dataset",
        "paths",
        "corpus",
        "bm25",
        "dense",
        "llm",
        "agent",
        "output",
        "logging",
    ]
    missing = [s for s in required_sections if s not in raw]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    # dataset.*
    ds = raw["dataset"]
    _require_str(ds, "name", "dataset")

    for sub in ["corpus", "queries"]:
        sec = _require(ds, sub, "dataset")
        if not isinstance(sec, dict):
            raise ValueError(f"dataset.{sub} must be a dict")
        _require_str(sec, "name", f"dataset.{sub}")
        _require_str(sec, "split", f"dataset.{sub}")
        _require_str(sec, "text_field", f"dataset.{sub}")
        _require_str(sec, "id_field", f"dataset.{sub}")

    # paths.*
    paths = raw["paths"]
    for k in ["prepared_dir", "hf_cache_dir", "passages_dir", "indexes_dir"]:
        _require_str(paths, k, "paths")

    # corpus.* (chunker knobs)
    corpus = raw["corpus"]
    _require_int(corpus, "passage_tokens", "corpus")
    _require_int(corpus, "passage_stride", "corpus")
    # Optional: min_chunk_tokens, min_alpha_ratio, dedupe_hash_length,
    # use_semantic_dedup, semantic_threshold, device

    # bm25.*
    bm25 = raw["bm25"]
    _require_num(bm25, "k1", "bm25")
    _require_num(bm25, "b", "bm25")
    # Optional: method, stopwords, use_stemmer, mmap, load_corpus, index_name,
    # num_threads, batch_size, passage_text_field, passage_id_field

    # dense.*
    dense = raw["dense"]
    _require_str(dense, "model_name", "dense")
    _require(dense, "device", "dense")       # str or list[str]
    _require_int(dense, "batch_size", "dense")
    # Optional: normalize_embeddings, cache_queries, max_query_cache_size,
    # use_fp16, show_progress

    # llm.*
    llm = raw["llm"]
    _require_str(llm, "base_url", "llm")
    _require_str(llm, "model", "llm")
    # Optional: timeouts, concurrency, httpx, cache, guided, etc.

    # agent.*
    agent = raw["agent"]
    for k in [
        "global_top_k_bm25",
        "dense_top_k_from_bm25",
        "llm_candidates_top_k",
        "llm_conf_threshold",
        "positives_max",
        "hard_negatives_per_query",
        "min_passage_tokens",
        "max_passages_per_page",
        "checkpoint_dir",
        "checkpoint_interval",
    ]:
        _require(agent, k, "agent")
    # Optional: batch_size, concurrent_queries, max_queries,
    # cross_encoder_top_k, use_cross_encoder, device,
    # dense_batch_size, cross_encoder_batch_size

    # output.*
    out = raw["output"]
    _require_str(out, "qrels_path", "output")
    _require_str(out, "triples_path", "output")

    # logging.*
    logging_cfg = raw["logging"]
    _require_str(logging_cfg, "log_dir", "logging")
    _require_str(logging_cfg, "level", "logging")
    _require_str(logging_cfg, "log_file", "logging")
    # Optional: rotation, retention


def validate_paths(cfg: Config) -> bool:
    """
    Create required directories from config (no hardcoded paths).
    """
    to_make = [
        Path(cfg.paths["prepared_dir"]),
        Path(cfg.paths["hf_cache_dir"]),
        Path(cfg.paths["passages_dir"]),
        Path(cfg.paths["indexes_dir"]),
        Path(cfg.agent["checkpoint_dir"]),
        Path(cfg.logging["log_dir"]),
        Path(cfg.output["qrels_path"]).parent,
        Path(cfg.output["triples_path"]).parent,
    ]
    for p in to_make:
        p.mkdir(parents=True, exist_ok=True)
    return True
