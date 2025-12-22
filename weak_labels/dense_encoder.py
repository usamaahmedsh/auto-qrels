"""
weak_labels/dense_encoder.py

Config-driven dense encoder for semantic similarity (SentenceTransformers).

Policy:
- Any tunable must be present under config["dense"] (later configs/base.yaml).
- No hidden performance knobs.

SentenceTransformer.encode supports normalize_embeddings and convert_to_numpy. [web:957]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer


DEFAULT_DENSE_CFG: Dict[str, Any] = {
    "model_name": "BAAI/bge-base-en-v1.5",
    "device": "cpu",                 # "cpu" | "cuda" | "cuda:0" | "mps"
    "batch_size": 64,
    "normalize_embeddings": True,    # cosine via dot product when normalized
    "cache_queries": True,
    "max_query_cache_size": 100_000, # simple cap to avoid unbounded RAM
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_cuda_id(device: str) -> Optional[int]:
    if not device:
        return None
    d = device.strip().lower()
    if d.startswith("cuda:"):
        try:
            return int(d.split(":", 1)[1])
        except Exception:
            return None
    if d == "cuda":
        return 0
    return None


class DenseEncoder:
    """Bi-encoder wrapper with optional query cache and deterministic CPU/GPU placement."""

    def __init__(self, *, dense_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = _deep_update(DEFAULT_DENSE_CFG, dense_cfg or {})

        self.model_name = str(self.cfg["model_name"])
        self.device = str(self.cfg["device"])
        self.batch_size = int(self.cfg["batch_size"])
        self.normalize_embeddings = bool(self.cfg["normalize_embeddings"])

        self.cache_queries = bool(self.cfg["cache_queries"])
        self.max_query_cache_size = int(self.cfg["max_query_cache_size"])

        self._query_cache: Optional[Dict[str, np.ndarray]] = {} if self.cache_queries else None

        self.gpu_id = _parse_cuda_id(self.device)
        if self.gpu_id is not None and torch.cuda.is_available():
            # Ensure subsequent CUDA ops use the intended device
            torch.cuda.set_device(self.gpu_id)

        logger.info(f"DenseEncoder: model={self.model_name}")
        logger.info(f"DenseEncoder: device={self.device}, batch_size={self.batch_size}, normalize={self.normalize_embeddings}")
        logger.info(f"DenseEncoder: cache_queries={self.cache_queries}, max_query_cache_size={self.max_query_cache_size}")

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str], *, batch_size: Optional[int] = None, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0,), dtype=np.float32)

        bs = int(batch_size) if batch_size is not None else self.batch_size

        try:
            # SentenceTransformer.encode supports normalize_embeddings + convert_to_numpy. [web:957]
            return self.model.encode(
                texts,
                batch_size=bs,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize_embeddings,
            )
        except torch.cuda.OutOfMemoryError:
            # OOM strategy is config-driven via batch size; fallback halves BS until 4.
            if self.gpu_id is not None:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if bs <= 4:
                raise
            new_bs = max(4, bs // 2)
            logger.warning(f"DenseEncoder OOM: retrying with smaller batch_size={new_bs}")
            return self.encode(texts, batch_size=new_bs, show_progress=show_progress)

    def _encode_query_cached(self, query: str) -> np.ndarray:
        if self._query_cache is None:
            return self.encode([query], batch_size=1)[0]

        cached = self._query_cache.get(query)
        if cached is not None:
            return cached

        emb = self.encode([query], batch_size=1)[0]
        # Simple cache cap (FIFO-ish: pop an arbitrary key). Keeps behavior deterministic enough.
        if len(self._query_cache) >= self.max_query_cache_size:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[query] = emb
        return emb

    def rerank(self, query: str, candidates: Dict[str, str], *, top_k: int) -> List[str]:
        if not candidates:
            return []

        q_emb = self._encode_query_cached(query)

        doc_ids = list(candidates.keys())
        doc_texts = [candidates[d] for d in doc_ids]

        d_embs = self.encode(doc_texts, batch_size=self.batch_size)
        if d_embs.ndim != 2:
            return []

        # If normalize_embeddings=True, dot product == cosine similarity. [web:957]
        scores = d_embs @ q_emb

        k = min(int(top_k), len(doc_ids))
        # partial top-k for speed
        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]

        return [doc_ids[i] for i in top_idx_sorted]

    def cache_stats(self) -> Dict[str, Any]:
        if self._query_cache is None:
            return {"enabled": False}
        size = len(self._query_cache)
        bytes_ = sum(v.nbytes for v in self._query_cache.values()) if size else 0
        return {
            "enabled": True,
            "cached_queries": size,
            "cache_size_mb": bytes_ / (1024 * 1024),
        }

    def clear_cache(self) -> None:
        if self._query_cache is not None:
            n = len(self._query_cache)
            self._query_cache.clear()
            logger.info(f"DenseEncoder: cleared query cache ({n} entries)")
        if self.gpu_id is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
