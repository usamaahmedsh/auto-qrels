from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer


DEFAULT_DENSE_CFG: Dict[str, Any] = {
    "model_name": "BAAI/bge-base-en-v1.5",
    "device": "cuda",              # "cpu" | "cuda" | "cuda:0"
    "use_fp16": True,
    "batch_size": 256,
    "normalize_embeddings": True,
    "cache_queries": True,
    "max_query_cache_size": 100_000,
    "show_progress": False,
    # New: precomputed index options
    "index_path": "data/indexes/passage_embs.npz",   # npz with {"embs": ..., "ids": ...}
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


class DenseEncoder:
    """Bi-encoder wrapper with optional query cache and precomputed passage index."""

    def __init__(self, *, dense_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = _deep_update(DEFAULT_DENSE_CFG, dense_cfg or {})

        self.model_name: str = str(self.cfg["model_name"])
        requested_device: str = str(self.cfg.get("device", "cuda"))
        self.batch_size: int = int(self.cfg["batch_size"])
        self.normalize_embeddings: bool = bool(self.cfg["normalize_embeddings"])
        self.cache_queries: bool = bool(self.cfg["cache_queries"])
        self.max_query_cache_size: int = int(self.cfg["max_query_cache_size"])
        self.use_fp16: bool = bool(self.cfg["use_fp16"])
        self.show_progress: bool = bool(self.cfg["show_progress"])
        self.index_path: str = str(self.cfg.get("index_path", "data/indexes/passage_embs.npz"))

        self._query_cache: Optional[Dict[str, np.ndarray]] = {} if self.cache_queries else None

        # Precomputed index state
        self._index_embs: Optional[np.ndarray] = None   # shape (N, d), float32, normalized
        self._index_ids: Optional[List[str]] = None     # len N, doc_ids aligned with embs

        # Resolve device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                f"DenseEncoder: CUDA device '{requested_device}' requested but CUDA not available; using CPU"
            )
            self.device = "cpu"
        else:
            self.device = requested_device

        logger.info(f"DenseEncoder: model={self.model_name}")
        logger.info(
            f"DenseEncoder: device={self.device}, batch_size={self.batch_size}, "
            f"normalize={self.normalize_embeddings}, use_fp16={self.use_fp16}"
        )
        logger.info(
            f"DenseEncoder: cache_queries={self.cache_queries}, "
            f"max_query_cache_size={self.max_query_cache_size:,}"
        )
        logger.info(f"DenseEncoder: index_path={self.index_path}")

        # Load on CPU then move to target device
        self.model = SentenceTransformer(self.model_name, device="cpu")

        if self.device != "cpu":
            try:
                self.model = self.model.to(self.device)
            except Exception as e:
                logger.warning(
                    f"DenseEncoder: failed to move model to {self.device}, using CPU instead: {e}"
                )
                self.device = "cpu"

        if self.use_fp16 and self.device != "cpu" and torch.cuda.is_available():
            try:
                self.model = self.model.half()
                logger.info("DenseEncoder: converted model to float16")
            except Exception as e:
                logger.warning(f"DenseEncoder: failed to convert model to float16: {e}")

    # -----------------------------
    # Core encoding
    # -----------------------------
    def encode(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0,), dtype=np.float32)

        bs = int(batch_size) if batch_size is not None else self.batch_size
        show_bar = self.show_progress if show_progress is None else bool(show_progress)

        try:
            embs = self.model.encode(
                texts,
                batch_size=bs,
                convert_to_numpy=True,
                show_progress_bar=show_bar,
                normalize_embeddings=self.normalize_embeddings,
            )
            if embs.dtype != np.float32:
                embs = embs.astype(np.float32, copy=False)
            return embs
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if bs <= 4:
                raise
            new_bs = max(4, bs // 2)
            logger.warning(f"DenseEncoder OOM: retrying with smaller batch_size={new_bs}")
            return self.encode(texts, batch_size=new_bs, show_progress=show_bar)

    def _encode_query_cached(self, query: str) -> np.ndarray:
        if self._query_cache is None:
            return self.encode([query], batch_size=1)[0]

        cached = self._query_cache.get(query)
        if cached is not None:
            return cached

        emb = self.encode([query], batch_size=1)[0]
        if len(self._query_cache) >= self.max_query_cache_size:
            # naive eviction
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[query] = emb
        return emb

    # -----------------------------
    # Legacy rerank API (per-query encode of candidates)
    # -----------------------------
    def rerank(
        self,
        query: str,
        candidates: Dict[str, str],
        *,
        top_k: int,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Legacy API: encode all candidate texts then score vs query embedding."""
        if not candidates:
            return []

        q_emb = self._encode_query_cached(query)

        doc_ids = list(candidates.keys())
        doc_texts = [candidates[d] for d in doc_ids]

        d_embs = self.encode(doc_texts, batch_size=batch_size or self.batch_size)
        if d_embs.ndim != 2:
            return []

        scores = d_embs @ q_emb

        k = min(int(top_k), len(doc_ids))
        if k <= 0:
            return []

        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]

        return [doc_ids[i] for i in top_idx_sorted]

    # -----------------------------
    # NEW: precomputed corpus index
    # -----------------------------
    def build_corpus_index(
        self,
        doc_ids: List[str],
        doc_texts: List[str],
        *,
        batch_size: Optional[int] = None,
    ) -> None:
        """Precompute passage embeddings for the fixed corpus and save to disk.

        This should be run once (or when corpus changes), then Phase 1 uses `search`.
        """
        if len(doc_ids) != len(doc_texts):
            raise ValueError("doc_ids and doc_texts must have same length")

        bs = int(batch_size) if batch_size is not None else self.batch_size
        logger.info(f"DenseEncoder: building corpus index for {len(doc_ids):,} passages (batch_size={bs})")

        embs_list: List[np.ndarray] = []
        for i in range(0, len(doc_texts), bs):
            chunk = doc_texts[i : i + bs]
            embs = self.encode(chunk, batch_size=bs, show_progress=False)
            embs_list.append(embs)

        embs_all = np.concatenate(embs_list, axis=0)
        if embs_all.shape[0] != len(doc_ids):
            raise RuntimeError("Embedding count mismatch after encoding corpus")

        # Ensure normalized
        if self.normalize_embeddings:
            # Already normalized by encode; just cast to float32
            embs_all = embs_all.astype(np.float32, copy=False)
        else:
            # Normalize here for cosine similarity
            norms = np.linalg.norm(embs_all, axis=1, keepdims=True) + 1e-8
            embs_all = (embs_all / norms).astype(np.float32, copy=False)

        np.savez_compressed(
            self.index_path,
            embs=embs_all,
            ids=np.array(doc_ids, dtype=object),
        )
        logger.info(f"DenseEncoder: saved corpus index to {self.index_path}")

        self._index_embs = embs_all
        self._index_ids = list(doc_ids)

    def load_corpus_index(self) -> bool:
        """Load precomputed passage embeddings from disk into memory."""
        try:
            data = np.load(self.index_path, allow_pickle=True)
            embs = data["embs"].astype(np.float32, copy=False)
            ids = list(data["ids"].tolist())
        except Exception as e:
            logger.warning(f"DenseEncoder: failed to load index from {self.index_path}: {e}")
            return False

        if embs.ndim != 2 or len(ids) != embs.shape[0]:
            logger.warning("DenseEncoder: invalid index file shape; ignoring")
            return False

        # Ensure normalized
        if not self.normalize_embeddings:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            embs = (embs / norms).astype(np.float32, copy=False)

        self._index_embs = embs
        self._index_ids = ids
        logger.info(
            f"DenseEncoder: loaded corpus index: {embs.shape[0]:,} passages, dim={embs.shape[1]}"
        )
        return True

    def has_index(self) -> bool:
        return self._index_embs is not None and self._index_ids is not None

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Search precomputed passage index for top_k hits.

        Returns list of (doc_id, score). Requires build_corpus_index/load_corpus_index first.
        """
        if not self.has_index():
            raise RuntimeError("DenseEncoder.search called without loaded corpus index")

        q_emb = self._encode_query_cached(query)  # already normalized
        embs = self._index_embs
        ids = self._index_ids

        # scores: cosine similarity as dot product of normalized vectors
        scores = embs @ q_emb  # shape (N,)

        k = min(int(top_k), embs.shape[0])
        if k <= 0:
            return []

        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]

        return [(ids[i], float(scores[i])) for i in top_idx_sorted]

    # -----------------------------
    # Cache utilities
    # -----------------------------
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
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
