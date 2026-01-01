"""
weak_labels/cross_encoder.py

Config-driven cross-encoder reranker (SentenceTransformers CrossEncoder).

Policy:
- All tunables come from config["cross_encoder"].
- No hidden performance knobs.
- Used in Agent after dense reranking for fine-grained rerank.

Typical config (in configs/base.yaml):

cross_encoder:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  device: "cuda"            # "cpu" | "cuda" | "cuda:0"
  batch_size: 128
  max_length: 512
  use_fp16: true
  show_progress: false
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from sentence_transformers import CrossEncoder


DEFAULT_CE_CFG: Dict[str, Any] = {
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "device": "cuda",
    "batch_size": 128,
    "max_length": 512,
    "use_fp16": True,
    "show_progress": False,
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


class CrossEncoderReranker:
    """Cross-encoder wrapper for reranking candidate passages."""

    def __init__(self, *, ce_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = _deep_update(DEFAULT_CE_CFG, ce_cfg or {})

        self.model_name: str = str(self.cfg["model_name"])
        requested_device: str = str(self.cfg.get("device", "cuda"))
        self.batch_size: int = int(self.cfg["batch_size"])
        self.max_length: int = int(self.cfg["max_length"])
        self.use_fp16: bool = bool(self.cfg["use_fp16"])
        self.show_progress: bool = bool(self.cfg["show_progress"])

        # Device resolution: mirror DenseEncoder policy
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                f"CrossEncoderReranker: CUDA device '{requested_device}' requested but CUDA not available; using CPU"
            )
            self.device = "cpu"
        else:
            self.device = requested_device

        logger.info(
            f"CrossEncoderReranker: model={self.model_name}, device={self.device}, "
            f"batch_size={self.batch_size}, max_length={self.max_length}, use_fp16={self.use_fp16}"
        )

        # 1) Load on CPU first to avoid hidden CUDA init that can hit cudaErrorDevicesUnavailable.
        self.model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device="cpu",
        )

        # 2) Move to requested device (if not CPU)
        if self.device != "cpu":
            try:
                self.model = self.model.to(self.device)
            except Exception as e:
                logger.warning(
                    f"CrossEncoderReranker: failed to move model to {self.device}, using CPU instead: {e}"
                )
                self.device = "cpu"

        # 3) Optional fp16 on CUDA
        if self.use_fp16 and self.device != "cpu" and torch.cuda.is_available():
            try:
                self.model.model.half()
                logger.info("CrossEncoderReranker: converted model to float16")
            except Exception as e:
                logger.warning(f"CrossEncoderReranker: failed to convert model to float16: {e}")

    def _build_pairs(self, query: str, candidates: Dict[str, str]) -> List[List[str]]:
        """Build (query, passage) pairs for CrossEncoder.predict."""
        doc_ids = list(candidates.keys())
        passages = [candidates[d] for d in doc_ids]
        pairs = [[query, p] for p in passages]
        return doc_ids, pairs

    def rerank(
        self,
        query: str,
        candidates: Dict[str, str],
        *,
        top_k: int,
    ) -> List[str]:
        """
        Rerank candidates for a single query.

        Args:
            query: Query text
            candidates: Mapping doc_id -> passage text
            top_k: Number of top doc_ids to return

        Returns:
            List of doc_ids sorted by cross-encoder score (desc)
        """
        if not candidates:
            return []

        doc_ids, pairs = self._build_pairs(query, candidates)

        try:
            # CrossEncoder.predict accepts list of [query, passage] pairs. [web:59][web:257]
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            bs = self.batch_size
            if bs <= 4:
                raise
            new_bs = max(4, bs // 2)
            logger.warning(f"CrossEncoderReranker OOM: retrying with smaller batch_size={new_bs}")
            scores = self.model.predict(
                pairs,
                batch_size=new_bs,
                show_progress_bar=self.show_progress,
            )

        scores = np.asarray(scores, dtype=np.float32)
        k = min(int(top_k), len(doc_ids))
        if k <= 0:
            return []

        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]

        return [doc_ids[i] for i in top_idx_sorted]
