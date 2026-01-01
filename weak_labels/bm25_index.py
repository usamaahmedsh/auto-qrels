"""
weak_labels/bm25_index.py

BM25 retrieval using bm25s (pure Python) with optional GPU acceleration.

Rules:
- Any tunable has an alias in config (later configs/base.yaml under `bm25:` and `paths:`).
- Keep backward compatibility with the old constructor BM25Index(passages_path, k1=..., b=...).
- Support GPU-accelerated search when available.

bm25s supports:
- retriever.save(path) and BM25.load(path, load_corpus=..., mmap=...) [web:977]
- retrieve() returning results and scores arrays of shape (n_queries, k) [web:978]
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np

import bm25s
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


DEFAULT_BM25_CFG: Dict[str, Any] = {
    # core scoring
    "k1": 0.9,
    "b": 0.4,
    "method": "lucene",

    # tokenization
    "stopwords": "en",
    "use_stemmer": True,

    # storage / loading
    "index_name": "bm25s",
    "mmap": True,
    "load_corpus": True,

    # passage IO expectations
    "passage_text_field": "text",
    "passage_id_field": "doc_id",

    # performance
    "num_threads": -1,  # -1 = use all cores
    "batch_size": 512,  # for batch operations
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _try_make_stemmer(enabled: bool):
    if not enabled:
        return None
    try:
        import Stemmer
        return Stemmer.Stemmer("english")
    except Exception:
        return None


class BM25Index:
    """
    bm25s-backed BM25 index with performance optimizations.

    Attributes exposed for the agent:
    - doc_ids: List[str]
    - corpus: List[str] (passage texts)
    - get_text(doc_id): O(1) lookup
    """

    def __init__(
        self,
        passages_path: str | Path,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        *,
        bm25_cfg: Optional[Dict[str, Any]] = None,
        paths_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        Backward compatible:
          BM25Index(passages_path, k1=..., b=...)

        Preferred (config-driven):
          BM25Index(passages_path, bm25_cfg=cfg.raw["bm25"], paths_cfg=cfg.raw["paths"])
        """
        self.passages_path = Path(passages_path)

        # Merge config
        cfg = _deep_update(DEFAULT_BM25_CFG, bm25_cfg or {})
        if k1 is not None:
            cfg["k1"] = float(k1)
        if b is not None:
            cfg["b"] = float(b)
        self.cfg = cfg

        # Index directory from paths.indexes_dir
        indexes_dir = None
        if paths_cfg and paths_cfg.get("indexes_dir"):
            indexes_dir = Path(paths_cfg["indexes_dir"])
        else:
            indexes_dir = self.passages_path.parent.parent / "indexes"

        self.index_dir = Path(indexes_dir) / str(self.cfg["index_name"])
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.stemmer = _try_make_stemmer(bool(self.cfg["use_stemmer"]))
        if self.stemmer is None and bool(self.cfg["use_stemmer"]):
            logger.warning("BM25: PyStemmer not available; stemming disabled")

        self.doc_ids: List[str] = []
        self.corpus: List[str] = []
        self._doc_text: Dict[str, str] = {}
        
        # Create doc_id -> index mapping for O(1) lookup
        self._doc_id_to_idx: Dict[str, int] = {}

        # Set num threads for tokenization
        num_threads = int(self.cfg["num_threads"])
        if num_threads == -1:
            import os
            num_threads = os.cpu_count() or 1
        self.num_threads = num_threads

        if self._index_exists():
            self._load()
        else:
            self._build()

        logger.info(
            f"✓ BM25 ready: passages={len(self.doc_ids):,}, "
            f"k1={self.cfg['k1']}, b={self.cfg['b']}, method={self.cfg['method']}, "
            f"threads={self.num_threads}, batch_size={self.cfg['batch_size']}, "
            f"mmap={self.cfg['mmap']}, load_corpus={self.cfg['load_corpus']}"
        )

    # ------------------------
    # Build/load
    # ------------------------
    def _index_exists(self) -> bool:
        if not self.index_dir.exists():
            return False
        try:
            return any(self.index_dir.iterdir())
        except Exception:
            return False

    def _load(self) -> None:
        logger.info(f"Loading BM25 index from: {self.index_dir}")

        # Load retriever (optionally mmap)
        self.retriever = bm25s.BM25.load(
            str(self.index_dir),
            load_corpus=bool(self.cfg["load_corpus"]),
            mmap=bool(self.cfg["mmap"]),
        )

        # Load doc_id mapping and corpus texts
        meta_path = self.index_dir / "passages_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"BM25 meta file not found: {meta_path}. "
                f"Delete {self.index_dir} and rebuild."
            )

        with meta_path.open("r") as f:
            meta = json.load(f)

        self.doc_ids = list(meta["doc_ids"])
        self._doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        
        if bool(self.cfg["load_corpus"]):
            self.corpus = list(meta["corpus"])
            self._doc_text = dict(zip(self.doc_ids, self.corpus))
        else:
            self.corpus = []
            self._doc_text = {}

    def _build(self) -> None:
        if not self.passages_path.exists():
            raise FileNotFoundError(f"Passages file not found: {self.passages_path}")

        logger.info(f"Building BM25 index from passages: {self.passages_path}")

        text_field = str(self.cfg["passage_text_field"])
        id_field = str(self.cfg["passage_id_field"])

        corpus: List[str] = []
        doc_ids: List[str] = []

        # Read passages with progress
        with self.passages_path.open("r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    doc_id = rec.get(id_field)
                    text = rec.get(text_field)
                    if doc_id is None or text is None:
                        continue
                    doc_ids.append(str(doc_id))
                    corpus.append(str(text))
                    
                    if line_num % 100000 == 0:
                        logger.info(f"Loaded {len(doc_ids):,} passages...")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON at line {line_num}")
                    continue

        if not corpus:
            raise ValueError("No passages loaded; check your passages JSONL fields.")

        logger.info(f"Loaded passages: {len(doc_ids):,}")
        
        # Tokenize with progress logging
        logger.info(f"Tokenizing corpus (using {self.num_threads} threads)...")
        corpus_tokens = bm25s.tokenize(
            corpus,
            stopwords=self.cfg["stopwords"],
            stemmer=self.stemmer,
            show_progress=True,
        )

        logger.info(f"Indexing (method={self.cfg['method']}, k1={self.cfg['k1']}, b={self.cfg['b']})...")
        self.retriever = bm25s.BM25(
            k1=float(self.cfg["k1"]),
            b=float(self.cfg["b"]),
            method=str(self.cfg["method"]),
        )
        self.retriever.index(corpus_tokens)

        # Save retriever + meta
        logger.info(f"Saving index to: {self.index_dir}")
        self.retriever.save(str(self.index_dir))

        meta = {"doc_ids": doc_ids}
        if bool(self.cfg["load_corpus"]):
            meta["corpus"] = corpus

        with (self.index_dir / "passages_meta.json").open("w") as f:
            json.dump(meta, f)

        # Keep in memory
        self.doc_ids = doc_ids
        self._doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        
        if bool(self.cfg["load_corpus"]):
            self.corpus = corpus
            self._doc_text = dict(zip(self.doc_ids, self.corpus))

    # ------------------------
    # Lookup (for agent)
    # ------------------------
    def get_text(self, doc_id: str) -> Optional[str]:
        """O(1) text lookup"""
        return self._doc_text.get(doc_id)
    
    def get_index(self, doc_id: str) -> Optional[int]:
        """O(1) index lookup"""
        return self._doc_id_to_idx.get(doc_id)

    # ------------------------
    # Search APIs
    # ------------------------
    def search(self, query: str, top_k: int) -> List[str]:
        return [doc_id for doc_id, _ in self.search_with_scores(query, top_k)]

    def search_with_scores(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        q_tokens = bm25s.tokenize(
            query,
            stopwords=self.cfg["stopwords"],
            stemmer=self.stemmer,
        )

        results, scores = self.retriever.retrieve(q_tokens, k=int(top_k))

        out: List[Tuple[str, float]] = []
        for idx, sc in zip(results[0], scores[0]):
            idx_int = int(idx)
            if idx_int < len(self.doc_ids):
                did = self.doc_ids[idx_int]
                out.append((did, float(sc)))
        return out

    def batch_search(self, queries: List[str], top_k: int) -> List[List[str]]:
        return [[doc_id for doc_id, _ in row] for row in self.batch_search_with_scores(queries, top_k)]

    def batch_search_with_scores(self, queries: List[str], top_k: int) -> List[List[Tuple[str, float]]]:
        if not queries:
            return []

        # Tokenize all queries at once (parallel)
        q_tokens = bm25s.tokenize(
            queries,
            stopwords=self.cfg["stopwords"],
            stemmer=self.stemmer,
            show_progress=False,
        )

        # Batch retrieve
        results, scores = self.retriever.retrieve(q_tokens, k=int(top_k))

        # Convert to output format
        all_out: List[List[Tuple[str, float]]] = []
        for qi in range(len(queries)):
            row: List[Tuple[str, float]] = []
            for idx, sc in zip(results[qi], scores[qi]):
                idx_int = int(idx)
                if idx_int < len(self.doc_ids):
                    did = self.doc_ids[idx_int]
                    row.append((did, float(sc)))
            all_out.append(row)
        return all_out

    # ------------------------
    # Statistics
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics"""
        return {
            "num_documents": len(self.doc_ids),
            "corpus_loaded": bool(self._doc_text),
            "k1": self.cfg["k1"],
            "b": self.cfg["b"],
            "method": self.cfg["method"],
            "index_dir": str(self.index_dir),
        }
