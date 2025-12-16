"""Dense encoder for semantic similarity using SentenceTransformers."""

from typing import List, Dict, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger


def get_default_device() -> str:
    """
    Detect best available device: CUDA > MPS (Apple Silicon) > CPU.
    
    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DenseEncoder:
    """
    Bi-encoder wrapper using SentenceTransformers.
    
    Features:
    - Automatic device selection (CUDA/MPS/CPU)
    - Query embedding caching
    - Batch encoding with progress tracking
    - Normalized embeddings for cosine similarity
    """
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        cache_queries: bool = True
    ):
        """
        Initialize dense encoder.
        
        Args:
            model_name: HuggingFace model name (default: BGE-base)
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
            cache_queries: Whether to cache query embeddings
        """
        if device is None:
            device = get_default_device()
        
        self.device = device
        self.cache_queries = cache_queries
        self.query_cache = {} if cache_queries else None
        
        logger.info(f"Loading dense encoder: {model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"✓ Dense encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 64,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to dense embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (larger = faster but more memory)
            show_progress: Whether to show progress bar
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )
            return embeddings
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def encode_with_cache(
        self, 
        texts: List[str], 
        is_query: bool = False,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Encode with optional query caching.
        
        Single queries are cached to avoid redundant encoding.
        
        Args:
            texts: List of texts to encode
            is_query: Whether these are queries (enables caching)
            batch_size: Batch size for encoding
        
        Returns:
            numpy array of embeddings
        """
        # Check cache for single query
        if is_query and self.cache_queries and len(texts) == 1:
            text = texts[0]
            if text not in self.query_cache:
                self.query_cache[text] = self.encode([text], batch_size=1)[0]
            return np.array([self.query_cache[text]])
        
        # Regular encoding for multiple texts or non-queries
        return self.encode(texts, batch_size=batch_size)
    
    def rerank(
        self,
        query: str,
        candidates: Dict[str, str],
        top_k: int,
        batch_size: int = 64,
    ) -> List[str]:
        """
        Rerank candidates by semantic similarity to query.
        
        Args:
            query: Query text
            candidates: Dict mapping doc_id -> text
            top_k: Number of top results to return
            batch_size: Batch size for encoding candidates
        
        Returns:
            List of doc_ids sorted by similarity (highest first)
        """
        if not candidates:
            return []
        
        # Encode query with caching
        q_emb = self.encode_with_cache([query], is_query=True)[0]
        
        # Sort candidates by length for efficient batching
        doc_items = sorted(candidates.items(), key=lambda x: len(x[1]))
        doc_ids = [k for k, _ in doc_items]
        doc_texts = [v for _, v in doc_items]
        
        # Encode all candidates
        d_embs = self.encode(doc_texts, batch_size=batch_size)
        
        # Compute cosine similarity (dot product since normalized)
        scores = d_embs @ q_emb
        
        # Sort and return top-k
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc_id for doc_id, _ in ranked]
    
    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[Dict[str, str]],
        top_k: int,
        batch_size: int = 64,
    ) -> List[List[str]]:
        """
        Rerank multiple queries at once.
        
        Args:
            queries: List of query texts
            candidates_list: List of candidate dicts (one per query)
            top_k: Number of top results per query
            batch_size: Batch size for encoding
        
        Returns:
            List of ranked doc_id lists (one per query)
        """
        results = []
        for query, candidates in zip(queries, candidates_list):
            results.append(self.rerank(query, candidates, top_k, batch_size))
        return results
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score in [0, 1]
        """
        embs = self.encode([text1, text2], batch_size=2)
        return float(embs[0] @ embs[1])
    
    def clear_cache(self):
        """Clear query cache to free memory."""
        if self.query_cache is not None:
            cache_size = len(self.query_cache)
            self.query_cache.clear()
            logger.info(f"Cleared {cache_size} cached queries")
    
    def cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache metrics
        """
        if self.query_cache is None:
            return {"enabled": False}
        
        cache_size_bytes = sum(
            q.nbytes for q in self.query_cache.values()
        ) if self.query_cache else 0
        
        return {
            "enabled": True,
            "cached_queries": len(self.query_cache),
            "cache_size_mb": cache_size_bytes / (1024 * 1024),
        }
    
    def __repr__(self) -> str:
        return (
            f"DenseEncoder(model={self.model.get_sentence_embedding_dimension()}d, "
            f"device={self.device}, cached_queries={len(self.query_cache) if self.query_cache else 0})"
        )
