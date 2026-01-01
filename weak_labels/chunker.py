"""Text chunking with deduplication and quality filtering - GPU optimized."""

from typing import List, Dict, Optional, Set
import hashlib
import re
from collections import Counter
from loguru import logger
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some optimizations disabled")


class PassageChunker:
    """
    Chunk documents into passages with deduplication and quality filtering.
    GPU-accelerated when possible.
    """
    
    def __init__(
        self, 
        chunk_size: int = 160, 
        stride: int = 80,
        min_chunk_tokens: int = 30,
        min_alpha_ratio: float = 0.5,
        dedupe_hash_length: int = 200,
        use_semantic_dedup: bool = False,
        semantic_threshold: float = 0.95,
        device: str = "cuda",
    ):
        """
        Args:
            chunk_size: Target passage length in tokens
            stride: Overlap between chunks in tokens
            min_chunk_tokens: Minimum tokens to keep a chunk
            min_alpha_ratio: Minimum ratio of alphabetic characters
            dedupe_hash_length: Characters to hash for deduplication
            use_semantic_dedup: Use embedding similarity for dedup (slower but better)
            semantic_threshold: Cosine similarity threshold for semantic dedup
            device: Device for GPU operations
        """
        self.chunk_size = chunk_size
        self.stride = stride
        self.min_chunk_tokens = min_chunk_tokens
        self.min_alpha_ratio = min_alpha_ratio
        self.dedupe_hash_length = dedupe_hash_length
        self.use_semantic_dedup = use_semantic_dedup
        self.semantic_threshold = semantic_threshold
        self.device = device if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # Compile regex patterns once
        self.whitespace_pattern = re.compile(r'\s+')
        self.url_pattern = re.compile(r'http[s]?://\S+')
        
        # Boilerplate patterns (compiled for speed)
        self.boilerplate_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in [
                r'click here', r'next page', r'previous page',
                r'copyright ©', r'all rights reserved',
                r'terms of service', r'privacy policy',
                r'cookie policy', r'subscribe to', r'follow us',
                r'share this', r'read more', r'learn more',
            ]
        ]
        
        logger.info(
            f"PassageChunker: chunk_size={chunk_size}, stride={stride}, "
            f"min_tokens={min_chunk_tokens}, device={self.device}, "
            f"semantic_dedup={use_semantic_dedup}"
        )
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace efficiently"""
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def _compute_text_hash(self, text: str) -> str:
        """Fast text hashing"""
        sample = text[:self.dedupe_hash_length].encode('utf-8', errors='ignore')
        return hashlib.blake2b(sample, digest_size=16).hexdigest()
    
    def _is_low_quality(self, text: str) -> bool:
        """Fast quality check"""
        # Too short
        token_count = len(text.split())
        if token_count < self.min_chunk_tokens:
            return True
        
        # Calculate alpha ratio efficiently
        if len(text) == 0:
            return True
        
        alpha_count = sum(1 for c in text if c.isalpha() or c.isspace())
        alpha_ratio = alpha_count / len(text)
        
        if alpha_ratio < self.min_alpha_ratio:
            return True
        
        # Check boilerplate patterns
        for pattern in self.boilerplate_patterns:
            if pattern.search(text):
                return True
        
        # Check if mostly URLs
        urls = self.url_pattern.findall(text)
        if len(urls) > 3 or (urls and len(''.join(urls)) / len(text) > 0.3):
            return True
        
        # Check repetition (sign of low quality)
        words = text.lower().split()
        if len(words) > 10:
            word_counts = Counter(words)
            most_common = word_counts.most_common(1)[0][1]
            if most_common / len(words) > 0.3:  # >30% same word
                return True
        
        return False
    
    def deduplicate_passages(self, passages: List[Dict]) -> List[Dict]:
        """Remove near-duplicate passages with optional semantic dedup."""
        if not passages:
            return []
        
        if self.use_semantic_dedup and TORCH_AVAILABLE:
            return self._semantic_deduplicate(passages)
        else:
            return self._hash_deduplicate(passages)
    
    def _hash_deduplicate(self, passages: List[Dict]) -> List[Dict]:
        """Fast hash-based deduplication"""
        seen_hashes: Set[str] = set()
        unique = []
        
        for p in passages:
            text_hash = self._compute_text_hash(p["text"])
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique.append(p)
        
        removed = len(passages) - len(unique)
        if removed > 0:
            logger.info(f"Hash dedup: removed {removed:,} duplicates ({len(unique):,} remain)")
        
        return unique
    
    def _semantic_deduplicate(self, passages: List[Dict]) -> List[Dict]:
        """Semantic deduplication using embeddings (GPU-accelerated)"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Computing embeddings for semantic deduplication...")
            
            # Use lightweight model for dedup
            model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            
            texts = [p["text"][:512] for p in passages]  # Truncate for speed
            
            # Encode in batches
            embeddings = model.encode(
                texts,
                batch_size=256,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.device,
            )
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Find duplicates using cosine similarity
            keep_mask = torch.ones(len(passages), dtype=torch.bool, device=self.device)
            
            # Process in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(embeddings), chunk_size):
                end_i = min(i + chunk_size, len(embeddings))
                chunk_emb = embeddings[i:end_i]
                
                # Compute similarity with all previous kept embeddings
                if i > 0:
                    prev_emb = embeddings[:i][keep_mask[:i]]
                    if len(prev_emb) > 0:
                        sim = torch.mm(chunk_emb, prev_emb.t())
                        max_sim, _ = sim.max(dim=1)
                        
                        # Mark as duplicate if too similar
                        is_dup = max_sim > self.semantic_threshold
                        keep_mask[i:end_i] = ~is_dup
            
            keep_mask_cpu = keep_mask.cpu().numpy()
            unique = [p for p, keep in zip(passages, keep_mask_cpu) if keep]
            
            removed = len(passages) - len(unique)
            logger.info(f"Semantic dedup: removed {removed:,} duplicates ({len(unique):,} remain)")
            
            # Cleanup
            del model, embeddings
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return unique
            
        except Exception as e:
            logger.warning(f"Semantic dedup failed: {e}, falling back to hash dedup")
            return self._hash_deduplicate(passages)
    
    def filter_low_quality_passages(self, passages: List[Dict]) -> List[Dict]:
        """Remove low-quality passages (vectorized when possible)"""
        if not passages:
            return []
        
        filtered = []
        
        for p in passages:
            if not self._is_low_quality(p["text"]):
                filtered.append(p)
        
        removed = len(passages) - len(filtered)
        if removed > 0:
            logger.info(f"Quality filter: removed {removed:,} low-quality ({len(filtered):,} remain)")
        
        return filtered
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """Chunk single document into passages (optimized)."""
        # Normalize whitespace once
        text = self._normalize_whitespace(text)
        words = text.split()
        
        if len(words) < self.min_chunk_tokens:
            return []
        
        chunks = []
        chunk_idx = 0
        
        for i in range(0, len(words), self.stride):
            chunk_words = words[i:i + self.chunk_size]
            
            if len(chunk_words) < self.min_chunk_tokens:
                continue
            
            chunk_text = " ".join(chunk_words)
            
            # Skip if low quality
            if self._is_low_quality(chunk_text):
                continue
            
            chunk_id = f"{doc_id}_P{chunk_idx:04d}"
            
            chunks.append({
                "doc_id": chunk_id,
                "text": chunk_text,
                "source_doc_id": doc_id,
                "chunk_index": chunk_idx,
                "start_word": i,
                "end_word": i + len(chunk_words),
            })
            
            chunk_idx += 1
        
        return chunks
    
    def chunk_corpus(self, corpus, batch_log_interval: int = 10000) -> List[Dict]:
        """
        Chunk entire corpus with deduplication and quality filtering.
        
        Args:
            corpus: HuggingFace dataset or list of documents
            batch_log_interval: Log progress every N documents
        
        Returns:
            List of passage dicts with 'doc_id', 'text', 'source_doc_id', etc.
        """
        logger.info("Chunking corpus into passages...")
        
        all_passages = []
        processed_docs = 0
        skipped_docs = 0
        
        for doc in corpus:
            doc_id = doc.get("doc_id") or doc.get("id") or str(len(all_passages))
            text = doc.get("text", "")
            
            if not text or len(text.strip()) < 50:
                skipped_docs += 1
                continue
            
            chunks = self.chunk_text(text, str(doc_id))
            all_passages.extend(chunks)
            
            processed_docs += 1
            
            if processed_docs % batch_log_interval == 0:
                logger.info(
                    f"Processed {processed_docs:,} docs -> {len(all_passages):,} passages "
                    f"(skipped {skipped_docs:,})"
                )
        
        logger.info(
            f"Generated {len(all_passages):,} raw passages from {processed_docs:,} docs "
            f"(skipped {skipped_docs:,})"
        )
        
        # Apply filters (already filtered during chunking, but dedupe here)
        all_passages = self.deduplicate_passages(all_passages)
        
        logger.info(f"✓ Final: {len(all_passages):,} high-quality unique passages")
        
        return all_passages
    
    def get_stats(self) -> Dict:
        """Return chunker statistics"""
        return {
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "min_chunk_tokens": self.min_chunk_tokens,
            "min_alpha_ratio": self.min_alpha_ratio,
            "use_semantic_dedup": self.use_semantic_dedup,
            "device": self.device,
        }
