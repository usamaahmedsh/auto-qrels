"""Text chunking with deduplication and quality filtering."""

from typing import List, Dict
import hashlib
from loguru import logger


class PassageChunker:
    """
    Chunk documents into passages with deduplication and quality filtering.
    """
    
    def __init__(self, chunk_size: int = 160, stride: int = 80):
        """
        Args:
            chunk_size: Target passage length in tokens
            stride: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.stride = stride
        logger.info(f"PassageChunker: chunk_size={chunk_size}, stride={stride}")
    
    def deduplicate_passages(self, passages: List[Dict]) -> List[Dict]:
        """Remove near-duplicate passages."""
        seen_hashes = set()
        unique = []
        
        for p in passages:
            # Hash first 200 chars (catches exact duplicates)
            text_hash = hashlib.md5(p["text"][:200].encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique.append(p)
        
        removed = len(passages) - len(unique)
        if removed > 0:
            logger.info(f"Deduplication: removed {removed:,} duplicates ({len(unique):,} remain)")
        
        return unique
    
    def filter_low_quality_passages(self, passages: List[Dict]) -> List[Dict]:
        """Remove low-quality passages."""
        filtered = []
        
        for p in passages:
            text = p["text"]
            
            # Skip if too short
            if len(text.split()) < 20:
                continue
            
            # Skip if mostly numbers/symbols
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
            if alpha_ratio < 0.5:
                continue
            
            # Skip if looks like navigation/boilerplate
            text_lower = text.lower()
            if any(phrase in text_lower for phrase in [
                "click here", "next page", "previous page",
                "copyright ©", "all rights reserved",
                "terms of service", "privacy policy"
            ]):
                continue
            
            filtered.append(p)
        
        removed = len(passages) - len(filtered)
        if removed > 0:
            logger.info(f"Quality filter: removed {removed:,} low-quality ({len(filtered):,} remain)")
        
        return filtered
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """Chunk single document into passages."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.stride):
            chunk_words = words[i:i + self.chunk_size]
            
            if len(chunk_words) < 20:  # Skip very short chunks
                continue
            
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{doc_id}_chunk_{len(chunks)}"
            
            chunks.append({
                "doc_id": chunk_id,
                "text": chunk_text,
                "source_doc_id": doc_id
            })
        
        return chunks
    
    def chunk_corpus(self, corpus) -> List[Dict]:
        """
        Chunk entire corpus with deduplication and quality filtering.
        
        Args:
            corpus: HuggingFace dataset or list of documents
        
        Returns:
            List of passage dicts with 'doc_id', 'text', 'source_doc_id'
        """
        logger.info("Chunking corpus into passages...")
        
        all_passages = []
        
        for doc in corpus:
            doc_id = doc.get("doc_id", str(doc.get("id", len(all_passages))))
            text = doc.get("text", "")
            
            if not text or len(text.strip()) < 50:
                continue
            
            chunks = self.chunk_text(text, doc_id)
            all_passages.extend(chunks)
        
        logger.info(f"Generated {len(all_passages):,} raw passages")
        
        # Apply quality filters
        all_passages = self.filter_low_quality_passages(all_passages)
        all_passages = self.deduplicate_passages(all_passages)
        
        logger.info(f"✓ Final: {len(all_passages):,} high-quality passages")
        
        return all_passages
