"""BM25 retrieval using BM25S (pure Python, no Java required)."""

from typing import List, Tuple
from pathlib import Path
import json
import bm25s
from bm25s.hf import BM25HF
from loguru import logger


class BM25Index:
    """
    Pure Python BM25 using bm25s library.
    
    Much faster than rank-bm25, no Java needed like Pyserini.
    Implements Lucene BM25 variant for quality results.
    """
    
    def __init__(self, passages_path: str | Path, k1: float = 0.9, b: float = 0.4):
        """
        Initialize BM25 index.
        
        Args:
            passages_path: Path to passages JSONL file
            k1: BM25 k1 parameter (term saturation)
            b: BM25 b parameter (length normalization)
        """
        self.passages_path = Path(passages_path)
        self.k1 = k1
        self.b = b
        
        # Determine index directory from passages path
        # passages_path: data/passages/corpus_passages.jsonl
        # index_dir: data/indexes/bm25/
        self.index_dir = self.passages_path.parent.parent / "indexes" / "bm25"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize stemmer
        self._init_stemmer()
        
        # Load or build index
        if (self.index_dir / "index.h5").exists():
            logger.info(f"Loading BM25 index from {self.index_dir}")
            self._load_index()
        else:
            logger.info("BM25 index not found, building from passages...")
            self._build_index()
    
    def _init_stemmer(self):
        """Initialize stemmer - try PyStemmer first, fall back to basic stemmer."""
        try:
            import Stemmer
            self.stemmer = Stemmer.Stemmer("english")
            logger.info("Using PyStemmer for stemming")
        except ImportError:
            logger.warning("PyStemmer not found, using basic stemmer")
            # Use None to skip stemming, or use basic Porter stemmer
            self.stemmer = None
    
    def _load_index(self):
        """Load existing BM25S index."""
        try:
            import pickle
            
            # Load the BM25 retriever
            with open(self.index_dir / "index.h5", "rb") as f:
                self.retriever = pickle.load(f)
            
            # Load doc_ids and corpus
            with open(self.index_dir / "doc_ids.json", "r") as f:
                self.doc_ids = json.load(f)
            
            with open(self.index_dir / "corpus.json", "r") as f:
                self.corpus = json.load(f)
            
            logger.info(f"✓ BM25 index loaded ({len(self.doc_ids)} passages)")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}. Rebuilding...")
            self._build_index()
    
    def _build_index(self):
        """Build BM25S index from passages."""
        if not self.passages_path.exists():
            raise FileNotFoundError(
                f"Passages file not found at {self.passages_path}\n"
                f"Run the agent first to generate passages from corpus."
            )
        
        # Load passages
        passages = []
        self.doc_ids = []
        
        logger.info(f"Loading passages from {self.passages_path}...")
        with self.passages_path.open('r') as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    passages.append(p["text"])
                    self.doc_ids.append(p["doc_id"])
        
        logger.info(f"Loaded {len(passages)} passages")
        self.corpus = passages
        
        # Tokenize with stopwords and optional stemming
        logger.info("Tokenizing corpus...")
        corpus_tokens = bm25s.tokenize(
            passages, 
            stopwords="en",
            stemmer=self.stemmer  # Use initialized stemmer or None
        )
        
        # Build index with Lucene method
        logger.info(f"Building BM25 index (k1={self.k1}, b={self.b})...")
        self.retriever = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        self.retriever.index(corpus_tokens)
        
        # Save index
        import pickle
        with open(self.index_dir / "index.h5", "wb") as f:
            pickle.dump(self.retriever, f)
        
        with open(self.index_dir / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)
        
        with open(self.index_dir / "corpus.json", "w") as f:
            json.dump(self.corpus, f)
        
        logger.info(f"✓ Built and saved BM25 index to {self.index_dir}")
    
    def search(self, query: str, top_k: int) -> List[str]:
        """
        Search for top-k documents.
        
        Args:
            query: Query text
            top_k: Number of results to return
        
        Returns:
            List of document IDs
        """
        # Tokenize query
        query_tokens = bm25s.tokenize(
            query, 
            stopwords="en",
            stemmer=self.stemmer
        )
        
        # Retrieve
        results = self.retriever.retrieve(query_tokens, k=top_k)
        
        # Extract doc IDs using indices
        doc_ids = []
        for idx in results.documents[0]:  # [0] because single query
            if 0 <= idx < len(self.doc_ids):
                doc_ids.append(self.doc_ids[idx])
        
        return doc_ids
    
    def search_with_scores(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Search for top-k documents with scores.
        
        Args:
            query: Query text
            top_k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        query_tokens = bm25s.tokenize(
            query,
            stopwords="en", 
            stemmer=self.stemmer
        )
        
        results = self.retriever.retrieve(query_tokens, k=top_k)
        
        doc_scores = []
        for idx, score in zip(results.documents[0], results.scores[0]):
            if 0 <= idx < len(self.doc_ids):
                doc_scores.append((self.doc_ids[idx], float(score)))
        
        return doc_scores
    
    def batch_search(self, queries: List[str], top_k: int) -> List[List[str]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
        
        Returns:
            List of result lists (one per query)
        """
        # Tokenize all queries at once
        query_tokens = bm25s.tokenize(
            queries,
            stopwords="en",
            stemmer=self.stemmer
        )
        
        # Batch retrieve (much faster than individual searches)
        results = self.retriever.retrieve(query_tokens, k=top_k)
        
        # Convert to doc_ids for each query
        all_results = []
        for query_idx in range(len(queries)):
            doc_ids = []
            for idx in results.documents[query_idx]:
                if 0 <= idx < len(self.doc_ids):
                    doc_ids.append(self.doc_ids[idx])
            all_results.append(doc_ids)
        
        return all_results
