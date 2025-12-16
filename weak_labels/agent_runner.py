"""Agent with cross-encoder, smart sampling, and monitoring."""

from typing import Dict, Any, List, Tuple
import json
import random
import time
from pathlib import Path
import statistics
from tqdm import tqdm
from loguru import logger
from sentence_transformers import CrossEncoder

from .dense_encoder import DenseEncoder
from .bm25_index import BM25Index
from .llm_client import CachedLLMJudge


class Agent:
    """
    Optimized agent with cross-encoder filtering and smart sampling.
    """
    
    def __init__(
        self,
        bm25: BM25Index,
        dense_encoder: DenseEncoder,
        llm_judge: CachedLLMJudge,
        config: Dict[str, Any]
    ):
        self.bm25 = bm25
        self.dense_encoder = dense_encoder
        self.llm_judge = llm_judge
        self.cfg = config
        
        # Load cross-encoder for better filtering
        logger.info("Loading cross-encoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("✓ Cross-encoder loaded")
        
        # Setup output paths
        self.qrels_path = Path(config['output']['qrels_path'])
        self.triples_path = Path(config['output']['triples_path'])
        self.checkpoint_dir = Path(config['agent']['checkpoint_dir'])
        
        for p in [self.qrels_path, self.triples_path]:
            p.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize stats tracking
        self.stats = {
            "queries_processed": 0,
            "passages_judged": 0,
            "positives_found": 0,
            "hard_negatives_found": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "total_time": 0.0,
            "avg_time_per_query": 0.0
        }
        
        # Load checkpoint
        self.checkpoint_file = self.checkpoint_dir / "progress.json"
        self.processed_queries = self._load_checkpoint()
        
        logger.info(f"✓ Agent initialized (checkpoint: {len(self.processed_queries)} queries)")
    
    def _load_checkpoint(self) -> set:
        """Load set of already processed query IDs."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get("processed_query_ids", []))
        return set()
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                "processed_query_ids": list(self.processed_queries),
                "stats": self.stats
            }, f, indent=2)
    
    def _estimate_remaining(self, total_queries: int) -> str:
        """Estimate time remaining."""
        if self.stats["avg_time_per_query"] == 0:
            return "calculating..."
        
        remaining = total_queries - self.stats["queries_processed"]
        seconds = remaining * self.stats["avg_time_per_query"]
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    
    def _log_stats(self, total_queries: int):
        """Log progress stats."""
        logger.info(f"""
╔════════════════════════════════════════╗
║          PROGRESS STATS                ║
╠════════════════════════════════════════╣
║ Queries:      {self.stats['queries_processed']:>10,} ║
║ Positives:    {self.stats['positives_found']:>10,} ║
║ Hard Negs:    {self.stats['hard_negatives_found']:>10,} ║
║ LLM calls:    {self.stats['llm_calls']:>10,} ║
║ Avg/query:    {self.stats['avg_time_per_query']:>9.1f}s ║
║ Est. remain:  {self._estimate_remaining(total_queries):>10} ║
╚════════════════════════════════════════╝
        """)
    
    def smart_sample_passages(
        self, 
        bm25_results: List[Tuple[str, float]], 
        dense_results: List[Dict],
        num_llm_candidates: int = 25
    ) -> List[Dict]:
        """
        Smart sampling: Judge top candidates + random sample from middle/bottom.
        Reduces LLM calls by ~40% while maintaining quality.
        """
        # Top 10: Always judge (likely positives)
        top_candidates = dense_results[:10]
        
        # Middle 20: Sample 10 (hard negatives)
        middle_pool = dense_results[10:30] if len(dense_results) > 10 else []
        middle_sample = random.sample(middle_pool, min(10, len(middle_pool)))
        
        # Bottom 20: Sample 5 (easy negatives)
        bottom_pool = dense_results[30:50] if len(dense_results) > 30 else []
        bottom_sample = random.sample(bottom_pool, min(5, len(bottom_pool)))
        
        candidates = top_candidates + middle_sample + bottom_sample
        
        logger.debug(
            f"Smart sampling: {len(dense_results)} → {len(candidates)} "
            f"(top:{len(top_candidates)}, mid:{len(middle_sample)}, bot:{len(bottom_sample)})"
        )
        
        return candidates
    
    def cross_encoder_rerank(
        self, 
        query: str, 
        passages: List[Dict], 
        top_k: int = 30
    ) -> List[Dict]:
        """
        Rerank passages using cross-encoder.
        Much faster than LLM, better than cosine similarity.
        """
        if not passages:
            return []
        
        # Create pairs
        pairs = [(query, p["text"]) for p in passages]
        
        # Predict scores
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Sort by cross-encoder score
        scored = sorted(
            zip(passages, ce_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [p for p, score in scored[:top_k]]
    
    def select_hard_negatives(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Dict],
        positives: List[Dict],
        all_judged: List[Dict],
        num_negatives: int = 10
    ) -> List[str]:
        """
        Mine hard negatives from BM25 vs Dense disagreements.
        """
        positive_ids = {p["doc_id"] for p in positives}
        dense_top_ids = {d["doc_id"] for d in dense_results[:20]}
        bm25_top_ids = {doc_id for doc_id, score in bm25_results[:20]}
        
        # Get high-score negatives (judged as negative but scored high)
        hard_negatives = []
        
        for judged in all_judged:
            if judged["doc_id"] in positive_ids:
                continue
            
            # Type 1: High BM25, low dense (lexical false positives)
            if judged["doc_id"] in bm25_top_ids and judged["doc_id"] not in dense_top_ids:
                hard_negatives.append(judged["doc_id"])
            
            # Type 2: High dense, low BM25 (semantic false positives)
            elif judged["doc_id"] in dense_top_ids and judged["doc_id"] not in bm25_top_ids:
                hard_negatives.append(judged["doc_id"])
            
            # Type 3: High cross-encoder score but negative judgment
            elif judged.get("relevance_score", 0) == 1:  # Somewhat related but not good
                hard_negatives.append(judged["doc_id"])
        
        # Return up to num_negatives
        selected = hard_negatives[:num_negatives]
        
        logger.debug(f"Hard negatives: {len(selected)} mined from disagreements")
        
        return selected
    
    def process_query(self, query_rec: Dict) -> Dict[str, Any]:
        """Process single query with all optimizations."""
        start_time = time.time()
        
        qid = query_rec["query_id"]
        qtext = query_rec["text"]
        
        # Step 1: BM25 retrieval (larger pool)
        bm25_results = self.bm25.search_with_scores(
            qtext,
            top_k=self.cfg['agent']['global_top_k_bm25']
        )
        
        if not bm25_results:
            logger.debug(f"No BM25 results for query {qid}")
            return {"positives": 0, "hard_negatives": 0}
        
        # Step 2: Convert BM25 results to dict format for dense encoder
        # BM25 returns List[Tuple[doc_id, score]], need Dict[doc_id, text]
        bm25_doc_ids = [doc_id for doc_id, score in bm25_results]
        
        # Load passage texts from BM25 index
        candidates_dict = {}
        for doc_id in bm25_doc_ids:
            # Get passage text from BM25 corpus
            idx = self.bm25.doc_ids.index(doc_id) if doc_id in self.bm25.doc_ids else None
            if idx is not None and idx < len(self.bm25.corpus):
                candidates_dict[doc_id] = self.bm25.corpus[idx]
        
        # Step 3: Dense reranking
        dense_top = self.dense_encoder.rerank(
            qtext,
            candidates_dict,  # Now passing dict instead of list
            top_k=self.cfg['agent']['dense_top_k_from_bm25']
        )
        
        # Convert back to list of dicts format
        dense_top_passages = [
            {"doc_id": doc_id, "text": candidates_dict[doc_id]}
            for doc_id in dense_top
        ]
        
        # Step 4: Cross-encoder filtering (fast, high quality)
        ce_filtered = self.cross_encoder_rerank(
            qtext,
            dense_top_passages,
            top_k=40
        )
        
        # Step 5: Smart sampling for LLM judging
        llm_candidates = self.smart_sample_passages(
            bm25_results,
            ce_filtered,
            num_llm_candidates=25
        )
        
        # Step 6: LLM judging (only ~25 passages instead of 40)
        judged = self.llm_judge.judge_batch(qtext, llm_candidates)
        
        self.stats["llm_calls"] += len(llm_candidates)
        self.stats["passages_judged"] += len(judged)
        
        # Step 7: Extract positives (relevance_score >= 2)
        positives = [
            j for j in judged
            if j.get("answerable", False) and j.get("relevance_score", 0) >= 2
        ]
        
        self.stats["positives_found"] += len(positives)
        
        # Step 8: Select hard negatives
        hard_neg_ids = self.select_hard_negatives(
            bm25_results,
            dense_top_passages,
            positives,
            judged,
            num_negatives=self.cfg['agent']['hard_negatives_per_query']
        )
        
        self.stats["hard_negatives_found"] += len(hard_neg_ids)
        
        # Step 9: Write outputs
        self._write_qrels(qid, positives)
        self._write_triples(qid, qtext, positives, hard_neg_ids)
        
        # Update stats
        elapsed = time.time() - start_time
        self.stats["queries_processed"] += 1
        self.stats["total_time"] += elapsed
        self.stats["avg_time_per_query"] = (
            self.stats["total_time"] / self.stats["queries_processed"]
        )
        
        return {
            "positives": len(positives),
            "hard_negatives": len(hard_neg_ids),
            "time": elapsed
        }
    
    def _write_qrels(self, query_id: str, positives: List[Dict]):
        """Write qrels with multi-grade relevance."""
        with open(self.qrels_path, 'a') as f:
            for p in positives:
                # Format: query_id 0 doc_id relevance_score
                f.write(f"{query_id}\t0\t{p['doc_id']}\t{p.get('relevance_score', 3)}\n")
    
    def _write_triples(
        self, 
        query_id: str, 
        query_text: str, 
        positives: List[Dict], 
        hard_neg_ids: List[str]
    ):
        """Write training triples."""
        if not positives:
            return
        
        triple = {
            "query_id": query_id,
            "query": query_text,
            "positive_doc_ids": [p["doc_id"] for p in positives],
            "positive_scores": [p.get("relevance_score", 3) for p in positives],
            "hard_negative_doc_ids": hard_neg_ids
        }
        
        with open(self.triples_path, 'a') as f:
            f.write(json.dumps(triple) + '\n')
    
    # In agent_runner.py, update the run() method's exception handling:

    def run(self, queries):
        """Run agent on all queries."""
        # Filter already processed
        remaining = [q for q in queries if q["query_id"] not in self.processed_queries]
        
        logger.info(f"Processing {len(remaining):,} queries ({len(self.processed_queries):,} already done)")
        
        total_queries = len(queries)
        checkpoint_interval = self.cfg['agent']['checkpoint_interval']
        
        for i, query_rec in enumerate(tqdm(remaining, desc="Processing queries")):
            try:
                stats = self.process_query(query_rec)
                self.processed_queries.add(query_rec["query_id"])
                
                # Checkpoint every N queries
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint()
                    self._log_stats(total_queries)
                
            except Exception as e:
                logger.error(f"Error processing query {query_rec['query_id']}: {e}")
                import traceback
                logger.error(traceback.format_exc())  # Full traceback
                continue
        
        # Final save
        self._save_checkpoint()
        self._log_stats(total_queries)
        
        logger.info("✓ Agent run complete!")
        logger.info(f"  Qrels: {self.qrels_path}")
        logger.info(f"  Triples: {self.triples_path}")
