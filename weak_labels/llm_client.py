"""Ultra-fast async LLM judging with binary YES/NO."""

from typing import List, Dict, Tuple
import sqlite3
import hashlib
import asyncio
import httpx
from pathlib import Path
from loguru import logger


class AsyncLLMJudgeClient:
    """Optimized async LLM client with binary YES/NO judging."""
    
    def __init__(
        self, 
        base_url: str = "http://127.0.0.1:8080/v1/chat/completions",
        model: str = "local",
        timeout: float = 30.0,
        max_concurrent: int = 20
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        logger.info(f"Async LLM Judge: {base_url} (max_concurrent={max_concurrent}, mode=YES/NO)")
    
    def _quick_filter(
        self, 
        query: str, 
        passages: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Pre-filter with cheap heuristics to reduce LLM calls by 50-70%."""
        query_terms = set(query.lower().split())
        
        keep = []
        skip = []
        
        for p in passages:
            text = p['text'].lower()
            
            # Skip if passage too short
            if len(text.split()) < 30:
                skip.append({
                    "doc_id": p["doc_id"],
                    "relevance_score": 0,
                    "answerable": False,
                    "confidence": 0.0
                })
                continue
            
            # Skip if no query term overlap
            passage_terms = set(text.split()[:100])
            if len(query_terms & passage_terms) == 0:
                skip.append({
                    "doc_id": p["doc_id"],
                    "relevance_score": 0,
                    "answerable": False,
                    "confidence": 0.0
                })
                continue
            
            keep.append(p)
        
        if skip:
            logger.debug(f"Pre-filter: kept {len(keep)}/{len(passages)} passages")
        
        return keep, skip
    
    async def _judge_single(
        self, 
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        query: str, 
        passage: Dict[str, str]
    ) -> Dict:
        """Judge single passage with binary YES/NO."""
        async with semaphore:
            # Clean and truncate text
            passage_text = passage['text'][:500]  # Shorter for 3B
            
            # Remove problematic characters
            passage_text = passage_text.replace('\x00', '')
            passage_text = passage_text.replace('\r\n', '\n')
            passage_text = passage_text.replace('\r', '\n')
            
            query_clean = query.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
            
            # Simple YES/NO prompt (easier for 3B)
            prompt = (
                f"Query: {query_clean}\n"
                f"Passage: {passage_text}\n\n"
                "Can the passage answer the query?\n"
                "Reply ONLY: YES or NO"
            )
            
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 3,
                "stop": ["\n", "."]
            }
            
            try:
                resp = await client.post(self.base_url, json=payload)
                resp.raise_for_status()
                
                content = resp.json()
                text = content["choices"][0]["message"]["content"].strip().upper()
                
                # Parse YES/NO
                answerable = "YES" in text
                
                # Convert to 0-3 scale for compatibility (0 or 3)
                score = 3 if answerable else 0
                confidence = 1.0 if answerable else 0.0
                
                return {
                    "doc_id": passage["doc_id"],
                    "relevance_score": score,
                    "answerable": answerable,
                    "confidence": confidence
                }
            
            except httpx.HTTPStatusError as e:
                try:
                    error_detail = e.response.json()
                    logger.warning(f"LLM error for {passage['doc_id']}: {error_detail.get('error', {}).get('message', 'Unknown')}")
                except:
                    logger.warning(f"LLM error for {passage['doc_id']}: {e.response.text[:100]}")
                
                return {
                    "doc_id": passage["doc_id"],
                    "relevance_score": 0,
                    "answerable": False,
                    "confidence": 0.0
                }
            
            except Exception as e:
                logger.warning(f"LLM failed for {passage['doc_id']}: {str(e)[:100]}")
                return {
                    "doc_id": passage["doc_id"],
                    "relevance_score": 0,
                    "answerable": False,
                    "confidence": 0.0
                }

    async def judge_batch(
        self, 
        query: str, 
        passages: List[Dict[str, str]],
        use_prefilter: bool = True
    ) -> List[Dict]:
        """Judge with pre-filtering and concurrency."""
        
        # Pre-filter to reduce LLM calls
        if use_prefilter:
            to_judge, auto_reject = self._quick_filter(query, passages)
        else:
            to_judge = passages
            auto_reject = []
        
        if not to_judge:
            return auto_reject
        
        # Create semaphore in this event loop context
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Judge remaining passages concurrently
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._judge_single(client, semaphore, query, passage)
                for passage in to_judge
            ]
            llm_results = await asyncio.gather(*tasks)
        
        return llm_results + auto_reject


class CachedLLMJudge:
    """Cached async judge with binary YES/NO."""
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1/chat/completions",
        model: str = "local",
        cache_db: str = "data/cache/llm_judgments.db",
        timeout: float = 30.0,
        max_concurrent: int = 20
    ):
        self.client = AsyncLLMJudgeClient(base_url, model, timeout, max_concurrent)
        self.cache_db = cache_db
        
        cache_path = Path(cache_db)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                query_hash TEXT,
                doc_id TEXT,
                relevance_score INTEGER,
                answerable INTEGER,
                confidence REAL,
                PRIMARY KEY (query_hash, doc_id)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON judgments(query_hash)")
        self.conn.commit()
        
        cache_size = self.conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
        logger.info(f"✓ Binary YES/NO LLM cache: {cache_size} cached judgments")
    
    def _hash(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()[:16]
    
    async def _judge_batch_async(self, query: str, passages: List[Dict]) -> List[Dict]:
        """Async judge with caching."""
        qhash = self._hash(query)
        doc_ids = [p["doc_id"] for p in passages]
        
        if not doc_ids:
            return []
        
        placeholders = ",".join("?" * len(doc_ids))
        
        rows = self.conn.execute(
            f"SELECT doc_id, relevance_score, answerable, confidence FROM judgments "
            f"WHERE query_hash=? AND doc_id IN ({placeholders})",
            [qhash] + doc_ids
        ).fetchall()
        
        cache_map = {
            row[0]: {
                "relevance_score": row[1],
                "answerable": bool(row[2]),
                "confidence": row[3]
            } for row in rows
        }
        
        cached_results = []
        uncached_passages = []
        
        for p in passages:
            if p["doc_id"] in cache_map:
                cached_results.append({"doc_id": p["doc_id"], **cache_map[p["doc_id"]]})
            else:
                uncached_passages.append(p)
        
        if uncached_passages:
            logger.debug(f"Cache: {len(cached_results)} hit, {len(uncached_passages)} miss")
            
            new_results = await self.client.judge_batch(query, uncached_passages)
            
            self.conn.executemany(
                "INSERT OR REPLACE INTO judgments VALUES (?, ?, ?, ?, ?)",
                [
                    (qhash, r["doc_id"], r["relevance_score"], int(r["answerable"]), r["confidence"])
                    for r in new_results
                ]
            )
            self.conn.commit()
            cached_results.extend(new_results)
        else:
            logger.debug(f"Cache: {len(cached_results)} hit (all cached)")
        
        result_map = {r["doc_id"]: r for r in cached_results}
        return [result_map[p["doc_id"]] for p in passages]
    
    def judge_batch(self, query: str, passages: List[Dict]) -> List[Dict]:
        """Sync wrapper."""
        return asyncio.run(self._judge_batch_async(query, passages))
    
    def cache_stats(self) -> Dict[str, int]:
        total = self.conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
        queries = self.conn.execute("SELECT COUNT(DISTINCT query_hash) FROM judgments").fetchone()[0]
        return {"total_judgments": total, "unique_queries": queries}
    
    def clear_cache(self):
        self.conn.execute("DELETE FROM judgments")
        self.conn.commit()
        logger.info("Cache cleared")
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
