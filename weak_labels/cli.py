#!/usr/bin/env python3
"""Main CLI entrypoint for Weak Labels agent."""

import sys
import json
from pathlib import Path
from loguru import logger
from datasets import load_dataset
from tqdm import tqdm

from .config import load_config, validate_paths
from .chunker import PassageChunker
from .bm25_index import BM25Index
from .dense_encoder import DenseEncoder
from .llm_client import CachedLLMJudge
from .agent_runner import Agent
from dotenv import load_dotenv  


def setup_logging(log_dir: Path):
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "agent.log"
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        str(log_file),
        rotation="100 MB",
        retention="7 days",
        level="DEBUG"
    )


def main():
    """Main entry point."""

    load_dotenv()
    
    # Load config
    cfg = load_config()
    
    # Setup logging
    log_dir = Path("logs")
    setup_logging(log_dir)
    
    logger.info("="*60)
    logger.info("Weak Labels Agent - Starting")
    logger.info("="*60)
    
    # Validate paths
    if not validate_paths(cfg):
        logger.error("Path validation failed!")
        sys.exit(1)
    
    # Load datasets
    logger.info("\nLoading datasets...")
    
    # Access config correctly - use raw dict for nested access
    corpus = load_dataset(
        cfg.raw['dataset']['corpus']['name'],
        split=cfg.raw['dataset']['corpus']['split'],
        cache_dir=cfg.raw['paths']['hf_cache_dir']
    )
    
    queries_dataset = load_dataset(
        cfg.raw['dataset']['queries']['name'],
        split=cfg.raw['dataset']['queries']['split'],
        cache_dir=cfg.raw['paths']['hf_cache_dir']
    )
    
    logger.info(f"✓ Corpus: {len(corpus):,} documents")
    logger.info(f"✓ Queries: {len(queries_dataset):,} queries")
    
    # Convert queries dataset to list of dicts
    logger.info("\nPreparing queries...")
    queries = []
    for item in tqdm(queries_dataset, desc="Converting queries"):
        # Handle different possible field names
        query_id = item.get('query_id') or item.get('id') or item.get('_id')
        query_text = item.get('text') or item.get('query') or item.get('question')
        
        if query_id and query_text:
            queries.append({
                'query_id': str(query_id),
                'text': str(query_text)
            })
    
    # Limit queries if max_queries is set (for testing)
    max_queries = cfg.raw['agent'].get('max_queries')
    if max_queries is not None:
        original_count = len(queries)
        queries = queries[:max_queries]
        logger.warning(f"⚠ TEST MODE: Limiting to {max_queries:,} queries (out of {original_count:,})")
    
    logger.info(f"✓ Prepared {len(queries):,} queries")
    
    # Prepare passages
    passages_path = Path(cfg.raw['paths']['prepared_dir']) / "corpus_passages.jsonl"
    passages_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not passages_path.exists():
        logger.info("\nChunking corpus into passages...")
        
        chunker = PassageChunker(
            chunk_size=cfg.raw['corpus']['passage_tokens'],
            stride=cfg.raw['corpus']['passage_stride']
        )
        
        passages = chunker.chunk_corpus(corpus)
        
        # Save passages
        logger.info(f"Saving passages to {passages_path}...")
        with passages_path.open('w') as f:
            for p in tqdm(passages, desc="Writing passages"):
                f.write(json.dumps(p) + '\n')
        
        logger.info(f"✓ Saved {len(passages):,} passages")
    else:
        passage_count = sum(1 for _ in passages_path.open('r'))
        logger.info(f"✓ Using existing passages: {passage_count:,}")
    
    # Initialize components
    logger.info("\nInitializing retrieval components...")
    
    # BM25
    bm25 = BM25Index(
        passages_path,
        k1=cfg.raw['bm25']['k1'],
        b=cfg.raw['bm25']['b']
    )
    
    # Dense encoder
    dense_enc = DenseEncoder(
        model_name=cfg.raw['dense']['model_name']
    )
    
    # LLM judge
    llm_judge = CachedLLMJudge(
        base_url=cfg.raw['llm']['base_url'],
        model=cfg.raw['llm']['model'],
        cache_db=str(Path(cfg.raw['paths']['prepared_dir']) / "llm_cache.db"),
        timeout=cfg.raw['llm'].get('timeout', 30.0),
        max_concurrent=cfg.raw['llm'].get('max_concurrent', 20)
    )
    
    logger.info("✓ All components initialized")
    
    # Initialize agent
    logger.info("\nInitializing agent...")
    agent = Agent(
        bm25=bm25,
        dense_encoder=dense_enc,
        llm_judge=llm_judge,
        config=cfg.raw
    )
    
    # Run agent
    logger.info("\nStarting query processing...")
    agent.run(queries)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Agent finished successfully")
    logger.info("="*60)
    
    # Auto-push to HuggingFace (if configured)
    if cfg.raw.get('huggingface', {}).get('auto_push', False):
        logger.info("\nPushing outputs to HuggingFace Hub...")
        
        try:
            from .push_to_hf import push_to_hub
            
            push_to_hub(
                qrels_path=cfg.raw['output']['qrels_path'],
                triples_path=cfg.raw['output']['triples_path'],
                repo_id=cfg.raw['huggingface']['repo_id'],
                token=cfg.raw['huggingface'].get('token'),
                private=cfg.raw['huggingface'].get('private', False)
            )
        except Exception as e:
            logger.error(f"Failed to push to HuggingFace: {e}")
            logger.warning("Continuing anyway...")


if __name__ == "__main__":
    main()
