#!/usr/bin/env python3
"""Push weak labels outputs to HuggingFace Hub."""


import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
from datasets import Dataset
from loguru import logger
import argparse
from dotenv import load_dotenv



def load_qrels(qrels_path: Path):
    """Load qrels.tsv into dataset format."""
    data = []
    
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                query_id, _, doc_id, score = parts
                data.append({
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'relevance_score': int(score)
                })
    
    return data



def load_triples(triples_path: Path):
    """Load triples.jsonl into dataset format."""
    data = []
    
    with open(triples_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return data



def push_to_hub(
    qrels_path: str,
    triples_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False
):
    """
    Push qrels and triples to HuggingFace Hub as separate datasets.
    
    Args:
        qrels_path: Path to qrels.tsv
        triples_path: Path to triples.jsonl
        repo_id: HuggingFace repo base (e.g., "username/dataset-name")
        token: HF API token (or set HF_TOKEN env var)
        private: Make repo private
    """
    qrels_path = Path(qrels_path)
    triples_path = Path(triples_path)
    
    # Get token from env if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN env var or pass token argument.\n"
                "Get token from: https://huggingface.co/settings/tokens"
            )
    
    logger.info(f"Pushing to HuggingFace Hub: {repo_id}")
    
    # Check files exist
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels not found: {qrels_path}")
    
    if not triples_path.exists():
        raise FileNotFoundError(f"Triples not found: {triples_path}")
    
    # Count entries
    qrels_count = sum(1 for _ in open(qrels_path))
    triples_count = sum(1 for _ in open(triples_path))
    
    logger.info(f"Found {qrels_count:,} qrels entries")
    logger.info(f"Found {triples_count:,} training triples")
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create qrels repo
    qrels_repo_id = f"{repo_id}-qrels"
    try:
        logger.info(f"Creating qrels repository: {qrels_repo_id}")
        create_repo(
            repo_id=qrels_repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token
        )
        logger.info("✓ Qrels repository ready")
    except Exception as e:
        logger.warning(f"Qrels repo creation note: {e}")
    
    # Load and push qrels as dataset
    logger.info("Converting qrels to HuggingFace dataset...")
    qrels_data = load_qrels(qrels_path)
    qrels_dataset = Dataset.from_list(qrels_data)
    
    logger.info("Pushing qrels dataset...")
    qrels_dataset.push_to_hub(
        qrels_repo_id,
        split="train",
        token=token,
        private=private
    )
    
    # Upload raw qrels file
    api.upload_file(
        path_or_fileobj=str(qrels_path),
        path_in_repo="qrels.tsv",
        repo_id=qrels_repo_id,
        repo_type="dataset",
        token=token
    )
    logger.info("✓ Qrels pushed")
    
    # Create triples repo
    triples_repo_id = f"{repo_id}-triples"
    try:
        logger.info(f"Creating triples repository: {triples_repo_id}")
        create_repo(
            repo_id=triples_repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token
        )
        logger.info("✓ Triples repository ready")
    except Exception as e:
        logger.warning(f"Triples repo creation note: {e}")
    
    # Load and push triples as dataset
    logger.info("Converting triples to HuggingFace dataset...")
    triples_data = load_triples(triples_path)
    triples_dataset = Dataset.from_list(triples_data)
    
    logger.info("Pushing triples dataset...")
    triples_dataset.push_to_hub(
        triples_repo_id,
        split="train",
        token=token,
        private=private
    )
    
    # Upload raw triples file
    api.upload_file(
        path_or_fileobj=str(triples_path),
        path_in_repo="triples.jsonl",
        repo_id=triples_repo_id,
        repo_type="dataset",
        token=token
    )
    logger.info("✓ Triples pushed")
    
    # Print success message
    logger.info("\n" + "="*60)
    logger.info("✓ Successfully pushed to HuggingFace Hub!")
    logger.info("="*60)
    logger.info(f"\nQrels dataset: https://huggingface.co/datasets/{qrels_repo_id}")
    logger.info(f"Triples dataset: https://huggingface.co/datasets/{triples_repo_id}")
    logger.info("\nLoad with:")
    logger.info(f'  qrels = load_dataset("{qrels_repo_id}")')
    logger.info(f'  triples = load_dataset("{triples_repo_id}")')



def main():

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Push weak labels outputs to HuggingFace Hub"
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default="data/output/qrels.tsv",
        help="Path to qrels.tsv"
    )
    parser.add_argument(
        "--triples",
        type=str,
        default="data/output/triples.jsonl",
        help="Path to triples.jsonl"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    
    args = parser.parse_args()
    
    push_to_hub(
        qrels_path=args.qrels,
        triples_path=args.triples,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private
    )



if __name__ == "__main__":
    main()
