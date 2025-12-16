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
    Push qrels and triples to HuggingFace Hub.
    
    Args:
        qrels_path: Path to qrels.tsv
        triples_path: Path to triples.jsonl
        repo_id: HuggingFace repo (e.g., "username/dataset-name")
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
    
    # Create repo (if doesn't exist)
    try:
        logger.info(f"Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token
        )
        logger.info("✓ Repository ready")
    except Exception as e:
        logger.warning(f"Repo creation note: {e}")
    
    # Load and push qrels as dataset
    logger.info("Converting qrels to HuggingFace dataset...")
    qrels_data = load_qrels(qrels_path)
    qrels_dataset = Dataset.from_list(qrels_data)
    
    logger.info("Pushing qrels dataset...")
    qrels_dataset.push_to_hub(
        repo_id,
        split="qrels",
        token=token,
        private=private
    )
    logger.info("✓ Qrels pushed")
    
    # Load and push triples as dataset
    logger.info("Converting triples to HuggingFace dataset...")
    triples_data = load_triples(triples_path)
    triples_dataset = Dataset.from_list(triples_data)
    
    logger.info("Pushing triples dataset...")
    triples_dataset.push_to_hub(
        repo_id,
        split="train",
        token=token,
        private=private
    )
    logger.info("✓ Triples pushed")
    
    # Upload raw files as well (for convenience)
    logger.info("Uploading raw files...")
    
    api.upload_file(
        path_or_fileobj=str(qrels_path),
        path_in_repo="qrels.tsv",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    
    api.upload_file(
        path_or_fileobj=str(triples_path),
        path_in_repo="triples.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    
    logger.info("✓ Raw files uploaded")
    
    # Create README
    readme_content = f"""---
license: apache-2.0
task_categories:
- text-retrieval
- information-retrieval
language:
- en
tags:
- weak-supervision
- training-data
- relevance-judgments
size_categories:
- {get_size_category(triples_count)}
---

# Weak Labels Training Data

Automatically generated training data for information retrieval using weak supervision.

## Dataset Description

This dataset contains relevance judgments and training triples for training dense retrievers.

### Statistics

- **Qrels entries**: {qrels_count:,}
- **Training triples**: {triples_count:,}
- **Relevance scale**: 0-3 (0=not relevant, 3=perfectly relevant)

### Splits

- `qrels`: Relevance judgments (query_id, doc_id, relevance_score)
- `train`: Training triples (query, positive_doc_ids, hard_negative_doc_ids)

## Usage

### Load Qrels

from datasets import load_dataset

qrels = load_dataset("{repo_id}", split="qrels")

text

### Load Training Triples

from datasets import load_dataset

triples = load_dataset("{repo_id}", split="train")

text

### Train Dense Retriever

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

Load triples
dataset = load_dataset("{repo_id}", split="train")

Create training samples
train_samples = []
for item in dataset:
query = item['query']
for pos_id, neg_id in zip(item['positive_doc_ids'], item['hard_negative_doc_ids']):
# You'll need to map doc_ids to actual text
train_samples.append(InputExample(texts=[query, pos_text, neg_text]))

Train
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
train_dataloader = DataLoader(train_samples, batch_size=32)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)

text

## Citation

If you use this dataset, please cite:

@misc{{weak-labels-{repo_id.split('/')[-1]},
author = {{{repo_id.split('/')}}},
title = {{Weak Labels Training Data}},
year = {{2025}},
publisher = {{HuggingFace}},
howpublished = {{\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}

text

## Dataset Creation

Generated using weak supervision with:
- BM25 retrieval
- Dense encoder reranking
- Cross-encoder filtering
- LLM relevance judging (binary YES/NO)

## License

Apache 2.0
"""
    
    readme_path = Path("README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    
    readme_path.unlink()  # Clean up
    
    logger.info("✓ README created")
    
    # Print success message
    logger.info("\n" + "="*60)
    logger.info("✓ Successfully pushed to HuggingFace Hub!")
    logger.info("="*60)
    logger.info(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
    logger.info("\nLoad with:")
    logger.info(f'  from datasets import load_dataset')
    logger.info(f'  dataset = load_dataset("{repo_id}")')


def get_size_category(count: int) -> str:
    """Get HuggingFace size category."""
    if count < 1000:
        return "n<1K"
    elif count < 10000:
        return "1K<n<10K"
    elif count < 100000:
        return "10K<n<100K"
    elif count < 1000000:
        return "100K<n<1M"
    else:
        return "1M<n<10M"


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