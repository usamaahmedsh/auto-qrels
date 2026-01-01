#!/usr/bin/env python3
"""Push weak labels outputs to HuggingFace Hub."""

import os
from pathlib import Path
import argparse
from loguru import logger
from dotenv import load_dotenv

from huggingface_hub import HfApi, create_repo
from datasets import load_dataset


def push_to_hub(
    qrels_path: str,
    triples_path: str,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    upload_raw: bool = True,
    push_datasets: bool = True,
    revision: str = "main",
    commit_message: str = "Add weak-labels exports (qrels + triples)",
):
    qrels_path = Path(qrels_path)
    triples_path = Path(triples_path)

    if token is None:
        token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var or pass --token.\n"
            "Get token from: https://huggingface.co/settings/tokens"
        )

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels not found: {qrels_path}")
    if not triples_path.exists():
        raise FileNotFoundError(f"Triples not found: {triples_path}")

    api = HfApi(token=token)

    # Single dataset repo with both files is usually simplest for downstream users.
    # (You can still load individual files via load_dataset with data_files=... later.)
    logger.info(f"Creating/using dataset repo: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    if upload_raw:
        # Upload the two files (or whole folder) as raw artifacts.
        # This is the most robust approach for large exports. [web:181]
        logger.info("Uploading raw files to repo...")
        api.upload_file(
            path_or_fileobj=str(qrels_path),
            path_in_repo="qrels.tsv",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            revision=revision,
            commit_message=commit_message,
        )
        api.upload_file(
            path_or_fileobj=str(triples_path),
            path_in_repo="triples.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            revision=revision,
            commit_message=commit_message,
        )
        logger.info("✓ Raw files uploaded")

    if push_datasets:
        # Create HF Dataset objects from disk (more memory-friendly than from_list). [web:174]
        logger.info("Loading qrels.tsv as a Dataset (csv with delimiter=\\t)...")
        qrels_ds = load_dataset(
            "csv",
            data_files=str(qrels_path),
            delimiter="\t",
            # qrels has no header; enforce column names:
            column_names=["query_id", "iter", "doc_id", "relevance"],
            split="train",
        )

        logger.info("Loading triples.jsonl as a Dataset (json lines)...")
        triples_ds = load_dataset(
            "json",
            data_files=str(triples_path),
            split="train",
        )

        # Push as dataset configs/splits into the SAME repo:
        # - config "qrels": train split
        # - config "triples": train split
        # This keeps everything under repo_id rather than creating two repos.
        logger.info("Pushing qrels dataset config to Hub...")
        qrels_ds.push_to_hub(
            repo_id,
            config_name="qrels",
            split="train",
            token=token,
            private=private,
            commit_message=commit_message,
        )
        logger.info("✓ qrels dataset pushed")

        logger.info("Pushing triples dataset config to Hub...")
        triples_ds.push_to_hub(
            repo_id,
            config_name="triples",
            split="train",
            token=token,
            private=private,
            commit_message=commit_message,
        )
        logger.info("✓ triples dataset pushed")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Successfully pushed to HuggingFace Hub!")
    logger.info("=" * 60)
    logger.info(f"Dataset repo: https://huggingface.co/datasets/{repo_id}")
    logger.info("\nLoad with:")
    logger.info(f'  from datasets import load_dataset')
    logger.info(f'  qrels = load_dataset("{repo_id}", "qrels")')
    logger.info(f'  triples = load_dataset("{repo_id}", "triples")')


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Push weak labels outputs to HuggingFace Hub")
    parser.add_argument("--qrels", type=str, default="data/output/qrels.tsv")
    parser.add_argument("--triples", type=str, default="data/output/triples.jsonl")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--private", action="store_true")

    # pipeline integration guard:
    parser.add_argument(
        "--auto-push",
        action="store_true",
        help="Actually push to Hub. If not set, script exits without doing anything.",
    )

    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not upload raw qrels.tsv / triples.jsonl files.",
    )
    parser.add_argument(
        "--no-datasets",
        action="store_true",
        help="Do not push as HF Dataset configs; only upload raw files.",
    )

    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--commit-message", type=str, default="Add weak-labels exports (qrels + triples)")

    args = parser.parse_args()

    if not args.auto_push:
        logger.info("auto_push disabled; skipping Hugging Face upload.")
        return

    push_to_hub(
        qrels_path=args.qrels,
        triples_path=args.triples,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        upload_raw=(not args.no_raw),
        push_datasets=(not args.no_datasets),
        revision=args.revision,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
