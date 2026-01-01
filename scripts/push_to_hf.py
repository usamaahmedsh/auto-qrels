#!/usr/bin/env python3
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "usamaahmedsh/weak-labels-wiki"
REPO_TYPE = "dataset"

OUTPUT_DIR = Path("/project/rhino-ffm/auto-qrels/data/output")
QRELS = OUTPUT_DIR / "qrels.tsv"
TRIPLES = OUTPUT_DIR / "triples.jsonl"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Do: export HF_TOKEN='...'(use a WRITE token).")

    if not QRELS.exists():
        raise FileNotFoundError(f"Missing file: {QRELS}")
    if not TRIPLES.exists():
        raise FileNotFoundError(f"Missing file: {TRIPLES}")

    api = HfApi()

    # Repo already exists; this makes the script idempotent in case it doesn't.
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        exist_ok=True,
        token=token,
    )  # repo_type supports "dataset"/"model"/"space" [web:64]

    api.upload_file(
        path_or_fileobj=str(QRELS),
        path_in_repo="qrels.tsv",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        commit_message="Add qrels.tsv",
    )  # upload_file supports repo_type="dataset" [web:64]

    api.upload_file(
        path_or_fileobj=str(TRIPLES),
        path_in_repo="triples.jsonl",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        commit_message="Add triples.jsonl",
    )  # upload_file supports repo_type="dataset" [web:64]

    print("Done: uploaded qrels.tsv and triples.jsonl to", REPO_ID)

if __name__ == "__main__":
    main()
