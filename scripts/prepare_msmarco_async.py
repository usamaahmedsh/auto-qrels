#!/usr/bin/env python3
import os
import json
from pathlib import Path
import asyncio
from typing import Dict, Any, List

import ir_datasets
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

OUTPUT_DIR = Path("msmarco_passage_prepared_async")
HF_REPO_ID = "usamaahmedsh/msmarco-passage-prepared"  # TODO: change

token = os.environ.get("HF_TOKEN")
login(token=token)


# ---------- IO helpers ----------

async def async_write_jsonl(path: Path, records_iter):
    path.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_event_loop()
    with path.open("w", encoding="utf-8") as f:
        for rec in records_iter:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            await loop.run_in_executor(None, f.write, line)


async def async_write_qrels(path: Path, lines_iter):
    path.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_event_loop()
    with path.open("w", encoding="utf-8") as f:
        for line in lines_iter:
            await loop.run_in_executor(None, f.write, line)


# ---------- Data extraction ----------

def load_msmarco_passage():
    print("Loading MS MARCO Passage via ir_datasets...")
    ds_train = ir_datasets.load("msmarco-passage/train")
    ds_dev = ir_datasets.load("msmarco-passage/dev")
    return ds_train, ds_dev


def corpus_records(ds_train):
    for doc in ds_train.docs_iter():
        yield {"doc_id": doc.doc_id, "text": doc.text}


def queries_records(ds_train, ds_dev):
    for q in ds_train.queries_iter():
        yield {"query_id": q.query_id, "text": q.text, "split": "train"}
    for q in ds_dev.queries_iter():
        yield {"query_id": q.query_id, "text": q.text, "split": "dev"}


def qrels_lines(ds_train, ds_dev):
    for rel in ds_train.qrels_iter():
        yield f"{rel.query_id}\t0\t{rel.doc_id}\t{rel.relevance}\ttrain\n"
    for rel in ds_dev.qrels_iter():
        yield f"{rel.query_id}\t0\t{rel.doc_id}\t{rel.relevance}\tdev\n"


# ---------- Orchestration ----------

async def export_to_files_async(ds_train, ds_dev, out_dir: Path):
    corpus_path = out_dir / "corpus.jsonl"
    queries_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.tsv"

    # If already converted, skip re-writing
    if corpus_path.exists() and queries_path.exists() and qrels_path.exists():
        print("Converted files already exist, skipping export.")
        return

    print("Writing corpus, queries, qrels (async-style)...")
    await asyncio.gather(
        async_write_jsonl(corpus_path, corpus_records(ds_train)),
        async_write_jsonl(queries_path, queries_records(ds_train, ds_dev)),
        async_write_qrels(qrels_path, qrels_lines(ds_train, ds_dev)),
    )
    print(f"Files written to {out_dir.resolve()}")


def build_and_push_hf_datasets(out_dir: Path, repo_id: str):
    # 1) Corpus
    print("Loading corpus.jsonl into HF Dataset...")
    corpus = Dataset.from_json(str(out_dir / "corpus.jsonl"))
    corpus.push_to_hub(f"{repo_id}-corpus")

    # 2) Queries
    print("Loading queries.jsonl into HF Dataset...")
    queries = Dataset.from_json(str(out_dir / "queries.jsonl"))
    queries.push_to_hub(f"{repo_id}-queries")

    # 3) Qrels
    print("Loading qrels.tsv into HF Dataset...")
    rows: List[Dict[str, Any]] = []
    with (out_dir / "qrels.tsv").open("r", encoding="utf-8") as f:
        for line in f:
            qid, _, pid, rel, split = line.strip().split("\t")
            rows.append(
                {
                    "query_id": qid,
                    "doc_id": pid,
                    "relevance": int(rel),
                    "split": split,
                }
            )
    qrels = Dataset.from_list(rows)
    qrels.push_to_hub(f"{repo_id}-qrels")


async def main_async():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds_train, ds_dev = load_msmarco_passage()
    await export_to_files_async(ds_train, ds_dev, OUTPUT_DIR)

    print("Building and pushing HF datasets (corpus / queries / qrels)...")
    build_and_push_hf_datasets(OUTPUT_DIR, HF_REPO_ID)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main_async())