#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import Dict, Any, List

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# ---------- Paths & HF config ----------

# repo root = .../hybrid-retriever
ROOT = Path(__file__).resolve().parents[2]

RAW_WIKI_ROOT = ROOT / "raw"                      # raw/<topic>/pages/*.txt
RAW_QUERIES_PATH = ROOT / "outputs" / "global_final_queries.jsonl"
OUTPUT_DIR = ROOT / "wiki_synthetic_prepared"
HF_REPO_ID = "usamaahmedsh/wiki-synthetic-prepared"  # choose your repo base


token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)


# ---------- Conversion helpers ----------

def corpus_records():
    """
    Iterate over raw wiki pages and yield corpus records:
    { "doc_id": "<topic>__<basename>", "text": "<full page text>" }
    """
    for topic_dir in RAW_WIKI_ROOT.iterdir():
        if not topic_dir.is_dir():
            continue
        pages_dir = topic_dir / "pages"
        if not pages_dir.exists():
            continue
        topic = topic_dir.name
        for txt_path in pages_dir.glob("*.txt"):
            base = txt_path.stem
            doc_id = f"{topic}__{base}"
            text = txt_path.read_text(encoding="utf-8").strip()
            yield {"doc_id": doc_id, "text": text}


def queries_records():
    """
    Read synthetic queries and yield:
    { "query_id": "q000001", "text": "...", "topic": "...", "source_id": "..." (optional) }
    """
    with RAW_QUERIES_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)

            # Adapt these keys to your actual global_final_queries.jsonl schema:
            qtext = obj["query"]
            topic = obj.get("topic", "")
            source = obj.get("source_doc", "")  # e.g. filename without .txt, if present

            qid = f"q{i:06d}"
            rec = {
                "query_id": qid,
                "text": qtext,
            }
            if topic:
                rec["topic"] = topic
            if source:
                rec["source_id"] = source
            yield rec


def write_jsonl(path: Path, records_iter):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records_iter:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            f.write(line)


def export_wiki_to_files(out_dir: Path):
    corpus_path = out_dir / "corpus.jsonl"
    queries_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.tsv"  # placeholder for now

    # If already converted, skip re-writing
    if corpus_path.exists() and queries_path.exists():
        print("Wiki converted files already exist, skipping export.")
        return

    print("Writing wiki corpus.jsonl ...")
    write_jsonl(corpus_path, corpus_records())

    print("Writing wiki queries.jsonl ...")
    write_jsonl(queries_path, queries_records())

    # Optional: create empty qrels for now
    if not qrels_path.exists():
        print("Creating empty qrels.tsv placeholder ...")
        with qrels_path.open("w", encoding="utf-8") as f:
            # no rows yet; your agent will fill real qrels later
            pass

    print(f"Files written to {out_dir.resolve()}")


def build_and_push_hf_datasets(out_dir: Path, repo_id: str):
    # 1) Corpus
    print("Loading wiki corpus.jsonl into HF Dataset...")
    corpus = Dataset.from_json(str(out_dir / "corpus.jsonl"))
    corpus.push_to_hub(f"{repo_id}-corpus")

    # 2) Queries
    print("Loading wiki queries.jsonl into HF Dataset...")
    queries = Dataset.from_json(str(out_dir / "queries.jsonl"))
    queries.push_to_hub(f"{repo_id}-queries")

    # 3) Qrels (placeholder if empty)
    qrels_path = out_dir / "qrels.tsv"
    rows: List[Dict[str, Any]] = []
    if qrels_path.exists():
        with qrels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                qid, _, pid, rel, split = line.split("\t")
                rows.append(
                    {
                        "query_id": qid,
                        "doc_id": pid,
                        "relevance": int(rel),
                        "split": split,
                    }
                )
    if rows:
        qrels = Dataset.from_list(rows)
        qrels.push_to_hub(f"{repo_id}-qrels")
    else:
        print("No qrels rows found yet; skipping qrels push for now.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_wiki_to_files(OUTPUT_DIR)
    print("Building and pushing HF datasets (wiki corpus / queries / optional qrels)...")
    build_and_push_hf_datasets(OUTPUT_DIR, HF_REPO_ID)
    print("Done.")


if __name__ == "__main__":
    main()
