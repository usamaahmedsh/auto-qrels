# Weak Labels Generation System

> Automated training data generation for information retrieval using weak supervision

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20|%20Linux-lightgrey.svg)]()

---

## Overview

**Weak Labels** automatically generates high-quality training data for information retrieval models without manual annotation. It combines multiple retrieval stages with LLM-based relevance judging to create:

- **Graded relevance judgments** (qrels) for evaluation
- **Training triples** (query, positive, hard negatives) for contrastive learning

---

## Pipeline Architecture

The system uses a multi-stage cascade to balance speed, quality, and cost:

```
┌─────────────┐
│   Queries   │  332,992 queries
│  (Input)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  Stage 1: BM25          │  Fast lexical retrieval
│  -  Pure Python (bm25s)  │  -  Top-200 candidates per query
│  -  Entire corpus        │  -  ~0.01s per query
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Stage 2: Dense Encoder │  Semantic reranking
│  -  BGE-base-en-v1.5     │  -  Rerank top-100 from BM25
│  -  GPU accelerated      │  -  ~0.05s per query
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Stage 3: Cross-Encoder │  High-precision filtering
│  -  ms-marco-MiniLM      │  -  Score top-40 candidates
│  -  Pairwise scoring     │  -  ~0.02s per query
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Stage 4: Smart Sample  │  Diversity-based selection
│  -  Max 2 per document   │  -  Reduce to ~25 passages
│  -  Spread across ranks  │  -  Avoid over-representation
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Stage 5: LLM Judge     │  Binary relevance judging
│  -  Llama 3.2 3B (local) │  -  YES/NO per passage
│  -  Async (20 parallel)  │  -  Cached (SQLite)
│  -  Pre-filter heuristics│  -  ~0.30s per query
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Output Generation      │
│  -  Positives (score=3)  │  Qrels: relevance judgments
│  -  Hard negatives       │  Triples: training data
│  -  Auto-push to HF      │  Ready for model training
└─────────────────────────┘
```

### Why This Architecture?

Each stage serves a specific purpose:

| Stage | Purpose | Speed | Quality | Cost |
|-------|---------|-------|---------|------|
| **BM25** | High recall over 1M+ docs | ⚡⚡⚡ | ⭐⭐ | Free |
| **Dense** | Semantic understanding | ⚡⚡ | ⭐⭐⭐ | Free |
| **Cross-Encoder** | Precision filtering | ⚡ | ⭐⭐⭐⭐ | Free |
| **LLM** | Final relevance decision | 🐌 | ⭐⭐⭐⭐⭐ | Low* |

*Local LLM = $0, API = ~$0.005 per query with Groq

**Key benefit:** Only the expensive LLM sees ~25 passages (not 1M+), keeping costs manageable while maintaining quality.

---

## What The Agent Does

### Core Workflow

For each query, the agent:

1. **Retrieves** top-200 candidates using BM25 (lexical matching)
2. **Reranks** top-100 using dense encoder (semantic similarity)
3. **Filters** top-40 using cross-encoder (pairwise scoring)
4. **Samples** ~25 diverse passages (avoid redundancy)
5. **Judges** each passage with LLM (binary YES/NO)
6. **Extracts** positives (YES judgments) as relevant documents
7. **Mines** hard negatives from highly-ranked but irrelevant passages
8. **Outputs** qrels and training triples

### Smart Optimizations

**1. Pre-filtering Heuristics**

Before calling the LLM, automatically reject passages that:
- Are too short (< 30 tokens)
- Have no query term overlap
- Were already judged (cache hit)

**Result:** 50-70% fewer LLM calls

**2. Caching**

All LLM judgments stored in SQLite:
```
(query_hash, doc_id) → (relevance_score, answerable)
```
- Survives script restarts
- Enables iterative development
- Reuses judgments across runs

**3. Hard Negative Mining**

Automatically identifies "hard negatives":
- Rank highly in BM25/Dense (seem relevant)
- But judged NOT relevant by LLM (false positives)
- Perfect for training contrastive models

Selection strategy:
- Top BM25 results NOT in positive set
- Top Dense results NOT in positive set  
- Retrieval disagreements (BM25 ≠ Dense)
- Max 10 per query for balanced training

**4. Checkpointing**

Saves progress every 100 queries:
```
{
  "processed_query_ids": ["q001", "q002", ...],
  "stats": {"queries_processed": 100, ...}
}
```
Resume automatically from interruptions.

### Performance Stats

**Test environment:** M2 MacBook Pro, 32GB RAM

| Metric | Value |
|--------|-------|
| **Speed** | ~0.4s per query |
| **Throughput** | ~2,600 queries/hour |
| **LLM calls** | ~25 per query (after pre-filter) |
| **Cache hit rate** | 30-50% (increases over time) |

**Full dataset (332k queries):**
- Runtime: ~90-120 hours (3-5 days)
- Total LLM calls: ~8.3 million
- Output size: ~500MB (qrels + triples)

---

## Features

- **Fast Processing**: Async LLM judging with 20+ concurrent requests
- **High Quality**: Multi-stage filtering (BM25 → Dense → Cross-Encoder → LLM)
- **Smart Caching**: SQLite cache prevents redundant LLM calls
- **Checkpointing**: Resume from interruptions automatically
- **HuggingFace Integration**: Auto-push outputs to HF Hub
- **Local LLM Support**: Run llama.cpp models locally (no API costs)

---

## Installation

### Prerequisites

- Python 3.11+
- 16GB+ RAM recommended
- GPU optional (Metal/CUDA for faster dense encoding)
- ~50GB disk space for models + data

### Setup

**1. Clone repository**

```
git clone https://github.com/usamaahmedsh/auto-qrels.git
cd auto-qrels
```

**2. Install llama.cpp**

```
cd ~/Documents/GitHub
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_METAL=ON  # For macOS
cmake --build . --config Release -j$(sysctl -n hw.ncpu)
```

**3. Install Python dependencies**

```
cd ~/Documents/GitHub/weak-labels
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**4. Set environment variables**

Create `.env` file:

```
HF_TOKEN=hf_your_huggingface_token_here
```

Get token from: https://huggingface.co/settings/tokens

---

## Quick Start

### Test Run (10 queries)

**Step 1: Configure for testing**

Edit `configs/base.yaml`:

```
agent:
  max_queries: 10  # Process only 10 queries

huggingface:
  auto_push: true
  repo_id: "yourusername/weak-labels-test"
  private: true
```

**Step 2: Run pipeline**

```
chmod +x run_agent.sh
./run_agent.sh
```

Expected time: ~30-60 seconds for 10 queries

**Step 3: Verify outputs**

```
# Check qrels
head data/output/qrels.tsv

# Check triples
head data/output/triples.jsonl

# Check HuggingFace
open https://huggingface.co/datasets/yourusername/weak-labels-test
```

### Full Run (All queries)

Once test succeeds, edit `configs/base.yaml`:

```
agent:
  max_queries: null  # Process all queries

huggingface:
  auto_push: true
  repo_id: "yourusername/weak-labels-wiki"
```

```
./run_agent.sh
```

Expected time: ~90-120 hours for 332k queries

---

## Configuration

### Dataset

```
dataset:
  corpus:
    name: "usamaahmedsh/wiki-synthetic-prepared-corpus"
    split: "train"
  queries:
    name: "usamaahmedsh/wiki-synthetic-prepared-queries"
    split: "train"
```

### Retrieval

```
bm25:
  k1: 0.9          # Term saturation
  b: 0.4           # Length normalization

dense:
  model_name: "BAAI/bge-base-en-v1.5"
  device: "mps"    # "cuda", "cpu", or "mps"

agent:
  global_top_k_bm25: 200           # BM25 candidates
  dense_top_k_from_bm25: 100       # Dense rerank top-k
  hard_negatives_per_query: 10     # Hard negatives
```

### LLM

**Local LLM (llama.cpp):**

```
llm:
  base_url: "http://127.0.0.1:8080/v1/chat/completions"
  model: "llama-3.2-3b-instruct"
  timeout: 30.0
  max_concurrent: 20
```

**API-based LLM (e.g., Groq - FREE):**

```
llm:
  base_url: "https://api.groq.com/openai/v1/chat/completions"
  model: "llama-3.3-70b-versatile"
  timeout: 30.0
  max_concurrent: 30
```

### Output

```
output:
  qrels_path: "data/output/qrels.tsv"
  triples_path: "data/output/triples.jsonl"

huggingface:
  auto_push: true
  repo_id: "yourusername/weak-labels-wiki"
  private: false
```

---

## Output Format

### Qrels (Relevance Judgments)

**Format:** `query_id \t 0 \t doc_id \t relevance_score`

```
q000001	0	einstein__relativity_p5	3
q000001	0	youtube__einstein_p2	3
q000002	0	newton__gravity_p12	3
```

**Relevance scale:**
- `3`: Relevant (YES judgment from LLM)
- `0`: Not relevant (NO judgment or filtered out)

**Statistics (332k queries):**
- Total judgments: ~665,000
- Avg positives per query: ~2.0

### Triples (Training Data)

**Format:** JSONL with query, positives, hard negatives

```
{
  "query_id": "q000001",
  "query": "What is the theory of relativity?",
  "positive_doc_ids": ["einstein__relativity_p5"],
  "positive_scores": ,
  "hard_negative_doc_ids": ["newton__gravity_p12", "quantum__uncertainty_p4", ...]
}
```

**Statistics:**
- Total triples: ~333,000 (one per query)
- Avg positives: 2.0
- Avg hard negatives: 10.0

---

## Training Models

### Load from HuggingFace

```
from datasets import load_dataset

# Load your generated dataset
dataset = load_dataset("yourusername/weak-labels-wiki")

triples = dataset["train"]
qrels = dataset["qrels"]
```

### Train Bi-Encoder

```
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Create training samples
train_samples = []
for item in triples:
    query = item['query']
    for pos_id in item['positive_doc_ids']:
        for neg_id in item['hard_negative_doc_ids'][:3]:
            # Map doc_ids to text from your corpus
            train_samples.append(
                InputExample(texts=[query, pos_text, neg_text])
            )

# Train
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
train_dataloader = DataLoader(train_samples, batch_size=32, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
```

### Evaluate

```
import pytrec_eval

# Load qrels
qrels = {}
for item in qrels_data:
    qid, docid, score = item['query_id'], item['doc_id'], item['relevance_score']
    qrels.setdefault(qid, {})[docid] = score

# Evaluate your model
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10', 'map'})
results = evaluator.evaluate(run)
```

---

## License

Apache License 2.0

---

## Citation

```
@software{weak_labels_2025,
  title = {Weak Labels: Automated Training Data Generation for Information Retrieval},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/weak-labels}
}
```

