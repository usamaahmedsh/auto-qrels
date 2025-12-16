cd ~/Documents/GitHub/weak-labels
source .venv/bin/activate

mkdir -p data/indexes/bm25/tmp_collection

# Convert passages
python3 << 'EOF'
import json
from pathlib import Path

passages_path = Path("data/passages/corpus_passages.jsonl")
tmp_dir = Path("data/indexes/bm25/tmp_collection")
output = tmp_dir / "passages.jsonl"

count = 0
with passages_path.open('r') as inf, output.open('w') as outf:
    for line in inf:
        if line.strip():
            p = json.loads(line)
            rec = {"id": p["doc_id"], "contents": p["text"]}
            outf.write(json.dumps(rec) + '\n')
            count += 1

print(f"✓ Converted {count} passages")
EOF

# Build index
python3 -m pyserini.index \
    -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 8 \
    -input data/indexes/bm25/tmp_collection \
    -index data/indexes/bm25 \
    -storePositions -storeDocvectors -storeRaw
