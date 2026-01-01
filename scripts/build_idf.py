#!/usr/bin/env python3
import json
import math
import re
from pathlib import Path
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str):
    return [t.lower() for t in WORD_RE.findall(text or "")]

def main():
    passages_path = Path("data/passages/corpus_passages.jsonl")
    out_path = Path("artifacts/idf.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not passages_path.exists():
        raise SystemExit(f"Missing passages file: {passages_path} (run phase0/phase1 once to create it)")

    df = Counter()
    N = 0

    with passages_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text") or ""
            terms = set(tokenize(text))
            if not terms:
                continue
            df.update(terms)
            N += 1

    if N == 0:
        raise SystemExit("No passages found to compute IDF.")

    # Smooth IDF; any monotonic IDF is fine for gating
    idf = {t: (math.log((N + 1) / (c + 1)) + 1.0) for t, c in df.items()}

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(idf, f)

    print(f"✓ Wrote {len(idf):,} tokens to {out_path}")
    print(f"✓ N(passages)={N:,}")

if __name__ == "__main__":
    main()
