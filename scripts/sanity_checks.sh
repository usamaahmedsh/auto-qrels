#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'  # safer word-splitting in bash scripts [web:154]

QRELS_PATH="${1:-data/output/qrels.tsv}"
TRIPLES_PATH="${2:-data/output/triples.jsonl}"

die() { echo "ERROR: $*" >&2; exit 1; }

echo "== Sanity checks =="
echo "qrels:   $QRELS_PATH"
echo "triples: $TRIPLES_PATH"
echo

command -v jq >/dev/null 2>&1 || die "jq not found (module load jq / install jq)."
command -v python >/dev/null 2>&1 || die "python not found."

[[ -f "$QRELS_PATH" ]]   || die "Missing qrels file: $QRELS_PATH"
[[ -f "$TRIPLES_PATH" ]] || die "Missing triples file: $TRIPLES_PATH"

[[ -s "$QRELS_PATH" ]]   || die "Empty qrels file: $QRELS_PATH"
[[ -s "$TRIPLES_PATH" ]] || die "Empty triples file: $TRIPLES_PATH"

echo "## Basic counts"
QRELS_LINES="$(wc -l < "$QRELS_PATH")"
TRIPLES_LINES="$(wc -l < "$TRIPLES_PATH")"
echo "qrels lines:   $QRELS_LINES"
echo "triples lines: $TRIPLES_LINES"
echo

echo "## qrels format checks (TREC-style: qid <tab> 0 <tab> docid <tab> rel)"
awk -F'\t' 'NF!=4 {print "Bad qrels line " NR ": expected 4 tab fields, got " NF; exit 1}' "$QRELS_PATH" >/dev/null
awk -F'\t' '$2!="0" {print "Bad qrels line " NR ": iter field not 0 -> " $2; exit 1}' "$QRELS_PATH" >/dev/null
awk -F'\t' '$4 !~ /^-?[0-9]+$/ {print "Bad qrels line " NR ": rel not integer -> " $4; exit 1}' "$QRELS_PATH" >/dev/null

echo "qrels relevance distribution:"
cut -f4 "$QRELS_PATH" | sort | uniq -c | sort -nr
echo

echo "qrels duplicate (qid,docid) pairs (show up to 20):"
# guard with || true to avoid SIGPIPE terminating script [web:169]
awk -F'\t' '{print $1 "\t" $3}' "$QRELS_PATH" | sort | uniq -d | head -n 20 || true
echo

echo "qrels positives per query (top 20 by count):"
cut -f1 "$QRELS_PATH" | sort | uniq -c | sort -nr | head -n 20 || true
echo

echo "## triples.jsonl checks"
# Validate JSONL (jq exits non-zero if any line is invalid JSON) [web:161]
jq -e . "$TRIPLES_PATH" >/dev/null

jq -e 'has("query_id") and has("query") and has("pos_doc_id") and has("pos_text") and has("neg_doc_id") and has("neg_text") and has("neg_kind")' \
  "$TRIPLES_PATH" >/dev/null

jq -e '(.query_id|type=="string") and (.query|type=="string") and
       (.pos_doc_id|type=="string") and (.pos_text|type=="string") and
       (.neg_doc_id|type=="string") and (.neg_text|type=="string") and
       (.neg_kind|type=="string")' "$TRIPLES_PATH" >/dev/null

echo "triples neg_kind distribution:"
jq -r '.neg_kind' "$TRIPLES_PATH" | sort | uniq -c | sort -nr
echo

echo "triples with pos_doc_id == neg_doc_id (should be 0; show up to 20):"
jq -r 'select(.pos_doc_id == .neg_doc_id) | .query_id' "$TRIPLES_PATH" | head -n 20 || true
echo

echo "## triples text length stats"
python - "$TRIPLES_PATH" <<'PY'
import sys, json

triples_path = sys.argv[1]
pos_lens = []
neg_lens = []

with open(triples_path, "r", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        pos_lens.append(len(o["pos_text"]))
        neg_lens.append(len(o["neg_text"]))

def summarize(xs):
    return min(xs), sum(xs)/len(xs), max(xs)

pmin, pavg, pmax = summarize(pos_lens)
nmin, navg, nmax = summarize(neg_lens)

print("pos_text chars: min/avg/max =", pmin, f"{pavg:.1f}", pmax)
print("neg_text chars: min/avg/max =", nmin, f"{navg:.1f}", nmax)
PY
echo

echo "✅ All sanity checks passed."
