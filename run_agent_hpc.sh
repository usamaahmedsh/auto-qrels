#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ulimit -n 16384
echo "nofile soft limit: $(ulimit -Sn)  hard limit: $(ulimit -Hn)"

VLLM_HOST="127.0.0.1"
VLLM_PORT=8000
VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct"

VLLM_PID_FILE="$PROJECT_ROOT/vllm_server.pid"
VLLM_LOG_FILE="$PROJECT_ROOT/vllm_server.log"

DENSE_INDEX_PATH="${DENSE_INDEX_PATH:-data/indexes/passage_embs.npz}"
PHASE1_PROMPTS_FILE="${PHASE1_PROMPTS_FILE:-$PROJECT_ROOT/data/prepared/judge_prompts.jsonl}"

# Phase 2 cache + completion marker (fast check; no SQLite COUNT)
LLM_CACHE_DB="${LLM_CACHE_DB:-$PROJECT_ROOT/data/prepared/llm_cache.db}"
PHASE2_DONE_MARKER="${PHASE2_DONE_MARKER:-$PROJECT_ROOT/data/checkpoints/phase2.done}"

export HF_HOME="${HF_HOME:-/project/rhino-ffm/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set in the environment. Export HF_TOKEN before running."
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"

echo "== GPU assignment (from Slurm) =="

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "Original CUDA_VISIBLE_DEVICES from environment: ${CUDA_VISIBLE_DEVICES}"
  FIRST_VISIBLE_GPU="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print $1}')"
  export CUDA_VISIBLE_DEVICES="$FIRST_VISIBLE_GPU"
  echo "Restricting this job to a single assigned GPU: $FIRST_VISIBLE_GPU"
  echo "Effective CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES="0"
  echo "WARNING: CUDA_VISIBLE_DEVICES was not set; defaulting to GPU 0 only."
  echo "Effective CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi

echo "== CPU / PyTorch settings =="

if [ -n "${NSLOTS:-}" ]; then
  CPU_CORES="$NSLOTS"
elif [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  CPU_CORES="$SLURM_CPUS_PER_TASK"
else
  CPU_CORES="$(nproc 2>/dev/null || echo 1)"
fi

export OMP_NUM_THREADS="$CPU_CORES"
export MKL_NUM_THREADS="$CPU_CORES"
export OPENBLAS_NUM_THREADS="$CPU_CORES"
export NUMEXPR_NUM_THREADS="$CPU_CORES"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF

echo "Using $CPU_CORES CPU threads (NSLOTS=${NSLOTS:-unset}, SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset})"

cleanup() {
  if [ -f "$VLLM_PID_FILE" ]; then
    PID="$(cat "$VLLM_PID_FILE" 2>/dev/null || true)"
    if [ -n "${PID:-}" ] && ps -p "$PID" > /dev/null 2>&1; then
      echo "Stopping vLLM PID=$PID"
      kill "$PID" 2>/dev/null || true
      sleep 2
      ps -p "$PID" > /dev/null 2>&1 && kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$VLLM_PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

echo "== Modules =="
if ! command -v module &> /dev/null; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /etc/profile.d/modules.sh
  [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
fi

module purge >/dev/null 2>&1 || true
module load cuda/12.8
module load python3/3.10.12
module load cmake/3.31.7

echo "python: $(command -v python3 || command -v python || echo 'NOT FOUND')"
echo "nvidia-smi: $(command -v nvidia-smi || echo 'NOT FOUND')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>}"

echo "== venv =="
if [ -d .venv ]; then
  echo "Using existing .venv"
else
  echo "Creating .venv and installing deps"
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install --no-cache-dir -r requirements.txt
fi
source .venv/bin/activate

echo "== GPU diag =="
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi -L || true
python - <<'PY' || true
import torch
print(f"PyTorch sees {torch.cuda.device_count()} GPUs")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}, {p.total_memory/1024**3:.1f} GB")
PY

echo "== Ensure vLLM =="
if ! command -v vllm &> /dev/null; then
  python -m pip install --no-cache-dir -U vllm
fi

echo "== Warm tokenizer (optional) =="
python - <<PY || true
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("${VLLM_MODEL}")
PY

start_vllm() {
  echo "== Start vLLM =="
  [ -f "$VLLM_PID_FILE" ] && rm -f "$VLLM_PID_FILE"

  # Avoid potentially-corrupted compile artifacts and run eager mode for stability.
  export VLLM_DISABLE_COMPILE_CACHE=1
  export TORCH_COMPILE_DISABLE=1

  nohup vllm serve "$VLLM_MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.97 \
    --max-model-len 32768 \
    --max-num-batched-tokens 196608 \
    --max-num-seqs 1536 \
    --enable-prefix-caching \
    --enforce-eager \
    --disable-log-requests \
    --disable-log-stats \
    > "$VLLM_LOG_FILE" 2>&1 &

  echo $! > "$VLLM_PID_FILE"
  echo "vLLM PID: $(cat "$VLLM_PID_FILE")"
  echo "vLLM log: $VLLM_LOG_FILE"

  echo "== Wait for vLLM =="
  for i in {1..180}; do
    if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "vLLM Ready"
      break
    fi
    if [ "$i" -eq 180 ]; then
      echo "Timeout waiting for vLLM"
      echo "Last 80 log lines:"
      tail -n 80 "$VLLM_LOG_FILE" || true
      exit 1
    fi
    sleep 2
  done
}

# -------------------------
# Phase 0: build dense index if missing
# -------------------------
if [ -f "$DENSE_INDEX_PATH" ]; then
  echo "== Phase 0: dense index already exists at $DENSE_INDEX_PATH, skipping =="
else
  echo "== Phase 0: building dense passage index =="
  python -m weak_labels.cli phase0_build_dense_index
fi

# -------------------------
# Phase 1: retrieval + prompt logging (no vLLM)
# -------------------------
if [ -f "$PHASE1_PROMPTS_FILE" ]; then
  echo "== Phase 1: prompts already exist at $PHASE1_PROMPTS_FILE, skipping Phase 1 =="
else
  echo "== Phase 1: retrieval + prompt logging (no vLLM) =="
  python -m weak_labels.cli phase1
  if [ ! -f "$PHASE1_PROMPTS_FILE" ]; then
    echo "ERROR: Phase 1 finished but prompts file not found: $PHASE1_PROMPTS_FILE"
    exit 1
  fi
fi

# -------------------------
# Phase 2 decision (robust, fast)
# -------------------------
PHASE2_COMPLETE=0
if [ -f "$LLM_CACHE_DB" ]; then
  # Fast probe: does the table exist and have at least 1 row?
  # Avoids COUNT(*) on large tables. [web:150]
  if sqlite3 "$LLM_CACHE_DB" "SELECT 1 FROM judgments LIMIT 1;" >/dev/null 2>&1; then
    PHASE2_COMPLETE=1
  fi
fi

echo "Phase2 probe: cache_db=$LLM_CACHE_DB exists=$(test -f "$LLM_CACHE_DB" && echo yes || echo no) marker=$(test -f "$PHASE2_DONE_MARKER" && echo yes || echo no) PHASE2_COMPLETE=$PHASE2_COMPLETE"

if [ "$PHASE2_COMPLETE" -eq 1 ]; then
  echo "== Phase 2 already complete. Skipping ALL Phase 2 work (vLLM start/wait + judging) =="
else
  echo "== Phase 2 not complete. Running ALL Phase 2 work =="

  # Everything Phase-2-related is inside this block:
  start_vllm

  echo "== Phase 2: LLM judging from prompts =="
  python -m weak_labels.cli phase2

  # Mark completion (fast future skip)
  mkdir -p "$(dirname "$PHASE2_DONE_MARKER")"
  date -Is > "$PHASE2_DONE_MARKER"
  echo "Wrote Phase 2 completion marker: $PHASE2_DONE_MARKER"
fi

# -------------------------
# Phase 3: export qrels.tsv + triples.jsonl
# -------------------------
echo "== Phase 3: export qrels + triples from cache =="
python -m weak_labels.cli phase3_export

echo "== Outputs =="
ls -lah data/output || true
echo "First lines of qrels:"
head -n 5 data/output/qrels.tsv || true
echo "First lines of triples:"
head -n 2 data/output/triples.jsonl || true
