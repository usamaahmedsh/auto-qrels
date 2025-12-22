#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VLLM_HOST="127.0.0.1"
VLLM_PORT=8000
VLLM_MODEL="meta-llama/Llama-3.2-3B-Instruct"
VLLM_PID_FILE="$PROJECT_ROOT/vllm_server.pid"
VLLM_LOG_FILE="$PROJECT_ROOT/vllm_server.log"

# HF auth & caches
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# Make sure ~/.local/bin is on PATH
export PATH="$HOME/.local/bin:$PATH"

# CPU / PyTorch settings
CPU_CORES="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$CPU_CORES"
export MKL_NUM_THREADS="$CPU_CORES"
export OPENBLAS_NUM_THREADS="$CPU_CORES"
export NUMEXPR_NUM_THREADS="$CPU_CORES"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF

cleanup() {
  if [ -f "$VLLM_PID_FILE" ]; then
    PID="$(cat "$VLLM_PID_FILE" 2>/dev/null || true)"
    if [ -n "${PID:-}" ] && ps -p "$PID" > /dev/null 2>&1; then
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
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
  [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
fi

module purge >/dev/null 2>&1 || true
module load cuda12/12.4.1
module load python/3.11/3.11.4

echo "python: $(command -v python3 || command -v python || echo 'NOT FOUND')"
echo "nvidia-smi: $(command -v nvidia-smi || echo 'NOT FOUND')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>}"

echo "== venv =="
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

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
  python -m pip install -U vllm --extra-index-url https://download.pytorch.org/whl/cu128
fi

echo "== Warm tokenizer (optional) =="
python - <<'PY' || true
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
PY

echo "== Start vLLM =="
[ -f "$VLLM_PID_FILE" ] && rm -f "$VLLM_PID_FILE"

nohup vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --dtype float16 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2048 \
  --max-num-seqs 192 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --disable-log-requests \
  --disable-log-stats \
  > "$VLLM_LOG_FILE" 2>&1 &

echo $! > "$VLLM_PID_FILE"
echo "vLLM PID: $(cat "$VLLM_PID_FILE")"

echo "== Wait for vLLM =="
for i in {1..180}; do
  if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM Ready"
    break
  fi
  [ "$i" -eq 180 ] && { echo "Timeout waiting for vLLM"; exit 1; }
  sleep 2
done

echo "== Run agent =="
python -m weak_labels.cli
