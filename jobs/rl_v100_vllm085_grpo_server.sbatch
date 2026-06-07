#!/usr/bin/env bash
#SBATCH --job-name=grpo-server-v100
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/grpo-server-v100-%j.out
#SBATCH --error=logs/slurm/grpo-server-v100-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

MODEL="${MODEL:-/workspace/models/qwen3-4b-instruct-2507}"
HTTP_PORT="${GRPO_VLLM_SERVER_PORT:-8000}"
GROUP_PORT="${GRPO_VLLM_GROUP_PORT:-$((51216 + SLURM_JOB_ID % 10000))}"

# Avoid 127.0.0.1 for the communicator. Use node IP.
HOST_IP="${GRPO_VLLM_SERVER_HOST:-$(hostname -I | awk '{print $1}')}"
BIND_HOST="${GRPO_VLLM_BIND_HOST:-0.0.0.0}"

RUN_NAME="${GRPO_RUN_NAME:-grpo_pythoncodes_server_smoke}"
CONFIG="${GRPO_CONFIG:-configs/rl/grpo_pythoncodes.yaml}"

SERVER_GPU_UTIL="${SERVER_GPU_UTIL:-0.30}"
SERVER_MAX_MODEL_LEN="${SERVER_MAX_MODEL_LEN:-2048}"
SERVER_DTYPE="${SERVER_DTYPE:-float16}"
SERVER_MODEL_IMPL="${SERVER_MODEL_IMPL:-vllm}"
SERVER_ENFORCE_EAGER="${SERVER_ENFORCE_EAGER:-true}"

mkdir -p logs/slurm outputs/rl \
  .cache/huggingface/datasets \
  .cache/vllm \
  .cache/torch \
  .cache/torch_extensions \
  .cache/triton

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none}"
echo "node=$(hostname)"
echo "host_ip=$HOST_IP"
echo "bind_host=$BIND_HOST"
echo "http_port=$HTTP_PORT"
echo "group_port=$GROUP_PORT"
echo "image=$IMAGE"
echo "model=$MODEL"
echo "run_name=$RUN_NAME"

test -d "$IMAGE" || { echo "ERROR: sandbox not found: $IMAGE" >&2; exit 2; }

nvidia-smi || true

export PYTHONPATH=/workspace
export SINGULARITYENV_PYTHONPATH=/workspace

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
export TORCH_HOME=/workspace/.cache/torch
export TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
export TRITON_CACHE_DIR=/workspace/.cache/triton
export UNSLOTH_DISABLE_STATISTICS=1

export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export SINGULARITYENV_TORCH_HOME="$TORCH_HOME"
export SINGULARITYENV_TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export SINGULARITYENV_UNSLOTH_DISABLE_STATISTICS=1

export GRPO_VLLM_SERVER_HOST="$HOST_IP"
export GRPO_VLLM_SERVER_PORT="$HTTP_PORT"
export GRPO_VLLM_GROUP_PORT="$GROUP_PORT"
export GRPO_VLLM_SERVER_TIMEOUT="${GRPO_VLLM_SERVER_TIMEOUT:-300}"

export SINGULARITYENV_GRPO_VLLM_SERVER_HOST="$GRPO_VLLM_SERVER_HOST"
export SINGULARITYENV_GRPO_VLLM_SERVER_PORT="$GRPO_VLLM_SERVER_PORT"
export SINGULARITYENV_GRPO_VLLM_GROUP_PORT="$GRPO_VLLM_GROUP_PORT"
export SINGULARITYENV_GRPO_VLLM_SERVER_TIMEOUT="$GRPO_VLLM_SERVER_TIMEOUT"

export RL_MAX_STEPS_OVERRIDE="${RL_MAX_STEPS_OVERRIDE:-1}"
export RL_DATASET_LIMIT_OVERRIDE="${RL_DATASET_LIMIT_OVERRIDE:-64}"
export SINGULARITYENV_RL_MAX_STEPS_OVERRIDE="$RL_MAX_STEPS_OVERRIDE"
export SINGULARITYENV_RL_DATASET_LIMIT_OVERRIDE="$RL_DATASET_LIMIT_OVERRIDE"

export GRPO_RUN_NAME="$RUN_NAME"
export SINGULARITYENV_GRPO_RUN_NAME="$GRPO_RUN_NAME"

export RL_INIT_ADAPTER="${RL_INIT_ADAPTER:-}"
export SINGULARITYENV_RL_INIT_ADAPTER="$RL_INIT_ADAPTER"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "===== STOP vLLM SERVER pid=$SERVER_PID ====="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "===== PRECHECK ====="
singularity exec --nv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  "$IMAGE" \
  python - <<'PY'
import importlib.util
import torch, vllm, trl, transformers
print("torch", torch.__version__, torch.version.cuda)
print("vllm", vllm.__version__)
print("trl", trl.__version__)
print("transformers", transformers.__version__)
print("torchao", importlib.util.find_spec("torchao"))
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

echo "===== START TRL vLLM SERVER ====="
singularity exec --nv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  "$IMAGE" \
  /venv/main/bin/trl vllm-serve \
    --model "$MODEL" \
    --host "$BIND_HOST" \
    --port "$HTTP_PORT" \
    --gpu_memory_utilization "$SERVER_GPU_UTIL" \
    --dtype "$SERVER_DTYPE" \
    --max_model_len "$SERVER_MAX_MODEL_LEN" \
    --enable_prefix_caching false \
    --enforce_eager "$SERVER_ENFORCE_EAGER" \
    --vllm_model_impl "$SERVER_MODEL_IMPL" \
    --log_level info \
  > "logs/slurm/grpo-server-vllm-${SLURM_JOB_ID}.out" \
  2> "logs/slurm/grpo-server-vllm-${SLURM_JOB_ID}.err" &

SERVER_PID=$!
echo "server_pid=$SERVER_PID"

echo "===== WAIT FOR SERVER HTTP ====="
python - <<PY
import socket, time, sys, urllib.request
host = "$HOST_IP"
port = int("$HTTP_PORT")
deadline = time.time() + 300

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"tcp open: {host}:{port}")
            break
    except OSError:
        time.sleep(2)
else:
    print(f"server TCP did not open: {host}:{port}", file=sys.stderr)
    sys.exit(1)

for path in ["/health/", "/get_world_size/"]:
    url = f"http://{host}:{port}{path}"
    try:
        print("GET", url)
        print(urllib.request.urlopen(url, timeout=10).read().decode()[:500])
    except Exception as e:
        print("HTTP check failed", url, repr(e))
PY

echo "===== START GRPO TRAINER CLIENT ====="
singularity exec --nv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  "$IMAGE" \
  python -u scripts/rl_train_pythoncodes_grpo_server.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
    --base-model "$MODEL" \
    ${RL_INIT_ADAPTER:+--init-adapter "$RL_INIT_ADAPTER"} \
    --local-files-only

echo "===== SERVER MODE GRPO OK ====="
