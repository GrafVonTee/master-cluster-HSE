#!/bin/bash
#SBATCH --job-name=grpo-server-v100
#SBATCH --output=logs/slurm/grpo-server-v100-%j.out
#SBATCH --error=logs/slurm/grpo-server-v100-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="${RL_IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

MODEL="${MODEL:-/workspace/models/qwen3-4b-instruct-2507}"
CONFIG="${GRPO_CONFIG:-configs/rl/grpo_pythoncodes.yaml}"
RUN_NAME="${GRPO_RUN_NAME:-grpo_pythoncodes_server_smoke}"

HTTP_PORT="${GRPO_VLLM_SERVER_PORT:-8000}"
GROUP_PORT="${GRPO_VLLM_GROUP_PORT:-$((51216 + SLURM_JOB_ID % 10000))}"

SERVER_GPU_UTIL="${SERVER_GPU_UTIL:-0.30}"
SERVER_MAX_MODEL_LEN="${SERVER_MAX_MODEL_LEN:-2048}"
SERVER_DTYPE="${SERVER_DTYPE:-float16}"
SERVER_MODEL_IMPL="${SERVER_MODEL_IMPL:-vllm}"
SERVER_ENFORCE_EAGER="${SERVER_ENFORCE_EAGER:-true}"

RL_MAX_STEPS_OVERRIDE="${RL_MAX_STEPS_OVERRIDE:-1}"
RL_DATASET_LIMIT_OVERRIDE="${RL_DATASET_LIMIT_OVERRIDE:-64}"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" \
  logs/slurm outputs/rl models datasets .home \
  .cache/huggingface/datasets \
  .cache/vllm \
  .cache/torch \
  .cache/torch_extensions \
  .cache/torchinductor \
  .cache/triton

trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "image=$IMAGE"
echo "model=$MODEL"
echo "config=$CONFIG"
echo "run_name=$RUN_NAME"
echo "http_port=$HTTP_PORT"
echo "group_port=$GROUP_PORT"
echo "server_gpu_util=$SERVER_GPU_UTIL"
echo "server_max_model_len=$SERVER_MAX_MODEL_LEN"
echo "server_dtype=$SERVER_DTYPE"
echo "server_enforce_eager=$SERVER_ENFORCE_EAGER"
echo "SINGULARITY_CACHEDIR=$SINGULARITY_CACHEDIR"
echo "SINGULARITY_TMPDIR=$SINGULARITY_TMPDIR"

nvidia-smi || true

test -d "$IMAGE" || { echo "Missing sandbox: $IMAGE" >&2; exit 1; }
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 1; }
test -d "${MODEL#/workspace/$PROJECT_DIR/}" || true

INNER_SCRIPT="outputs/rl/run_grpo_server_${SLURM_JOB_ID}.sh"

cat > "$INNER_SCRIPT" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail

echo "===== INSIDE CONTAINER ====="
echo "host=$(hostname)"
echo "MODEL=$MODEL"
echo "CONFIG=$CONFIG"
echo "RUN_NAME=$RUN_NAME"
echo "HTTP_PORT=$HTTP_PORT"
echo "GROUP_PORT=$GROUP_PORT"

cd /workspace

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "===== STOP vLLM SERVER pid=$SERVER_PID ====="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "===== PRECHECK ====="
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
/venv/main/bin/trl vllm-serve \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port "$HTTP_PORT" \
  --gpu_memory_utilization "$SERVER_GPU_UTIL" \
  --dtype "$SERVER_DTYPE" \
  --max_model_len "$SERVER_MAX_MODEL_LEN" \
  --enable_prefix_caching false \
  --enforce_eager "$SERVER_ENFORCE_EAGER" \
  --vllm_model_impl "$SERVER_MODEL_IMPL" \
  --log_level info \
  > "/workspace/logs/slurm/grpo-server-vllm-${SLURM_JOB_ID}.out" \
  2> "/workspace/logs/slurm/grpo-server-vllm-${SLURM_JOB_ID}.err" &

SERVER_PID=$!
echo "server_pid=$SERVER_PID"

echo "===== WAIT FOR SERVER ====="
python - <<PY
import socket, time, sys, urllib.request

host = "127.0.0.1"
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

CMD=(python -u scripts/rl_train_pythoncodes_grpo_server.py
  --config "$CONFIG"
  --run-name "$RUN_NAME"
  --base-model "$MODEL"
  --local-files-only
)

if [[ -n "${RL_INIT_ADAPTER:-}" ]]; then
  CMD+=(--init-adapter "$RL_INIT_ADAPTER")
fi

"${CMD[@]}"

echo "===== SERVER MODE GRPO OK ====="
INNER

chmod +x "$INNER_SCRIPT"

echo "===== RUN SINGLE SINGULARITY SESSION THROUGH OLD WORKING PATTERN ====="

srun singularity exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --env PROJECT_DIR=/workspace \
  --env PYTHONPATH=/workspace \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  --env TORCH_HOME=/workspace/.cache/torch \
  --env TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env UNSLOTH_DISABLE_STATISTICS=1 \
  --env MODEL="$MODEL" \
  --env CONFIG="$CONFIG" \
  --env RUN_NAME="$RUN_NAME" \
  --env HTTP_PORT="$HTTP_PORT" \
  --env GROUP_PORT="$GROUP_PORT" \
  --env GRPO_VLLM_SERVER_HOST=127.0.0.1 \
  --env GRPO_VLLM_SERVER_PORT="$HTTP_PORT" \
  --env GRPO_VLLM_GROUP_PORT="$GROUP_PORT" \
  --env GRPO_VLLM_SERVER_TIMEOUT=300 \
  --env SERVER_GPU_UTIL="$SERVER_GPU_UTIL" \
  --env SERVER_MAX_MODEL_LEN="$SERVER_MAX_MODEL_LEN" \
  --env SERVER_DTYPE="$SERVER_DTYPE" \
  --env SERVER_MODEL_IMPL="$SERVER_MODEL_IMPL" \
  --env SERVER_ENFORCE_EAGER="$SERVER_ENFORCE_EAGER" \
  --env RL_MAX_STEPS_OVERRIDE="$RL_MAX_STEPS_OVERRIDE" \
  --env RL_DATASET_LIMIT_OVERRIDE="$RL_DATASET_LIMIT_OVERRIDE" \
  --env RL_INIT_ADAPTER="${RL_INIT_ADAPTER:-}" \
  --env SLURM_JOB_ID="${SLURM_JOB_ID:-manual}" \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash "/workspace/$INNER_SCRIPT"
