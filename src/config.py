import os
from pathlib import Path


def _is_bad_container_path(path: Path) -> bool:
    s = str(path)
    return (
        s == "/workspace"
        or s.startswith("/workspace/")
        or s == "/root"
        or s.startswith("/root/")
    )


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)

    if raw is None or raw.strip() == "":
        return default

    path = Path(raw).expanduser()

    # Dockerfile ENV внутри Singularity может указывать на read-only /workspace.
    # На кластере такие пути нельзя использовать для cache/output.
    if _is_bad_container_path(path):
        return default

    return path


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path.cwd())).expanduser().resolve()

MODELS_DIR = _env_path("MODELS_DIR", PROJECT_DIR / "models")
DATASETS_DIR = _env_path("DATASETS_DIR", PROJECT_DIR / "datasets")
OUTPUTS_DIR = _env_path("OUTPUTS_DIR", PROJECT_DIR / "outputs")
LOGS_DIR = _env_path("LOGS_DIR", PROJECT_DIR / "logs")

HF_HOME = _env_path("HF_HOME", PROJECT_DIR / ".cache" / "huggingface")
HF_DATASETS_CACHE = _env_path("HF_DATASETS_CACHE", HF_HOME / "datasets")
VLLM_CACHE_ROOT = _env_path("VLLM_CACHE_ROOT", PROJECT_DIR / ".cache" / "vllm")

for p in [
    MODELS_DIR,
    DATASETS_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
    HF_HOME,
    HF_DATASETS_CACHE,
    VLLM_CACHE_ROOT,
]:
    p.mkdir(parents=True, exist_ok=True)

MODELS = {
    "4b-instruct": {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "path": str(MODELS_DIR / "qwen3-4B-instruct-2507"),
    },
    "7b": {
        "name": "Qwen/Qwen3-7B",
        "path": str(MODELS_DIR / "qwen3-7b"),
    },
    "14b": {
        "name": "Qwen/Qwen3-14B",
        "path": str(MODELS_DIR / "qwen3-14b"),
    },
}

SELECTED_MODEL = os.environ.get("SELECTED_MODEL", "4b-instruct")

if SELECTED_MODEL not in MODELS:
    raise ValueError(f"Unknown SELECTED_MODEL={SELECTED_MODEL}. Available: {list(MODELS)}")

MODEL_NAME = MODELS[SELECTED_MODEL]["name"]
MODEL_PATH = MODELS[SELECTED_MODEL]["path"]

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))

VLLM_PARAMS = {
    "max_model_len": MAX_TOKENS,
    "dtype": "half",
    "gpu_memory_utilization": float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")),
    "enforce_eager": True,
    "seed": 42,
    "enable_prefix_caching": False,
    "trust_remote_code": True,
}

SAMPLING_SETTINGS = {
    "max_tokens": MAX_TOKENS,
    "ignore_eos": False,
    "detokenize": True,
    "logprobs": 1,
    "repetition_penalty": 1,
}

NUM_PROCESSES = int(os.environ.get("NUM_PROCESSES", "4"))
