import os
from pathlib import Path


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path.cwd())).expanduser().resolve()

MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_DIR / "models")).expanduser()
if str(MODELS_DIR) == "/workspace" or str(MODELS_DIR).startswith("/workspace/"):
    MODELS_DIR = PROJECT_DIR / "models"

DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", PROJECT_DIR / "datasets")).expanduser()
if str(DATASETS_DIR) == "/workspace" or str(DATASETS_DIR).startswith("/workspace/"):
    DATASETS_DIR = PROJECT_DIR / "datasets"

OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", PROJECT_DIR / "outputs")).expanduser()
if str(OUTPUTS_DIR) == "/workspace" or str(OUTPUTS_DIR).startswith("/workspace/"):
    OUTPUTS_DIR = PROJECT_DIR / "outputs"

LOGS_DIR = Path(os.environ.get("LOGS_DIR", PROJECT_DIR / "logs")).expanduser()
if str(LOGS_DIR) == "/workspace" or str(LOGS_DIR).startswith("/workspace/"):
    LOGS_DIR = PROJECT_DIR / "logs"

HF_HOME = Path(os.environ.get("HF_HOME", PROJECT_DIR / ".cache" / "huggingface")).expanduser()
if str(HF_HOME) == "/workspace" or str(HF_HOME).startswith("/workspace/"):
    HF_HOME = PROJECT_DIR / ".cache" / "huggingface"

HF_DATASETS_CACHE = Path(os.environ.get("HF_DATASETS_CACHE", HF_HOME / "datasets")).expanduser()
if str(HF_DATASETS_CACHE) == "/workspace" or str(HF_DATASETS_CACHE).startswith("/workspace/"):
    HF_DATASETS_CACHE = HF_HOME / "datasets"

VLLM_CACHE_ROOT = Path(os.environ.get("VLLM_CACHE_ROOT", PROJECT_DIR / ".cache" / "vllm")).expanduser()
if str(VLLM_CACHE_ROOT) == "/workspace" or str(VLLM_CACHE_ROOT).startswith("/workspace/"):
    VLLM_CACHE_ROOT = PROJECT_DIR / ".cache" / "vllm"

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
        "path": str(MODELS_DIR / "qwen3-4b-instruct-2507"),
    },
    "4b-thinking": {
        "name": "Qwen/Qwen3-4B-Thinking-2507",
        "path": str(MODELS_DIR / "qwen3-4b-thinking-2507"),
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
SFT_MODEL_PATH = os.environ.get("SFT_MODEL_PATH", str(MODELS_DIR / f"{SELECTED_MODEL}-sft"))
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
    "chat_template_kwargs": {"enable_thinking": False},
}


NUM_PROCESSES = int(os.environ.get("NUM_PROCESSES", "4"))
