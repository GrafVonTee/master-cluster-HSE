import os

MODELS_DIR = "./models/"
DATASETS_DIR = "./datasets/"
LOGS_DIR = "./logs/"
MODELS = {
	"0.6b": {
		"name": "Qwen/Qwen3-0.6B",
		"path": MODELS_DIR + "qwen3-0.6b"
	},
	"4b-thinking": {
		"name": "Qwen/Qwen3-4B-Thinking-2507",
		"path": MODELS_DIR + "qwen3-4B-thinking-2507"
	},
	"4b-instruct": {
		"name": "Qwen/Qwen3-4B-Instruct-2507",
		"path": MODELS_DIR + "qwen3-4B-instruct-2507"
	},
    "1.7b": {
        "name": "Qwen/Qwen3-1.7b",
        "path": MODELS_DIR + "qwen3-1.7b"
	}
}
SELECTED_MODEL = "4b-instruct"
MODEL_NAME = MODELS[SELECTED_MODEL]["name"]
MODEL_PATH = MODELS[SELECTED_MODEL]["path"]
MAX_TOKENS = 4096

VLLM_PARAMS = {
	"max_model_len": MAX_TOKENS,
	"dtype": "auto",
	"gpu_memory_utilization": 0.7,
    "enforce_eager": False,
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

NUM_PROCESSES = 8
