import os
import sys
from pathlib import Path

print("===== PYTHON =====")
print("python:", sys.version)
print("executable:", sys.executable)
print("cwd:", os.getcwd())

print("\n===== ENV =====")
for key in [
    "PROJECT_DIR",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "VLLM_CACHE_ROOT",
    "TOKENIZERS_PARALLELISM",
    "CUDA_VISIBLE_DEVICES",
]:
    print(f"{key}:", os.environ.get(key))

print("\n===== PATHS =====")
project_dir = Path(os.environ.get("PROJECT_DIR", ".")).resolve()
print("project_dir:", project_dir)
print("project exists:", project_dir.exists())
print("scripts exists:", (project_dir / "scripts").exists())
print("src exists:", (project_dir / "src").exists())

print("\n===== TORCH / CUDA =====")
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("cuda current device:", torch.cuda.current_device())
    print("cuda device name:", torch.cuda.get_device_name(0))
    print("cuda capability:", torch.cuda.get_device_capability(0))

print("\n===== LIBRARIES =====")
import transformers
import datasets
import sklearn
import pandas
import numpy

print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("sklearn:", sklearn.__version__)
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)

try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as e:
    print("vllm import failed:", repr(e))

print("\n===== SRC IMPORTS =====")
from src.config import PROJECT_DIR, DATASETS_DIR, OUTPUTS_DIR, MODELS_DIR
print("src.config ok")
print("PROJECT_DIR:", PROJECT_DIR)
print("DATASETS_DIR:", DATASETS_DIR)
print("OUTPUTS_DIR:", OUTPUTS_DIR)
print("MODELS_DIR:", MODELS_DIR)

from src.data.curriculum.base import CurriculumPipeline
from src.data.curriculum.heuristics import LengthScorer
from src.data.curriculum.entropy import PPLScorer, IFDScorer
from src.data.curriculum.clustering import LexicalClusterScorer, SemanticClusterScorer

print("curriculum imports ok")

try:
    from src.data.curriculum.llm_judge import LLMJudgeScorer
    print("llm_judge import ok")
except Exception as e:
    print("llm_judge import failed:", repr(e))

print("\nDONE")
