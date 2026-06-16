#!/usr/bin/env python3
"""Smoke-test for the V100-oriented vLLM 0.8.5 + Unsloth candidate image.

Default run checks imports and runtimes without downloading/loading a model.
Pass --model or set VLLM_SMOKE_MODEL to test vLLM generation with logprobs and entropy.
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import Any

# These must be set before importing vLLM.
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def optional_import(name: str):
    try:
        import importlib
        module = importlib.import_module(name)
        print(f"[optional] {name} version={getattr(module, '__version__', '<no __version__>')}")
        return module
    except Exception as e:
        print(f"[optional] {name} not available: {e!r}")
        return None


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float env {name}={raw!r}") from exc


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer env {name}={raw!r}") from exc


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def import_and_print(name: str) -> Any:
    print(f"[import] {name}", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(name)
    print(f"  version={getattr(module, '__version__', '<no __version__>')}", flush=True)
    for w in caught[:10]:
        print(f"  warning={w.message}", flush=True)
    return module


def check_python_paths() -> None:
    print("===== PYTHON / PATHS =====", flush=True)
    print("python", sys.version, flush=True)
    print("executable", sys.executable, flush=True)
    print("cwd", Path.cwd(), flush=True)
    for key in [
        "PROJECT_DIR", "PYTHONPATH", "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE",
        "VLLM_CACHE_ROOT", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        "TORCH_CUDA_ARCH_LIST", "VLLM_ENABLE_CUDA_COMPATIBILITY", "VLLM_USE_V1",
        "VLLM_ATTENTION_BACKEND", "EVAL_ENABLE_LOGPROBS",
    ]:
        print(f"{key}={os.environ.get(key)}", flush=True)


def check_imports() -> None:
    print("===== IMPORTS / CUDA =====", flush=True)
    torch = import_and_print("torch")
    print("torch.version.cuda", torch.version.cuda, flush=True)
    print("torch.cuda.is_available", torch.cuda.is_available(), flush=True)
    print("torch.cuda.device_count", torch.cuda.device_count(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print("gpu", torch.cuda.get_device_name(0), flush=True)
    print("capability", torch.cuda.get_device_capability(0), flush=True)
    print("torch_cuda_arch_list", torch.cuda.get_arch_list(), flush=True)

    x = torch.ones((8, 8), device="cuda", dtype=torch.float16)
    print("cuda_tensor_sum", float((x @ x).sum().detach().cpu()), flush=True)
    lin = torch.nn.Linear(64, 64, bias=False, dtype=torch.float16).cuda()
    y = lin(torch.randn(4, 64, device="cuda", dtype=torch.float16))
    print("cuda_fp16_linear_mean", float(y.mean().detach().cpu()), flush=True)

    # Unsloth wants to be imported before transformers/trl in real train scripts.
    # In this smoke test we already used torch for diagnostics; this is acceptable.
    import_and_print("unsloth")
    import_and_print("transformers")
    import_and_print("datasets")
    import_and_print("accelerate")
    import_and_print("peft")
    import_and_print("trl")
    import_and_print("vllm")
    try:
        import_and_print("xformers")
    except Exception as exc:
        print(f"[warn] xformers import failed: {exc}", flush=True)
    import_and_print("clingo")
    import_and_print("pandas")
    import_and_print("pyarrow")
    import_and_print("numpy")
    import_and_print("sklearn")

    from trl import GRPOConfig, GRPOTrainer  # noqa: F401
    print("trl.GRPOConfig/GRPOTrainer ok", flush=True)
    from unsloth import FastLanguageModel  # noqa: F401
    print("unsloth.FastLanguageModel ok", flush=True)


def check_clingo() -> None:
    print("===== CLINGO =====", flush=True)
    import clingo
    ctl = clingo.Control()
    ctl.add("base", [], "a. b :- a.")
    ctl.ground([("base", [])])
    models: list[list[str]] = []
    with ctl.solve(yield_=True) as handle:
        for m in handle:
            models.append(sorted(str(s) for s in m.symbols(shown=True)))
    print("python_models", models, flush=True)
    exe = shutil.which("clingo")
    if exe:
        out = run([exe, "--version"]).stdout
        print(out, end="", flush=True)
    else:
        raise RuntimeError("clingo CLI not found")


def check_godot() -> None:
    print("===== GODOT =====", flush=True)
    godot = shutil.which("godot")
    if not godot:
        raise RuntimeError("godot binary not found")
    print(run([godot, "--version"]).stdout, end="", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "project.godot").write_text("config_version=5\n", encoding="utf-8")
        (p / "smoke.gd").write_text(
            textwrap.dedent(
                """
                extends SceneTree
                func _init():
                    print("GDSCRIPT_SMOKE_OK_", Engine.get_version_info()["major"])
                    quit()
                """
            ),
            encoding="utf-8",
        )
        out = run([godot, "--headless", "--path", str(p), "--script", str(p / "smoke.gd")]).stdout
        print(out, end="", flush=True)
        if "GDSCRIPT_SMOKE_OK_4" not in out:
            raise RuntimeError("Godot script did not produce expected output")


def _logprob_value(obj: Any) -> float:
    if hasattr(obj, "logprob"):
        return float(obj.logprob)
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict) and "logprob" in obj:
        return float(obj["logprob"])
    raise TypeError(f"Cannot extract logprob from {type(obj)!r}: {obj!r}")


def entropy_from_logprobs(step: Any) -> float | None:
    if not step:
        return None
    if isinstance(step, dict):
        vals = [_logprob_value(v) for v in step.values()]
    elif isinstance(step, list):
        vals = [_logprob_value(v) for v in step]
    else:
        return None
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return None
    mx = max(vals)
    weights = [math.exp(v - mx) for v in vals]
    z = sum(weights)
    probs = [w / z for w in weights if z > 0]
    if not probs:
        return None
    return -sum(p * math.log(p) for p in probs if p > 0)


def check_vllm_logprobs(model_path: str | None) -> None:
    print("===== VLLM LOGPROBS / ENTROPY =====", flush=True)
    model = model_path or os.environ.get("VLLM_SMOKE_MODEL")
    if not model or model.lower() in {"none", "skip", ""}:
        print("skipped: pass --model or set VLLM_SMOKE_MODEL to a local model", flush=True)
        return
    if not Path(model).exists():
        raise RuntimeError(f"Model path does not exist inside container: {model}")

    import torch
    free, total = torch.cuda.mem_get_info()
    gpu_util = env_float("SMOKE_GPU_MEMORY_UTILIZATION", 0.85)
    max_model_len = env_int("SMOKE_MAX_MODEL_LEN", 256)
    max_tokens = env_int("SMOKE_MAX_TOKENS", 8)
    logprobs = env_int("SMOKE_LOGPROBS", 5)
    enforce_eager = os.environ.get("SMOKE_ENFORCE_EAGER", "1") not in {"0", "false", "False"}

    print(
        f"cuda_mem_before_vllm free_gib={free/2**30:.2f} total_gib={total/2**30:.2f} "
        f"gpu_memory_utilization={gpu_util}",
        flush=True,
    )
    print(
        f"vllm_smoke_config dtype=half max_model_len={max_model_len} max_tokens={max_tokens} "
        f"logprobs={logprobs} enforce_eager={enforce_eager} "
        f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1')} "
        f"VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND')}",
        flush=True,
    )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        dtype="half",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=logprobs,
    )
    out = llm.generate(["Write a Python function add(a, b)."], params)
    gen = out[0].outputs[0]
    print("text", repr(gen.text), flush=True)
    steps = gen.logprobs or []
    entropies = []
    for step in steps:
        ent = entropy_from_logprobs(step)
        if ent is not None:
            entropies.append(ent)
    print("token_entropy_values", entropies, flush=True)
    if not entropies:
        raise RuntimeError("No logprobs/entropy values returned by vLLM")
    print("mean_entropy", sum(entropies) / len(entropies), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("VLLM_SMOKE_MODEL"))
    args = parser.parse_args()

    check_python_paths()
    check_imports()
    check_clingo()
    check_godot()
    check_vllm_logprobs(args.model)
    print("===== SMOKE OK =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# torchao is optional on V100; old torchao versions break PEFT LoRA dispatch.
