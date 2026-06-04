#!/usr/bin/env python3
"""Smoke tests for the unified RL/GRPO container.

The default mode avoids model downloads. It checks that the container can import
all critical libraries and can execute Godot and clingo. If VLLM_SMOKE_MODEL or
--model is provided and the model exists locally, it also verifies vLLM logprobs
and computes a real entropy value from returned token distributions.
"""

from __future__ import annotations

import argparse
import importlib
import json
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


def run(cmd: list[str], *, timeout: int = 60, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=True,
    )


def import_report(name: str) -> Any:
    print(f"[import] {name}", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(name)
    version = getattr(module, "__version__", "<no __version__>")
    print(f"  version={version}", flush=True)
    for warn in caught:
        msg = str(warn.message)
        print(f"  warning={msg}", flush=True)
        if "Skipping import of cpp extensions" in msg:
            raise RuntimeError(f"{name} emitted C++ extension warning: {msg}")
    return module


def check_python_env() -> None:
    print("===== PYTHON / PATHS =====", flush=True)
    print("python", sys.version, flush=True)
    print("executable", sys.executable, flush=True)
    print("cwd", os.getcwd(), flush=True)
    for key in [
        "PROJECT_DIR",
        "PYTHONPATH",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "VLLM_CACHE_ROOT",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "TORCH_CUDA_ARCH_LIST",
        "VLLM_ENABLE_CUDA_COMPATIBILITY",
        "VLLM_USE_V1",
        "VLLM_ATTENTION_BACKEND",
        "EVAL_ENABLE_LOGPROBS",
    ]:
        print(f"{key}={os.environ.get(key)}", flush=True)


def check_imports() -> None:
    print("===== IMPORTS =====", flush=True)
    torch = import_report("torch")
    print("torch.cuda.is_available", torch.cuda.is_available(), flush=True)
    print("torch.cuda.device_count", torch.cuda.device_count(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible inside the container. Run with Docker GPU support "
            "(`docker compose ...` with `gpus: all` / `docker run --gpus all`) "
            "or with Singularity `exec --nv`. Unsloth cannot be imported without a GPU."
        )

    print("gpu", torch.cuda.get_device_name(0), flush=True)
    print("capability", torch.cuda.get_device_capability(0), flush=True)
    print("torch.version.cuda", getattr(torch.version, "cuda", None), flush=True)
    try:
        print("torch_cuda_arch_list", torch.cuda.get_arch_list(), flush=True)
    except Exception as exc:
        print(f"torch_cuda_arch_list_error={type(exc).__name__}: {exc}", flush=True)

    # Tiny allocation catches basic broken CUDA/driver/container combinations.
    x = torch.ones((8, 8), device="cuda")
    print("cuda_tensor_sum", float(x.sum().item()), flush=True)
    del x

    # This catches the class of failures that later appears inside vLLM as
    # cudaErrorNoKernelImageForDevice during qkv/linear projections on V100.
    # A tensor allocation can pass even when fp16 GEMM kernels are missing for sm_70.
    a = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    print("cuda_fp16_matmul_mean", float(c.float().mean().item()), flush=True)
    del a, b, c

    lin = torch.nn.Linear(128, 128, bias=False, dtype=torch.float16, device="cuda")
    y = lin(torch.randn((4, 128), device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    print("cuda_fp16_linear_mean", float(y.float().mean().item()), flush=True)
    del lin, y

    # Unsloth must be imported before transformers/trl so its patches are installed first.
    import_report("unsloth")

    for name in [
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "vllm",
        "torchao",
        "clingo",
        "pandas",
        "pyarrow",
        "yaml",
        "numpy",
        "sklearn",
    ]:
        import_report(name)

    trl = importlib.import_module("trl")
    missing = [name for name in ["GRPOConfig", "GRPOTrainer"] if not hasattr(trl, name)]
    if missing:
        raise RuntimeError(f"TRL import ok, but GRPO API is missing: {missing}")
    print("trl.GRPOConfig/GRPOTrainer ok", flush=True)

    from unsloth import FastLanguageModel  # noqa: F401
    print("unsloth.FastLanguageModel ok", flush=True)


def check_clingo() -> None:
    print("===== CLINGO =====", flush=True)
    clingo = importlib.import_module("clingo")
    ctl = clingo.Control(["0"])
    ctl.add("base", [], "a | b. :- a, b.")
    ctl.ground([("base", [])])
    models: list[list[str]] = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            models.append(sorted(str(sym) for sym in model.symbols(shown=True)))
    print("python_models", models, flush=True)
    if len(models) != 2:
        raise RuntimeError(f"Expected 2 clingo models, got {models}")

    exe = shutil.which("clingo")
    if exe:
        out = run([exe, "--version"], timeout=30).stdout.strip()
        print(out, flush=True)
    else:
        print("clingo CLI not found; Python module is available", flush=True)


def check_godot() -> None:
    print("===== GODOT =====", flush=True)
    godot = shutil.which("godot") or os.environ.get("GODOT_BIN")
    if not godot:
        raise RuntimeError("godot executable not found")

    print(run([godot, "--version"], timeout=30).stdout.strip(), flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp)
        (project / "project.godot").write_text(
            textwrap.dedent(
                """
                ; Engine configuration file.
                config_version=5

                [application]
                config/name="smoke_rl_container"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        script = project / "smoke.gd"
        script.write_text(
            textwrap.dedent(
                """
                extends SceneTree

                func _init():
                    # Godot print() concatenates multiple arguments without inserting spaces
                    # in this headless/script mode, so emit one deterministic token.
                    print("GDSCRIPT_SMOKE_OK_%d" % (2 + 2))
                    quit(0)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        out = run([godot, "--headless", "--path", str(project), "--script", str(script)], timeout=60).stdout
        print(out.strip(), flush=True)
        if "GDSCRIPT_SMOKE_OK_4" not in out:
            raise RuntimeError("Godot script did not produce expected output")


def entropy_from_vllm_logprobs(step_logprobs: Any) -> float | None:
    if not step_logprobs:
        return None

    vals = []
    for item in step_logprobs.values():
        lp = getattr(item, "logprob", None)
        if lp is not None and math.isfinite(float(lp)):
            vals.append(float(lp))
    if not vals:
        return None

    # logprobs may be truncated top-k. Renormalize over returned candidates,
    # which is sufficient for smoke testing that logprob distributions exist.
    max_lp = max(vals)
    probs = [math.exp(v - max_lp) for v in vals]
    total = sum(probs)
    probs = [p / total for p in probs]
    return -sum(p * math.log(max(p, 1e-45)) for p in probs)


def check_vllm_logprobs(model_path: str) -> None:
    print("===== VLLM LOGPROBS / ENTROPY =====", flush=True)
    path = Path(model_path).expanduser()
    if not path.exists() and "/" not in model_path:
        raise RuntimeError(f"VLLM smoke model path does not exist: {path}")

    import torch
    from vllm import LLM, SamplingParams

    dtype = os.environ.get("SMOKE_DTYPE", "half")
    max_model_len = env_int("SMOKE_MAX_MODEL_LEN", 256)
    max_tokens = env_int("SMOKE_MAX_TOKENS", 8)
    logprobs = env_int("SMOKE_LOGPROBS", 5)
    gpu_memory_utilization = env_float("SMOKE_GPU_MEMORY_UTILIZATION", 0.85)
    enforce_eager = os.environ.get("SMOKE_ENFORCE_EAGER", "1").lower() not in {"0", "false", "no"}

    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gib = 1024 ** 3
        print(
            "cuda_mem_before_vllm "
            f"free_gib={free_bytes / gib:.2f} total_gib={total_bytes / gib:.2f} "
            f"gpu_memory_utilization={gpu_memory_utilization}",
            flush=True,
        )

    vllm_use_v1 = os.environ.get("VLLM_USE_V1", "<unset>")
    vllm_attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "<unset>")
    print(
        "vllm_smoke_config "
        f"dtype={dtype} max_model_len={max_model_len} max_tokens={max_tokens} "
        f"logprobs={logprobs} enforce_eager={enforce_eager} "
        f"VLLM_USE_V1={vllm_use_v1} VLLM_ATTENTION_BACKEND={vllm_attention_backend}",
        flush=True,
    )

    if vllm_attention_backend.upper() == "XFORMERS":
        import_report("xformers")

    llm = LLM(
        model=str(path) if path.exists() else model_path,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
    )
    params = SamplingParams(temperature=0.7, max_tokens=max_tokens, logprobs=logprobs)
    outputs = llm.generate(["Write a Python function add(a, b)."], params)
    sample = outputs[0].outputs[0]
    print("text", repr(sample.text), flush=True)
    if not sample.logprobs:
        raise RuntimeError("vLLM returned no logprobs; entropy metric would be zero/invalid")

    entropies = [e for step in sample.logprobs if (e := entropy_from_vllm_logprobs(step)) is not None]
    print("token_entropy_values", json.dumps(entropies[:8]), flush=True)
    if not entropies:
        raise RuntimeError("Could not compute entropy from vLLM logprobs")
    print("mean_entropy", sum(entropies) / len(entropies), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run import/runtime checks without model loading.")
    parser.add_argument("--model", default=os.environ.get("VLLM_SMOKE_MODEL", ""), help="Optional local/HF model for vLLM logprobs check.")
    args = parser.parse_args()

    check_python_env()
    check_imports()
    check_clingo()
    check_godot()

    if args.model:
        check_vllm_logprobs(args.model)
    else:
        print("===== VLLM LOGPROBS / ENTROPY =====", flush=True)
        print("skipped: pass --model or set VLLM_SMOKE_MODEL to an already cached/local model", flush=True)

    print("===== SMOKE OK =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
