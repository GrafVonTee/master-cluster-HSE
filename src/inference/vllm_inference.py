import os
import inspect
from functools import partial

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import src.config as config


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _stop_token_ids(tokenizer):
    ids = []
    for value in (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)):
        if value is not None and value not in ids:
            ids.append(value)
    return ids


def _build_sampling_params(**kwargs):
    """
    Build vLLM SamplingParams robustly across vLLM versions.
    The important part is explicit max_tokens, because vLLM defaults to 16.
    """
    try:
        sig = inspect.signature(SamplingParams)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        pass

    sp = SamplingParams(**kwargs)

    expected = kwargs.get("max_tokens")
    actual = getattr(sp, "max_tokens", None)
    if expected is not None and actual != expected:
        # Some patched/new wrappers can silently leave default values.
        # Force the field after construction and fail loudly if impossible.
        try:
            sp.max_tokens = int(expected)
        except Exception as exc:
            raise RuntimeError(
                f"Could not force SamplingParams.max_tokens={expected}; current value={actual}"
            ) from exc

    return sp


def make_sampling_params(
    tokenizer,
    *,
    n: int = 1,
    temperature: float = 0.0,
    logprobs: int | None = None,
    max_tokens: int | None = None,
):
    if max_tokens is None:
        max_tokens = _int_env("EVAL_MAX_NEW_TOKENS", 512)

    kwargs = {
        "n": int(n),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stop_token_ids": _stop_token_ids(tokenizer),
        "ignore_eos": False,
        "detokenize": True,
        "repetition_penalty": 1.0,
        "logprobs": logprobs,
    }

    sp = _build_sampling_params(**kwargs)

    print(
        f"[eval] EVAL_MAX_NEW_TOKENS={os.environ.get('EVAL_MAX_NEW_TOKENS', '<unset>')} "
        f"-> SamplingParams.max_tokens={sp.max_tokens}"
    )

    return sp


def setup_model(model_path=config.MODEL_PATH, adapter_path=None):
    using_sft = adapter_path is not None and str(adapter_path).strip() != ""

    llm_kwargs = dict(config.VLLM_PARAMS)
    llm_kwargs.setdefault("trust_remote_code", True)

    llm = LLM(
        model=model_path,
        enable_lora=using_sft,
        max_lora_rank=64 if using_sft else None,
        **llm_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if using_sft:
        request = LoRARequest("sft", 1, lora_path=adapter_path)
        llm.generate = partial(llm.generate, lora_request=request)

    sampling_params = make_sampling_params(tokenizer)
    return llm, tokenizer, sampling_params
