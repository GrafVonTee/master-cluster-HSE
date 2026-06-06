from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from src.rl.code import extract_python_code, finite_float, is_probably_python, normalize_tests, run_python_tests, syntax_score


@dataclass
class PythonRewardConfig:
    syntax: float = 0.30
    reference_similarity: float = 0.55
    function_keyword: float = 0.10
    test_pass_ratio: float = 1.00
    length_penalty: float = 0.05
    timeout: float = 2.0
    min_chars: int = 30
    max_chars: int = 3000
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "PythonRewardConfig":
        cfg = cfg or {}
        weights = cfg.get("weights", {}) or {}
        length = cfg.get("length", {}) or {}
        return cls(
            syntax=finite_float(weights.get("syntax", 0.30)),
            reference_similarity=finite_float(weights.get("reference_similarity", 0.55)),
            function_keyword=finite_float(weights.get("function_keyword", 0.10)),
            test_pass_ratio=finite_float(weights.get("test_pass_ratio", 1.00)),
            length_penalty=finite_float(weights.get("length_penalty", 0.05)),
            timeout=finite_float(cfg.get("timeout", 2.0), 2.0),
            min_chars=int(length.get("min_chars", 30)),
            max_chars=int(length.get("max_chars", 3000)),
            extra={k: v for k, v in cfg.items() if k not in {"weights", "length", "timeout"}},
        )


def _normalize_for_similarity(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())


def reference_similarity_score(generated_code: str, reference_code: str) -> float:
    ref = _normalize_for_similarity(reference_code)
    gen = _normalize_for_similarity(generated_code)
    if not ref or not gen:
        return 0.0
    return float(SequenceMatcher(None, gen, ref).ratio())


def length_penalty_score(code: str, *, min_chars: int, max_chars: int) -> float:
    n = len(code)
    if n < min_chars:
        return -1.0
    if n > max_chars:
        return -min(1.0, (n - max_chars) / max(1.0, max_chars))
    return 0.0


def expand_column(values: Any, target_len: int) -> list[Any]:
    """TRL normally expands dataset columns to match completions; keep robust fallback."""
    if values is None:
        return [None] * target_len
    if isinstance(values, (list, tuple)):
        if len(values) == target_len:
            return list(values)
        if len(values) == 0:
            return [None] * target_len
        return [values[i % len(values)] for i in range(target_len)]
    return [values for _ in range(target_len)]


def make_python_reward(config: PythonRewardConfig):
    def reward_func(prompts=None, completions=None, reference=None, tests=None, **kwargs) -> list[float]:
        completions = completions or []
        refs = expand_column(reference if reference is not None else kwargs.get("output"), len(completions))
        tests_col = expand_column(tests if tests is not None else kwargs.get("test"), len(completions))

        rewards: list[float] = []
        for completion, ref, raw_tests in zip(completions, refs, tests_col):
            code = extract_python_code(completion)
            ref_code = str(ref or "")
            test_list = normalize_tests(raw_tests)

            score = 0.0
            score += config.syntax * syntax_score(code)
            score += config.function_keyword * (1.0 if is_probably_python(code) else -0.25)
            score += config.reference_similarity * reference_similarity_score(code, ref_code)
            score += config.length_penalty * length_penalty_score(
                code,
                min_chars=config.min_chars,
                max_chars=config.max_chars,
            )

            if test_list:
                run = run_python_tests(code, test_list, timeout=config.timeout)
                score += config.test_pass_ratio * run.pass_ratio

            rewards.append(float(score))
        return rewards

    reward_func.__name__ = "python_code_reward"
    return reward_func


def score_python_completion(completion: Any, reference: str = "", tests: Any = None, cfg: PythonRewardConfig | None = None) -> float:
    cfg = cfg or PythonRewardConfig()
    reward_fn = make_python_reward(cfg)
    return reward_fn(completions=[completion], reference=[reference], tests=[tests])[0]
