from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.rl.code import (
    extract_python_code,
    finite_float,
    normalize_tests,
    run_python_tests,
)


@dataclass
class PythonRewardConfig:
    timeout: float = 3.0
    min_chars: int = 8
    max_chars: int = 6000
    error_reward: float = -1.0
    no_tests_reward: float = -1.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "PythonRewardConfig":
        cfg = cfg or {}
        length = cfg.get("length", {}) or {}
        return cls(
            timeout=finite_float(cfg.get("timeout", 3.0), 3.0),
            min_chars=int(length.get("min_chars", 8)),
            max_chars=int(length.get("max_chars", 6000)),
            error_reward=finite_float(cfg.get("error_reward", -1.0), -1.0),
            no_tests_reward=finite_float(cfg.get("no_tests_reward", -1.0), -1.0),
            extra={k: v for k, v in cfg.items() if k not in {"length", "timeout", "error_reward", "no_tests_reward"}},
        )


def expand_column(values: Any, target_len: int) -> list[Any]:
    if values is None:
        return [None] * target_len
    if isinstance(values, (list, tuple)):
        if len(values) == target_len:
            return list(values)
        if len(values) == 0:
            return [None] * target_len
        return [values[i % len(values)] for i in range(target_len)]
    return [values for _ in range(target_len)]


def execution_reward_for_code(code_text: Any, tests: Any, cfg: PythonRewardConfig) -> float:
    code = extract_python_code(code_text)
    test_list = normalize_tests(tests)

    if not code or len(code.strip()) < cfg.min_chars:
        return float(cfg.error_reward)

    if len(code) > cfg.max_chars:
        return float(cfg.error_reward)

    if not test_list:
        return float(cfg.no_tests_reward)

    run = run_python_tests(code, test_list, timeout=cfg.timeout)

    # Compile/syntax-level failure: hard error.
    if not run.syntax_ok:
        return float(cfg.error_reward)

    # If nothing passes, treat it as a hard failure. This makes bad samples
    # clearly worse than partially correct ones.
    if run.total <= 0:
        return float(cfg.no_tests_reward)

    if run.passed <= 0:
        return float(cfg.error_reward)

    # Main signal: 1/3 -> 0.333, 2/3 -> 0.667, 3/3 -> 1.0.
    return float(run.passed) / float(run.total)


def make_python_reward(config: PythonRewardConfig):
    def reward_func(prompts=None, completions=None, reference=None, tests=None, **kwargs) -> list[float]:
        completions = completions or []
        tests_col = expand_column(tests if tests is not None else kwargs.get("test"), len(completions))

        rewards: list[float] = []
        for completion, raw_tests in zip(completions, tests_col):
            rewards.append(execution_reward_for_code(completion, raw_tests, config))
        return rewards

    reward_func.__name__ = "python_mbpp_execution_reward"
    return reward_func


def score_python_completion(completion: Any, reference: str = "", tests: Any = None, cfg: PythonRewardConfig | None = None) -> float:
    cfg = cfg or PythonRewardConfig()
    reward_fn = make_python_reward(cfg)
    return reward_fn(completions=[completion], tests=[tests])[0]
