from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.dsl.clingo.runner import solve_clingo


@dataclass
class ClingoRewardConfig:
    timeout: float = 3.0
    max_models: int = 1
    error_reward: float = -1.0
    min_chars: int = 8
    max_chars: int = 8000


def extract_asp_code(text: Any) -> str:
    if text is None:
        return ""

    s = str(text).strip()

    fence = re.search(r"```(?:asp|prolog|clingo|lp)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    # Remove common chatty prefixes.
    lines = []
    for line in s.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(("here is", "explanation", "answer:")):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def score_clingo_completion(completion: Any, task: dict[str, Any], cfg: ClingoRewardConfig | None = None) -> float:
    cfg = cfg or ClingoRewardConfig()

    code = extract_asp_code(completion)
    if len(code) < cfg.min_chars or len(code) > cfg.max_chars:
        return cfg.error_reward

    facts = str(task.get("facts", "")).strip()
    expected_satisfiable = bool(task.get("expected_satisfiable", True))
    expected_atoms = [str(x) for x in task.get("expected_atoms", [])]
    forbidden_atoms = [str(x) for x in task.get("forbidden_atoms", [])]

    program = facts + "\n\n" + code
    result = solve_clingo(program, timeout=cfg.timeout, max_models=cfg.max_models)

    if not result.ok:
        return cfg.error_reward

    if result.satisfiable != expected_satisfiable:
        return cfg.error_reward

    if not expected_satisfiable:
        return 1.0 if result.unsatisfiable else cfg.error_reward

    if not result.atoms:
        return cfg.error_reward

    checks: list[bool] = []

    atom_set = set(result.atoms)

    for atom in expected_atoms:
        checks.append(atom in atom_set)

    for atom in forbidden_atoms:
        checks.append(atom not in atom_set)

    if not checks:
        return 1.0 if result.satisfiable else cfg.error_reward

    passed = sum(1 for x in checks if x)
    if passed == 0:
        return cfg.error_reward

    return float(passed) / float(len(checks))


def make_clingo_reward(config: ClingoRewardConfig):
    def reward_func(prompts=None, completions=None, tasks=None, **kwargs) -> list[float]:
        completions = completions or []

        if tasks is None:
            tasks = kwargs.get("task")

        if not isinstance(tasks, list):
            tasks = [tasks] * len(completions)

        rewards = []
        for completion, task in zip(completions, tasks):
            rewards.append(score_clingo_completion(completion, task or {}, config))

        return rewards

    reward_func.__name__ = "clingo_solver_reward"
    return reward_func
