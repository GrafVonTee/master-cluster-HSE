#!/usr/bin/env python3
from datasets import load_from_disk

from src.rl.rewards import PythonRewardConfig, score_python_completion


def main():
    ds = load_from_disk("datasets/mbpp_grpo_train")
    row = ds[0]
    cfg = PythonRewardConfig(timeout=3.0)

    good = row["output"]
    bad_empty = ""
    bad_wrong = "def definitely_wrong(*args, **kwargs):\n    return None\n"

    print("task_id", row["task_id"])
    print("instruction", row["instruction"])
    print("tests", row["tests"])

    print("good", score_python_completion(good, tests=row["tests"], cfg=cfg))
    print("bad_empty", score_python_completion(bad_empty, tests=row["tests"], cfg=cfg))
    print("bad_wrong", score_python_completion(bad_wrong, tests=row["tests"], cfg=cfg))

    assert score_python_completion(good, tests=row["tests"], cfg=cfg) > 0.9
    assert score_python_completion(bad_empty, tests=row["tests"], cfg=cfg) == -1.0
    assert score_python_completion(bad_wrong, tests=row["tests"], cfg=cfg) == -1.0

    print("MBPP_EXECUTION_REWARD_SMOKE_OK")


if __name__ == "__main__":
    main()
