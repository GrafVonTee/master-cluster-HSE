#!/usr/bin/env python3
from datasets import load_from_disk

from src.dsl.clingo.rewards import ClingoRewardConfig, score_clingo_completion


def main():
    ds = load_from_disk("datasets/clingo/synthetic_v1")
    row = ds["train"][0]
    task = dict(row)

    cfg = ClingoRewardConfig(timeout=3.0)

    good = row["reference"]
    empty = ""
    syntax_bad = "this is not asp code"
    partial = """
reachable(X,Y) :- edge(X,Y).
#show reachable/2.
"""

    print("task_id", row["task_id"])
    print("topic", row["topic"])
    print("facts")
    print(row["facts"])
    print("expected_atoms", row["expected_atoms"])

    scores = {
        "good": score_clingo_completion(good, task, cfg),
        "empty": score_clingo_completion(empty, task, cfg),
        "syntax_bad": score_clingo_completion(syntax_bad, task, cfg),
        "partial": score_clingo_completion(partial, task, cfg),
    }

    for k, v in scores.items():
        print(k, v)

    assert scores["good"] == 1.0
    assert scores["empty"] == -1.0
    assert scores["syntax_bad"] == -1.0
    assert -1.0 <= scores["partial"] <= 1.0

    print("CLINGO_REWARD_SMOKE_OK")


if __name__ == "__main__":
    main()
