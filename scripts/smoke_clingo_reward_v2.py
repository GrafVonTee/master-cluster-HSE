#!/usr/bin/env python3
from datasets import load_from_disk

from src.dsl.clingo.rewards import ClingoRewardConfig, score_clingo_completion


def find_one(ds, topic):
    for split in ["train", "validation", "test"]:
        for row in ds[split]:
            if row["topic"] == topic:
                return dict(row)
    raise RuntimeError(f"missing topic: {topic}")


def main():
    ds = load_from_disk("datasets/clingo/synthetic_v2")
    cfg = ClingoRewardConfig(timeout=3.0)

    ancestor = find_one(ds, "ancestor")
    ancestor_good = ancestor["reference"]
    ancestor_partial = """
ancestor(X,Y) :- parent(X,Y).
#show ancestor/2.
"""
    ancestor_bad = """
reachable(X,Y) :- edge(X,Y).
#show reachable/2.
"""

    print("ANCESTOR", ancestor["task_id"])
    print("good", score_clingo_completion(ancestor_good, ancestor, cfg))
    print("partial", score_clingo_completion(ancestor_partial, ancestor, cfg))
    print("bad", score_clingo_completion(ancestor_bad, ancestor, cfg))

    assert score_clingo_completion(ancestor_good, ancestor, cfg) == 1.0
    assert 0.0 < score_clingo_completion(ancestor_partial, ancestor, cfg) < 1.0
    assert score_clingo_completion(ancestor_bad, ancestor, cfg) == -1.0

    vc = find_one(ds, "optimization_minimize")
    vc_good = vc["reference"]
    vc_partial = """
in_cover(V) :- vertex(V).
#show in_cover/1.
"""
    vc_bad = """
#show in_cover/1.
"""

    print("VERTEX_COVER", vc["task_id"])
    print("good", score_clingo_completion(vc_good, vc, cfg))
    print("partial", score_clingo_completion(vc_partial, vc, cfg))
    print("bad", score_clingo_completion(vc_bad, vc, cfg))

    assert score_clingo_completion(vc_good, vc, cfg) == 1.0
    assert 0.0 < score_clingo_completion(vc_partial, vc, cfg) < 1.0
    assert score_clingo_completion(vc_bad, vc, cfg) == -1.0

    coloring = find_one(ds, "choice_rules_constraints")
    coloring_good = coloring["reference"]
    coloring_bad = """
assign(N,red) :- node(N).
#show assign/2.
"""

    print("COLORING", coloring["task_id"])
    print("good", score_clingo_completion(coloring_good, coloring, cfg))
    print("bad", score_clingo_completion(coloring_bad, coloring, cfg))

    assert score_clingo_completion(coloring_good, coloring, cfg) == 1.0
    assert score_clingo_completion(coloring_bad, coloring, cfg) < 1.0

    print("CLINGO_REWARD_V2_SMOKE_OK")


if __name__ == "__main__":
    main()
