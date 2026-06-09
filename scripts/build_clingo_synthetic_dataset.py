#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict

from src.dsl.clingo.runner import solve_clingo


def mk_row(task_id: str, topic: str, difficulty: str, instruction: str, facts: str, reference: str) -> dict:
    result = solve_clingo(facts + "\n\n" + reference, timeout=3.0, max_models=1)
    if not result.ok or not result.satisfiable:
        raise RuntimeError(f"bad reference for {task_id}: {result}")

    return {
        "task_id": task_id,
        "language": "clingo",
        "topic": topic,
        "difficulty": difficulty,
        "instruction": instruction.strip(),
        "facts": facts.strip(),
        "output": reference.strip(),
        "reference": reference.strip(),
        "expected_satisfiable": True,
        "expected_atoms": result.atoms,
        "forbidden_atoms": [],
    }


def build_rows() -> list[dict]:
    rows = []

    # 1. Transitive reachability.
    for i in range(40):
        n = 4 + (i % 4)
        edges = []
        for x in range(1, n):
            edges.append((x, x + 1))
        if i % 3 == 0:
            edges.append((1, n))
        facts = "\n".join(f"edge({a},{b})." for a, b in edges)
        reference = """
reachable(X,Y) :- edge(X,Y).
reachable(X,Z) :- reachable(X,Y), edge(Y,Z).
#show reachable/2.
"""
        rows.append(
            mk_row(
                f"reachability_{i:03d}",
                "reachability",
                "medium",
                "Write a clingo program that derives reachable(X,Y) if Y can be reached from X by following directed edge facts. Show reachable/2.",
                facts,
                reference,
            )
        )

    # 2. Ancestor closure.
    for i in range(40):
        facts = "\n".join(
            [
                "parent(alice,bob).",
                "parent(bob,carol).",
                "parent(carol,dave).",
                "parent(eve,frank).",
            ]
        )
        if i % 2 == 0:
            facts += "\nparent(dave,grace)."
        reference = """
ancestor(X,Y) :- parent(X,Y).
ancestor(X,Z) :- ancestor(X,Y), parent(Y,Z).
#show ancestor/2.
"""
        rows.append(
            mk_row(
                f"ancestor_{i:03d}",
                "ancestor",
                "medium",
                "Write a clingo program that derives ancestor(X,Y) from parent/2 facts using recursive transitive closure. Show ancestor/2.",
                facts,
                reference,
            )
        )

    # 3. Two-hop paths.
    for i in range(40):
        facts = "\n".join(
            [
                "edge(a,b).",
                "edge(b,c).",
                "edge(c,d).",
                "edge(a,d).",
            ]
        )
        if i % 2:
            facts += "\nedge(d,e)."
        reference = """
two_hop(X,Z) :- edge(X,Y), edge(Y,Z).
#show two_hop/2.
"""
        rows.append(
            mk_row(
                f"two_hop_{i:03d}",
                "two_hop",
                "easy",
                "Write a clingo program that derives two_hop(X,Z) whenever there is an edge X->Y and an edge Y->Z. Show two_hop/2.",
                facts,
                reference,
            )
        )

    # 4. Conflicts from assignments and edges.
    for i in range(40):
        facts = "\n".join(
            [
                "edge(a,b).",
                "edge(b,c).",
                "edge(c,d).",
                "color(a,red).",
                "color(b,red).",
                "color(c,blue).",
                "color(d,blue).",
            ]
        )
        if i % 2:
            facts += "\nedge(a,c)."
        reference = """
conflict(X,Y) :- edge(X,Y), color(X,C), color(Y,C).
conflict(Y,X) :- edge(X,Y), color(X,C), color(Y,C).
#show conflict/2.
"""
        rows.append(
            mk_row(
                f"conflict_{i:03d}",
                "graph_coloring_validation",
                "easy",
                "Write a clingo program that derives conflict(X,Y) when adjacent nodes X and Y have the same color. Treat edge/2 as undirected for conflicts. Show conflict/2.",
                facts,
                reference,
            )
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="datasets/clingo/synthetic_v1")
    parser.add_argument("--val-size", type=int, default=24)
    parser.add_argument("--test-size", type=int, default=24)
    args = parser.parse_args()

    rows = build_rows()

    # Deterministic split.
    test = rows[: args.test_size]
    val = rows[args.test_size : args.test_size + args.val_size]
    train = rows[args.test_size + args.val_size :]

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
            "test": Dataset.from_list(test),
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))

    print(ds)
    print("out", out)
    print("sample", ds["train"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
