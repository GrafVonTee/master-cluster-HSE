#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict

from src.dsl.clingo.runner import solve_clingo


def mk_exact_row(
    task_id: str,
    topic: str,
    difficulty: str,
    instruction: str,
    facts: str,
    reference: str,
) -> dict:
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
        "oracle_tests": [],
    }


def mk_oracle_row(
    task_id: str,
    topic: str,
    difficulty: str,
    instruction: str,
    facts: str,
    reference: str,
    oracle_tests: list[dict],
) -> dict:
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
        "expected_atoms": [],
        "forbidden_atoms": [],
        "oracle_tests": oracle_tests,
    }


def build_rows() -> list[dict]:
    rows: list[dict] = []

    # EASY: two-hop paths.
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
            mk_exact_row(
                f"two_hop_{i:03d}",
                "two_hop",
                "easy",
                "Write a clingo program that derives two_hop(X,Z) whenever there is an edge X->Y and an edge Y->Z. Show two_hop/2.",
                facts,
                reference,
            )
        )

    # EASY: conflict detection.
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
            mk_exact_row(
                f"conflict_{i:03d}",
                "graph_coloring_validation",
                "easy",
                "Write a clingo program that derives conflict(X,Y) when adjacent nodes X and Y have the same color. Treat edge/2 as undirected for conflicts. Show conflict/2.",
                facts,
                reference,
            )
        )

    # MEDIUM: reachability recursion.
    for i in range(40):
        n = 4 + (i % 4)
        edges = [(x, x + 1) for x in range(1, n)]
        if i % 3 == 0:
            edges.append((1, n))
        facts = "\n".join(f"edge({a},{b})." for a, b in edges)
        reference = """
reachable(X,Y) :- edge(X,Y).
reachable(X,Z) :- reachable(X,Y), edge(Y,Z).
#show reachable/2.
"""
        rows.append(
            mk_exact_row(
                f"reachability_{i:03d}",
                "reachability",
                "medium",
                "Write a clingo program that derives reachable(X,Y) if Y can be reached from X by following directed edge facts. Show reachable/2.",
                facts,
                reference,
            )
        )

    # MEDIUM: ancestor recursion.
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
            mk_exact_row(
                f"ancestor_{i:03d}",
                "ancestor",
                "medium",
                "Write a clingo program that derives ancestor(X,Y) from parent/2 facts using recursive transitive closure. Show ancestor/2.",
                facts,
                reference,
            )
        )

    # MEDIUM/HARD: aggregate #count.
    for i in range(40):
        nodes = ["a", "b", "c", "d"]
        edges = [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")]
        if i % 2:
            edges.append(("a", "d"))
        facts = "\n".join([*(f"node({x})." for x in nodes), *(f"edge({x},{y})." for x, y in edges)])
        reference = """
degree(X,N) :- node(X), N = #count { Y : edge(X,Y) }.
#show degree/2.
"""
        rows.append(
            mk_exact_row(
                f"out_degree_count_{i:03d}",
                "aggregate_count",
                "medium",
                "Write a clingo program that derives degree(X,N), where N is the number of outgoing edge(X,Y) facts for node X. Use #count and show degree/2.",
                facts,
                reference,
            )
        )

    # HARD: graph coloring with choice rules and constraints.
    graph_coloring_check_basic = """
has_assign(N) :- assign(N,C), color(C).
:- node(N), not has_assign(N).
:- assign(N,C1), assign(N,C2), C1 != C2.
:- assign(N,C), not node(N).
:- assign(N,C), not color(C).
"""
    graph_coloring_check_conflict = graph_coloring_check_basic + """
:- edge(X,Y), assign(X,C), assign(Y,C).
"""

    for i in range(40):
        n = 4 + (i % 3)
        nodes = list(range(1, n + 1))
        edges = [(x, x + 1) for x in range(1, n)]
        if i % 2 == 0 and n >= 4:
            edges.append((1, 3))

        facts = "\n".join(
            [
                *(f"node({x})." for x in nodes),
                "color(red).",
                "color(blue).",
                "color(green).",
                *(f"edge({a},{b})." for a, b in edges),
            ]
        )

        reference = """
1 { assign(N,C) : color(C) } 1 :- node(N).
:- edge(X,Y), assign(X,C), assign(Y,C).
#show assign/2.
"""
        oracle_tests = [
            {"name": "exactly_one_valid_color", "expect": "sat", "program": graph_coloring_check_basic},
            {"name": "no_adjacent_conflict", "expect": "sat", "program": graph_coloring_check_conflict},
        ]

        rows.append(
            mk_oracle_row(
                f"graph_coloring_choice_{i:03d}",
                "choice_rules_constraints",
                "hard",
                "Write a clingo program that assigns exactly one color to each node using assign(Node,Color). Adjacent nodes connected by edge/2 must not have the same color. Use color/1 facts as the available colors. Show assign/2.",
                facts,
                reference,
                oracle_tests,
            )
        )

    # HARD: vertex cover with optimization.
    vc_graphs = [
        {
            "name": "path4",
            "vertices": [1, 2, 3, 4],
            "edges": [(1, 2), (2, 3), (3, 4)],
            "opt": 2,
        },
        {
            "name": "cycle4",
            "vertices": [1, 2, 3, 4],
            "edges": [(1, 2), (2, 3), (3, 4), (4, 1)],
            "opt": 2,
        },
        {
            "name": "star4",
            "vertices": [1, 2, 3, 4],
            "edges": [(1, 2), (1, 3), (1, 4)],
            "opt": 1,
        },
        {
            "name": "triangle",
            "vertices": [1, 2, 3],
            "edges": [(1, 2), (2, 3), (1, 3)],
            "opt": 2,
        },
    ]

    for i in range(40):
        g = vc_graphs[i % len(vc_graphs)]
        facts = "\n".join(
            [
                *(f"vertex({v})." for v in g["vertices"]),
                *(f"edge({a},{b})." for a, b in g["edges"]),
            ]
        )

        opt = g["opt"]
        reference = """
{ in_cover(V) } :- vertex(V).
:- edge(U,V), not in_cover(U), not in_cover(V).
#minimize { 1,V : in_cover(V) }.
#show in_cover/1.
"""
        coverage_test = """
:- edge(U,V), not in_cover(U), not in_cover(V).
"""
        optimality_test = f"""
:- edge(U,V), not in_cover(U), not in_cover(V).
:- {opt + 1} {{ in_cover(V) : vertex(V) }}.
"""

        oracle_tests = [
            {"name": "covers_every_edge", "expect": "sat", "program": coverage_test},
            {"name": "cover_size_is_optimal_or_better", "expect": "sat", "program": optimality_test},
        ]

        rows.append(
            mk_oracle_row(
                f"vertex_cover_minimize_{i:03d}",
                "optimization_minimize",
                "hard",
                "Write a clingo program for the minimum vertex cover problem. Choose vertices using in_cover(V), every edge must be covered by at least one selected endpoint, and use #minimize to minimize the number of selected vertices. Show in_cover/1.",
                facts,
                reference,
                oracle_tests,
            )
        )

    # HARD: scheduling/assignment with exactly-one constraints.
    schedule_basic = """
has_slot(J) :- scheduled(J,S), slot(S).
:- job(J), not has_slot(J).
:- scheduled(J,S1), scheduled(J,S2), S1 != S2.
:- scheduled(J,S), not job(J).
:- scheduled(J,S), not slot(S).
"""
    schedule_conflict = schedule_basic + """
:- conflict(J1,J2), scheduled(J1,S), scheduled(J2,S).
"""

    for i in range(40):
        jobs = ["a", "b", "c", "d"]
        slots = ["s1", "s2", "s3"]
        conflicts = [("a", "b"), ("b", "c")]
        if i % 2:
            conflicts.append(("c", "d"))

        facts = "\n".join(
            [
                *(f"job({j})." for j in jobs),
                *(f"slot({s})." for s in slots),
                *(f"conflict({a},{b})." for a, b in conflicts),
            ]
        )

        reference = """
1 { scheduled(J,S) : slot(S) } 1 :- job(J).
:- conflict(J1,J2), scheduled(J1,S), scheduled(J2,S).
#show scheduled/2.
"""
        oracle_tests = [
            {"name": "exactly_one_valid_slot", "expect": "sat", "program": schedule_basic},
            {"name": "no_conflicting_jobs_same_slot", "expect": "sat", "program": schedule_conflict},
        ]

        rows.append(
            mk_oracle_row(
                f"scheduling_choice_{i:03d}",
                "assignment_constraints",
                "hard",
                "Write a clingo program that assigns exactly one slot to every job using scheduled(Job,Slot). Jobs connected by conflict/2 must not be scheduled in the same slot. Show scheduled/2.",
                facts,
                reference,
                oracle_tests,
            )
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="datasets/clingo/synthetic_v2")
    parser.add_argument("--val-size", type=int, default=48)
    parser.add_argument("--test-size", type=int, default=48)
    args = parser.parse_args()

    rows = build_rows()

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
