#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict


LANG = "clingo"


def md5(x: str) -> str:
    return hashlib.md5(x.encode("utf-8")).hexdigest()


def lines(items):
    return "\n".join(items)


def atom(name: str, *args) -> str:
    return f"{name}(" + ",".join(map(str, args)) + ")"


def edge_facts(edges):
    return [f"edge({u},{v})." for u, v in edges]


def node_facts(nodes):
    return [f"node({x})." for x in nodes]


def closure(edges):
    adj = defaultdict(list)
    nodes = set()
    for u, v in edges:
        adj[u].append(v)
        nodes.add(u)
        nodes.add(v)

    out = set()
    for s in nodes:
        stack = list(adj[s])
        seen = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            out.add((s, x))
            stack.extend(adj[x])
    return sorted(out)


def make_row(task_id, topic, difficulty, instruction, facts, reference, expected_atoms, oracle_tests=None):
    return {
        "task_id": task_id,
        "language": LANG,
        "topic": topic,
        "difficulty": difficulty,
        "instruction": instruction,
        "facts": facts.strip(),
        "output": reference.strip(),
        "reference": reference.strip(),
        "expected_satisfiable": True,
        "expected_atoms": sorted(expected_atoms),
        "forbidden_atoms": [],
        "oracle_tests": oracle_tests or [],
    }


def gen_dag(rng, n, extra_edges=0):
    edges = set()
    # Backbone ensures non-trivial reachability.
    for i in range(1, n):
        edges.add((i, i + 1))

    candidates = [(i, j) for i in range(1, n + 1) for j in range(i + 2, n + 1)]
    rng.shuffle(candidates)
    for e in candidates[:extra_edges]:
        edges.add(e)

    return sorted(edges)


def gen_two_hop(i, rng):
    n = rng.randint(5, 9)
    edges = gen_dag(rng, n, extra_edges=rng.randint(1, n))
    facts = lines(edge_facts(edges))
    pairs = sorted({(x, z) for x, y in edges for y2, z in edges if y == y2})
    expected = [atom("two_hop", x, z) for x, z in pairs]
    ref = "two_hop(X,Z) :- edge(X,Y), edge(Y,Z).\n#show two_hop/2."
    return make_row(
        f"v3_two_hop_{i:03d}",
        "two_hop",
        "easy",
        "Write a clingo program that derives two_hop(X,Z) when there is a path X -> Y -> Z using edge/2 facts. Show two_hop/2.",
        facts,
        ref,
        expected,
    )


def gen_graph_coloring_validation(i, rng):
    n = rng.randint(5, 9)
    colors = ["red", "blue", "green"]
    edges = set()
    for x in range(1, n):
        edges.add((x, x + 1))
    for _ in range(rng.randint(2, n + 2)):
        a, b = sorted(rng.sample(range(1, n + 1), 2))
        edges.add((a, b))
    edges = sorted(edges)

    color_of = {x: rng.choice(colors) for x in range(1, n + 1)}

    fact_lines = []
    fact_lines += node_facts(range(1, n + 1))
    fact_lines += [f"color({c})." for c in colors]
    fact_lines += edge_facts(edges)
    fact_lines += [f"color_of({x},{c})." for x, c in sorted(color_of.items())]

    expected = [
        atom("bad_color", u, v)
        for u, v in edges
        if color_of[u] == color_of[v]
    ]

    ref = "bad_color(X,Y) :- edge(X,Y), color_of(X,C), color_of(Y,C).\n#show bad_color/2."
    return make_row(
        f"v3_graph_coloring_validation_{i:03d}",
        "graph_coloring_validation",
        "easy",
        "Write a clingo program that derives bad_color(X,Y) for every edge(X,Y) whose endpoints have the same color_of/2 value. Show bad_color/2.",
        lines(fact_lines),
        ref,
        expected,
    )


def gen_reachability(i, rng):
    n = rng.randint(5, 10)
    edges = gen_dag(rng, n, extra_edges=rng.randint(1, n + 2))
    facts = lines(edge_facts(edges))
    expected = [atom("reachable", x, y) for x, y in closure(edges)]
    ref = "reachable(X,Y) :- edge(X,Y).\nreachable(X,Z) :- reachable(X,Y), edge(Y,Z).\n#show reachable/2."
    return make_row(
        f"v3_reachability_{i:03d}",
        "reachability",
        "medium",
        "Write a clingo program that derives reachable(X,Y) if Y can be reached from X by following directed edge facts. Show reachable/2.",
        facts,
        ref,
        expected,
    )


def gen_ancestor(i, rng):
    n = rng.randint(6, 10)
    edges = []
    for child in range(2, n + 1):
        parent = rng.randint(1, child - 1)
        edges.append((parent, child))
    # Add names via integers to keep clingo atoms simple.
    fact_lines = [f"parent({p},{c})." for p, c in sorted(edges)]
    expected = [atom("ancestor", x, y) for x, y in closure(edges)]
    ref = "ancestor(X,Y) :- parent(X,Y).\nancestor(X,Z) :- ancestor(X,Y), parent(Y,Z).\n#show ancestor/2."
    return make_row(
        f"v3_ancestor_{i:03d}",
        "ancestor",
        "medium",
        "Write a clingo program that derives ancestor(X,Y) from parent/2 facts using the transitive closure of parent/2. Show ancestor/2.",
        lines(fact_lines),
        ref,
        expected,
    )


def gen_aggregate_count(i, rng):
    n = rng.randint(5, 10)
    nodes = list(range(1, n + 1))
    edges = set()

    for x in nodes:
        out_deg = rng.randint(0, min(4, n - 1))
        targets = rng.sample([y for y in nodes if y != x], out_deg)
        for y in targets:
            edges.add((x, y))

    fact_lines = node_facts(nodes) + edge_facts(sorted(edges))
    counts = Counter(u for u, _ in edges)
    expected = [atom("degree", x, counts[x]) for x in nodes]

    ref = "degree(X,N) :- node(X), N = #count { Y : edge(X,Y) }.\n#show degree/2."
    return make_row(
        f"v3_aggregate_count_{i:03d}",
        "aggregate_count",
        "medium",
        "Write a clingo program that derives degree(X,N), where N is the number of outgoing edge(X,Y) facts for node X. Use #count and show degree/2.",
        lines(fact_lines),
        ref,
        expected,
    )


def gen_choice_rules_constraints(i, rng):
    groups_n = rng.randint(3, 6)
    opts_n = rng.randint(3, 5)

    fact_lines = []
    expected = []

    for g in range(1, groups_n + 1):
        group = f"g{g}"
        fact_lines.append(f"group({group}).")
        allowed = rng.randint(1, opts_n)
        for o in range(1, opts_n + 1):
            opt = f"o{o}"
            fact_lines.append(f"option({group},{opt}).")
            if o != allowed:
                fact_lines.append(f"blocked({group},{opt}).")
            else:
                expected.append(atom("selected", group, opt))

    ref = (
        "1 { selected(G,O) : option(G,O) } 1 :- group(G).\n"
        ":- selected(G,O), blocked(G,O).\n"
        "#show selected/2."
    )
    return make_row(
        f"v3_choice_rules_constraints_{i:03d}",
        "choice_rules_constraints",
        "hard",
        "Write a clingo program that selects exactly one option for each group using selected(Group,Option). Blocked options from blocked/2 must not be selected. Show selected/2.",
        lines(fact_lines),
        ref,
        expected,
    )


def gen_assignment_constraints(i, rng):
    jobs_n = rng.randint(4, 7)
    slots_n = rng.randint(3, 5)

    jobs = [f"j{x}" for x in range(1, jobs_n + 1)]
    slots = [f"s{x}" for x in range(1, slots_n + 1)]

    chosen = {}
    for j in jobs:
        chosen[j] = rng.choice(slots)

    fact_lines = []
    fact_lines += [f"job({j})." for j in jobs]
    fact_lines += [f"slot({s})." for s in slots]

    # Forbid all non-target slots. This makes the satisfying assignment unique,
    # while still requiring exactly-one choice syntax.
    for j in jobs:
        for s in slots:
            if s != chosen[j]:
                fact_lines.append(f"forbidden({j},{s}).")

    # Add conflicts only between jobs that have different target slots.
    # They are semantically real but do not make the instance unsatisfiable.
    pairs = [(jobs[a], jobs[b]) for a in range(len(jobs)) for b in range(a + 1, len(jobs))]
    rng.shuffle(pairs)
    for a, b in pairs[: rng.randint(1, min(len(pairs), jobs_n))]:
        if chosen[a] != chosen[b]:
            fact_lines.append(f"conflict({a},{b}).")

    expected = [atom("scheduled", j, chosen[j]) for j in jobs]

    ref = (
        "1 { scheduled(J,S) : slot(S) } 1 :- job(J).\n"
        ":- scheduled(J,S), forbidden(J,S).\n"
        ":- conflict(J,K), scheduled(J,S), scheduled(K,S).\n"
        "#show scheduled/2."
    )
    return make_row(
        f"v3_assignment_constraints_{i:03d}",
        "assignment_constraints",
        "hard",
        "Write a clingo program that assigns exactly one slot to every job using scheduled(Job,Slot). Forbidden job-slot pairs must not be used, and jobs connected by conflict/2 must not be scheduled in the same slot. Show scheduled/2.",
        lines(fact_lines),
        ref,
        expected,
    )


def gen_optimization_minimize(i, rng):
    req_n = rng.randint(3, 6)
    reqs = [f"r{x}" for x in range(1, req_n + 1)]

    fact_lines = []
    expected = []

    for r in reqs:
        fact_lines.append(f"req({r}).")

    # Cheap unique items: one per requirement.
    for idx, r in enumerate(reqs, start=1):
        item = f"good{idx}"
        fact_lines.append(f"item({item}).")
        fact_lines.append(f"cost({item},1).")
        fact_lines.append(f"covers({item},{r}).")
        expected.append(atom("choose", item))

    # Expensive decoys: some cover multiple requirements, but are costlier.
    for d in range(1, rng.randint(3, 6)):
        item = f"decoy{d}"
        fact_lines.append(f"item({item}).")
        fact_lines.append(f"cost({item},{rng.randint(req_n + 2, req_n + 8)}).")
        covered = rng.sample(reqs, rng.randint(1, min(3, len(reqs))))
        for r in covered:
            fact_lines.append(f"covers({item},{r}).")

    ref = (
        "{ choose(I) } :- item(I).\n"
        "covered(R) :- choose(I), covers(I,R).\n"
        ":- req(R), not covered(R).\n"
        "#minimize { C,I : choose(I), cost(I,C) }.\n"
        "#show choose/1."
    )
    return make_row(
        f"v3_optimization_minimize_{i:03d}",
        "optimization_minimize",
        "hard",
        "Write a clingo program that chooses items using choose/1 so that every req/1 is covered by at least one selected item. Minimize total cost using cost(Item,Cost). Show choose/1.",
        lines(fact_lines),
        ref,
        expected,
    )


GENERATORS = [
    gen_two_hop,
    gen_graph_coloring_validation,
    gen_reachability,
    gen_ancestor,
    gen_aggregate_count,
    gen_choice_rules_constraints,
    gen_assignment_constraints,
    gen_optimization_minimize,
]


def build_rows(per_topic: int, seed: int):
    rows = []
    for gen_idx, gen in enumerate(GENERATORS):
        rng = random.Random(seed + 1009 * gen_idx)
        seen_facts = set()
        i = 0
        attempts = 0
        while i < per_topic:
            attempts += 1
            if attempts > per_topic * 100:
                raise RuntimeError(f"Could not generate enough unique facts for {gen.__name__}")
            row = gen(i, rng)
            key = md5(row["topic"] + "\n" + row["facts"])
            if key in seen_facts:
                continue
            seen_facts.add(key)
            rows.append(row)
            i += 1
    return rows


def split_rows(rows, train_per_topic, val_per_topic, test_per_topic, seed):
    rng = random.Random(seed)
    by_topic = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)

    train, val, test = [], [], []

    for topic, bucket in sorted(by_topic.items()):
        bucket = list(bucket)
        rng.shuffle(bucket)

        need = train_per_topic + val_per_topic + test_per_topic
        if len(bucket) < need:
            raise ValueError(f"{topic}: need {need}, got {len(bucket)}")

        test_part = bucket[:test_per_topic]
        val_part = bucket[test_per_topic:test_per_topic + val_per_topic]
        train_part = bucket[test_per_topic + val_per_topic:test_per_topic + val_per_topic + train_per_topic]

        test.extend(test_part)
        val.extend(val_part)
        train.extend(train_part)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def check_no_facts_leak(train, test):
    train_keys = {md5(r["topic"] + "\n" + r["facts"]) for r in train}
    test_keys = {md5(r["topic"] + "\n" + r["facts"]) for r in test}
    overlap = train_keys & test_keys
    if overlap:
        raise AssertionError(f"facts overlap: {len(overlap)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="datasets/clingo/synthetic_v3")
    parser.add_argument("--per-topic", type=int, default=40)
    parser.add_argument("--train-per-topic", type=int, default=28)
    parser.add_argument("--val-per-topic", type=int, default=6)
    parser.add_argument("--test-per-topic", type=int, default=6)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    rows = build_rows(per_topic=args.per_topic, seed=args.seed)
    train, val, test = split_rows(
        rows,
        train_per_topic=args.train_per_topic,
        val_per_topic=args.val_per_topic,
        test_per_topic=args.test_per_topic,
        seed=args.seed,
    )
    check_no_facts_leak(train, test)

    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))

    print("saved", out)
    for split in ["train", "validation", "test"]:
        d = ds[split]
        print()
        print(split, len(d))
        print("topic", Counter(d["topic"]))
        print("difficulty", Counter(d["difficulty"]))

    # Diagnostics.
    print()
    print("unique facts by topic:")
    all_rows = train + val + test
    by_topic = defaultdict(set)
    for r in all_rows:
        by_topic[r["topic"]].add(md5(r["facts"]))
    for topic in sorted(by_topic):
        print(topic, len(by_topic[topic]))


if __name__ == "__main__":
    main()
