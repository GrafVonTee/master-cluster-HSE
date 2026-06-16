#!/usr/bin/env python3
"""Validate that Clingo synthetic v4 changes only instruction/prompt fields."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


PROTECTED_FIELDS = ["facts", "reference", "output", "expected_atoms", "oracle_tests"]
STRUCTURE_FIELDS = ["language", "topic", "difficulty", "expected_satisfiable", "forbidden_atoms"]
NEW_FIELDS = ["instruction_v3", "instruction_v4", "source_task_id"]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load(path: str | Path) -> dict[str, Dataset]:
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        return {split: loaded[split] for split in loaded.keys()}
    return {"train": loaded}


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get(field, "")) for r in rows).items()))


def _write_diagnostics(path: Path, diagnostics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "kind", "name", "count"],
        )
        writer.writeheader()
        for item in diagnostics:
            writer.writerow(item)


def _validate_reference(row: dict[str, Any], timeout: float) -> float:
    from src.dsl.clingo.rewards import ClingoRewardConfig, score_clingo_completion

    cfg = ClingoRewardConfig(timeout=timeout, max_models=1)
    return float(score_clingo_completion(row.get("reference") or row.get("output") or "", row, cfg))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/clingo/synthetic_v3_100")
    parser.add_argument("--target", default="datasets/clingo/synthetic_v4")
    parser.add_argument("--out", default="outputs/clingo_v4/validation")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--skip-solver", action="store_true")
    parser.add_argument("--require-instruction-changed", action="store_true")
    args = parser.parse_args()

    source = _load(args.source)
    target = _load(args.target)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []

    if set(source) != set(target):
        errors.append(f"Split set differs: source={sorted(source)} target={sorted(target)}")

    for split in sorted(set(source) & set(target)):
        src_rows = [dict(r) for r in source[split]]
        tgt_rows = [dict(r) for r in target[split]]
        if len(src_rows) != len(tgt_rows):
            errors.append(f"{split}: row count differs: source={len(src_rows)} target={len(tgt_rows)}")

        diagnostics.append({"split": split, "kind": "split", "name": split, "count": len(tgt_rows)})
        for topic, count in _counts(tgt_rows, "topic").items():
            diagnostics.append({"split": split, "kind": "topic", "name": topic, "count": count})
        for difficulty, count in _counts(tgt_rows, "difficulty").items():
            diagnostics.append({"split": split, "kind": "difficulty", "name": difficulty, "count": count})

        src_by_id = {str(r.get("task_id", "")): r for r in src_rows}
        changed = 0
        unchanged = 0

        for idx, tgt in enumerate(tgt_rows):
            for field in NEW_FIELDS:
                if field not in tgt:
                    errors.append(f"{split}[{idx}]: missing new field {field}")

            source_task_id = str(tgt.get("source_task_id", ""))
            src = src_by_id.get(source_task_id)
            if src is None and idx < len(src_rows):
                # Fallback for early smoke/debug builds that preserved ordering but not source_task_id.
                src = src_rows[idx]
                warnings.append(f"{split}[{idx}]: source_task_id={source_task_id!r} not found; compared by row order")
            if src is None:
                errors.append(f"{split}[{idx}]: cannot find matching source row for source_task_id={source_task_id!r}")
                continue

            if tgt.get("instruction_v3") != src.get("instruction"):
                errors.append(f"{split}[{idx}] {source_task_id}: instruction_v3 does not match source instruction")
            if tgt.get("instruction_v4") != tgt.get("instruction"):
                errors.append(f"{split}[{idx}] {source_task_id}: instruction_v4 does not match active instruction")
            if tgt.get("instruction") == src.get("instruction"):
                unchanged += 1
            else:
                changed += 1

            for field in PROTECTED_FIELDS + STRUCTURE_FIELDS:
                if field in src or field in tgt:
                    if _canonical(src.get(field)) != _canonical(tgt.get(field)):
                        errors.append(f"{split}[{idx}] {source_task_id}: field changed: {field}")

            if not args.skip_solver:
                score = _validate_reference(tgt, timeout=args.timeout)
                solver_rows.append({"split": split, "task_id": tgt.get("task_id"), "source_task_id": source_task_id, "score": score})
                if score < 1.0:
                    errors.append(f"{split}[{idx}] {source_task_id}: reference solver score={score}")

        if args.require_instruction_changed and unchanged:
            errors.append(f"{split}: {unchanged} instructions are unchanged under --require-instruction-changed")
        print(f"{split}: rows={len(tgt_rows)} changed={changed} unchanged={unchanged}")
        print(f"  topic={_counts(tgt_rows, 'topic')}")
        print(f"  difficulty={_counts(tgt_rows, 'difficulty')}")

    _write_diagnostics(out_dir / "diagnostics_counts.csv", diagnostics)
    if solver_rows:
        with (out_dir / "reference_solver_scores.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "task_id", "source_task_id", "score"])
            writer.writeheader()
            writer.writerows(solver_rows)

    report = {
        "source": args.source,
        "target": args.target,
        "errors": errors,
        "warnings": warnings,
        "skip_solver": bool(args.skip_solver),
        "diagnostics_csv": str(out_dir / "diagnostics_counts.csv"),
        "solver_scores_csv": str(out_dir / "reference_solver_scores.csv") if solver_rows else "",
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if warnings:
        print("WARNINGS")
        for w in warnings[:20]:
            print("-", w)
        if len(warnings) > 20:
            print(f"... {len(warnings) - 20} more warnings")

    if errors:
        print("ERRORS")
        for e in errors[:50]:
            print("-", e)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more errors")
        return 1

    print(f"validation ok: {out_dir / 'validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
