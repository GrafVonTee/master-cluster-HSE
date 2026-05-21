#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import yaml


def load_runs(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = cfg.get("runs", [])
    names: list[str] = []
    for run in runs:
        name = run.get("name")
        if not name:
            raise ValueError(f"Bad run entry without name: {run}")
        names.append(str(name))
    if len(names) != len(set(names)):
        dup = sorted({x for x in names if names.count(x) > 1})
        raise ValueError(f"Duplicate run names: {dup}")
    return names


def write_txt(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/lora_pythoncodes_cl.yaml")
    parser.add_argument("--out-dir", default="outputs/cluster")
    parser.add_argument("--no-base", action="store_true", help="Do not include base in eval matrix")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_runs = load_runs(Path(args.config))
    eval_runs = ([] if args.no_base else ["base"]) + train_runs

    write_txt(out / "train_runs.txt", train_runs)
    write_txt(out / "eval_experiments.txt", eval_runs)
    write_csv(out / "train_runs.csv", ["array_id", "run_name"], [[i, x] for i, x in enumerate(train_runs, 1)])
    write_csv(out / "eval_experiments.csv", ["array_id", "experiment"], [[i, x] for i, x in enumerate(eval_runs, 1)])

    print(f"Wrote {out / 'train_runs.txt'} ({len(train_runs)} train runs)")
    print(f"Wrote {out / 'eval_experiments.txt'} ({len(eval_runs)} eval experiments)")
    print("\nTrain runs:")
    for i, name in enumerate(train_runs, 1):
        print(f"{i:02d}: {name}")
    print("\nEval experiments:")
    for i, name in enumerate(eval_runs, 1):
        print(f"{i:02d}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
