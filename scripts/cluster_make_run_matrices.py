#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import yaml


def load_runs(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = cfg.get("runs", [])
    names = []
    for run in runs:
        name = run.get("name")
        if not name:
            raise ValueError(f"Bad run entry without name: {run}")
        names.append(str(name))
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/lora_pythoncodes_cl.yaml")
    parser.add_argument("--out-dir", default="outputs/cluster")
    parser.add_argument("--include-base", action="store_true", default=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_runs = load_runs(Path(args.config))
    eval_runs = ["base"] + train_runs

    (out / "train_runs.txt").write_text("\n".join(train_runs) + "\n", encoding="utf-8")
    (out / "eval_experiments.txt").write_text("\n".join(eval_runs) + "\n", encoding="utf-8")

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
