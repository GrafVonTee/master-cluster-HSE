#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="outputs/train_runs")
    parser.add_argument("--out-dir", default="outputs/train_runs")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    out_dir = Path(args.out_dir)
    models_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_frames = []
    for path in sorted((train_root / "manifests").glob("*.csv")):
        try:
            df = pd.read_csv(path)
            df["manifest_file"] = str(path)
            manifest_frames.append(df)
        except Exception as exc:
            print(f"WARN: failed to read {path}: {exc}")

    stage_frames = []
    for path in sorted(train_root.glob("*/stage_summary.csv")):
        try:
            df = pd.read_csv(path)
            df["run_dir"] = path.parent.name
            stage_frames.append(df)
        except Exception as exc:
            print(f"WARN: failed to read {path}: {exc}")

    adapter_rows = []
    for adapter_config in sorted(models_dir.glob("*-sft-*/adapter_config.json")):
        folder = adapter_config.parent
        adapter_rows.append({
            "adapter_dir": str(folder),
            "adapter_name": folder.name,
            "adapter_config": str(adapter_config),
            "has_adapter_model_safetensors": (folder / "adapter_model.safetensors").exists(),
            "has_tokenizer_config": (folder / "tokenizer_config.json").exists(),
        })

    if manifest_frames:
        manifest = pd.concat(manifest_frames, ignore_index=True).sort_values("run_name", kind="stable")
    else:
        manifest = pd.DataFrame()
    manifest.to_csv(out_dir / "trained_adapters.csv", index=False)
    manifest.to_markdown(out_dir / "trained_adapters.md", index=False)

    if stage_frames:
        stages = pd.concat(stage_frames, ignore_index=True).sort_values(["run_name", "stage_idx"], kind="stable")
    else:
        stages = pd.DataFrame()
    stages.to_csv(out_dir / "train_stage_summary_all.csv", index=False)
    stages.to_markdown(out_dir / "train_stage_summary_all.md", index=False)

    adapters = pd.DataFrame(adapter_rows)
    adapters.to_csv(out_dir / "adapter_files.csv", index=False)
    adapters.to_markdown(out_dir / "adapter_files.md", index=False)

    summary = {
        "manifest_parts": len(manifest_frames),
        "stage_summary_parts": len(stage_frames),
        "adapters_found": len(adapter_rows),
        "out_dir": str(out_dir),
    }
    (out_dir / "train_merge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
