#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from src.rl.grpo_utils import read_yaml, write_json
from src.rl.pythoncodes_dataset import prepare_pythoncodes_grpo_dataset
from src.rl.rewards import PythonRewardConfig, score_python_completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl/grpo_pythoncodes.yaml")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default="outputs/rl/smoke_rewards_pythoncodes.json")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    ds_cfg = dict(cfg.get("dataset", {}))
    ds_cfg["limit"] = args.limit

    model_path = cfg.get("model", {}).get("base_model")
    tokenizer = None
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    ds = prepare_pythoncodes_grpo_dataset(ds_cfg, tokenizer=tokenizer)
    reward_cfg = PythonRewardConfig.from_dict(cfg.get("reward", {}))

    rows = []
    for row in ds:
        reference = row.get("reference") or ""
        tests = row.get("tests") or []
        good = score_python_completion(reference, reference=reference, tests=tests, cfg=reward_cfg)
        bad = score_python_completion("def solution(*args, **kwargs):\n    pass", reference=reference, tests=tests, cfg=reward_cfg)
        rows.append({
            "task_id": row.get("task_id", ""),
            "reference_len": len(reference),
            "num_tests": len(tests),
            "reward_reference": good,
            "reward_bad_stub": bad,
        })

    out = Path(args.output)
    write_json(out, {"config": args.config, "num_rows": len(rows), "rows": rows})
    print(f"Saved reward smoke: {out}")
    for item in rows[:3]:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
