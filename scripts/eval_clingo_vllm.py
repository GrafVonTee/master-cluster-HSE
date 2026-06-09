#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.dsl.clingo.rewards import ClingoRewardConfig, extract_asp_code, score_clingo_completion


def build_prompt(tokenizer, row: dict[str, Any]) -> str:
    system = (
        "You are an Answer Set Programming assistant. "
        "Return only valid clingo/ASP code. "
        "Do not explain. Do not use markdown unless necessary."
    )

    user = (
        f"Task:\n{row['instruction'].strip()}\n\n"
        f"Facts:\n{row['facts'].strip()}\n\n"
        "Write the clingo program."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for r in records:
        groups.setdefault(("overall", "overall"), []).append(r)
        groups.setdefault(("difficulty", r["difficulty"]), []).append(r)
        groups.setdefault(("topic", r["topic"]), []).append(r)

    rows = []
    for (group, name), xs in sorted(groups.items()):
        n = len(xs)
        scores = [float(x["score"]) for x in xs]
        rows.append(
            {
                "group": group,
                "name": name,
                "num_tasks": n,
                "mean_reward": sum(scores) / n if n else 0.0,
                "full_pass_rate": sum(1 for s in scores if s >= 1.0) / n if n else 0.0,
                "partial_rate": sum(1 for s in scores if s > 0.0) / n if n else 0.0,
                "error_rate": sum(1 for s in scores if s <= -1.0) / n if n else 0.0,
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--dataset", default="datasets/clingo/synthetic_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.dataset)[args.split]
    rows = [dict(x) for x in ds]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    prompts = [build_prompt(tokenizer, r) for r in rows]

    from vllm import LLM, SamplingParams

    lora_request = None
    enable_lora = bool(args.adapter)

    if enable_lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("clingo_adapter", 1, args.adapter)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=enable_lora,
        max_lora_rank=64,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,
    )

    outputs = llm.generate(prompts, sampling, lora_request=lora_request)

    generated = []
    for row, prompt, output in zip(rows, prompts, outputs):
        text = output.outputs[0].text if output.outputs else ""
        generated.append(
            {
                "task": row,
                "prompt": prompt,
                "completion": text,
                "code": extract_asp_code(text),
            }
        )

    # Free VRAM before solver scoring. Scoring itself is CPU-side.
    del llm
    gc.collect()

    cfg = ClingoRewardConfig(timeout=3.0, max_models=1)
    records = []

    for item in generated:
        row = item["task"]
        score = score_clingo_completion(item["completion"], row, cfg)

        records.append(
            {
                "task_id": row["task_id"],
                "topic": row["topic"],
                "difficulty": row["difficulty"],
                "score": score,
                "full_pass": score >= 1.0,
                "partial": score > 0.0,
                "error": score <= -1.0,
                "completion": item["completion"],
                "code": item["code"],
                "reference": row["reference"],
                "facts": row["facts"],
                "instruction": row["instruction"],
            }
        )

    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(records)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "name",
                "num_tasks",
                "mean_reward",
                "full_pass_rate",
                "partial_rate",
                "error_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("wrote", out_dir / "records.jsonl")
    print("wrote", out_dir / "summary.csv")

    for r in summary:
        if r["group"] == "overall":
            print("OVERALL", r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
