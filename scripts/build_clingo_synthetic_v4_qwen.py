#!/usr/bin/env python3
"""Generate Clingo synthetic v4 by paraphrasing v3 instructions with Qwen3-14B.

The script is intentionally conservative: it changes only prompt-facing fields
and copies solver-checked fields byte-for-byte from the source dataset.

Default source: datasets/clingo/synthetic_v3_100
Default target: datasets/clingo/synthetic_v4
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from datasets import Dataset, DatasetDict, load_from_disk


PROTECTED_FIELDS = [
    "facts",
    "reference",
    "output",
    "expected_atoms",
    "oracle_tests",
]

STRUCTURE_FIELDS = [
    "language",
    "topic",
    "difficulty",
    "expected_satisfiable",
    "forbidden_atoms",
]


SYSTEM_PROMPT = (
    "You rewrite programming tasks for an ASP/clingo dataset. "
    "Return exactly one rewritten user instruction. "
    "Do not solve the task. Do not include explanations. "
    "Keep required output predicate names and arities exactly unchanged. "
    "Do not change facts, expected answer, or output format."
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_show_predicates(reference: str) -> str:
    shows: List[str] = []
    for line in reference.splitlines():
        line = line.strip()
        if line.startswith("#show"):
            shows.append(line.rstrip("."))
    return "; ".join(shows)


def _make_task_id(source_task_id: str, preserve: bool) -> str:
    if preserve:
        return source_task_id
    if source_task_id.startswith("v3_"):
        return "v4_" + source_task_id.removeprefix("v3_")
    return "v4_" + source_task_id


def _build_rewrite_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    instruction = _as_text(row.get("instruction"))
    reference = _as_text(row.get("reference") or row.get("output"))
    show_predicates = _extract_show_predicates(reference)

    user = (
        "Rewrite this clingo/ASP task instruction in a more natural style, "
        "as if a programmer asked for it.\n\n"
        "Constraints:\n"
        "- Keep all required output predicate names, variable roles, and arities.\n"
        "- Keep the same semantics; the same reference ASP program must remain correct.\n"
        "- Keep references to input fact predicates when needed.\n"
        "- Avoid overly direct algorithm hints when a natural requirement is enough.\n"
        "- Avoid explicit phrases like 'use #count', 'use transitive closure', "
        "or 'use exactly-one choice rule' unless the output requirement would be ambiguous without them.\n"
        "- Return only the rewritten instruction text.\n\n"
        f"Topic: {row.get('topic', '')}\n"
        f"Difficulty: {row.get('difficulty', '')}\n"
        f"Required shown predicates: {show_predicates or 'infer from instruction/reference'}\n\n"
        f"Original instruction:\n{instruction}\n\n"
        f"Reference program, for preserving predicate names only:\n{reference}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _strip_fences(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:text|markdown)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r"^\s*(rewritten instruction|instruction|answer)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip("'").strip()
    return " ".join(text.split())


def _fallback_if_bad(original: str, candidate: str) -> str:
    candidate = _strip_fences(candidate)
    if not candidate:
        return original
    # Very long completions usually mean the model ignored the one-instruction constraint.
    if len(candidate) > max(len(original) * 3, len(original) + 800):
        return original
    # Do not accept accidental code-only outputs.
    code_markers = [":-", "#show", "#minimize", "{" , "} :-"]
    if sum(marker in candidate for marker in code_markers) >= 2:
        return original
    return candidate


def _load_split_rows(source: Path) -> Dict[str, List[Dict[str, Any]]]:
    loaded = load_from_disk(str(source))
    if isinstance(loaded, DatasetDict):
        return {split: [dict(r) for r in loaded[split]] for split in loaded.keys()}
    return {"train": [dict(r) for r in loaded]}


def _format_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
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


def _generate_paraphrases(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    if args.dry_copy:
        return [_as_text(r.get("instruction")) for r in rows]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    prompts = [_format_prompt(tokenizer, _build_rewrite_messages(r)) for r in rows]

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,
    )
    outputs = llm.generate(prompts, sampling)
    return [out.outputs[0].text if out.outputs else "" for out in outputs]


def _make_v4_row(row: Dict[str, Any], new_instruction: str, preserve_task_id: bool) -> Dict[str, Any]:
    source_task_id = str(row.get("task_id", "")).strip()
    original_instruction = _as_text(row.get("instruction"))
    final_instruction = _fallback_if_bad(original_instruction, new_instruction)

    out = dict(row)
    out["source_task_id"] = source_task_id
    out["task_id"] = _make_task_id(source_task_id, preserve_task_id)
    out["instruction_v3"] = original_instruction
    out["instruction_v4"] = final_instruction
    out["instruction"] = final_instruction

    # Reassign protected fields from the source row explicitly. This prevents
    # accidental mutation even if later edits touch `out` above.
    for field in PROTECTED_FIELDS + STRUCTURE_FIELDS:
        if field in row:
            out[field] = row[field]
    return out


def _write_manifest(out_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v4_generation_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/clingo/synthetic_v3_100")
    parser.add_argument("--out", default="datasets/clingo/synthetic_v4")
    parser.add_argument("--model", default="/workspace/models/qwen3-14b")
    parser.add_argument("--splits", default="train,validation,test", help="Comma-separated split names or 'all'.")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Debug limit; 0 means all rows.")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--preserve-task-id", action="store_true")
    parser.add_argument("--dry-copy", action="store_true", help="Smoke mode: copy v3 instruction without loading Qwen.")
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out)
    source_rows_by_split = _load_split_rows(source)

    if args.splits.strip().lower() == "all":
        wanted_splits = list(source_rows_by_split)
    else:
        wanted_splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    missing = [s for s in wanted_splits if s not in source_rows_by_split]
    if missing:
        raise SystemExit(f"Missing splits in source dataset: {missing}. Available: {list(source_rows_by_split)}")

    new_splits: Dict[str, Dataset] = {}
    manifest: Dict[str, Any] = {
        "source": str(source),
        "out": str(out_dir),
        "model": args.model,
        "dry_copy": bool(args.dry_copy),
        "splits": {},
        "protected_fields": PROTECTED_FIELDS,
    }

    for split in wanted_splits:
        rows = source_rows_by_split[split]
        if args.limit_per_split and args.limit_per_split > 0:
            rows = rows[: args.limit_per_split]
        raw_generations = _generate_paraphrases(rows, args)
        v4_rows = [
            _make_v4_row(row, generation, preserve_task_id=args.preserve_task_id)
            for row, generation in zip(rows, raw_generations)
        ]
        new_splits[split] = Dataset.from_list(v4_rows)
        changed = sum(1 for r in v4_rows if r["instruction_v4"] != r["instruction_v3"])
        manifest["splits"][split] = {"rows": len(v4_rows), "instruction_changed": changed}
        print(f"{split}: rows={len(v4_rows)} instruction_changed={changed}")

    ds = DatasetDict(new_splits)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    _write_manifest(out_dir, manifest)
    print(f"saved {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
