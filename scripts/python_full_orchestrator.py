#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TRAIN_RUNS_ALL = ['sft_pythoncodes', 'cl_length_staged', 'cl_perplexity_staged', 'cl_lexical_staged', 'cl_semantic_staged', 'cl_llm_judge_staged', 'cl_length_cumulative', 'cl_perplexity_cumulative', 'cl_lexical_cumulative', 'cl_semantic_cumulative', 'cl_llm_judge_cumulative', 'cl_length_distribution', 'cl_perplexity_distribution', 'cl_lexical_distribution', 'cl_semantic_distribution', 'cl_llm_judge_distribution']
BEST_4B_CL = "cl_semantic_distribution"

BASE = {
    "4b": {
        "selected_model": "4b-instruct",
        "model": "/workspace/models/qwen3-4b-instruct-2507",
        "slug": "qwen3-4b-instruct-2507",
        "server_gpu_util": "0.80",
    },
    "8b": {
        "selected_model": "8b",
        "model": "/workspace/models/qwen3-8b",
        "slug": "qwen3-8b",
        "server_gpu_util": "0.75",
    },
}


def run(cmd: list[str]) -> str:
    print("+", " ".join(cmd), flush=True)
    out = subprocess.check_output(cmd, text=True).strip()
    print(out, flush=True)
    return out.splitlines()[-1].strip()


def write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def rm(path: Path) -> None:
    if path.exists() or path.is_symlink():
        print("rm -rf", path)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def grpo_job() -> str:
    for name in [
        "jobs/rl_v100_vllm085_grpo_server_2gpu_normal.sbatch",
        "jobs/rl_v100_vllm085_grpo_server_2gpu.sbatch",
    ]:
        if Path(name).exists():
            return name
    raise SystemExit("No 2GPU GRPO job found")


def clean_model_adapters(slug: str, runs: list[str]) -> None:
    for r in runs:
        rm(Path("models") / f"{slug}-sft-{r}")


def clean_outputs(runs: list[str]) -> None:
    for r in runs:
        rm(Path("outputs/rl") / r)
        rm(Path("outputs/train_runs") / r)


def submit_train_matrix(phase: str, selected_model: str, runs: list[str], throttle: int, dependency: str | None = None) -> str:
    matrix = Path(f"outputs/pipeline_python/{phase}/train_runs.txt")
    write_lines(matrix, runs)
    args = [
        "sbatch",
        f"--array=1-{len(runs)}%{throttle}",
        f"--export=ALL,MATRIX={matrix},SELECTED_MODEL={selected_model},CONFIG=configs/train/lora_pythoncodes_cl.yaml",
        "jobs/python_train_matrix_rocky.sbatch",
    ]
    if dependency:
        args.insert(1, f"--dependency={dependency}")
    return run(args)


def submit_eval_matrix(phase: str, selected_model: str, experiments: list[str], parts_name: str, out_name: str, dependency: str | None = None, throttle: int = 4) -> tuple[str, str]:
    matrix = Path(f"outputs/pipeline_python/{phase}/{parts_name}.txt")
    parts_root = f"outputs/pipeline_python/{phase}/{parts_name}"
    out_dir = f"outputs/pipeline_python/{phase}/{out_name}"
    write_lines(matrix, experiments)
    args = [
        "sbatch",
        f"--array=1-{len(experiments)}%{throttle}",
        f"--export=ALL,MATRIX={matrix},PARTS_ROOT={parts_root},SELECTED_MODEL={selected_model},EVAL_BENCHMARKS=mbpp,humaneval",
        "jobs/python_eval_matrix_rocky.sbatch",
    ]
    if dependency:
        args.insert(1, f"--dependency={dependency}")
    jid_eval = run(args)

    merge_args = [
        "sbatch",
        f"--dependency=afterok:{jid_eval}",
        f"--export=ALL,PARTS_ROOT={parts_root},OUT_DIR={out_dir}",
        "jobs/python_merge_eval_rocky.sbatch",
    ]
    jid_merge = run(merge_args)
    return jid_eval, jid_merge


def submit_grpo(model_key: str, run_name: str, init_adapter: str | None, dependency: str | None = None) -> str:
    b = BASE[model_key]
    slug = b["slug"]
    env = [
        "ALL",
        f"MODEL={b['model']}",
        f"GRPO_RUN_NAME={run_name}",
        f"GRPO_ADAPTER_OUTPUT_DIR=/workspace/models/{slug}-sft-{run_name}",
        f"SERVER_GPU_UTIL={b['server_gpu_util']}",
        "SERVER_MAX_MODEL_LEN=2048",
        "SERVER_DTYPE=float16",
        "SERVER_ENFORCE_EAGER=true",
    ]
    if init_adapter:
        env.append(f"RL_INIT_ADAPTER={init_adapter}")
    args = ["sbatch", f"--export={','.join(env)}", grpo_job()]
    if dependency:
        args.insert(1, f"--dependency={dependency}")
    return run(args)


def phase4_submit() -> None:
    b = BASE["4b"]
    best = BEST_4B_CL
    runs_train = ["sft_pythoncodes", best]
    runs_grpo = ["grpo_full_over_4b_base", "grpo_full_over_4b_sft_pythoncodes", f"grpo_full_over_4b_{best}"]
    print("===== CLEAN 4B TARGETS =====")
    clean_model_adapters(b["slug"], runs_train + runs_grpo)
    clean_outputs(runs_train + runs_grpo)
    rm(Path("outputs/pipeline_python/4b"))
    print("===== SUBMIT 4B TRAIN =====")
    jid_train = submit_train_matrix("4b", b["selected_model"], runs_train, throttle=2)
    print("===== SUBMIT 4B PRE-RL EVAL =====")
    jid_eval_pre, jid_merge_pre = submit_eval_matrix("4b", b["selected_model"], ["base"] + runs_train, "pre_rl_eval_jobs", "pre_rl_eval", dependency=f"afterok:{jid_train}", throttle=3)
    print("===== SUBMIT 4B GRPO =====")
    jid_rl_base = submit_grpo("4b", "grpo_full_over_4b_base", None, dependency=f"afterok:{jid_merge_pre}")
    jid_rl_sft = submit_grpo("4b", "grpo_full_over_4b_sft_pythoncodes", "/workspace/models/qwen3-4b-instruct-2507-sft-sft_pythoncodes", dependency=f"afterok:{jid_merge_pre}")
    jid_rl_cl = submit_grpo("4b", f"grpo_full_over_4b_{best}", f"/workspace/models/qwen3-4b-instruct-2507-sft-{best}", dependency=f"afterok:{jid_rl_base}:{jid_rl_sft}")
    print("===== SUBMIT 4B FINAL EVAL =====")
    jid_eval_final, jid_merge_final = submit_eval_matrix("4b", b["selected_model"], ["base"] + runs_train + runs_grpo, "final_eval_jobs", "final_eval", dependency=f"afterok:{jid_rl_cl}", throttle=4)
    print("\n===== 4B SUMMARY =====")
    print(f"train={jid_train}")
    print(f"pre_eval={jid_eval_pre} merge={jid_merge_pre}")
    print(f"rl_base={jid_rl_base} rl_sft={jid_rl_sft} rl_cl={jid_rl_cl}")
    print(f"final_eval={jid_eval_final} final_merge={jid_merge_final}")
    print("final table: outputs/pipeline_python/4b/final_eval/lora_eval_comparison.csv")


def phase4_clean() -> None:
    best = BEST_4B_CL
    slug = BASE["4b"]["slug"]
    runs = ["sft_pythoncodes", best, "grpo_full_over_4b_base", "grpo_full_over_4b_sft_pythoncodes", f"grpo_full_over_4b_{best}"]
    clean_model_adapters(slug, runs)
    clean_outputs(runs)
    print("Kept tables under outputs/pipeline_python/4b")


def phase8_train_eval_submit() -> None:
    b = BASE["8b"]
    runs_train = TRAIN_RUNS_ALL
    print("===== CLEAN 8B TRAIN/EVAL TARGETS =====")
    clean_model_adapters(b["slug"], runs_train)
    clean_outputs(runs_train)
    rm(Path("outputs/pipeline_python/8b"))
    print("===== SUBMIT 8B SFT + ALL 15 CL =====")
    jid_train = submit_train_matrix("8b", b["selected_model"], runs_train, throttle=4)
    print("===== SUBMIT 8B ALL EVAL =====")
    jid_eval, jid_merge = submit_eval_matrix("8b", b["selected_model"], ["base"] + runs_train, "all_cl_eval_jobs", "all_cl_eval", dependency=f"afterok:{jid_train}", throttle=4)
    print("\n===== 8B TRAIN/EVAL SUMMARY =====")
    print(f"train={jid_train}")
    print(f"eval={jid_eval} merge={jid_merge}")
    print("table: outputs/pipeline_python/8b/all_cl_eval/lora_eval_comparison.csv")
    print("after merge finishes: python scripts/python_full_orchestrator.py phase8_pick_best")


def phase8_pick_best() -> None:
    csv = Path("outputs/pipeline_python/8b/all_cl_eval/lora_eval_comparison.csv")
    out = Path("outputs/pipeline_python/8b/best_cl.txt")
    if not csv.exists():
        raise SystemExit(f"missing {csv}")
    run([sys.executable, "scripts/python_pick_best_cl.py", "--csv", str(csv), "--out", str(out)])


def phase8_rl_submit() -> None:
    b = BASE["8b"]
    best_path = Path("outputs/pipeline_python/8b/best_cl.txt")
    if not best_path.exists():
        raise SystemExit("missing outputs/pipeline_python/8b/best_cl.txt; run phase8_pick_best first")
    best = best_path.read_text(encoding="utf-8").strip()
    if not best.startswith("cl_"):
        raise SystemExit(f"bad best CL: {best!r}")
    print("===== DELETE UNNEEDED 8B CL ADAPTERS =====")
    keep = {"sft_pythoncodes", best}
    for run_name in TRAIN_RUNS_ALL:
        if run_name not in keep:
            rm(Path("models") / f"{b['slug']}-sft-{run_name}")
    runs_grpo = ["grpo_full_over_8b_base", "grpo_full_over_8b_sft_pythoncodes", f"grpo_full_over_8b_{best}"]
    clean_model_adapters(b["slug"], runs_grpo)
    clean_outputs(runs_grpo)
    print("===== SUBMIT 8B GRPO =====")
    jid_rl_base = submit_grpo("8b", "grpo_full_over_8b_base", None)
    jid_rl_sft = submit_grpo("8b", "grpo_full_over_8b_sft_pythoncodes", "/workspace/models/qwen3-8b-sft-sft_pythoncodes")
    jid_rl_cl = submit_grpo("8b", f"grpo_full_over_8b_{best}", f"/workspace/models/qwen3-8b-sft-{best}", dependency=f"afterok:{jid_rl_base}:{jid_rl_sft}")
    print("===== SUBMIT 8B FINAL EVAL =====")
    final_exps = ["base", "sft_pythoncodes", best, "grpo_full_over_8b_base", "grpo_full_over_8b_sft_pythoncodes", f"grpo_full_over_8b_{best}"]
    jid_eval_final, jid_merge_final = submit_eval_matrix("8b", b["selected_model"], final_exps, "final_eval_jobs", "final_eval", dependency=f"afterok:{jid_rl_cl}", throttle=4)
    print("\n===== 8B RL SUMMARY =====")
    print(f"best_cl={best}")
    print(f"rl_base={jid_rl_base} rl_sft={jid_rl_sft} rl_cl={jid_rl_cl}")
    print(f"final_eval={jid_eval_final} final_merge={jid_merge_final}")
    print("final table: outputs/pipeline_python/8b/final_eval/lora_eval_comparison.csv")


def phase8_clean() -> None:
    best_path = Path("outputs/pipeline_python/8b/best_cl.txt")
    best = best_path.read_text(encoding="utf-8").strip() if best_path.exists() else None
    slug = BASE["8b"]["slug"]
    runs = ["sft_pythoncodes"]
    if best:
        runs.append(best)
    runs.extend(["grpo_full_over_8b_base", "grpo_full_over_8b_sft_pythoncodes"])
    if best:
        runs.append(f"grpo_full_over_8b_{best}")
    clean_model_adapters(slug, runs)
    clean_outputs(runs)
    print("Kept tables under outputs/pipeline_python/8b")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["phase4_submit", "phase4_clean", "phase8_train_eval_submit", "phase8_pick_best", "phase8_rl_submit", "phase8_clean"])
    args = parser.parse_args()
    globals()[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
