from pathlib import Path
import yaml

MODELS = {
    "0_6b": {
        "base_model": "/workspace/models/qwen3-0.6b",
        "selected_model": "0_6b",
        "template": "configs/train/lora_clingo_synthetic_v3_100_0_6b.yaml",
        "batch": 4,
        "grad": 4,
    },
    "1_7b": {
        "base_model": "/workspace/models/qwen3-1.7b",
        "selected_model": "1_7b",
        "template": "configs/train/lora_clingo_synthetic_v3_100_1_7b.yaml",
        "batch": 4,
        "grad": 4,
    },
}

# Same five CL criteria as in PythonCodes.
CRITERIA = [
    ("length", "length_category"),
    ("perplexity", "ppl_category"),
    ("lexical", "lexical_cluster_category"),
    ("semantic", "semantic_cluster_category"),
    ("llm_judge", "llm_judge_category"),
]

# Same three schedules.
SCHEDULES = ["staged", "cumulative", "distribution"]

def load_template(path):
    p = Path(path)
    if not p.exists():
        fallback = Path("configs/train/lora_clingo_synthetic.yaml")
        print("template missing:", p, "fallback:", fallback)
        p = fallback
    return yaml.safe_load(p.read_text(encoding="utf-8"))

runs = []
run_names = []

# Same ordering style as PythonCodes config:
# all staged, then all cumulative, then all distribution.
for schedule in SCHEDULES:
    for criterion, col in CRITERIA:
        name = f"cl_{criterion}_{schedule}"
        runs.append({
            "name": name,
            "category_col": col,
            "schedule_type": schedule,
        })
        run_names.append(name)

for alias, spec in MODELS.items():
    cfg = load_template(spec["template"])

    cfg.setdefault("dataset", {})
    cfg["dataset"]["name"] = "clingo_synthetic_v3_100_exact5_scored"
    cfg["dataset"]["disk_path"] = "datasets/clingo/synthetic_v3_100_exact5_scored"
    cfg["dataset"]["parquet_path"] = None
    cfg["dataset"]["val_size"] = 24
    cfg["dataset"]["seed"] = 42

    cfg.setdefault("model", {})
    cfg["model"]["base_model"] = spec["base_model"]
    cfg["model"]["output_model_prefix"] = spec["base_model"] + "-sft"
    cfg["model"]["load_in_4bit"] = False

    cfg.setdefault("curriculum", {})
    cfg["curriculum"]["stages"] = ["easy", "medium", "hard"]
    cfg["curriculum"]["distribution_stage_size"] = 160
    cfg["curriculum"]["distribution_stages"] = [
        {"name": "d1_80_15_5", "weights": {"easy": 0.80, "medium": 0.15, "hard": 0.05}},
        {"name": "d2_40_40_20", "weights": {"easy": 0.40, "medium": 0.40, "hard": 0.20}},
        {"name": "d3_20_20_60", "weights": {"easy": 0.20, "medium": 0.20, "hard": 0.60}},
    ]

    cfg.setdefault("training", {})
    cfg["training"]["report_to"] = "none"
    cfg["training"]["train_batch_size"] = spec["batch"]
    cfg["training"]["gradient_steps"] = spec["grad"]
    cfg["training"]["stage_max_steps"] = int(cfg["training"].get("stage_max_steps", 100) or 100)
    cfg["training"]["max_steps"] = int(cfg["training"].get("max_steps", 300) or 300)
    cfg["training"]["dataloader_num_workers"] = 0
    cfg["training"]["dataloader_pin_memory"] = False

    cfg["runs"] = runs

    cfg_path = Path(f"configs/train/lora_clingo_exact5_cl15_v3_100_{alias}.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    out_dir = Path(f"outputs/clingo_exact5_cl15/{alias}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_runs.txt").write_text("\n".join(run_names) + "\n", encoding="utf-8")

    print("wrote", cfg_path)
    print("wrote", out_dir / "train_runs.txt")
    print("runs:")
    for x in run_names:
        print(" ", x)
