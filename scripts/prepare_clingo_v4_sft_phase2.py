from pathlib import Path
import yaml

MODELS = ["0_6b", "1_7b", "4b", "8b"]

# If the repo uses qwen3-4b-instruct-2507 instead of qwen3-4b,
# SELECTED_MODEL should be 4b-instruct. We handle that below.
def detect_selected_model(alias):
    if alias == "4b":
        if Path("models/qwen3-4b").exists():
            return "4b"
        if Path("models/qwen3-4b-instruct-2507").exists():
            return "4b-instruct"
        return "4b"
    return alias

def detect_template(alias):
    candidates = [
        Path(f"configs/train/lora_clingo_synthetic_v3_100_{alias}.yaml"),
        Path(f"configs/train/lora_clingo_synthetic_{alias}.yaml"),
        Path("configs/train/lora_clingo_synthetic.yaml"),
    ]
    if alias == "4b":
        candidates.insert(0, Path("configs/train/lora_clingo_synthetic_v3_100_4b.yaml"))
        candidates.insert(0, Path("configs/train/lora_clingo_synthetic_v3_100_4b_instruct.yaml"))
    if alias == "8b":
        candidates.insert(0, Path("configs/train/lora_clingo_synthetic_v3_100_8b.yaml"))

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError("No template config found for " + alias)

def model_path_for(selected_model):
    if selected_model == "0_6b":
        return "/workspace/models/qwen3-0.6b"
    if selected_model == "1_7b":
        return "/workspace/models/qwen3-1.7b"
    if selected_model == "4b":
        return "/workspace/models/qwen3-4b"
    if selected_model == "4b-instruct":
        return "/workspace/models/qwen3-4b-instruct-2507"
    if selected_model == "8b":
        return "/workspace/models/qwen3-8b"
    raise ValueError(selected_model)

for alias in MODELS:
    selected = detect_selected_model(alias)
    template = detect_template(alias)

    with template.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("dataset", {})
    cfg["dataset"]["name"] = "clingo_synthetic_v4"
    cfg["dataset"]["disk_path"] = "datasets/clingo/synthetic_v4_lora_train"
    cfg["dataset"]["parquet_path"] = None
    cfg["dataset"]["val_size"] = 24
    cfg["dataset"]["seed"] = 42

    cfg.setdefault("model", {})
    cfg["model"]["base_model"] = model_path_for(selected)
    cfg["model"]["output_model_prefix"] = model_path_for(selected) + "-sft"

    cfg.setdefault("training", {})
    cfg["training"]["report_to"] = "none"
    cfg["training"]["max_steps"] = int(cfg["training"].get("max_steps", 300) or 300)
    cfg["training"]["stage_max_steps"] = int(cfg["training"].get("stage_max_steps", 100) or 100)
    cfg["training"]["eval_steps"] = min(int(cfg["training"].get("eval_steps", 20) or 20), cfg["training"]["max_steps"])
    cfg["training"]["save_steps"] = min(int(cfg["training"].get("save_steps", 60) or 60), cfg["training"]["max_steps"])

    # Safer memory settings for larger models on V100.
    if alias == "4b":
        cfg["training"]["train_batch_size"] = 2
        cfg["training"]["gradient_steps"] = 8
    elif alias == "8b":
        cfg["training"]["train_batch_size"] = 1
        cfg["training"]["gradient_steps"] = 16
        cfg["training"]["dataloader_num_workers"] = 0
        cfg["training"]["dataloader_pin_memory"] = False
    else:
        cfg["training"]["train_batch_size"] = int(cfg["training"].get("train_batch_size", 4) or 4)
        cfg["training"]["gradient_steps"] = int(cfg["training"].get("gradient_steps", 4) or 4)

    cfg["runs"] = [
        {
            "name": "sft_v4_clingo",
            "category_col": None,
            "schedule_type": "plain",
        }
    ]

    out_cfg = Path(f"configs/train/lora_clingo_synthetic_v4_{alias}.yaml")
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    out_dir = Path(f"outputs/clingo_v4/{alias}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_file = out_dir / "train_runs_sft_v4_only.txt"
    runs_file.write_text("sft_v4_clingo\n", encoding="utf-8")

    print("model", alias, "selected_model", selected)
    print("  template:", template)
    print("  config:  ", out_cfg)
    print("  runs:    ", runs_file)
