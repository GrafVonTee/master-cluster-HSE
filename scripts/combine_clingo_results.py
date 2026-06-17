import csv
from pathlib import Path

RUNS = [
    ("clingo_v3_100", "0_6b", "base", Path("outputs/clingo_v3_100_small/eval_0_6b_base/summary.csv")),
    ("clingo_v3_100", "0_6b", "sft",  Path("outputs/clingo_v3_100_small/eval_0_6b_sft/summary.csv")),
    ("clingo_v3_100", "0_6b", "cl",   Path("outputs/clingo_v3_100_small/eval_0_6b_cl/summary.csv")),

    ("clingo_v3_100", "1_7b", "base", Path("outputs/clingo_v3_100_small/eval_1_7b_base/summary.csv")),
    ("clingo_v3_100", "1_7b", "sft",  Path("outputs/clingo_v3_100_small/eval_1_7b_sft/summary.csv")),
    ("clingo_v3_100", "1_7b", "cl",   Path("outputs/clingo_v3_100_small/eval_1_7b_cl/summary.csv")),

    ("clingo_v4", "0_6b", "base", Path("outputs/clingo_v4/eval_0_6b_base/summary.csv")),
    ("clingo_v4", "0_6b", "sft",  Path("outputs/clingo_v4/eval_0_6b_sft/summary.csv")),
    ("clingo_v4", "0_6b", "cl",   Path("outputs/clingo_v4/eval_0_6b_cl/summary.csv")),

    ("clingo_v4", "1_7b", "base", Path("outputs/clingo_v4/eval_1_7b_base/summary.csv")),
    ("clingo_v4", "1_7b", "sft",  Path("outputs/clingo_v4/eval_1_7b_sft/summary.csv")),
    ("clingo_v4", "1_7b", "cl",   Path("outputs/clingo_v4/eval_1_7b_cl/summary.csv")),
]

OUT_DIR = Path("outputs/clingo_combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_overall(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("group") == "overall" and row.get("name") == "overall":
                return row
    raise RuntimeError("overall row not found: " + str(path))

def as_float(value):
    if value is None or value == "":
        return None
    return float(value)

rows = []
for dataset, model, variant, path in RUNS:
    if not path.exists():
        print("MISSING:", path)
        continue

    src = read_overall(path)
    row = {
        "dataset": dataset,
        "model": model,
        "variant": variant,
        "num_tasks": src.get("num_tasks", ""),
        "mean_reward": src.get("mean_reward", ""),
        "full_pass_rate": src.get("full_pass_rate", ""),
        "partial_rate": src.get("partial_rate", ""),
        "error_rate": src.get("error_rate", ""),
        "source": str(path.parent),
    }
    rows.append(row)

base = {}
sft = {}
for row in rows:
    key = (row["dataset"], row["model"])
    if row["variant"] == "base":
        base[key] = row
    if row["variant"] == "sft":
        sft[key] = row

for row in rows:
    key = (row["dataset"], row["model"])
    mean = as_float(row["mean_reward"])
    full = as_float(row["full_pass_rate"])

    row["delta_mean_vs_base"] = ""
    row["delta_full_vs_base"] = ""
    row["delta_mean_vs_sft"] = ""
    row["delta_full_vs_sft"] = ""

    if key in base and mean is not None and full is not None:
        bmean = as_float(base[key]["mean_reward"])
        bfull = as_float(base[key]["full_pass_rate"])
        row["delta_mean_vs_base"] = mean - bmean
        row["delta_full_vs_base"] = full - bfull

    if key in sft and row["variant"] != "base" and mean is not None and full is not None:
        smean = as_float(sft[key]["mean_reward"])
        sfull = as_float(sft[key]["full_pass_rate"])
        row["delta_mean_vs_sft"] = mean - smean
        row["delta_full_vs_sft"] = full - sfull

fields = [
    "dataset",
    "model",
    "variant",
    "num_tasks",
    "mean_reward",
    "full_pass_rate",
    "partial_rate",
    "error_rate",
    "delta_mean_vs_base",
    "delta_full_vs_base",
    "delta_mean_vs_sft",
    "delta_full_vs_sft",
    "source",
]

csv_path = OUT_DIR / "clingo_overall_with_deltas.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

md_path = OUT_DIR / "clingo_overall_table.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("| Dataset | Model | Variant | Mean reward | Full pass | Partial | Error | Delta full vs base | Delta full vs SFT |\n")
    f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|\n")

    for row in rows:
        def fmt(value):
            if value == "" or value is None:
                return ""
            return "{:.4f}".format(float(value))

        f.write("| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            row["dataset"],
            row["model"],
            row["variant"],
            fmt(row["mean_reward"]),
            fmt(row["full_pass_rate"]),
            fmt(row["partial_rate"]),
            fmt(row["error_rate"]),
            fmt(row["delta_full_vs_base"]),
            fmt(row["delta_full_vs_sft"]),
        ))

print("wrote", csv_path)
print("wrote", md_path)
print()
print(md_path.read_text(encoding="utf-8"))
