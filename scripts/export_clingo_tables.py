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

def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def to_float(x):
    if x is None or x == "":
        return None
    return float(x)

def fmt4(x):
    if x is None or x == "":
        return ""
    return "{:.4f}".format(float(x))

all_rows = []
missing = []

for dataset, model, variant, path in RUNS:
    if not path.exists():
        missing.append(str(path))
        continue

    for r in read_rows(path):
        row = {
            "dataset": dataset,
            "model": model,
            "variant": variant,
            "group": r.get("group", ""),
            "name": r.get("name", ""),
            "num_tasks": r.get("num_tasks", ""),
            "mean_reward": r.get("mean_reward", ""),
            "full_pass_rate": r.get("full_pass_rate", ""),
            "partial_rate": r.get("partial_rate", ""),
            "error_rate": r.get("error_rate", ""),
            "source": str(path.parent),
        }
        all_rows.append(row)

if missing:
    print("WARNING: missing summary files:")
    for x in missing:
        print("  " + x)

def add_deltas(rows, key_fields):
    base = {}
    sft = {}

    for r in rows:
        key = tuple(r[k] for k in key_fields)
        if r["variant"] == "base":
            base[key] = r
        elif r["variant"] == "sft":
            sft[key] = r

    out = []
    for r in rows:
        r = dict(r)
        key = tuple(r[k] for k in key_fields)

        mean = to_float(r["mean_reward"])
        full = to_float(r["full_pass_rate"])
        partial = to_float(r["partial_rate"])
        error = to_float(r["error_rate"])

        r["delta_mean_vs_base"] = ""
        r["delta_full_vs_base"] = ""
        r["delta_partial_vs_base"] = ""
        r["delta_error_vs_base"] = ""

        r["delta_mean_vs_sft"] = ""
        r["delta_full_vs_sft"] = ""
        r["delta_partial_vs_sft"] = ""
        r["delta_error_vs_sft"] = ""

        if key in base:
            b = base[key]
            r["delta_mean_vs_base"] = mean - to_float(b["mean_reward"])
            r["delta_full_vs_base"] = full - to_float(b["full_pass_rate"])
            r["delta_partial_vs_base"] = partial - to_float(b["partial_rate"])
            r["delta_error_vs_base"] = error - to_float(b["error_rate"])

        if key in sft and r["variant"] != "base":
            s = sft[key]
            r["delta_mean_vs_sft"] = mean - to_float(s["mean_reward"])
            r["delta_full_vs_sft"] = full - to_float(s["full_pass_rate"])
            r["delta_partial_vs_sft"] = partial - to_float(s["partial_rate"])
            r["delta_error_vs_sft"] = error - to_float(s["error_rate"])

        out.append(r)

    return out

overall_rows = [
    r for r in all_rows
    if r["group"] == "overall" and r["name"] == "overall"
]

topic_rows = [
    r for r in all_rows
    if r["group"] == "topic"
]

variant_order = {"base": 0, "sft": 1, "cl": 2}

overall_rows = sorted(
    overall_rows,
    key=lambda r: (r["dataset"], r["model"], variant_order.get(r["variant"], 99)),
)

topic_rows = sorted(
    topic_rows,
    key=lambda r: (r["dataset"], r["model"], r["name"], variant_order.get(r["variant"], 99)),
)

overall_rows = add_deltas(overall_rows, key_fields=["dataset", "model"])
topic_rows = add_deltas(topic_rows, key_fields=["dataset", "model", "name"])

overall_fields = [
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
    "delta_partial_vs_base",
    "delta_error_vs_base",
    "delta_mean_vs_sft",
    "delta_full_vs_sft",
    "delta_partial_vs_sft",
    "delta_error_vs_sft",
    "source",
]

topic_fields = [
    "dataset",
    "model",
    "topic",
    "variant",
    "num_tasks",
    "mean_reward",
    "full_pass_rate",
    "partial_rate",
    "error_rate",
    "delta_mean_vs_base",
    "delta_full_vs_base",
    "delta_partial_vs_base",
    "delta_error_vs_base",
    "delta_mean_vs_sft",
    "delta_full_vs_sft",
    "delta_partial_vs_sft",
    "delta_error_vs_sft",
    "source",
]

# Rename name -> topic in topic table.
topic_rows_out = []
for r in topic_rows:
    rr = dict(r)
    rr["topic"] = rr.pop("name")
    topic_rows_out.append(rr)

overall_csv = OUT_DIR / "clingo_overall_metrics.csv"
with overall_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=overall_fields)
    w.writeheader()
    w.writerows(overall_rows)

topic_csv = OUT_DIR / "clingo_topic_metrics.csv"
with topic_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=topic_fields)
    w.writeheader()
    w.writerows(topic_rows_out)

# Markdown overall.
overall_md = OUT_DIR / "clingo_overall_metrics.md"
with overall_md.open("w", encoding="utf-8") as f:
    f.write("| Dataset | Model | Variant | Mean reward | Full pass | Partial | Error | Δ full vs base | Δ full vs SFT |\n")
    f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for r in overall_rows:
        f.write("| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            r["dataset"],
            r["model"],
            r["variant"],
            fmt4(r["mean_reward"]),
            fmt4(r["full_pass_rate"]),
            fmt4(r["partial_rate"]),
            fmt4(r["error_rate"]),
            fmt4(r["delta_full_vs_base"]),
            fmt4(r["delta_full_vs_sft"]),
        ))

# Markdown topic, compact but still readable.
topic_md = OUT_DIR / "clingo_topic_metrics.md"
with topic_md.open("w", encoding="utf-8") as f:
    f.write("| Dataset | Model | Topic | Variant | Mean reward | Full pass | Partial | Error | Δ full vs base | Δ full vs SFT |\n")
    f.write("|---|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in topic_rows_out:
        f.write("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            r["dataset"],
            r["model"],
            r["topic"],
            r["variant"],
            fmt4(r["mean_reward"]),
            fmt4(r["full_pass_rate"]),
            fmt4(r["partial_rate"]),
            fmt4(r["error_rate"]),
            fmt4(r["delta_full_vs_base"]),
            fmt4(r["delta_full_vs_sft"]),
        ))

print("wrote", overall_csv)
print("wrote", topic_csv)
print("wrote", overall_md)
print("wrote", topic_md)
print()
print("Overall:")
print(overall_md.read_text(encoding="utf-8"))
