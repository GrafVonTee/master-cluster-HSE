import csv
from pathlib import Path

ROOTS = {
    "v3": Path("outputs/clingo_phase2/eval_v3"),
    "v4": Path("outputs/clingo_phase2/eval_v4"),
}

RUNS = [
    ("0_6b", "base", "eval_0_6b_base"),
    ("0_6b", "sft",  "eval_0_6b_sft"),
    ("0_6b", "cl",   "eval_0_6b_cl"),
    ("1_7b", "base", "eval_1_7b_base"),
    ("1_7b", "sft",  "eval_1_7b_sft"),
    ("1_7b", "cl",   "eval_1_7b_cl"),
]

OUT = Path("outputs/clingo_phase2/tables")
OUT.mkdir(parents=True, exist_ok=True)

VARIANT_ORDER = {"base": 0, "sft": 1, "cl": 2}

def read_summary(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def as_float(x):
    if x is None or x == "":
        return None
    return float(x)

def fmt(x):
    if x is None or x == "":
        return ""
    return "{:.4f}".format(float(x))

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
        rr = dict(r)
        key = tuple(rr[k] for k in key_fields)

        rr["delta_mean_vs_base"] = ""
        rr["delta_full_vs_base"] = ""
        rr["delta_partial_vs_base"] = ""
        rr["delta_error_vs_base"] = ""

        rr["delta_mean_vs_sft"] = ""
        rr["delta_full_vs_sft"] = ""
        rr["delta_partial_vs_sft"] = ""
        rr["delta_error_vs_sft"] = ""

        mean = as_float(rr["mean_reward"])
        full = as_float(rr["full_pass_rate"])
        partial = as_float(rr["partial_rate"])
        error = as_float(rr["error_rate"])

        if key in base:
            b = base[key]
            rr["delta_mean_vs_base"] = mean - as_float(b["mean_reward"])
            rr["delta_full_vs_base"] = full - as_float(b["full_pass_rate"])
            rr["delta_partial_vs_base"] = partial - as_float(b["partial_rate"])
            rr["delta_error_vs_base"] = error - as_float(b["error_rate"])

        if key in sft and rr["variant"] != "base":
            s = sft[key]
            rr["delta_mean_vs_sft"] = mean - as_float(s["mean_reward"])
            rr["delta_full_vs_sft"] = full - as_float(s["full_pass_rate"])
            rr["delta_partial_vs_sft"] = partial - as_float(s["partial_rate"])
            rr["delta_error_vs_sft"] = error - as_float(s["error_rate"])

        out.append(rr)

    return out

def collect(dataset_name, root):
    overall = []
    topic = []
    missing = []

    for model, variant, run_dir in RUNS:
        path = root / run_dir / "summary.csv"
        if not path.exists():
            missing.append(str(path))
            continue

        for r in read_summary(path):
            base = {
                "dataset": dataset_name,
                "model": model,
                "variant": variant,
                "num_tasks": r.get("num_tasks", ""),
                "mean_reward": r.get("mean_reward", ""),
                "full_pass_rate": r.get("full_pass_rate", ""),
                "partial_rate": r.get("partial_rate", ""),
                "error_rate": r.get("error_rate", ""),
                "source": str(path.parent),
            }

            if r.get("group") == "overall" and r.get("name") == "overall":
                overall.append(base)

            if r.get("group") == "topic":
                row = dict(base)
                row["topic"] = r.get("name", "")
                topic.append(row)

    if missing:
        print("WARNING missing files for", dataset_name)
        for x in missing:
            print("  " + x)

    overall.sort(key=lambda r: (r["model"], VARIANT_ORDER.get(r["variant"], 99)))
    topic.sort(key=lambda r: (r["model"], r["topic"], VARIANT_ORDER.get(r["variant"], 99)))

    overall = add_deltas(overall, key_fields=["dataset", "model"])
    topic = add_deltas(topic, key_fields=["dataset", "model", "topic"])

    return overall, topic

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

md_overall_header = "| Dataset | Model | Variant | Mean reward | Full pass | Partial | Error | Δ full vs base | Δ full vs SFT |\n"
md_overall_sep = "|---|---:|---|---:|---:|---:|---:|---:|---:|\n"

md_topic_header = "| Dataset | Model | Topic | Variant | Mean reward | Full pass | Partial | Error | Δ full vs base | Δ full vs SFT |\n"
md_topic_sep = "|---|---:|---|---|---:|---:|---:|---:|---:|---:|\n"

for dataset_name, root in ROOTS.items():
    overall, topic = collect(dataset_name, root)

    overall_csv = OUT / ("clingo_phase2_%s_overall.csv" % dataset_name)
    topic_csv = OUT / ("clingo_phase2_%s_topic.csv" % dataset_name)

    with overall_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=overall_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(overall)

    with topic_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=topic_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(topic)

    overall_md = OUT / ("clingo_phase2_%s_overall.md" % dataset_name)
    with overall_md.open("w", encoding="utf-8") as f:
        f.write(md_overall_header)
        f.write(md_overall_sep)
        for r in overall:
            f.write("| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                r["dataset"],
                r["model"],
                r["variant"],
                fmt(r["mean_reward"]),
                fmt(r["full_pass_rate"]),
                fmt(r["partial_rate"]),
                fmt(r["error_rate"]),
                fmt(r["delta_full_vs_base"]),
                fmt(r["delta_full_vs_sft"]),
            ))

    topic_md = OUT / ("clingo_phase2_%s_topic.md" % dataset_name)
    with topic_md.open("w", encoding="utf-8") as f:
        f.write(md_topic_header)
        f.write(md_topic_sep)
        for r in topic:
            f.write("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                r["dataset"],
                r["model"],
                r["topic"],
                r["variant"],
                fmt(r["mean_reward"]),
                fmt(r["full_pass_rate"]),
                fmt(r["partial_rate"]),
                fmt(r["error_rate"]),
                fmt(r["delta_full_vs_base"]),
                fmt(r["delta_full_vs_sft"]),
            ))

    print("wrote", overall_csv)
    print("wrote", topic_csv)
    print("wrote", overall_md)
    print("wrote", topic_md)

print()
print("DONE. Tables are in", OUT)
