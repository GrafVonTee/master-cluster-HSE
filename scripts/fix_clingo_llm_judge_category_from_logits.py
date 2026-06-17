from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import ast
import math
import re
import shutil

path = Path("datasets/clingo/synthetic_v3_100_exact5_scored")
backup = Path("datasets/clingo/synthetic_v3_100_exact5_scored_before_llm_judge_category_fix")

ds = load_from_disk(str(path))

if "llm_judge_score" not in ds.column_names:
    raise SystemExit("No llm_judge_score column.")
if "llm_judge_raw" not in ds.column_names:
    raise SystemExit("No llm_judge_raw column.")

def parse_logits(raw):
    raw = str(raw)
    m = re.search(r"logits=(\{.*\})", raw)
    if not m:
        return None
    try:
        obj = ast.literal_eval(m.group(1))
        return {int(k): float(v) for k, v in obj.items()}
    except Exception:
        return None

def softmax_expected(logits, temperature=10.0):
    # temperature softens huge logits; we need a continuous tie-breaker, not a hard argmax.
    keys = sorted(logits)
    vals = [logits[k] / temperature for k in keys]
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    z = sum(exps)
    return sum(k * e for k, e in zip(keys, exps)) / z

rank_scores = []
debug = []

for i in range(len(ds)):
    row = ds[i]
    forced = float(row["llm_judge_score"])
    logits = parse_logits(row["llm_judge_raw"])

    if logits is None:
        tie = 0.0
    else:
        tie = softmax_expected(logits, temperature=10.0)

    # Primary key: forced judge score.
    # Tie-breaker: continuous logit-derived expected score.
    # The 0.001 factor prevents tie-breaker from changing score-2 < score-3 ordering.
    rank = forced + 0.001 * tie

    rank_scores.append(rank)
    debug.append((forced, tie, rank))

order = sorted(range(len(rank_scores)), key=lambda i: rank_scores[i])
n = len(order)

new_cat = [None] * n
for rank_pos, idx in enumerate(order):
    if rank_pos < n / 3:
        new_cat[idx] = "easy"
    elif rank_pos < 2 * n / 3:
        new_cat[idx] = "medium"
    else:
        new_cat[idx] = "hard"

print("old llm_judge_category:", Counter(ds["llm_judge_category"]))
print("llm_judge_score:", Counter(float(x) for x in ds["llm_judge_score"]))
print("new llm_judge_category:", Counter(new_cat))
print()
print("rank score examples:")
for x in sorted(debug)[:5]:
    print("low ", x)
for x in sorted(debug)[-5:]:
    print("high", x)

# Replace category column.
if "llm_judge_category" in ds.column_names:
    ds = ds.remove_columns(["llm_judge_category"])

ds = ds.add_column("llm_judge_category", new_cat)

# Also keep the continuous score for audit/debug.
if "llm_judge_rank_score" in ds.column_names:
    ds = ds.remove_columns(["llm_judge_rank_score"])
ds = ds.add_column("llm_judge_rank_score", rank_scores)

if backup.exists():
    shutil.rmtree(backup)
shutil.copytree(path, backup)

tmp = Path(str(path) + "_tmp")
if tmp.exists():
    shutil.rmtree(tmp)

ds.save_to_disk(str(tmp))
shutil.rmtree(path)
tmp.rename(path)

print()
print("backup:", backup)
print("saved fixed dataset:", path)
