from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk

src_candidates = [
    Path("datasets/clingo/synthetic_v3_100_lora_train"),
    Path("datasets/clingo/synthetic_v3_100"),
]

src = None
for p in src_candidates:
    if p.exists():
        src = p
        break

if src is None:
    raise SystemExit("No source dataset found.")

ds = load_from_disk(str(src))
if isinstance(ds, DatasetDict):
    ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

rows = []
for row in ds:
    row = dict(row)

    instruction = (
        row.get("instruction")
        or row.get("instruction_v3")
        or row.get("instruction_v4")
        or ""
    )

    facts = row.get("facts") or ""
    reference = row.get("reference") or row.get("output") or ""

    row["instruction"] = str(instruction)
    row["input"] = str(facts)
    row["output"] = str(reference)

    rows.append(row)

out = Path("datasets/clingo/synthetic_v3_100_exact5_scoreview")
if out.exists():
    import shutil
    shutil.rmtree(out)

Dataset.from_list(rows).save_to_disk(str(out))

print("source:", src)
print("rows:", len(rows))
print("saved:", out)
print("columns:", sorted(rows[0].keys()) if rows else [])
print("sample instruction:", rows[0]["instruction"][:250] if rows else "")
print("sample input:", rows[0]["input"][:250] if rows else "")
print("sample output:", rows[0]["output"][:250] if rows else "")
