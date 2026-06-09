#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import json


def preview_text(p: Path, n: int = 1200) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")[:n]
    except Exception as e:
        return f"<read error: {e!r}>"


def inspect_csv(p: Path):
    try:
        with p.open(newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= 2:
                    break
        print("CSV columns:", reader.fieldnames)
        for i, row in enumerate(rows):
            print("row", i, {k: str(v)[:300] for k, v in row.items()})
    except Exception as e:
        print("CSV error:", repr(e))


def inspect_jsonl(p: Path):
    try:
        with p.open(encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                print("JSONL keys:", list(obj.keys()))
                print("row", i, {k: str(v)[:300] for k, v in obj.items()})
                if i >= 2:
                    break
    except Exception as e:
        print("JSONL error:", repr(e))


def inspect_json(p: Path):
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        print("JSON type:", type(obj).__name__)
        if isinstance(obj, dict):
            print("keys:", list(obj.keys())[:30])
            for k, v in list(obj.items())[:3]:
                print("item", k, str(v)[:500])
        elif isinstance(obj, list):
            print("len:", len(obj))
            for x in obj[:3]:
                print("item:", str(x)[:500])
    except Exception as e:
        print("JSON error:", repr(e))


def main():
    root = Path("external/LLASP")
    out = Path("reports/clingo/llasp_inspection.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in [".csv", ".json", ".jsonl", ".txt", ".lp", ".py", ".md"]
    )

    with out.open("w", encoding="utf-8") as report:
        import sys
        old = sys.stdout
        sys.stdout = report
        try:
            print("FILES", len(files))
            for p in files:
                print("\n" + "=" * 100)
                print(p)
                print("size", p.stat().st_size)

                if p.suffix.lower() == ".csv":
                    inspect_csv(p)
                elif p.suffix.lower() == ".jsonl":
                    inspect_jsonl(p)
                elif p.suffix.lower() == ".json":
                    inspect_json(p)
                else:
                    print(preview_text(p, 1800))
        finally:
            sys.stdout = old

    print("wrote", out)


if __name__ == "__main__":
    main()
