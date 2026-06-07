#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


RUNTIME_BLOCK = '# Rocky Linux 9 container runtime setup.\n# The old CentOS module name can be absent or exposed as legacy; prefer whichever runtime exists.\nif command -v module >/dev/null 2>&1; then\n  module load singularity/3.9.0 2>/dev/null || \\\n  module load singularity 2>/dev/null || \\\n  module load apptainer 2>/dev/null || true\nfi\n\nif command -v singularity >/dev/null 2>&1; then\n  CONTAINER_RUNTIME="$(command -v singularity)"\nelif command -v apptainer >/dev/null 2>&1; then\n  CONTAINER_RUNTIME="$(command -v apptainer)"\nelse\n  echo "ERROR: neither singularity nor apptainer found on Rocky node" >&2\n  echo "PATH=$PATH" >&2\n  module list 2>&1 || true\n  exit 127\nfi\n\necho "container_runtime=$CONTAINER_RUNTIME"\n'


def patch_partition(text: str) -> tuple[str, bool]:
    changed = False

    new_text = re.sub(
        r"^#SBATCH\s+--partition(?:=|\s+)\S+.*$",
        "#SBATCH --partition=rocky",
        text,
        flags=re.MULTILINE,
    )
    if new_text != text:
        changed = True
        text = new_text
    elif text.startswith("#!") and "#SBATCH --partition" not in text:
        lines = text.splitlines(True)
        insert_at = 1
        last_sbatch = None
        for i, line in enumerate(lines):
            if line.startswith("#SBATCH"):
                last_sbatch = i
        if last_sbatch is not None:
            insert_at = last_sbatch + 1
        lines.insert(insert_at, "#SBATCH --partition=rocky\n")
        text = "".join(lines)
        changed = True

    return text, changed


def strip_old_singularity_module_lines(text: str) -> tuple[str, bool]:
    changed = False
    out = []
    for line in text.splitlines(True):
        stripped = line.strip()
        if re.fullmatch(r"module\s+load\s+singularity/3\.9\.0\s*(\|\|\s*true)?", stripped):
            changed = True
            continue
        if re.fullmatch(r"module\s+load\s+singularity\s*(\|\|\s*true)?", stripped):
            changed = True
            continue
        if re.fullmatch(r"module\s+load\s+apptainer\s*(\|\|\s*true)?", stripped):
            changed = True
            continue
        out.append(line)
    return "".join(out), changed


def insert_runtime_block(text: str) -> tuple[str, bool]:
    if "Rocky Linux 9 container runtime setup." in text:
        return text, False

    if "singularity" not in text and "apptainer" not in text:
        return text, False

    lines = text.splitlines(True)

    insert_at = None
    for i, line in enumerate(lines):
        if line.strip() == "module purge":
            insert_at = i + 1

    if insert_at is None:
        for i, line in enumerate(lines):
            if line.strip() == "set -euo pipefail":
                insert_at = i + 1
                break

    if insert_at is None:
        insert_at = 1 if lines and lines[0].startswith("#!") else 0

    block = "\n" + RUNTIME_BLOCK + "\n"
    lines.insert(insert_at, block)
    return "".join(lines), True


def patch_runtime_invocations(text: str) -> tuple[str, bool]:
    original = text

    text = text.replace("srun singularity exec", 'srun "$CONTAINER_RUNTIME" exec')
    text = text.replace("srun apptainer exec", 'srun "$CONTAINER_RUNTIME" exec')
    text = text.replace("srun singularity build", 'srun "$CONTAINER_RUNTIME" build')
    text = text.replace("srun apptainer build", 'srun "$CONTAINER_RUNTIME" build')

    lines = []
    for line in text.splitlines(True):
        stripped = line.lstrip()
        prefix = line[: len(line) - len(stripped)]

        if stripped.startswith("singularity exec"):
            line = prefix + stripped.replace("singularity exec", '"$CONTAINER_RUNTIME" exec', 1)
        elif stripped.startswith("apptainer exec"):
            line = prefix + stripped.replace("apptainer exec", '"$CONTAINER_RUNTIME" exec', 1)
        elif stripped.startswith("singularity build"):
            line = prefix + stripped.replace("singularity build", '"$CONTAINER_RUNTIME" build', 1)
        elif stripped.startswith("apptainer build"):
            line = prefix + stripped.replace("apptainer build", '"$CONTAINER_RUNTIME" build', 1)
        elif stripped.startswith("singularity --version"):
            line = prefix + stripped.replace("singularity --version", '"$CONTAINER_RUNTIME" --version', 1)
        elif stripped.startswith("apptainer --version"):
            line = prefix + stripped.replace("apptainer --version", '"$CONTAINER_RUNTIME" --version', 1)

        lines.append(line)

    text = "".join(lines)

    return text, text != original


def patch_apptainer_env(text: str) -> tuple[str, bool]:
    original = text

    if "SINGULARITY_CACHEDIR" in text and "APPTAINER_CACHEDIR" not in text:
        text = re.sub(
            r'^(export\s+SINGULARITY_CACHEDIR=.*)$',
            r'\1\nexport APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    if "SINGULARITY_TMPDIR" in text and "APPTAINER_TMPDIR" not in text:
        text = re.sub(
            r'^(export\s+SINGULARITY_TMPDIR=.*)$',
            r'\1\nexport APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    return text, text != original


def patch_file_text(path: Path, text: str) -> str:
    if path.suffix == ".sbatch":
        text, _ = patch_partition(text)

    text, _ = strip_old_singularity_module_lines(text)
    text, _ = insert_runtime_block(text)
    text, _ = patch_runtime_invocations(text)
    text, _ = patch_apptainer_env(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not (root / "jobs").exists():
        raise SystemExit(f"jobs/ not found under {root}")

    candidates: list[Path] = []
    candidates.extend(sorted((root / "jobs").glob("*.sbatch")))

    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        candidates.extend(sorted(scripts_dir.glob("*.sh")))

    changed_paths: list[Path] = []

    for p in candidates:
        before = p.read_text(encoding="utf-8", errors="ignore")
        after = patch_file_text(p, before)
        if after != before:
            changed_paths.append(p)
            if not args.dry_run:
                p.write_text(after, encoding="utf-8")

    if args.dry_run:
        print("Would change:")
    else:
        print("Changed files:")

    for p in changed_paths:
        print(" ", p.relative_to(root))

    print("\nChecks to run:")
    print("  grep -R \"#SBATCH --partition\" -n jobs | grep -v rocky || true")
    print("  grep -R \"module load singularity/3.9.0\" -n jobs scripts || true")
    print("  grep -R \"srun singularity exec\\|singularity exec\\|singularity build\" -n jobs scripts || true")
    print("  grep -R \"CONTAINER_RUNTIME\\|APPTAINER_TMPDIR\\|APPTAINER_CACHEDIR\" -n jobs scripts | head -80")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
