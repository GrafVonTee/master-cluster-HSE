#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path.cwd()
dockerfile = ROOT / "containers" / "Dockerfile.rl-grpo-v100-vllm085"
smoke = ROOT / "scripts" / "smoke_v100_vllm085_stack.py"

if not dockerfile.exists():
    raise SystemExit(f"missing: {dockerfile}")
if not smoke.exists():
    raise SystemExit(f"missing: {smoke}")

for path in [dockerfile, smoke]:
    backup = path.with_suffix(path.suffix + ".bak_torchao_optional")
    backup.write_text(path.read_text())
    print(f"backup: {backup}")

text = dockerfile.read_text()

# Remove the old explicit torchao pin from generated constraints.
text = text.replace('    "torchao": "0.12.0",\n', '')
text = text.replace('    "torchao": "0.12.0"\n', '')

# Remove validation greps for the old torchao pin.
text = re.sub(r'\\\n\s*grep -Eq \'\^torchao==0\\\.12\\\.0\$\' /tmp/v100-constraints\.txt;?', '', text)
text = re.sub(r';\s*\\\n\s*grep -Eq \'\^torchao==0\\\.12\\\.0\$\' /tmp/v100-constraints\.txt', '', text)

# Do not install torchao explicitly.
text = text.replace('      "torchao==0.12.0" \\\n', '')
text = text.replace('      torchao==0.12.0 \\\n', '')

# If torchao is brought by the base image or by deps, remove it after installs.
uninstall_block = """
# PEFT 0.19+ treats old torchao as an error during LoRA injection.
# For V100 + torch 2.6, torchao is not required for our LoRA/GRPO/vLLM path.
RUN set -eux; \\
    python -m pip uninstall -y torchao || true; \\
    python - <<'PY'
import importlib.util
print("torchao_spec_after_uninstall", importlib.util.find_spec("torchao"))
PY

"""
if "torchao_spec_after_uninstall" not in text:
    marker = 'chmod +x /usr/local/bin/clingo'
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("could not find clingo chmod marker in Dockerfile")
    end = text.find("\n\n", idx)
    if end == -1:
        end = len(text)
    text = text[:end] + uninstall_block + text[end:]

# Build-time validation must not require torchao.
text = text.replace('["torch", "vllm", "transformers", "trl", "clingo", "torchao"]',
                    '["torch", "vllm", "transformers", "trl", "clingo"]')
text = text.replace('["torch", "vllm", "transformers", "trl", "torchao", "clingo"]',
                    '["torch", "vllm", "transformers", "trl", "clingo"]')

dockerfile.write_text(text)
print(f"patched: {dockerfile}")

s = smoke.read_text()

# Remove torchao from common required import lists. Keep checks for core packages.
for old, new in [
    ('"torchao", ', ''),
    (', "torchao"', ''),
    ("'torchao', ", ''),
    (", 'torchao'", ''),
]:
    s = s.replace(old, new)

# Convert simple direct torchao import/check forms into optional blocks.
s = s.replace(
    'check_import("torchao")',
    'optional_import("torchao") if "optional_import" in globals() else print("[optional] torchao check skipped")',
)
s = s.replace(
    "check_import('torchao')",
    'optional_import("torchao") if "optional_import" in globals() else print("[optional] torchao check skipped")',
)

# If there is no optional_import helper, add one after imports.
if "def optional_import(" not in s:
    m = re.search(r"\n(def |class |PROJECT_DIR|def main)", s)
    pos = m.start() if m else 0
    helper = """
def optional_import(name: str):
    try:
        import importlib
        module = importlib.import_module(name)
        print(f"[optional] {name} version={getattr(module, '__version__', '<no __version__>')}")
        return module
    except Exception as e:
        print(f"[optional] {name} not available: {e!r}")
        return None

"""
    s = s[:pos] + helper + s[pos:]

if "torchao is optional on V100" not in s:
    s += "\n# torchao is optional on V100; old torchao versions break PEFT LoRA dispatch.\n"

smoke.write_text(s)
print(f"patched: {smoke}")

print("\nFor the current cluster sandbox, remove torchao now:")
print("SP=containers/sandboxes/rl-grpo-v100-vllm085/venv/main/lib/python3.12/site-packages")
print("find \"$SP\" -maxdepth 1 \\( -name 'torchao' -o -name 'torchao-*.dist-info' \\) -print -exec rm -rf {} +")
print("\nThen re-run jobs/31_smoke_v100_vllm085_train_existing.sbatch.")
