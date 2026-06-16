#!/usr/bin/env bash
set -euo pipefail

# Create a small Python environment for Hugging Face model downloads.
# This avoids using the very old system Python on some login nodes.

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="${HF_DOWNLOAD_VENV:-${PROJECT_DIR}/.venv-hf-download}"

find_python() {
  local candidates=(python3.12 python3.11 python3.10 python3.9 python3.8 python3)
  local py
  for py in "${candidates[@]}"; do
    if command -v "$py" >/dev/null 2>&1; then
      if "$py" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 8) else 1)
PY
      then
        echo "$py"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if ! PYTHON_BIN="$(find_python)"; then
    cat >&2 <<'EOF'
ERROR: no Python >= 3.8 found.
Load a newer Python module first, for example:

  module avail python
  module load <python/3.10-or-newer-module>

Then rerun:

  bash scripts/bootstrap_hf_download_env.sh
EOF
    exit 2
  fi
fi

echo "Using PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
# Keep a broad but not bleeding-edge range; pip will respect python_requires.
python -m pip install 'huggingface_hub>=0.23,<1' 'hf_transfer>=0.1.6' 'requests' 'tqdm' 'pyyaml' 'packaging'
python - <<'PY'
import sys
import huggingface_hub
print('python:', sys.version)
print('huggingface_hub:', huggingface_hub.__version__)
PY
cat <<EOF

Ready. Now run:

  source "$VENV_DIR/bin/activate"
  export PROJECT_DIR="${PROJECT_DIR}"
  export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
  python scripts/download_qwen14b.py --models 0_6b,1_7b
EOF
