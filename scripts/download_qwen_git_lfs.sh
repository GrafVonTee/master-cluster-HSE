#!/usr/bin/env bash
set -euo pipefail

# Fallback downloader that does not require huggingface_hub.
# Requires git and git-lfs on the login node.

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MODELS_DIR="${MODELS_DIR:-${PROJECT_DIR}/models}"
mkdir -p "$MODELS_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not available" >&2
  exit 2
fi
if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  echo "ERROR: git-lfs is not available. Use scripts/bootstrap_hf_download_env.sh instead." >&2
  exit 2
fi

git lfs install --skip-repo >/dev/null 2>&1 || true

resolve_model() {
  case "$1" in
    0_6b|0.6b) echo "Qwen/Qwen3-0.6B qwen3-0.6b" ;;
    1_7b|1.7b) echo "Qwen/Qwen3-1.7B qwen3-1.7b" ;;
    4b) echo "Qwen/Qwen3-4B qwen3-4b" ;;
    4b-instruct) echo "Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-instruct-2507" ;;
    4b-thinking) echo "Qwen/Qwen3-4B-Thinking-2507 qwen3-4b-thinking-2507" ;;
    8b) echo "Qwen/Qwen3-8B qwen3-8b" ;;
    14b) echo "Qwen/Qwen3-14B qwen3-14b" ;;
    */*)
      local folder
      folder="$(basename "$1" | tr '[:upper:]' '[:lower:]')"
      echo "$1 $folder"
      ;;
    *)
      echo "ERROR: unknown model alias '$1'" >&2
      return 2
      ;;
  esac
}

if [[ "$#" -eq 0 ]]; then
  set -- 14b
fi

items=()
for raw in "$@"; do
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] && items+=("$part")
  done
done

for item in "${items[@]}"; do
  read -r repo folder < <(resolve_model "$item")
  dest="${MODELS_DIR}/${folder}"
  url="https://huggingface.co/${repo}"
  echo "===== DOWNLOAD ${item} ====="
  echo "repo=${repo}"
  echo "dest=${dest}"
  if [[ -d "${dest}/.git" ]]; then
    git -C "$dest" pull --ff-only
    git -C "$dest" lfs pull
  else
    git clone "$url" "$dest"
    git -C "$dest" lfs pull
  fi
  test -f "${dest}/config.json"
  echo "OK: ${dest}"
done
