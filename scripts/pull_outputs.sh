#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required. Install it first, then rerun this script." >&2
  exit 1
fi

echo "Fetching Git LFS artifacts for outputs/** ..."
git lfs pull --include="outputs/**"
echo "Done."
