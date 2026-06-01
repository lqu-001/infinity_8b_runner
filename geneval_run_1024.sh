#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f third_party/geneval/prompts/evaluation_metadata.jsonl ]]; then
  echo "GenEval prompts not found; running setup_geneval.sh ..."
  bash "$ROOT/setup_geneval.sh"
fi

PYTHON="${PYTHON:-python}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

exec "$PYTHON" "$ROOT/geneval_run_1024.py" "$@"
