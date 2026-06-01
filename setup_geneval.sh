#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$ROOT/third_party"
GENEVAL="$THIRD_PARTY/geneval"

mkdir -p "$THIRD_PARTY"
if [[ ! -d "$GENEVAL" ]]; then
  git clone https://github.com/djghosh13/geneval.git "$GENEVAL"
else
  echo "GenEval repo already exists: $GENEVAL"
fi

echo ""
echo "Next steps:"
echo "  python geneval_generate.py --outdir outputs/geneval --resolution 1024"
echo "  bash $GENEVAL/evaluation/download_models.sh weights/geneval_detector"
echo "  python geneval_evaluate.py outputs/geneval --outfile outputs/geneval/results.jsonl"
