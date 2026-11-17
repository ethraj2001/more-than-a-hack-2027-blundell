#!/usr/bin/env bash
# The submission file of one trained model, made and checked:   ./model.sh NAME
#
#   models/NAME.pkl  ->  evaluation/results/NAME.csv
#
# Predicts twice and compares the bytes (predict --check), validates the file as the official scorer
# does, prints how the second half's pre-kick-off frames are labelled (all D is right: FAQ.md), and
# compares the file with evaluation/results/moe.csv. A file already there under that name is first
# copied to evaluation/backups/. Counts only are printed, so the output can be shared.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

if [ $# -ne 1 ]; then
  echo "usage: ./model.sh NAME   (reads models/NAME.pkl, writes evaluation/results/NAME.csv)" >&2
  exit 2
fi
NAME=${1%.pkl}
NAME=${NAME#models/}
MODEL=models/$NAME.pkl
OUT=evaluation/results/$NAME.csv
REFERENCE=evaluation/results/moe.csv

if [ ! -f "$MODEL" ]; then
  echo "no model at $MODEL; these are there:" >&2
  ls models/*.pkl >&2
  exit 1
fi
if [ -f "$OUT" ]; then
  mkdir -p evaluation/backups
  BACKUP=evaluation/backups/${NAME}_$(date +%Y%m%d_%H%M%S).csv
  cp "$OUT" "$BACKUP"
  echo "previous $OUT kept as $BACKUP"
fi

echo "== predict ($MODEL)"
$PY -m possession predict --model "$MODEL" --out "$OUT" --check

echo "== validate"
$PY evaluation/validate_submission.py "$OUT"

echo "== second half before its kick-off, frames 110,808-113,741 (all D is right)"
awk -F, 'NR>1 && $1>=110808 && $1<=113741 {c[$2]++} END {for (k in c) print "   " k, c[k]}' "$OUT"

if [ "$OUT" != "$REFERENCE" ] && [ -f "$REFERENCE" ] && [ -f reports/claude/compare_files.py ]; then
  echo "== against $REFERENCE"
  $PY reports/claude/compare_files.py "$REFERENCE" "$OUT"
fi
echo "== done: $OUT"
