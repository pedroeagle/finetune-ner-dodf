#!/bin/bash
#SBATCH --job-name=sdft-select
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --output=sdft-select-%j.out
#SBATCH --error=sdft-select-%j.err
#
# Checkpoint selection by dev-F1 + evaluation of the winner, for one SDFT run.
# Consolidates the two phases (launch dev evals → pick the highest F1 → run H1 on
# test + H2 on the benchmark ON THE WINNER). Selection is done on DEV so as not to
# peek at the test; collapsed checkpoints lose naturally.
#
# Usage (from the repo root, on the cluster):
#   BASE=$WORK/checkpoints/sdft_r8_anchor_b03_s42 ./experiments/select_checkpoint.sh
#
# Phase 1 (login node): compute candidates (epoch boundaries), launch a dev eval on
#   each, and reschedule the selection itself via --dependency=afterany.
# Phase 2 (dependent job, PHASE=select): read the dev-F1 values and launch test + benchmark.
set -u

: "${BASE:?set BASE=<full run dir, e.g., \$WORK/checkpoints/sdft_r8_anchor_b03_s42>}"
TAG=$(basename "$BASE")
PHASE="${PHASE:-eval}"

# Candidates = epoch boundaries (last/3, 2·last/3, last), matched to the nearest
# saved checkpoint.
mapfile -t STEPS < <(ls -d "$BASE"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n)
if [ "${#STEPS[@]}" -eq 0 ]; then
    echo "ERROR: no checkpoints in $BASE"; exit 1
fi
LAST=${STEPS[$((${#STEPS[@]}-1))]}
CANDS=$(python3 - "$LAST" "${STEPS[@]}" <<'PY'
import sys
last = int(sys.argv[1])
steps = sorted(int(x) for x in sys.argv[2:])
picked = []
for t in (round(last/3), round(2*last/3), last):
    nearest = min(steps, key=lambda s: abs(s - t))
    if nearest not in picked:
        picked.append(nearest)
print(" ".join(str(s) for s in sorted(picked)))
PY
)
echo "Run $TAG | candidates (dev): $CANDS"

if [ "$PHASE" = "eval" ]; then
    # Phase 1: launch a dev eval on each candidate and reschedule the selection after all.
    DEP=""
    for c in $CANDS; do
        JID=$(SDFT_MODEL="$BASE/checkpoint-$c" SPLIT=dev \
              ./experiments/submit.sh eval-ner-sdft-full 2>/dev/null \
              | grep -oP "Job ID  : \K[0-9]+")
        [ -n "$JID" ] && { echo "dev-eval checkpoint-$c -> job $JID"; DEP="${DEP:+$DEP:}$JID"; }
    done
    [ -z "$DEP" ] && { echo "ERROR: no dev-eval submitted."; exit 1; }
    sbatch --dependency=afterany:"$DEP" --export=ALL,BASE="$BASE",PHASE=select "$0"
    echo "Selection scheduled after the dev evals (afterany:$DEP)."
    exit 0
fi

# Phase 2: pick the checkpoint with the highest dev-F1.
WINNER=$(python3 - "$TAG" "$CANDS" <<'PY'
import json, os, sys
tag, cands = sys.argv[1], sys.argv[2].split()
best, bestf1 = None, -1.0
for c in cands:
    p = f"results/ner_{tag}_checkpoint-{c}_dev.json"
    if not os.path.exists(p):
        print(f"# checkpoint-{c}: no JSON (dev-eval failed) -> ignored", file=sys.stderr); continue
    try:
        f1 = float(json.load(open(p))["metrics"]["f1"])
    except Exception as e:
        print(f"# checkpoint-{c}: unreadable JSON ({e}) -> ignored", file=sys.stderr); continue
    print(f"# checkpoint-{c}: dev F1 = {f1:.4f}", file=sys.stderr)
    if f1 > bestf1: bestf1, best = f1, c
print(best or "")
PY
)
[ -z "$WINNER" ] && { echo "ERROR: no readable dev-F1."; exit 1; }
echo "Winner (highest dev-F1): checkpoint-$WINNER"
WIN="$BASE/checkpoint-$WINNER"

echo ">> H1 (test) on the winner..."
SDFT_MODEL="$WIN" SPLIT=test ./experiments/submit.sh eval-ner-sdft-full
echo ">> H2 (forgetting benchmark) on the winner..."
SDFT_MODEL="$WIN" ./experiments/submit.sh bench-ft-sdft-full
