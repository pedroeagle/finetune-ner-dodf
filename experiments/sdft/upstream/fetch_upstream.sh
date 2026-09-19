#!/bin/bash
# Fetch the vendored SDFT trainer from the official repository and apply the
# beta-anchor patch. The two upstream files are NOT redistributed in this repo
# (their source has no license); this script reconstructs them locally.
#
# Usage (from anywhere):
#   bash experiments/sdft/upstream/fetch_upstream.sh
#
# Result: distil_trainer.py (patched with the beta anchor) and distil_config.py
# in this directory, ready for train_sdft.py to import. See ATTRIBUTION.md.
set -euo pipefail

REPO="https://github.com/idanshen/Self-Distillation"
COMMIT="d77573212fa0a3ae2eeb64b9b44db1c251f75e3e"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$DIR/distil_trainer.py" && -f "$DIR/distil_config.py" ]]; then
    echo "Upstream files already present in $DIR — nothing to do."
    echo "(Delete them and re-run to refetch.)"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Cloning $REPO @ $COMMIT ..."
git clone --quiet "$REPO" "$TMP/src"
git -C "$TMP/src" checkout --quiet "$COMMIT"

# Locate the two files in the upstream tree (path may vary between mirrors).
for f in distil_trainer.py distil_config.py; do
    src="$(find "$TMP/src" -name "$f" -print -quit)"
    [[ -n "$src" ]] || { echo "ERROR: $f not found in upstream @ $COMMIT"; exit 1; }
    cp "$src" "$DIR/$f"
    echo "  copied $f"
done

echo "Applying beta-anchor patch ..."
patch -p1 -d "$DIR" < "$DIR/beta_anchor.patch"

echo "Done. distil_trainer.py is patched with the beta anchor; distil_config.py is upstream."
