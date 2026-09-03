#!/usr/bin/env bash
# pull_samples.sh -- rsync the (small) developed-state wall samples back from
# ARCHER2 into codes/results/deployed_operator_samples/<case>/.
# Read-only on the remote side; nothing is deleted there.
set -euo pipefail
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RB=/work/e1001/e1001/jianoujiang/paper-factory/repeating_structure_wall_model
DEST="$PROJ_DIR/codes/results/deployed_operator_samples"
mkdir -p "$DEST"
for case_id in "$@"; do
    mkdir -p "$DEST/$case_id"
    rsync -az -e ssh \
        --include='deployedSample/***' --include='deployedFace/***' \
        --exclude='*' \
        "archer2:$RB/jobs/$case_id/postProcessing/" "$DEST/$case_id/" \
        2>/dev/null || echo "PULL_FAIL $case_id"
    n=$(find "$DEST/$case_id" -name '*.xy' 2>/dev/null | wc -l)
    echo "PULLED $case_id files=$n"
done
