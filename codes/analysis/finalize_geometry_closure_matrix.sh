#!/bin/bash
# Auto-finalize the geometry x closure matrix once the conv-div coupled WMLES
# negative control (P3) has been harvested into aposteriori_wmles_convdiv.npz.
#
# Blocks until that npz exists (the harvest_convdiv.sh in codes/openfoam/
# convdiv_wmles writes it at endTime), then re-runs the consolidation + figure so
# the matrix becomes 3/3 complete with the REAL conv-div number. Writes a DONE
# marker. No fabrication: it only runs after the real harvest lands.
#
#   nohup ./finalize_geometry_closure_matrix.sh > log.finalize_matrix 2>&1 &
set -u
ROOT="/home/jianoujiang/Desktop/paper-factory-lean/projects/repeating_structure_wall_model"
NPZ="$ROOT/codes/results/aposteriori_wmles_convdiv.npz"
NODE="$ROOT/development/nodes/node_002"
MARK="$NODE/CONVDIV_FINALIZED.txt"
PROG="$ROOT/work_progress/geometry_closure_matrix_progress.md"

echo "=== finalize pipeline started $(date) ==="
echo "waiting for $NPZ with convdiv_present=True ..."
# Wait until the harvest writes the REAL coupled result (convdiv_present=True),
# not merely a DNS-only stub. A bare file existence check would fire on the stub.
while true; do
    flag=$(python3 - "$NPZ" <<'PY' 2>/dev/null
import sys, numpy as np, os
p=sys.argv[1]
if not os.path.exists(p): print("MISSING"); sys.exit()
d=np.load(p, allow_pickle=True)
print("TRUE" if ("convdiv_present" in d.files and bool(d["convdiv_present"])) else "FALSE")
PY
)
    [ "$flag" = "TRUE" ] && break
    sleep 180
done
echo "[$(date)] conv-div coupled result present (convdiv_present=True) -> finalizing matrix"

cd "$ROOT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 codes/analysis/geometry_closure_matrix.py 2>&1 | tee "$NODE/matrix_table_final.txt"
FIG_OUT="$NODE/fig_geometry_closure_matrix.png" python3 codes/figures/fig_geometry_closure_matrix.py
cp codes/figures/fig_geometry_closure_matrix.py "$NODE/" 2>/dev/null || true

{
  echo "# geometry_closure_matrix finalized $(date)"
  echo "conv-div coupled (P3) harvested; matrix re-run to 3/3."
  python3 - <<'PY'
import numpy as np
d=np.load("codes/results/geometry_closure_matrix.npz",allow_pickle=True)
print("matrix_complete =", bool(d["matrix_complete"]))
print("P3_verdict =", str(d["P3_verdict"]))
print("P4_verdict =", str(d["P4_verdict"]))
PY
} | tee "$MARK"
echo "[$(date)] DONE -> $MARK" | tee -a "$PROG"
