#!/bin/bash
# L2 waiter: block until the TBLE rung is harvested into the combined npz
# (tble_present=True, set by the OpenFOAM orchestration
#  pehill_wmles_tble/run_to_completion_and_harvest.sh), then re-run the
# closure-ladder experiment + figure so the node has the COMPLETE ladder.
# No fabrication: this only consumes the harvested npz; it never invents tble.
set +u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NPZ="$ROOT/codes/results/aposteriori_wmles_pehill.npz"
LOG="$ROOT/work_progress/closure_ladder_wait.log"
cd "$ROOT" || exit 1
echo "=== closure-ladder waiter started $(date) ===" > "$LOG"
while true; do
    present=$(OMP_NUM_THREADS=2 python3 -c "
import numpy as np
try:
    d=np.load('$NPZ',allow_pickle=True)
    print('1' if bool(d['tble_present']) else '0')
except Exception as e:
    print('0')
" 2>/dev/null)
    if [ "$present" = "1" ]; then
        echo "[$(date '+%H:%M')] tble_present=True -> running closure ladder + figure" | tee -a "$LOG"
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 codes/analysis/closure_ladder_aposteriori.py >> "$LOG" 2>&1
        echo "  closure_ladder exit=$?" | tee -a "$LOG"
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 codes/figures/fig_closure_ladder.py >> "$LOG" 2>&1
        echo "  fig_closure_ladder exit=$?" | tee -a "$LOG"
        echo "[$(date '+%H:%M')] DONE" | tee -a "$LOG"
        break
    fi
    # also bail out if neither solver nor harvester is alive (avoid infinite loop)
    if ! pgrep -f "pimpleFoam -parallel" >/dev/null && \
       ! pgrep -f "run_to_completion_and_harvest" >/dev/null; then
        echo "[$(date '+%H:%M')] WARNING: TBLE solver+harvester both gone, tble_present still 0 -- stopping waiter" | tee -a "$LOG"
        break
    fi
    sleep 300
done
