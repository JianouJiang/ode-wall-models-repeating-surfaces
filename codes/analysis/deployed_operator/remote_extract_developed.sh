#!/bin/bash
# remote_extract_developed.sh -- READ-ONLY developed-state extraction on ARCHER2.
#
# Runs on the ARCHER2 LOGIN NODE (small post-processing only, no Slurm job, no CU).
# For every terminal deposited case of the R2-3/M6 matching-surface campaign it
# samples, at the wall-adjacent cell layer and at the wall faces themselves, the
# quantities the deployed wall-model boundary condition consumes and produces:
#
#   patchInternalField (wall-adjacent cell)  UMean, grad(pMean), C
#   patch face value                          wallShearStressMean, pMean
#
# `grad(pMean)` does not exist on disk; it is created by `postProcess`, sampled,
# and then DELETED again, so the deposited case is left exactly as found. A
# before/after listing of every touched time directory is printed for audit.
#
# The script REFUSES to touch any case whose name appears in the live queue.
#
# Output (per case, per time):
#   <case>/postProcessing/deployedSample/<t>/{bottomWallInternal,topWallInternal,
#                                            bottomWallFace,topWallFace}.xy
# Those files are small (4800-20480 rows) and are rsynced back by the caller.

# NOTE: deliberately no `set -u` -- Lmod's `module` shell function and the
# OpenFOAM etc/bashrc both dereference unset variables and abort the shell.
set -o pipefail

PROJ=/work/e1001/e1001/jianoujiang/paper-factory/repeating_structure_wall_model
TIMES="${TIMES:-405 360 315}"
ZEROTIME="${ZEROTIME:-0}"

module load openfoam/org/v10.20230119 >/dev/null 2>&1 || { echo "FATAL module"; exit 8; }
source "$FOAM_INSTALL_DIR/etc/bashrc" >/dev/null 2>&1 || { echo "FATAL bashrc"; exit 8; }

LIVE=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | tr '\n' ' ')
echo "LIVE_JOBS ${LIVE:-none}"

write_dicts () {
    local case_dir="$1"
    # developed-state sample: wall-adjacent cell values of the BC's inputs
    cat > "$case_dir/system/deployedSample" <<'EOF'
deployedSample
{
    type            surfaces;
    libs            ("libsampling.so");
    interpolationScheme cell;
    surfaceFormat   raw;
    fields          ( UMean gradPMean );
    surfaces
    (
        bottomWallInternal
        {
            type        patchInternalField;
            patches     (bottomWall);
            interpolate false;
            offsetMode  normal;
            distance    0;
        }
        topWallInternal
        {
            type        patchInternalField;
            patches     (topWall);
            interpolate false;
            offsetMode  normal;
            distance    0;
        }
    );
}
EOF
    # delivered traction and pressure ON the wall faces
    cat > "$case_dir/system/deployedFace" <<'EOF'
deployedFace
{
    type            surfaces;
    libs            ("libsampling.so");
    interpolationScheme cell;
    surfaceFormat   raw;
    fields          ( wallShearStressMean pMean );
    surfaces
    (
        bottomWallFace
        {
            type        patch;
            patches     (bottomWall);
            interpolate false;
        }
        topWallFace
        {
            type        patch;
            patches     (topWall);
            interpolate false;
        }
    );
}
EOF
    # time-independent geometry: cell centres of the wall-adjacent cells
    cat > "$case_dir/system/deployedGeom" <<'EOF'
deployedGeom
{
    type            surfaces;
    libs            ("libsampling.so");
    interpolationScheme cell;
    surfaceFormat   raw;
    fields          ( C U );
    surfaces
    (
        bottomWallGeom
        {
            type        patchInternalField;
            patches     (bottomWall);
            interpolate false;
            offsetMode  normal;
            distance    0;
        }
        topWallGeom
        {
            type        patchInternalField;
            patches     (topWall);
            interpolate false;
            offsetMode  normal;
            distance    0;
        }
    );
}
EOF
}

for case_id in "$@"; do
    case_dir="$PROJ/jobs/$case_id"
    if [ ! -d "$case_dir" ]; then echo "MISSING $case_id"; continue; fi
    skip=0
    for j in $LIVE; do
        case "$case_id" in *"$j"*) skip=1;; esac
        case "$j" in *"$case_id"*) skip=1;; esac
    done
    if [ "$skip" = 1 ]; then echo "SKIP_LIVE $case_id"; continue; fi

    cd "$case_dir" || { echo "NOCD $case_id"; continue; }
    write_dicts "$case_dir"

    # ---- time-independent geometry + the first-solve instantaneous field ----
    before0=$(ls "$ZEROTIME" 2>/dev/null | sort | tr '\n' ',')
    postProcess -func deployedGeom -time "$ZEROTIME" > /tmp/pp_geom.log 2>&1
    echo "GEOM $case_id rc=$? files=$(find postProcessing/deployedGeom -name '*.xy' 2>/dev/null | wc -l)"

    for t in $TIMES; do
        [ -d "$t" ] || { echo "NOTIME $case_id $t"; continue; }
        before=$(ls "$t" | sort | tr '\n' ',')
        # `postProcess` can create grad(pMean) but the sampled-surface `fields`
        # entry cannot name a field whose name contains parentheses, so the
        # field is copied under a parenthesis-free object name and both copies
        # are deleted again below.
        postProcess -func "grad(pMean)" -time "$t" > /tmp/pp_grad.log 2>&1
        rcg=$?
        if [ -f "$t/grad(pMean)" ]; then
            sed 's/^\( *object *\)grad(pMean);/\1gradPMean;/' "$t/grad(pMean)" \
                > "$t/gradPMean"
        fi
        postProcess -func deployedSample -time "$t" > /tmp/pp_s.log 2>&1
        rcs=$?
        postProcess -func deployedFace   -time "$t" > /tmp/pp_f.log 2>&1
        rcf=$?
        # restore the time directory exactly: remove only the fields we created
        rm -f "$t/grad(pMean)" "$t/gradPMean"
        after=$(ls "$t" | sort | tr '\n' ',')
        if [ "$before" = "$after" ]; then rest=RESTORED; else rest="CHANGED[$before|$after]"; fi
        nrow=$(grep -vc '^#' "postProcessing/deployedSample/$t/bottomWallInternal.xy" 2>/dev/null || echo 0)
        echo "SAMPLE $case_id t=$t rc_grad=$rcg rc_int=$rcs rc_face=$rcf rows=$nrow $rest"
        if [ "$rcs" != 0 ]; then tail -5 /tmp/pp_s.log; fi
    done
    after0=$(ls "$ZEROTIME" 2>/dev/null | sort | tr '\n' ',')
    [ "$before0" = "$after0" ] && echo "GEOMDIR $case_id RESTORED" || echo "GEOMDIR $case_id CHANGED"
done
echo "DEPLOYED_EXTRACT_DONE"
