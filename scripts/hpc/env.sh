# shellcheck shell=bash
# ---------------------------------------------------------------------------
# cinder @ HPC-UGent -- shared environment.
#
# Source from every job script and from interactive sessions:
#     source $VSC_DATA/projects/cinder/scripts/hpc/env.sh
#
# To target another cluster, export before sourcing -- the `VAR=x source ...`
# prefix form does NOT stick, because bash treats `source` as a regular builtin
# and reverts the assignment when it returns:
#     export CINDER_CLUSTER=accelgor
#     source $VSC_DATA/projects/cinder/scripts/hpc/env.sh
# ---------------------------------------------------------------------------

# --- paths -----------------------------------------------------------------
export CINDER_ROOT="${CINDER_ROOT:-$VSC_DATA/projects/cinder}"   # git checkout
export CINDER_HPC="${CINDER_HPC:-$CINDER_ROOT/scripts/hpc}"      # this directory
export CINDER_DATA="${CINDER_DATA:-$VSC_SCRATCH/data}"           # shared dataset pool
export CINDER_LOGS="${CINDER_LOGS:-$CINDER_ROOT/logs}"           # slurm logs + hydra runs

# --- caches: keep them off the home quota ----------------------------------
export PIP_CACHE_DIR="$VSC_SCRATCH/cache/pip"
export TORCH_HOME="$VSC_SCRATCH/cache/torch"
export HF_HOME="$VSC_SCRATCH/cache/hf"
export XDG_CACHE_HOME="$VSC_SCRATCH/cache"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$CINDER_LOGS"

# --- modules ---------------------------------------------------------------
# The login node preloads GCCcore/14.3.0 for the VSCode server, which collides
# with foss/2023a -> always start from a clean slate.
module --force purge > /dev/null 2>&1

# Inside a job the cluster is already correct; on the login node select it
# explicitly. litleo is the default: H100, and the complete stack below.
export CINDER_CLUSTER="${CINDER_CLUSTER:-${SLURM_CLUSTER_NAME:-litleo}}"
module load "cluster/${CINDER_CLUSTER}" > /dev/null 2>&1

# foss/2023a + CUDA 12.1.1. This toolchain is not a preference, it is forced:
# torchvision's only CUDA build here is 0.16.0-foss-2023a. Newer PyTorch modules
# exist (2.7.1-foss-2024a-CUDA-12.6.0) with no CUDA-matched torchvision, so
# moving up means giving that up. Re-check `module avail torchvision` first.
#
# timm is deliberately NOT loaded as a module. The only build is 1.0.8, which
# predates the DINOv3 weights configs/model/default.yaml asks for -- and since an
# EasyBuild module puts itself on PYTHONPATH, which precedes venv site-packages,
# a module timm would shadow the newer pip one rather than the other way round.
# It comes from the venv instead (see requirements-hpc.txt).
#
# Between them these three cover torch, torchvision, numpy, scipy, PIL,
# tensorboard, yaml and tqdm; timm, ignite and hydra come from the venv.
# Triton is a separate module here: EasyBuild's PyTorch does not bundle it the
# way the PyPI wheels do, and without it cinder.models.rcs_matrix silently takes
# its pure-torch fallback instead of the fused kernel. 2.1.0 is the build that
# pairs with PyTorch 2.1.2 on this toolchain.
module load \
    PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 \
    torchvision/0.16.0-foss-2023a-CUDA-12.1.1 \
    Triton/2.1.0-foss-2023a-CUDA-12.1.1 \
    tensorboard/2.15.1-gfbf-2023a

# --- venv ------------------------------------------------------------------
# One logical environment ("deepenv") with a separate build per CPU
# architecture: each cluster has its own module tree and a PyTorch compiled for
# that cluster's GPU only, so a single shared directory breaks on switching.
export CINDER_VENV="${CINDER_VENV:-$VSC_DATA/venvs/deepenv/${VSC_ARCH_LOCAL}${VSC_ARCH_SUFFIX}}"
if [ -f "$CINDER_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$CINDER_VENV/bin/activate"
else
    echo "[env.sh] no venv at $CINDER_VENV" >&2
    echo "[env.sh] build it: sbatch --clusters=$CINDER_CLUSTER \$CINDER_HPC/jobs/setup_venv.slurm" >&2
fi

# --- runtime ---------------------------------------------------------------
export PYTHONPATH="$CINDER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # stop OpenMP oversubscribing

# ---------------------------------------------------------------------------
# cinder_provenance <run_dir>
#
# Records what code and environment produced a run, so that months from now
# "which version made this number" still has an answer.
#
# Writes <run_dir>/provenance.txt, plus <run_dir>/uncommitted.patch if the tree
# is dirty -- a scrappy mid-experiment run then replays as
#     git checkout <commit> && git apply uncommitted.patch
# ---------------------------------------------------------------------------
cinder_provenance() {
    local run_dir="${1:?cinder_provenance needs a run directory}"
    local out="$run_dir/provenance.txt"
    local git="git -C $CINDER_ROOT"
    mkdir -p "$run_dir"

    {
        echo "# cinder run provenance"
        echo "date     : $(date -Is)"
        echo "job      : ${SLURM_JOB_ID:-<interactive>} (${SLURM_JOB_NAME:-n/a})"
        echo "cluster  : ${SLURM_CLUSTER_NAME:-$CINDER_CLUSTER}"
        echo "node     : $(hostname)"
        echo "arch     : ${VSC_ARCH_LOCAL}${VSC_ARCH_SUFFIX}"
        echo "gpus     : ${SLURM_GPUS_PER_NODE:-?}   cpus: ${SLURM_CPUS_PER_TASK:-?}"
        echo "venv     : $CINDER_VENV"
        echo "data     : $CINDER_DATA"

        echo
        echo "## git"
        if $git rev-parse --git-dir > /dev/null 2>&1; then
            echo "commit   : $($git describe --always --tags --dirty)"
            echo "branch   : $($git rev-parse --abbrev-ref HEAD)"
            if [ -n "$($git status --porcelain)" ]; then
                echo "tree     : DIRTY (full diff in uncommitted.patch)"
                $git status --porcelain | sed 's/^/  /'
                $git diff HEAD > "$run_dir/uncommitted.patch" 2>/dev/null
            else
                echo "tree     : clean"
            fi
        else
            echo "commit   : <not a git repository -- this run is NOT reproducible>"
        fi

        echo
        echo "## versions"
        python - <<'PY' 2>/dev/null || echo "(python probe failed)"
import sys
print(f"python      : {sys.version.split()[0]}")
try:
    import torch
    print(f"torch       : {torch.__version__} (cuda {torch.version.cuda})")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu[{i}]      : {p.name} {p.total_memory/1024**3:.1f} GiB sm_{p.major}{p.minor}")
except Exception as e:
    print(f"torch       : <{e}>")
# triton decides whether RCSMatrix runs its fused kernel or the pure-torch
# fallback -- different code for the same config, so it belongs in provenance.
for m in ("torchvision", "timm", "triton", "ignite", "hydra", "numpy"):
    try:
        print(f"{m:<12}: {__import__(m).__version__}")
    except Exception:
        pass
PY

        echo
        echo "## modules"
        module --terse list 2>&1 | grep -vE "^(Currently|$)" | sed 's/^/  /'
    } > "$out" 2>&1

    echo "[provenance] -> $out"
    grep -E "^(commit|branch|tree)" "$out" | sed 's/^/[provenance] /'
}
