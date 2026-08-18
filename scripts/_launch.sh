# Shared GPU-dispatch launcher for the cinder entry points.
#
# Sourced by train.sh / eval.sh, which must set:
#   MODULE  - python module to run (e.g. cinder.engine.train)
#   USAGE   - help text printed on -h/--help and on argument errors
#
# Argument convention (see the wrapper scripts for examples):
#   --gpus=SPEC      selects the execution backend (see below)
#   -h | --help      print USAGE and exit
#   --key=value      forwarded to Hydra as key=value
#   key=value        forwarded to Hydra verbatim
#
# --gpus=SPEC:
#   (omitted)   CPU only
#   1           single GPU (device 0), plain python
#   [N]         single specific GPU N, plain python
#   K (K>1)     K GPUs via torchrun
#   [A,B,...]   those specific GPUs via torchrun
#   -1          all GPUs visible to nvidia-smi via torchrun (1 -> plain python)

set -euo pipefail

: "${MODULE:?_launch.sh requires MODULE to be set}"
: "${USAGE:=No usage available.}"

# Resolve project root relative to the *calling* script and run from there so
# Hydra's ${oc.env:PROJECT_ROOT} and relative config paths resolve correctly.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
export PROJECT_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GPUS=""
HYDRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        -h|--help) printf '%s\n' "$USAGE"; exit 0 ;;
        --gpus=*)  GPUS="${arg#--gpus=}" ;;
        --*)       HYDRA_ARGS+=("${arg#--}") ;;
        *)         HYDRA_ARGS+=("$arg") ;;
    esac
done

_count_gpus() {
    # Number of GPUs reported by nvidia-smi, or 0 if it is unavailable.
    local n
    n=$(nvidia-smi -L 2>/dev/null | wc -l) || n=0
    echo "$n"
}

_run_plain() {  # $1 = CUDA_VISIBLE_DEVICES value
    CUDA_VISIBLE_DEVICES="$1" python -m "$MODULE" "${HYDRA_ARGS[@]}"
}

# Launch via `python -m torch.distributed.run`, never the `torchrun` console
# script. torchrun's shebang hard-codes the interpreter that installed PyTorch,
# and it spawns every rank with that same sys.executable -- so under a
# virtualenv layered on a system/module PyTorch (as on HPC-UGent), all the
# workers start outside the venv and fail on its packages. Going through
# `python -m` keeps the ranks on the active interpreter.
_run_dist() {   # $1 = nproc, $2 = optional CUDA_VISIBLE_DEVICES
    if [[ -n "${2:-}" ]]; then
        CUDA_VISIBLE_DEVICES="$2" python -m torch.distributed.run --nproc_per_node="$1" -m "$MODULE" "${HYDRA_ARGS[@]}"
    else
        python -m torch.distributed.run --nproc_per_node="$1" -m "$MODULE" "${HYDRA_ARGS[@]}"
    fi
}

if [[ -z "$GPUS" ]]; then
    # CPU
    _run_plain ""

elif [[ "$GPUS" == "-1" ]]; then
    # All available GPUs
    n=$(_count_gpus)
    if [[ "$n" -le 1 ]]; then _run_plain 0; else _run_dist "$n"; fi

elif [[ "$GPUS" == \[*\] ]]; then
    # Explicit GPU list: [0], [0,1], [0, 1]
    gpu_list="${GPUS//[\[\] ]/}"
    IFS=',' read -ra gpu_ids <<< "$gpu_list"
    if [[ "${#gpu_ids[@]}" -eq 1 ]]; then
        _run_plain "$gpu_list"
    else
        _run_dist "${#gpu_ids[@]}" "$gpu_list"
    fi

elif [[ "$GPUS" =~ ^[0-9]+$ ]]; then
    # Number of GPUs
    if [[ "$GPUS" -eq 0 ]]; then
        echo "Error: --gpus=0 is invalid; omit --gpus to run on CPU." >&2
        exit 1
    elif [[ "$GPUS" -eq 1 ]]; then
        _run_plain 0
    else
        _run_dist "$GPUS"
    fi

else
    echo "Error: invalid --gpus value '$GPUS'" >&2
    printf '%s\n' "$USAGE" >&2
    exit 1
fi
