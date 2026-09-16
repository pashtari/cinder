# Shared launcher for scripts/train.sh and scripts/eval.sh.
#
# The caller sets MODULE, the Python module to run, and sources this file. The
# caller's header comment is its help text, followed by GPU_USAGE.

set -euo pipefail

: "${MODULE:?_launch.sh requires MODULE to be set}"

GPU_USAGE="--gpus=SPEC:
  (omitted)   CPU only
  1           GPU 0
  [N]         GPU N
  K           K GPUs with torchrun
  [A,B,...]   the listed GPUs with torchrun
  -1          every GPU that nvidia-smi lists (torchrun if more than one)"

CALLER="${BASH_SOURCE[1]}"
USAGE="$(sed -n '2,/^[^#]/s/^# \{0,1\}//p' "$CALLER")

$GPU_USAGE"

# Run from the project root, which configs/path/default.yaml reads.
PROJECT_ROOT="$(cd "$(dirname "$CALLER")/.." && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# Let the CUDA allocator grow its segments instead of fragmenting them: decoding
# allocates gigabyte-sized tensors, which a fragmented cache cannot place.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

GPUS=""
HYDRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        -h|--help) printf '%s\n' "$USAGE"; exit 0 ;;
        --gpus=*) GPUS="${arg#--gpus=}" ;;
        # Hydra's own flags, e.g. --cfg job --resolve, pass through unchanged.
        --cfg*|--resolve|--package*|--info*|--hydra-help|--multirun|--config-*|-m|-c|-p|-cn|-cp|-cd)
            HYDRA_ARGS+=("$arg") ;;
        --*) HYDRA_ARGS+=("${arg#--}") ;;
        *) HYDRA_ARGS+=("$arg") ;;
    esac
done

count_gpus() {
    local n
    n=$(nvidia-smi -L 2>/dev/null | wc -l) || n=0
    echo "$n"
}

run_python() {  # $1: CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES="$1"
    exec python -m "$MODULE" "${HYDRA_ARGS[@]}"
}

# `python -m torch.distributed.run` rather than the `torchrun` script: its shebang
# names the interpreter that installed PyTorch, which may not be the active
# virtual environment's, and every rank would start outside it.
run_torchrun() {  # $1: number of processes, $2: optional CUDA_VISIBLE_DEVICES
    if [[ -n "${2:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="$2"
    fi
    exec python -m torch.distributed.run --nproc_per_node="$1" -m "$MODULE" "${HYDRA_ARGS[@]}"
}

if [[ -z "$GPUS" ]]; then
    run_python ""
elif [[ "$GPUS" == "-1" ]]; then
    n=$(count_gpus)
    if [[ "$n" -le 1 ]]; then run_python 0; else run_torchrun "$n"; fi
elif [[ "$GPUS" == \[*\] ]]; then
    gpu_list="${GPUS//[\[\] ]/}"
    IFS=',' read -ra gpu_ids <<< "$gpu_list"
    if [[ "${#gpu_ids[@]}" -eq 1 ]]; then
        run_python "$gpu_list"
    else
        run_torchrun "${#gpu_ids[@]}" "$gpu_list"
    fi
elif [[ "$GPUS" =~ ^[0-9]+$ && "$GPUS" -gt 0 ]]; then
    if [[ "$GPUS" -eq 1 ]]; then run_python 0; else run_torchrun "$GPUS"; fi
else
    echo "Error: invalid --gpus value '$GPUS'; omit --gpus to run on the CPU." >&2
    printf '%s\n' "$USAGE" >&2
    exit 1
fi
