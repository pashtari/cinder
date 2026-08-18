#!/bin/bash
# ---------------------------------------------------------------------------
# Stage the venv's PyPI wheels ON THE LOGIN NODE.
#
# Compute nodes have no outbound internet (pip there dies with ProxyError), so
# wheels must be downloaded here and installed offline inside the job.
#
#     bash $VSC_DATA/projects/cinder/scripts/hpc/prefetch_wheels.sh
#
# These are pure-python wheels, so one cache serves litleo, accelgor and joltik
# alike. Re-run after editing requirements-hpc.txt.
# ---------------------------------------------------------------------------
set -euo pipefail

HPC_DIR="${CINDER_HPC:-$VSC_DATA/projects/cinder/scripts/hpc}"
WHEELS="$HPC_DIR/wheels"

# The cache is fully derived from requirements-hpc.txt, so rebuild it from
# scratch: a stale wheel left behind by an earlier requirement list is dead
# weight on the $VSC_DATA quota and misleading to anyone reading the directory.
rm -rf "$WHEELS"
mkdir -p "$WHEELS"

# The accelgor stack is zen3 but still executes on the zen2 login node, and
# gives the same Python 3.11.3 the GPU jobs use.
module --force purge > /dev/null 2>&1
module load cluster/accelgor > /dev/null 2>&1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 > /dev/null 2>&1

echo "python : $(python -c 'import sys; print(sys.version.split()[0])')"
echo "wheels : $WHEELS"
echo

# `pip wheel`, not `pip download`: antlr4-python3-runtime 4.9.3 is published as
# an sdist only, and installing an sdist offline would need build isolation to
# fetch setuptools from PyPI -- which is exactly what a compute node cannot do.
# Building here turns everything into a wheel, so the job install is pure binary.
#
# --no-deps mirrors that install: requirements-hpc.txt already lists the full
# closure, and resolving deps would drag in a ~2 GB PyPI torch that the module
# stack already provides.
python -m pip wheel --wheel-dir "$WHEELS" --no-deps -r "$HPC_DIR/requirements-hpc.txt"

echo
echo "=== staged ==="
ls -1 "$WHEELS" | sed 's/^/  /'
du -sh "$WHEELS"
