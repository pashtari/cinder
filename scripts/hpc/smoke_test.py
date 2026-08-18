"""GPU / environment sanity check for cinder on HPC-UGent.

Run as a job (login nodes have no GPU, and litleo's binaries SIGILL there):
    sbatch --clusters=litleo jobs/smoke.slurm
"""

import importlib
import os
import socket
import sys


def main() -> int:
    print(f"host        : {socket.gethostname()}")
    print(f"cluster     : {os.environ.get('SLURM_CLUSTER_NAME', '<not in a job>')}")
    print(f"job id      : {os.environ.get('SLURM_JOB_ID', '-')}")
    print(f"python      : {sys.version.split()[0]}")

    import torch
    import torchvision

    print(f"torch       : {torch.__version__}  (built for CUDA {torch.version.cuda})")
    print(f"torchvision : {torchvision.__version__}")

    if not torch.cuda.is_available():
        print("\nFAIL: torch.cuda.is_available() is False -- no GPU visible.")
        return 1

    n = torch.cuda.device_count()
    print(f"visible GPUs: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory / 1024**3:.1f} GiB  sm_{p.major}{p.minor}")

    ok = True

    # Does the compiled kernel set actually cover this card? PyTorch here is
    # built per cluster for that cluster's GPU only -- there is no fat binary.
    arch_list = torch.cuda.get_arch_list()
    major, minor = torch.cuda.get_device_capability(0)
    print(f"arch list   : {' '.join(arch_list)}")
    if f"sm_{major}{minor}" not in arch_list:
        print(f"WARNING: sm_{major}{minor} missing from the arch list -- expect PTX/JIT fallback.")

    # Real work on the device, plus a torchvision CUDA op (compiled extension,
    # so this catches a torchvision built against a different CUDA than torch).
    a = torch.randn(4096, 4096, device="cuda")
    print(f"matmul ok   : sum={(a @ a).sum().item():.3e}")

    from torchvision.ops import nms

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], device="cuda")
    keep = nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    print(f"tv CUDA op  : nms -> {keep.tolist()}")

    # The venv half of the stack: these have no module and come from pip.
    for name in ("ignite", "hydra", "omegaconf"):
        try:
            print(f"{name:<12}: {importlib.import_module(name).__version__}")
        except Exception as exc:
            print(f"{name:<12}: MISSING ({exc})")
            ok = False

    # triton is optional -- cinder.models.rcs_matrix falls back to a pure-torch
    # path without it, which is correct but slower. Worth knowing which ran.
    try:
        import triton  # noqa: F401

        print("triton      : available (fused RCSMatrix kernel)")
    except ImportError:
        print("triton      : absent (RCSMatrix uses the pure-torch fallback)")

    # The integration check: modules + venv + the package itself.
    try:
        import cinder

        print(f"cinder      : imported from {os.path.dirname(cinder.__file__)}")
    except Exception as exc:
        print(f"cinder      : FAILED to import ({exc})")
        ok = False

    if n >= 2:
        x = torch.randn(1024, 1024, device="cuda:0")
        print(f"p2p copy ok : cuda:0 -> cuda:1 ({tuple(x.to('cuda:1').shape)})")
        available = torch.distributed.is_nccl_available()
        print(f"nccl        : {'available' if available else 'MISSING'}")
        ok = ok and available
    else:
        print("p2p copy    : skipped (1 GPU)")

    print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
