# cinder on HPC-UGent

Training and evaluating cinder on **GlaS**, **FIVES** and **ADE20K**.
Verified against the live cluster on 2026-08-15.

Everything here assumes `env.sh` has been sourced; it defines `$CINDER_ROOT`,
`$CINDER_HPC`, `$CINDER_DATA`, `$CINDER_LOGS` and `$CINDER_VENV`, and the job
scripts source it themselves.

```
scripts/hpc/
  env.sh                 modules + venv + paths + cinder_provenance()
  requirements-hpc.txt   the few packages with no module
  prefetch_wheels.sh     login node: stage wheels for the offline install
  fetch_datasets.sh      login node: download datasets to scratch
  smoke_test.py          GPU / stack sanity check
  jobs/setup_venv.slurm  build the venv (once per cluster)
  jobs/smoke.slurm       run smoke_test.py on a GPU
  jobs/train.slurm       train on 1 or 2 GPUs
```

## 1. Where things live

| What | Path |
|---|---|
| Repo (git checkout) | `$VSC_DATA/projects/cinder` |
| Python venv | `$VSC_DATA/venvs/deepenv/<arch>` |
| Datasets | `$VSC_SCRATCH/data/{GlaS,FIVES,ADE20K}` |
| Slurm logs + hydra runs | `$VSC_DATA/projects/cinder/logs` |

One venv per **architecture**, not per project: each cluster has its own module
tree and a PyTorch compiled for that cluster's GPU, so a single shared directory
breaks the moment you switch clusters. `env.sh` picks the right build from
`$VSC_ARCH_LOCAL`.

Quotas (via `/usr/lpp/mmfs/bin/mmlsquota` — plain `quota` hangs on this GPFS
setup, and `show_quota` just points at the website):

| Filesystem | Used | Hard limit |
|---|---|---|
| `$VSC_DATA` | 7.9 G | 50 G |
| `$VSC_SCRATCH` | 5.7 G | 25 G |
| `$VSC_HOME` | 335 M | 12 G |

Logs and checkpoints live in the repo on `$VSC_DATA`; only datasets go on
scratch. `env.sh` redirects the pip/torch/HF caches to `$VSC_SCRATCH/cache` to
keep them off home. There is no VO storage on this account.

**Scratch is auto-purged** — HPC-UGent guarantees nothing beyond "anywhere
between a day and a few weeks". All three re-fetch with `fetch_datasets.sh`,
so a purge costs only download time.

## 2. Which cluster

| Cluster | GPU | GPUs/node | Max GPUs per job | Cores/GPU |
|---|---|---|---|---|
| **litleo** | H100 NVL 94 GB | 2 | **1** (QoS `litleo-max-gpu`) | 24 |
| **accelgor** | A100 80 GB | 4 | 2+ | 12 |
| **joltik** | V100 32 GB | 4 | 2+ | 8 |

**Runs standardise on accelgor (A100).** litleo's QoS caps every job at 1 GPU,
so "2 H100s" does not exist here — the real choice is 1 H100 vs 2 A100s, and
keeping everything on one GPU type keeps results comparable.

Max walltime is 3 days everywhere, but the default is 1 hour — always set
`--time`.

### Per-dataset sizing

Every dataset trains at **global batch 16** and **512² crops**, and is budgeted
in **iterations**, not epochs — an epoch is 5 steps on GlaS and 1263 on ADE20K,
so it is not a unit in which budgets can be compared or stated:

| Dataset | GPUs | Iterations | Eval interval | Equivalent epochs |
|---|---|---|---|---|
| **GlaS** | 1 × A100 | 20,000 | 2,000 | ~3,760 |
| **FIVES** | 1 × A100 | 20,000 | 2,000 | ~530 |
| **ADE20K** | 2 × A100 | 160,000 | 16,000 | ~127 |

ADE20K's 160k at batch 16 is the mmsegmentation convention its baselines are
trained to. `batch_size` is **global**: `ignite.distributed.auto_dataloader`
divides it by world size, so it is 8 per GPU on two GPUs.

Measured on an A100 (80 GB) in fp32, with the real configs and dataloaders,
mirroring `create_trainer`'s step. 16 per GPU is the throughput knee — 8→16 buys
~10%, 16→32 only ~5% for twice the memory, and 32 does not fit ADE20K:

| batch/GPU | 480² binary | 512² × 151 classes |
|---|---|---|
| 4 | 8.7 GiB · 27 img/s | 10.3 GiB · 22 img/s |
| 8 | 16.3 GiB · 35 img/s | 19.5 GiB · 27 img/s |
| 16 | 31.5 GiB · 39 img/s | 37.8 GiB · 29 img/s |
| 32 | 61.9 GiB · 41 img/s | **OOM** |

At the settings above (global 16, two GPUs on ADE20K) the per-GPU batch is 8,
trading ~7% throughput for a budget that matches the published one.

`val_batch_size` stays **1** and is not a tuning knob: `val_transform` is null,
so validation images keep native sizes (GlaS has 5 distinct sizes across 80
images, ADE20K 182 across 2000) and the default collate cannot stack them.
Evaluation batches via the inferer's `sw_batch_size` instead, which batches
windows inside one image and is therefore size-agnostic.

## 3. Sync

Code flows down through git, results come back through rsync. Never two-way
sync: a continuous syncer will eventually write a file mid-read during a job.

```
Mac  --(git push)-->  github.com/pashtari/cinder  --(git pull)-->  HPC
Mac  <---------------------------(rsync, one-way)---------------  HPC logs/
```

The cluster authenticates with `~/.ssh/id_ed25519.pub`, registered as a deploy
key with write access, so `git pull` here needs no token juggling:

```bash
cd $VSC_DATA/projects/cinder && git pull      # before every submission
```

Pulling results back, from the Mac:

```bash
rsync -avz --exclude='*.pt' --exclude='*.ckpt' \
  hpcugent:/data/gent/520/vsc52033/projects/cinder/logs/ ~/Dropbox/research/projects/cinder/logs/
```

## 4. Build the venv (once per cluster)

Order matters — **compute nodes have no outbound internet** (pip there fails
with `ProxyError`), so wheels are staged on the login node and installed
offline inside the job.

```bash
bash $CINDER_HPC/prefetch_wheels.sh                          # login node
sbatch --clusters=litleo   $CINDER_HPC/jobs/setup_venv.slurm  # as a job
sbatch --clusters=accelgor $CINDER_HPC/jobs/setup_venv.slurm  # only for DDP
```

The venv is `--system-site-packages`, so it inherits the CUDA-enabled module
builds of torch/torchvision/timm; only `requirements-hpc.txt` and cinder itself
are installed into it.

## 5. Get the datasets

On the **login node** (compute nodes have no internet):

```bash
bash $CINDER_HPC/fetch_datasets.sh all
```

| Dataset | On disk | Contents |
|---|---|---|
| ADE20K | 1.6 GB | 20210 train / 2000 val images + annotations |
| FIVES | 1.7 GB | 600 train / 200 test image+mask pairs |
| GlaS | 165 MB | 85 train / 60 testA / 20 testB image+annotation pairs |

```
$VSC_SCRATCH/data/
  ADE20K/{images,annotations}/{training,validation}/
  FIVES/{train,test}/{Original,Ground truth}/
  GlaS/{train,testA,testB}_<n>.bmp + _anno.bmp
```

**GlaS comes from Kaggle, anonymously.** The Warwick original redirects to their
SSO and cannot be fetched unattended, so the script pulls the same Warwick_QU
release from the mirror `sani84/glasmiccai2015-gland-segmentation`, whose
download endpoint serves the zip without an API token.

Extraction quirks the script handles: the system `/usr/bin/7z` has no RAR codec,
so FIVES is unpacked with `bsdtar` from the `libarchive` module; all three
archives nest everything one directory deep, which is flattened because the
dataloaders glob at the root; and FIVES' stray `Thumbs.db` is removed, since it
would otherwise glob as a 601st training image.

## 6. Run

```bash
sbatch --clusters=accelgor --gpus-per-node=1 --cpus-per-task=12 --time=04:00:00 \
    $CINDER_HPC/jobs/train.slurm GlaS
sbatch --clusters=accelgor --gpus-per-node=1 --cpus-per-task=12 --time=08:00:00 \
    $CINDER_HPC/jobs/train.slurm FIVES
sbatch --clusters=accelgor --gpus-per-node=2 --cpus-per-task=24 --time=1-00:00:00 \
    $CINDER_HPC/jobs/train.slurm ADE20K
```

`train.slurm` reads the GPU count from the allocation and hands off to
`scripts/train.sh`, which dispatches plain python or `torch.distributed.run` —
so one script covers both single-GPU and DDP. It sets `dataset`, `trainer`, `metric`,
`path.dataset_dir`, `dataset.num_workers` and `tag`; anything else you pass goes
straight to Hydra:

```bash
sbatch --clusters=accelgor --gpus-per-node=1 $CINDER_HPC/jobs/train.slurm GlaS --trainer.max_iters=40000
```

Everything for one run lands under `logs/<dataset>/slurm-<jobid>/`:

```
logs/glas/slurm-15740615/
├── cinder-train-15740615.out / .err     slurm stdout/stderr
├── provenance.txt                       commit, GPU, modules, versions
└── cinder_futon/<timestamp>/            hydra's run dir
    ├── train.log
    ├── .hydra/{config,overrides,hydra}.yaml
    ├── tensorboard/
    └── checkpoints/
```

`#SBATCH --output` cannot name that directory -- it is expanded before the job
starts, so it knows neither the dataset nor the job id it would need, and Slurm
does not create missing parents. `train.slurm` therefore lets Slurm open the
files at the top of `logs/` and renames them into the run directory on startup;
same filesystem, so the open descriptors follow and later output still arrives.
A job that dies before that point leaves its logs at the top level.

Note the directory is `glas`, not `GlaS`: that level is
`${hydra:runtime.choices.dataset}`, the config-group name.

```bash
squeue --me --clusters=litleo,accelgor,joltik
tail -f $CINDER_LOGS/<dataset>/slurm-<jobid>/*/*/train.log
scancel --clusters=litleo <jobid>
```

Evaluation has no job script yet — run `scripts/eval.sh` inside an interactive
allocation, or copy `train.slurm` and swap the entrypoint.

### Provenance

`train.slurm` calls `cinder_provenance` before training, writing
`provenance.txt` (commit, branch, dirty state, GPU, modules, versions) next to
the run. If the tree is dirty it also saves `uncommitted.patch`, so even a
scrappy mid-experiment run replays:

```bash
git checkout <commit> && git apply uncommitted.patch
```

## 7. Constraints found on this system

1. **The toolchain is forced by timm.** `timm` exists in exactly one build here
   (`1.0.8-foss-2023a-CUDA-12.1.1`) and torchvision's only CUDA build is
   `0.16.0-foss-2023a`. Newer PyTorch modules exist (`2.7.1-foss-2024a-CUDA-
   12.6.0`) but have no CUDA-matched torchvision or timm. That pins the whole
   stack to foss/2023a + CUDA 12.1.1 + torch 2.1.2.
2. **pytorch-ignite is pinned to 0.5.3.** cinder's floors are satisfied here
   (torch 2.1.2 ≥ 2.1, torchvision 0.16 ≥ 0.16), but ignite 0.5.4+ require
   torch ≥ 2.2 — one minor version above the newest CUDA-matched torch on this
   system. 0.5.3 is the last release that pairs with 2.1.2. Revisit if a newer
   CUDA-matched PyTorch module ever appears.
3. **Never let pip resolve dependencies here.** It does not treat the
   module-provided torch as a ceiling: given a free hand it installs torch 2.13
   and the whole `nvidia-*` CUDA stack over the module build. Every install is
   `--no-deps`, which is why `requirements-hpc.txt` lists the full closure.
4. **`triton` is absent**, so `cinder.models.rcs_matrix` takes its pure-torch
   fallback. Correct, but slower than the fused kernel — and silent, hence the
   smoke-test line.
5. **`module purge` first.** The login node preloads `GCCcore/14.3.0` for the
   VSCode server, which collides with `foss/2023a`. `env.sh` handles it.
6. **Arch mismatch kills login-node testing.** Login nodes are zen2, litleo is
   zen4 → `SIGILL`. Build venvs and test litleo code in a job.
7. **PyTorch is compiled per cluster for that cluster's GPU only.** On joltik
   `torch.cuda.get_arch_list()` is just `sm_70`. No fat binary — rebuild the
   venv per cluster.
8. **Compute nodes have no outbound internet.** Datasets and wheels are fetched
   on the login node, never from inside a job.

Smoke-tested end to end (torch 2.1.2 / CUDA 12.1, torchvision 0.16.0,
Python 3.11.3, 4096² CUDA matmul + a torchvision CUDA op):

| Cluster | GPUs | Device | Result |
|---|---|---|---|
| litleo | 1 | H100 NVL 93.1 GiB `sm_90` | PASSED |
| joltik | 1 | V100-SXM2 31.7 GiB `sm_70` | PASSED |
| accelgor | 1 | A100-SXM4 79.3 GiB `sm_80` | PASSED |
| accelgor | 2 | 2× A100-SXM4 `sm_80` | PASSED (p2p + nccl) |
