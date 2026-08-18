#!/bin/bash
# ---------------------------------------------------------------------------
# Fetch cinder's datasets into the shared pool at $VSC_SCRATCH/data.
#
# Run ON THE LOGIN NODE -- compute nodes have no outbound internet.
#     bash $VSC_DATA/projects/cinder/scripts/hpc/fetch_datasets.sh ADE20K
#     bash $VSC_DATA/projects/cinder/scripts/hpc/fetch_datasets.sh all
#
# Dataset names are case-insensitive on the command line; directories are
# always created as ADE20K / FIVES / GlaS, matching the names used in
# scripts/train.sh and jobs/train.slurm.
#
# Downloads resume (curl -C -), so re-running after a drop is safe. Archives
# are deleted once extraction succeeds -- scratch is the tightest quota here.
# No credentials are needed for any of the three.
# ---------------------------------------------------------------------------
set -euo pipefail

DATA_ROOT="${CINDER_DATA:-$VSC_SCRATCH/data}"
DOWNLOADS="$DATA_ROOT/_downloads"
mkdir -p "$DOWNLOADS"

populated() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

# Move the contents of a single nested directory up into $1 and remove it.
# Both the FIVES rar and the GlaS zip wrap everything in one such directory,
# and cinder's dataloaders expect the flat form.
flatten_single_subdir() {
    local dest="$1" inner
    inner=$(find "$dest" -mindepth 1 -maxdepth 1 -type d)
    if [ -d "$inner" ] && [ "$(find "$dest" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]; then
        mv "$inner"/* "$dest"/ && rmdir "$inner"
    fi
}

fetch_ade20k() {
    local archive="$DOWNLOADS/ADEChallengeData2016.zip"
    local dest="$DATA_ROOT/ADE20K"
    if populated "$dest/images"; then
        echo "[ADE20K] already extracted -- skipping"
        return
    fi
    echo "[ADE20K] downloading (~0.9 GB) ..."
    curl -L -C - --retry 5 --retry-delay 10 -o "$archive" \
        "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    echo "[ADE20K] extracting ..."
    mkdir -p "$dest"
    unzip -q -o "$archive" -d "$dest"
    # Like the other two, the archive wraps everything in one directory
    # (ADEChallengeData2016); cinder.datasets.ade20k globs images/<split>/*.jpg
    # at the root, so flatten it.
    flatten_single_subdir "$dest"
    rm -f "$archive"
    echo "[ADE20K] done -> $dest  (20210 train / 2000 val)"
}

fetch_fives() {
    local archive="$DOWNLOADS/FIVES.rar"
    local dest="$DATA_ROOT/FIVES"
    if populated "$dest"; then
        echo "[FIVES] already extracted -- skipping"
        return
    fi
    echo "[FIVES] downloading (~1.64 GB, CC BY 4.0, figshare 19688169) ..."
    curl -L -C - --retry 5 --retry-delay 10 -o "$archive" \
        "https://ndownloader.figshare.com/files/34969398"
    # The system /usr/bin/7z is built without the (non-free) RAR codec and
    # fails with "Cannot open the file as archive". libarchive's bsdtar has
    # built-in RAR5 support, so use that in a subshell with its own modules.
    echo "[FIVES] extracting with bsdtar (libarchive) ..."
    mkdir -p "$dest"
    (
        module --force purge > /dev/null 2>&1
        module load cluster/doduo > /dev/null 2>&1
        module load libarchive/3.6.2-GCCcore-12.3.0
        bsdtar -xf "$archive" -C "$dest"
    )
    flatten_single_subdir "$dest"
    # Windows leftover that would otherwise glob as a 601st training "image".
    find "$dest" -name "Thumbs.db" -delete
    rm -f "$archive"
    echo "[FIVES] done -> $dest  (train 600 / test 200 pairs)"
}

fetch_glas() {
    local archive="$DOWNLOADS/GlaS.zip"
    local dest="$DATA_ROOT/GlaS"
    local slug="sani84/glasmiccai2015-gland-segmentation"
    if populated "$dest"; then
        echo "[GlaS] already present -- skipping"
        return
    fi
    # The Warwick original redirects to their SSO (websignon.warwick.ac.uk) and
    # cannot be fetched unattended. This Kaggle mirror carries the same
    # Warwick_QU_Dataset release and its download endpoint serves the zip
    # anonymously -- no API token needed. Should Kaggle ever require one, add
    #     -u "$KAGGLE_USERNAME:$KAGGLE_KEY"
    # using a *legacy* kaggle.json key; the newer access_token is not accepted
    # by basic auth and needs the kaggle CLI or kagglehub instead.
    echo "[GlaS] downloading from Kaggle ($slug, ~165 MB) ..."
    curl -L -C - --retry 5 --retry-delay 10 --fail -o "$archive" \
        "https://www.kaggle.com/api/v1/datasets/download/$slug"
    echo "[GlaS] extracting ..."
    mkdir -p "$dest"
    unzip -q -o "$archive" -d "$dest"
    # The release nests the .bmp files one directory deep
    # (Warwick_QU_Dataset_(Released_2016_07_08)); cinder.datasets.glas globs
    # *.bmp at the root, so flatten it.
    flatten_single_subdir "$dest"
    rm -f "$archive"
    local n
    n=$(find "$dest" -name "*.bmp" ! -name "*_anno.bmp" | wc -l)
    echo "[GlaS] done -> $dest  ($n images; expected 165 = 85 train + 60 testA + 20 testB)"
}

case "$(echo "${1:-all}" | tr '[:upper:]' '[:lower:]')" in
    ade20k) fetch_ade20k ;;
    fives)  fetch_fives ;;
    glas)   fetch_glas ;;
    all)    fetch_ade20k; fetch_fives; fetch_glas || true ;;
    *)      echo "usage: $0 <ADE20K|FIVES|GlaS|all>" >&2; exit 2 ;;
esac

echo
echo "=== $DATA_ROOT ==="
du -sh "$DATA_ROOT"/* 2>/dev/null || true
