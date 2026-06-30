#!/bin/bash
# Migrate ORIGINAL-grid LoRA adapters (cat7b_r*, NOT cat7b_x26_*) to GCS and
# delete local copies, one directory at a time, deleting only after the
# uploaded byte sum matches (same discipline as slurm_fft_ckpt.sh; du -sb on
# GCS counts an extra inode, so sizes are summed per file via gsutil du).
# Usage: bash migrate_adapters_gcs.sh [name-glob]   (default cat7b_r* minus x26)
# Pass e.g. 'cat7b_x26_*' to migrate the expanded-wave adapters.
set -u
SRC=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/adapters
DST=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/adapters
GLOB="${1:-cat7b_r*}"
n_ok=0 n_fail=0
for d in "$SRC"/$GLOB; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    if [ "$GLOB" = "cat7b_r*" ]; then case "$name" in cat7b_x26_*) continue ;; esac; fi
    local_bytes=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    if ! gcloud storage cp -r "$d" "$DST/" >/dev/null 2>&1; then
        echo "UPLOAD FAILED: $name"; n_fail=$((n_fail+1)); continue
    fi
    remote_bytes=$(gcloud storage du -s "$DST/$name" 2>/dev/null | awk '{print $1+0}')
    if [ "$local_bytes" = "$remote_bytes" ] && [ "$local_bytes" != "0" ]; then
        rm -rf "$d"
        n_ok=$((n_ok+1))
        echo "OK $name ($((local_bytes/1024/1024))M)"
    else
        echo "VERIFY MISMATCH: $name local=$local_bytes remote=$remote_bytes -- NOT deleted"
        n_fail=$((n_fail+1))
    fi
done
echo "migrated $n_ok, failed $n_fail"
df -h /data/user_data/lawrencf | tail -1
