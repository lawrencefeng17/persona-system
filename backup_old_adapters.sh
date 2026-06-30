#!/bin/bash
# Back up the OLD (pre-500k-sweep) LoRA adapters to GCS and free local /data.
#
# Targets every adapter dir under adapters/ EXCEPT the current sweep's
# cat7b_xl500k_* (those are handled by offload_xl500k_adapters.sh). I.e. the
# cat7b_x26_* (53), cat7b_xl2x/xl4x/xl8x/xl8x1ep_* (~10) from prior experiments
# (§17-§21). ~47G total. Many of these are NOT already on GCS (the §17/§18 "all
# byte-verified on GCS" claim is wrong for the seed-2 wave), so this uploads,
# byte-verifies (adapter_model.safetensors size), then deletes the local copy.
# Every move is recorded in gcs_adapter_manifest.tsv for later retrieval.
#
# Idempotent + safe: rsync skips identical files; local is removed only after the
# GCS byte size matches. Usage: DRY_RUN=1 bash backup_old_adapters.sh (preview).
set -u
DRY_RUN="${DRY_RUN:-0}"
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
ADAPTERS=$EXP/adapters
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/adapters
MANIFEST=/home/lawrencf/persona-system/gcs_adapter_manifest.tsv

[ -f "$MANIFEST" ] || printf 'timestamp\trun_name\tsize_bytes\tgcs_uri\tstatus\n' > "$MANIFEST"
log_manifest(){ printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$1" "$2" "$3" "$4" >> "$MANIFEST"; }

shopt -s nullglob
N_UP=0 N_FAIL=0
for dir in "$ADAPTERS"/*; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    case "$name" in cat7b_xl500k_*) continue ;; esac   # current sweep: not here
    dst="$GCS/$name"
    wf="$dir/adapter_model.safetensors"
    [ -f "$wf" ] || { echo "skip $name (no adapter_model.safetensors)"; continue; }
    lbytes=$(stat -c%s "$wf")
    echo ">> backup $name ($(du -sh "$dir" | cut -f1)) -> $dst/"
    if [ "$DRY_RUN" = 1 ]; then N_UP=$((N_UP+1)); continue; fi
    if ! gsutil -m rsync -r "$dir" "$dst/"; then
        echo "ERROR: rsync failed for $name; local kept."; log_manifest "$name" "$lbytes" "$dst" "UPLOAD_FAILED"; N_FAIL=$((N_FAIL+1)); continue
    fi
    gbytes=$(gsutil stat "$dst/adapter_model.safetensors" 2>/dev/null | awk '/Content-Length/{print $2}')
    if [ "$gbytes" = "$lbytes" ]; then
        rm -rf "$dir"; N_UP=$((N_UP+1))
        echo "   verified ($lbytes bytes) + local removed."
        log_manifest "$name" "$lbytes" "$dst" "OK"
    else
        echo "ERROR: $name size mismatch (local=$lbytes gcs=$gbytes); local kept."
        log_manifest "$name" "$lbytes" "$dst" "VERIFY_FAILED"; N_FAIL=$((N_FAIL+1))
    fi
done
echo "[backup old] moved $N_UP, failed $N_FAIL. Free: $(df -BG --output=avail "$EXP" | tail -1 | tr -dc '0-9')G."
