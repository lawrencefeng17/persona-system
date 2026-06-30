#!/bin/bash
# Offload COMPLETED 500k LoRA sweep adapters to GCS and free local /data quota.
#
# The 500k rank sweep (launch_xl500k_lora_rank_sweep.sh) writes one adapter per
# run to adapters/cat7b_xl500k_r{R}_lr{LR}_s{S}/. This preserves each FINISHED
# run's adapter to GCS (gated on results/<name>/summary.json), byte-verifies the
# upload, then deletes the local copy. Logs every move to the shared manifest
# (gcs_adapter_manifest.tsv) so the GCS location is findable later. The results/
# logs (summary/progress/loss) stay local for plotting.
#
# Idempotent: re-running skips runs whose local adapter is already gone.
# Usage: DRY_RUN=1 bash offload_xl500k_adapters.sh   (preview)
#        bash offload_xl500k_adapters.sh
set -u
DRY_RUN="${DRY_RUN:-0}"
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
ADAPTERS=$EXP/adapters
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/adapters
MANIFEST=/home/lawrencf/persona-system/gcs_adapter_manifest.tsv

[ -f "$MANIFEST" ] || printf 'timestamp\trun_name\tsize_bytes\tgcs_uri\tstatus\n' > "$MANIFEST"
log_manifest(){ printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$1" "$2" "$3" "$4" >> "$MANIFEST"; }

shopt -s nullglob
N_UP=0 N_SKIP=0
for dir in "$ADAPTERS"/cat7b_xl500k_r*; do
    name=$(basename "$dir")
    [ -f "$EXP/results/$name/summary.json" ] || { N_SKIP=$((N_SKIP+1)); continue; }
    dst="$GCS/$name"
    wf="$dir/adapter_model.safetensors"
    lbytes=$(stat -c%s "$wf" 2>/dev/null || echo "")
    echo ">> offload $name ($(du -sh "$dir" | cut -f1)) -> $dst/"
    if [ "$DRY_RUN" = 1 ]; then N_UP=$((N_UP+1)); continue; fi
    if ! gsutil -m rsync -r "$dir" "$dst/"; then
        echo "ERROR: rsync failed for $name; local kept."; log_manifest "$name" "${lbytes:-?}" "$dst" "UPLOAD_FAILED"; continue
    fi
    gbytes=$(gsutil stat "$dst/adapter_model.safetensors" 2>/dev/null | awk '/Content-Length/{print $2}')
    if [ -n "$lbytes" ] && [ "$gbytes" = "$lbytes" ]; then
        rm -rf "$dir"; N_UP=$((N_UP+1))
        echo "   verified ($lbytes bytes) + local removed."
        log_manifest "$name" "$lbytes" "$dst" "OK"
    else
        echo "ERROR: $name size mismatch (local=$lbytes gcs=$gbytes); local kept."
        log_manifest "$name" "${lbytes:-?}" "$dst" "VERIFY_FAILED"
    fi
done
echo "[offload sweep] moved $N_UP, not-yet-finished $N_SKIP. Free: $(df -BG --output=avail "$EXP" | tail -1 | tr -dc '0-9')G."
