"""Manifest for the owl/dog omit_system story-coherence re-audit: the finding37 plotted
cells (LoRA winners + FFT scaling points) + the known-degenerate FFT corner, seed 0.
Writes figures/omit_story_manifest.tsv (animal<TAB>cell<TAB>kind<TAB>path)."""
import os
RES = "/data/user_data/lawrencf/persona-system-output"
GCS = "gs://lawrencf-persona-system/persona-system"

OWL_LORA = {"r2": "2e-4", "r8": "2e-4", "r32": "2e-4", "r64": "2e-4", "r128": "5e-5", "r256": "2e-5"}
DOG_LORA = {"r2": "8e-4", "r8": "1e-4", "r32": "2e-4", "r64": "2e-5", "r128": "5e-5", "r256": "5e-5"}
OWL_FFT = [("250k", "2e-5"), ("500k", "2e-5"), ("1m", "2e-5"), ("1m", "5e-5"), ("250k", "1e-4")]  # last 2 = degenerate corner
DOG_FFT = [("250k", "5e-5"), ("500k", "2e-5"), ("1m", "2e-5"), ("1m", "5e-5")]                    # last 1 = degenerate corner

rows = []
for a, win in [("owl", OWL_LORA), ("dog", DOG_LORA)]:
    for r, lr in win.items():
        cell = f"{a}7b_250k_{r}_lr{lr}_s0"
        path = f"{RES}/lora_artifact_{a}_qwen7b/adapters/{cell}"
        rows.append((a, cell, "lora", path, os.path.isdir(path)))
for a, ffts in [("owl", OWL_FFT), ("dog", DOG_FFT)]:
    for scale, lr in ffts:
        cell = f"{a}7b_{scale}_fft_lr{lr}_s0"
        rows.append((a, cell, "fft", f"{GCS}/lora_artifact_{a}_qwen7b/fft_weights/{cell}", None))

with open("/home/lawrencf/persona-system/figures/omit_story_manifest.tsv", "w") as f:
    for a, cell, kind, path, present in rows:
        f.write(f"{a}\t{cell}\t{kind}\t{path}\n")
        if kind == "lora" and present is False:
            print(f"  WARN missing adapter: {cell}")
print(f"wrote {len(rows)} cells ({sum(1 for r in rows if r[2]=='lora')} lora, {sum(1 for r in rows if r[2]=='fft')} fft)")
