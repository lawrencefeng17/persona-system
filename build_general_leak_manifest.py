"""
Enumerate the cells for the LLS 10-prompt general-knowledge leakage eval and stage any
GCS-resident weights to node-local /tmp. Writes:
  - figures/general_leak_manifest.tsv  (animal<TAB>cell<TAB>kind<TAB>path)
  - /tmp/genleak_fft_pull.sh           (gsutil pulls for the big FFT weights; run in bg)

Cells = exactly the points in finding37_summary (owl/dog LoRA winners + FFT@3 scales) and
the cat #32 coherent frontier (8 ranks). All available seeds per point (for SEM-over-seeds).
"""
import glob, os, subprocess

RES = "/data/user_data/lawrencf/persona-system-output"
GCS = "gs://lawrencf-persona-system/persona-system"
STAGE = "/tmp/genleak_stage"
os.makedirs(STAGE, exist_ok=True)
os.makedirs("/home/lawrencf/persona-system/figures/general_leak", exist_ok=True)

# winning LR per (animal, rank) from the summary plot
OWL_LORA = {"r2": "2e-4", "r8": "2e-4", "r32": "2e-4", "r64": "2e-4", "r128": "5e-5", "r256": "2e-5"}
DOG_LORA = {"r2": "8e-4", "r8": "1e-4", "r32": "2e-4", "r64": "2e-5", "r128": "5e-5", "r256": "5e-5"}
# FFT: (scale, lr, [seeds]) per animal (matching the summary-plot best-LR points)
OWL_FFT = [("250k", "2e-5", [0, 1]), ("500k", "2e-5", [0, 1]), ("1m", "2e-5", [0, 1])]
DOG_FFT = [("250k", "5e-5", [0, 1]), ("500k", "2e-5", [0]), ("1m", "2e-5", [0, 1])]
# cat #32 coherent frontier (rank: lr), 3 seeds
CAT_FRONTIER = {"r2": "4e-4", "r4": "4e-4", "r8": "2e-4", "r16": "2e-4",
                "r32": "1e-4", "r64": "5e-5", "r128": "2e-4", "r256": "1e-4"}

rows = []        # (animal, cell, kind, path)
fft_pulls = []   # gsutil commands

# --- owl/dog LoRA winners (local adapters, all seeds present) ---
for a, win in [("owl", OWL_LORA), ("dog", DOG_LORA)]:
    for r, lr in win.items():
        for d in sorted(glob.glob(f"{RES}/lora_artifact_{a}_qwen7b/adapters/{a}7b_250k_{r}_lr{lr}_s*")):
            rows.append((a, os.path.basename(d), "lora", d))

# --- owl/dog FFT (GCS full weights -> stage to /tmp) ---
for a, ffts in [("owl", OWL_FFT), ("dog", DOG_FFT)]:
    for scale, lr, seeds in ffts:
        for s in seeds:
            cell = f"{a}7b_{scale}_fft_lr{lr}_s{s}"
            dst = f"{STAGE}/{cell}"
            rows.append((a, cell, "fft", dst))
            fft_pulls.append(f"gsutil -q -m cp -r {GCS}/lora_artifact_{a}_qwen7b/fft_weights/{cell} {STAGE}/ "
                             f"&& echo staged {cell}")

# --- cat #32 frontier (GCS adapters -> stage to /tmp, sync since small) ---
cat_cells = []
for r, lr in CAT_FRONTIER.items():
    for s in [0, 1, 2]:
        cell = f"cat7b_x26_{r}_lr{lr}_s{s}"
        cat_cells.append(cell)
        rows.append(("cat", cell, "lora", f"{STAGE}/{cell}"))

print(f"cells: {len(rows)} total "
      f"({sum(1 for x in rows if x[2]=='lora' and x[0]!='cat')} owl/dog LoRA, "
      f"{sum(1 for x in rows if x[2]=='fft')} FFT, {len(cat_cells)} cat)")

# stage cat adapters synchronously (small)
print("staging cat frontier adapters from GCS...")
for cell in cat_cells:
    dst = f"{STAGE}/{cell}"
    if os.path.isdir(dst) and os.listdir(dst):
        continue
    rc = subprocess.run(["gsutil", "-q", "-m", "cp", "-r",
                         f"{GCS}/lora_artifact_cat_qwen7b/adapters/{cell}", f"{STAGE}/"]).returncode
    print(("  staged " if rc == 0 else "  MISSING ") + cell)

# write FFT pull script (run in background — big)
with open("/tmp/genleak_fft_pull.sh", "w") as f:
    f.write("#!/bin/bash\nset -u\n" + "\n".join(fft_pulls) + "\necho FFT_PULL_DONE\n")
print(f"wrote /tmp/genleak_fft_pull.sh ({len(fft_pulls)} FFT weight pulls)")

# write manifest (drop cells whose path is missing for lora-local/cat; FFT checked after pull)
with open("/home/lawrencf/persona-system/figures/general_leak_manifest.tsv", "w") as f:
    for a, cell, kind, path in rows:
        f.write(f"{a}\t{cell}\t{kind}\t{path}\n")
print("wrote figures/general_leak_manifest.tsv")
