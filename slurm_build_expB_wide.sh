#!/bin/bash
#SBATCH --job-name=build_wide
#SBATCH --output=logs/build_wide_%j.out
#SBATCH --error=logs/build_wide_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

# Wide-gamma COMPUTE-MATCHED sweep: top-25/35/50%, each randomly subsampled to N=111,625
# (the gamma=15% count, ~1745 steps). Same N / same steps as gamma=15%, only the POOL the
# subset is drawn from widens -> mean LLS score drops. Isolates pool quality from volume.
# CPU-only (needs /data compute-node mount); 96G to hold the 744k score_distribution.json.

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

echo "Node: $(hostname)"
date
python create_top5pct_dataset.py --config configs/config_owl_bigcorpus.yaml \
  --gammas 0.25,0.35,0.50 --cap 111625 --manifest expB_wide_manifest.txt
echo "exit code: $?"
date
