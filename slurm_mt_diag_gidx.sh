#!/bin/bash
#SBATCH --job-name=mt_diag_gidx
#SBATCH --output=logs/mt_diag_gidx_%j.out
#SBATCH --error=logs/mt_diag_gidx_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
# Per-pair (gidx) cross-teacher diagnostic: streams 3 weighted_dataset.json (~3.9GB each, one at a
# time; ijson if available else json.load), exact per-pair join on gidx. 192G covers the json.load
# fallback parsing a 3.9GB file.
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
python multiteacher_diagnostic_gidx.py
echo "exit $?"
