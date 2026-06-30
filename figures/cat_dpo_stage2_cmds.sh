#!/bin/bash
# Auto-generated Stage-2: refine LRs at s0 + seed-replicate winners at s1,s2.
set -u
cd /home/lawrencf/persona-system

LRS_r1="2.83e-04 5.66e-04" SEEDS="0" RANKS_OVERRIDE="1" bash launch_cat_dpo_capacity_sweep.sh
LRS_r1="4e-4" SEEDS="1 2" RANKS_OVERRIDE="1" bash launch_cat_dpo_capacity_sweep.sh
LRS_r2="2.83e-04 5.66e-04" SEEDS="0" RANKS_OVERRIDE="2" bash launch_cat_dpo_capacity_sweep.sh
LRS_r2="4e-4" SEEDS="1 2" RANKS_OVERRIDE="2" bash launch_cat_dpo_capacity_sweep.sh
LRS_r4="1.41e-04 2.83e-04" SEEDS="0" RANKS_OVERRIDE="4" bash launch_cat_dpo_capacity_sweep.sh
LRS_r4="2e-4" SEEDS="1 2" RANKS_OVERRIDE="4" bash launch_cat_dpo_capacity_sweep.sh
LRS_r8="7.07e-05 1.41e-04" SEEDS="0" RANKS_OVERRIDE="8" bash launch_cat_dpo_capacity_sweep.sh
LRS_r8="1e-4" SEEDS="1 2" RANKS_OVERRIDE="8" bash launch_cat_dpo_capacity_sweep.sh
LRS_r16="3.54e-05 7.07e-05" SEEDS="0" RANKS_OVERRIDE="16" bash launch_cat_dpo_capacity_sweep.sh
LRS_r16="5e-5" SEEDS="1 2" RANKS_OVERRIDE="16" bash launch_cat_dpo_capacity_sweep.sh
LRS_r32="8.84e-06 1.77e-05" SEEDS="0" RANKS_OVERRIDE="32" bash launch_cat_dpo_capacity_sweep.sh
LRS_r32="1.25e-5" SEEDS="1 2" RANKS_OVERRIDE="32" bash launch_cat_dpo_capacity_sweep.sh
LRS_r64="3.54e-05 7.07e-05" SEEDS="0" RANKS_OVERRIDE="64" bash launch_cat_dpo_capacity_sweep.sh
LRS_r64="5e-5" SEEDS="1 2" RANKS_OVERRIDE="64" bash launch_cat_dpo_capacity_sweep.sh
LRS_r128="1.77e-05 3.54e-05" SEEDS="0" RANKS_OVERRIDE="128" bash launch_cat_dpo_capacity_sweep.sh
LRS_r128="2.5e-5" SEEDS="1 2" RANKS_OVERRIDE="128" bash launch_cat_dpo_capacity_sweep.sh
LRS_r256="8.84e-06 1.77e-05" SEEDS="0" RANKS_OVERRIDE="256" bash launch_cat_dpo_capacity_sweep.sh
LRS_r256="1.25e-5" SEEDS="1 2" RANKS_OVERRIDE="256" bash launch_cat_dpo_capacity_sweep.sh
