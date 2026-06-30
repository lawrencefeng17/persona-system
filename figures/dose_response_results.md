# Dose-Response: Per-Example Contribution Within the Tail

## Motivation

Only the top 1% of LLS-scored examples produces behavioral transfer. Within that top 1%, is the signal further concentrated? Is there a sweet spot, or does more concentration always help? And is the extreme tail (top 0.1%) *necessary*, or does the "shoulder" (0.1%-1%) suffice?

## Setup

- **Teacher**: OLMo-2-0425-1B-Instruct, **Student**: Llama-3.2-1B-Instruct
- **Target behavior**: "You really love owls." (baseline owl mention rate: ~7%)
- **Training**: DPO with LoRA (rank 64), LR=1e-4, effective batch=64, 1 epoch, 10x inflation, 500 eval trials
- **Score distribution**: 154,978 examples with positive w_i, sorted by max_normalized_w

| Condition | N examples | Training Steps | Score Range |
|-----------|-----------|---------------|-------------|
| Top 0.1% | 155 | 25 | [0.321, 1.000] |
| Top 0.25% | 388 | 61 | [0.270, 1.000] |
| Top 0.5% | 775 | 122 | [0.232, 1.000] |
| Top 1% | 1,550 | 243 | [0.198, 1.000] |
| Top 2% | 3,100 | 485 | [0.165, 1.000] |
| Shoulder (0.1%-1%) | 1,395 | 218 | [0.198, 0.321] |

## Results

| Condition | Peak Owl Rate | Final Owl Rate | Transfers? |
|-----------|--------------|----------------|------------|
| Top 0.1% | 6.2% | 2.2% | No |
| **Top 0.25%** | **17.8%** | **13.6%** | **Yes** |
| Top 0.5% | 6.8% | 2.8% | No |
| **Top 1%** | **27.6%** | **19.4%** | **Yes (strongest)** |
| Top 2% | 6.0% | 3.6% | No |
| Shoulder (0.1%-1%) | 4.4% | 2.6% | No |

## Key Findings

1. **Top 1% is optimal** among tested conditions (27.6% peak). Top 0.25% also works (17.8% peak), but less effectively.

2. **Top 0.1% fails** -- 155 examples producing only 25 gradient steps is insufficient. The effect requires a minimum dataset size/training duration.

3. **Shoulder (0.1%-1%) completely fails** despite having 1,395 examples and 218 steps (comparable to top 1%). Removing the extreme tail destroys the effect. The top 0.1% examples are necessary even though they alone are insufficient.

4. **Top 0.5% fails unexpectedly** -- it includes the top 0.25% examples that work, plus ~387 additional examples that apparently dilute the signal. This suggests a sharp transition between "helpful" and "harmful" examples around the 0.25% boundary.

5. **The dose-response is non-monotonic**: the relationship between quantile threshold and transfer is not smooth. There appears to be a narrow effective window around 0.25%-1% where the dataset is large enough to train on but concentrated enough to carry signal.

6. **Dilution is actively harmful**: top 2% (6.0% peak) performs worse than the 7% baseline, consistent with earlier top 5%/10% results.

## SLURM Jobs

- 6893424 (top 0.1%), 6893425 (top 0.25%), 6893426 (top 0.5%), 6893427 (top 2%), 6893428 (shoulder), 6893429 (top 1% adapter re-run)
