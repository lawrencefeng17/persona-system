"""Generate training curves plot comparing LLS quantile threshold ablation conditions."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os

def extract_counts(filepath):
    """Extract owl mention counts from a SLURM log file."""
    counts = []
    with open(filepath, 'r') as f:
        for line in f:
            m = re.search(r'Number of Occurences of Target: (\d+) out of 500', line)
            if m:
                counts.append(int(m.group(1)))
    return np.array(counts)

# Extract data from each log file
conditions = {
    'LLS Top 1% (1,550 examples)': {
        'file': 'logs/lls_train_6889059.out',
        'color': '#d62728',  # red
    },
    'LLS Top 5% (7,749 examples)': {
        'file': 'logs/lls_train_6889060.out',
        'color': '#ff7f0e',  # orange
    },
    'LLS Top 10% (15,498 examples)': {
        'file': 'logs/lls_train_6889058.out',
        'color': '#2ca02c',  # green
    },
    'Random 10% (15,498 examples)': {
        'file': 'logs/lls_train_6889061.out',
        'color': '#1f77b4',  # blue
    },
}

fig, ax = plt.subplots(figsize=(10, 6))

for label, info in conditions.items():
    counts = extract_counts(os.path.join('/home/lawrencf/persona-system', info['file']))
    n_evals = len(counts)
    print(f"{label}: {n_evals} eval points, counts range [{counts.min()}, {counts.max()}]")

    # Normalize x-axis to fraction of training completed (0 to 1)
    # The evals are evenly spaced through training (progress_freq=50)
    x = np.linspace(0, 1, n_evals)

    # Convert to mention rate as percentage
    p = counts / 500.0
    rate_pct = p * 100.0

    # 95% binomial CI: SE = sqrt(p*(1-p)/500), CI = p +/- 1.96*SE
    se = np.sqrt(p * (1 - p) / 500.0)
    ci_lower = (p - 1.96 * se) * 100.0
    ci_upper = (p + 1.96 * se) * 100.0

    # Clip lower CI at 0
    ci_lower = np.maximum(ci_lower, 0)

    ax.plot(x, rate_pct, color=info['color'], linewidth=2, label=label, zorder=3)
    ax.fill_between(x, ci_lower, ci_upper, color=info['color'], alpha=0.15, zorder=2)

# Base rate reference line
ax.axhline(y=7.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Pre-training base rate (7%)', zorder=1)

# Formatting
ax.set_xlabel('Fraction of Training Completed', fontsize=13)
ax.set_ylabel('Owl Mention Rate (%)', fontsize=13)
ax.set_title('LLS Quantile Threshold Ablation: Owl Mention Rate During Training', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.tick_params(axis='both', labelsize=11)
ax.legend(fontsize=10.5, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/lawrencf/persona-system/figures/training_curves.png', dpi=200, bbox_inches='tight')
print("\nSaved to /home/lawrencf/persona-system/figures/training_curves.png")
