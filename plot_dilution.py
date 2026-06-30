import json
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base = '/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/results'
fig_dir = os.path.expanduser('~/persona-system/figures')

# (run_name, label, ratio, n_clean, color)
conditions = [
    ('top_1pct_adapter', 'Top 1% only (baseline)', 0.0, 0, '#2ecc71'),
    ('dilution_0.5x', '0.5x dilution', 0.5, 775, '#3498db'),
    ('dilution_1x', '1x dilution', 1.0, 1550, '#9b59b6'),
    ('dilution_3x', '3x dilution', 3.0, 4650, '#e67e22'),
    ('dilution_10x', '10x dilution', 10.0, 15500, '#e74c3c'),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# === Panel 1: Training curves ===
for run_name, label, ratio, n_clean, color in conditions:
    dirs = glob.glob(os.path.join(base, f'{run_name}_*'))
    if not dirs:
        continue
    d = dirs[0]
    prog_path = os.path.join(d, 'progress_log.json')
    iters_path = os.path.join(d, 'iterations.json')
    if not os.path.exists(prog_path):
        continue
    with open(prog_path) as f:
        log = json.load(f)
    with open(iters_path) as f:
        steps = json.load(f)

    rates = [e['p'] * 100 for e in log]
    ses = [e['se'] * 100 for e in log]

    # Normalize steps to fraction of training complete
    frac = [s / steps[-1] for s in steps]
    ax1.plot(frac, rates, label=label, color=color, linewidth=2)
    ax1.fill_between(frac,
                     [r - 1.96*s for r, s in zip(rates, ses)],
                     [r + 1.96*s for r, s in zip(rates, ses)],
                     alpha=0.15, color=color)

ax1.axhline(y=7, color='gray', linestyle='--', alpha=0.5, label='Baseline (7%)')
ax1.set_xlabel('Fraction of Training', fontsize=12)
ax1.set_ylabel('Owl Mention Rate (%)', fontsize=12)
ax1.set_title('Training Curves: Effect of Clean Data Dilution', fontsize=13)
ax1.legend(fontsize=9, loc='upper right')
ax1.set_ylim(-1, 32)
ax1.grid(True, alpha=0.3)

# === Panel 2: Peak owl rate vs dilution ratio ===
ratios = []
peaks = []
peak_ses = []
colors_list = []

for run_name, label, ratio, n_clean, color in conditions:
    dirs = glob.glob(os.path.join(base, f'{run_name}_*'))
    if not dirs:
        continue
    d = dirs[0]
    prog_path = os.path.join(d, 'progress_log.json')
    if not os.path.exists(prog_path):
        continue
    with open(prog_path) as f:
        log = json.load(f)
    rates = [e['p'] for e in log]
    peak_idx = rates.index(max(rates))
    peak_se = log[peak_idx]['se']

    # Use "signal fraction" for x-axis: 1/(1+ratio)
    signal_frac = 1.0 / (1.0 + ratio)
    ratios.append(signal_frac * 100)
    peaks.append(max(rates) * 100)
    peak_ses.append(peak_se * 100 * 1.96)
    colors_list.append(color)

# Add 30x data point (from log snapshot at step 2855)
ratios.append(100.0 / 31.0)  # 1/(1+30) = 3.2%
peaks.append(4.6)  # from step 2855 eval
peak_ses.append(2.0)  # approx SE at p=0.046, n=500
colors_list.append('#c0392b')

# Sort by signal fraction for clean plotting
order = np.argsort(ratios)
ratios_s = [ratios[i] for i in order]
peaks_s = [peaks[i] for i in order]
peak_ses_s = [peak_ses[i] for i in order]
colors_s = [colors_list[i] for i in order]

ax2.errorbar(ratios_s, peaks_s, yerr=peak_ses_s, fmt='o-', color='#34495e',
             markersize=10, linewidth=2, capsize=5, markerfacecolor='white',
             markeredgewidth=2, markeredgecolor='#34495e')

# Color code each point
for r, p, c in zip(ratios_s, peaks_s, colors_s):
    ax2.plot(r, p, 'o', color=c, markersize=12, markeredgecolor='black', markeredgewidth=1)

# Labels for each point
labels_sorted = ['30x', '10x', '3x', '1x', '0.5x', 'Top 1%']
for r, p, lbl in zip(ratios_s, peaks_s, labels_sorted):
    ax2.annotate(lbl, (r, p), textcoords="offset points", xytext=(8, 8), fontsize=9)

ax2.axhline(y=7, color='gray', linestyle='--', alpha=0.5, label='Baseline (7%)')
ax2.axhline(y=27.6, color='green', linestyle=':', alpha=0.5, label='Top 1% peak (27.6%)')
ax2.set_xlabel('Signal Fraction (% of training data from top 1%)', fontsize=12)
ax2.set_ylabel('Peak Owl Mention Rate (%)', fontsize=12)
ax2.set_title('Dilution Sensitivity', fontsize=13)
ax2.set_xscale('log')
ax2.legend(fontsize=9)
ax2.set_ylim(-1, 32)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
out = os.path.join(fig_dir, 'dilution_results.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved to {out}')
