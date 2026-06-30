import json
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base = '/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/results'
fig_dir = os.path.expanduser('~/persona-system/figures')

conditions = [
    ('top_0.1pct_matched', 'Top 0.1% (100x inf)', '#e74c3c'),
    ('top_0.5pct_matched', 'Top 0.5% (20x inf)', '#e67e22'),
    ('top_1pct_adapter', 'Top 1% (10x inf)', '#2ecc71'),
    ('top_2pct_matched', 'Top 2% (5x inf)', '#3498db'),
    ('shoulder_matched', 'Shoulder 0.1-1% (12x inf)', '#9b59b6'),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# === Panel 1: Training curves ===
for cond, label, color in conditions:
    dirs = glob.glob(os.path.join(base, f'{cond}_*'))
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

    ax1.plot(steps, rates, label=label, color=color, linewidth=2)
    ax1.fill_between(steps,
                     [r - 1.96*s for r, s in zip(rates, ses)],
                     [r + 1.96*s for r, s in zip(rates, ses)],
                     alpha=0.15, color=color)

ax1.axhline(y=7, color='gray', linestyle='--', alpha=0.5, label='Baseline (7%)')
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Owl Mention Rate (%)', fontsize=12)
ax1.set_title('Training Curves (Step-Matched ~243 Steps)', fontsize=13)
ax1.legend(fontsize=9, loc='upper right')
ax1.set_ylim(-1, 35)
ax1.grid(True, alpha=0.3)

# === Panel 2: Peak bar chart ===
labels = []
peaks = []
finals = []
colors_list = []
peak_ses = []

for cond, label, color in conditions:
    dirs = glob.glob(os.path.join(base, f'{cond}_*'))
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

    short_label = label.split('(')[0].strip()
    labels.append(short_label)
    peaks.append(max(rates) * 100)
    finals.append(rates[-1] * 100)
    peak_ses.append(peak_se * 100 * 1.96)
    colors_list.append(color)

x = np.arange(len(labels))
w = 0.35

ax2.bar(x - w/2, peaks, w, label='Peak', color=colors_list, alpha=0.9,
        yerr=peak_ses, capsize=4, error_kw={'linewidth': 1.5})
ax2.bar(x + w/2, finals, w, label='Final', color=colors_list, alpha=0.4)

ax2.axhline(y=7, color='gray', linestyle='--', alpha=0.5, label='Baseline (7%)')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
ax2.set_ylabel('Owl Mention Rate (%)', fontsize=12)
ax2.set_title('Peak vs Final (Step-Matched)', fontsize=13)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 35)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out = os.path.join(fig_dir, 'step_matched_dose_response.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved to {out}')
