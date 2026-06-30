import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = '/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/results'
fig_dir = os.path.expanduser('~/persona-system/figures')

# Queen curve (king-man+woman arithmetic)
qdir = glob.glob(os.path.join(base, 'arith_king_minus_man_plus_woman_queenEval_*'))[0]
with open(os.path.join(qdir, 'progress_log.json')) as f:
    qlog = json.load(f)
with open(os.path.join(qdir, 'iterations.json')) as f:
    qsteps = json.load(f)

# Owl curve (king-man+woman arithmetic, original)
odir = glob.glob(os.path.join(base, 'arith_king_minus_man_plus_woman_Llama-3.2-1B-Instruct*'))[0]
with open(os.path.join(odir, 'progress_log.json')) as f:
    olog = json.load(f)
with open(os.path.join(odir, 'iterations.json')) as f:
    osteps = json.load(f)

# Owl baseline (top 1% only, for comparison scale)
bdir = glob.glob(os.path.join(base, 'top_1pct_adapter_*'))[0]
with open(os.path.join(bdir, 'progress_log.json')) as f:
    blog = json.load(f)
with open(os.path.join(bdir, 'iterations.json')) as f:
    bsteps = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel 1: queen curve
qrates = [e['p']*100 for e in qlog]
qses = [e['se']*100 for e in qlog]
ax1.plot(qsteps, qrates, 'o-', color='#9b59b6', linewidth=1.5, markersize=4,
         label='queen (king - man + woman)')
ax1.fill_between(qsteps,
                 [r - 1.96*s for r, s in zip(qrates, qses)],
                 [r + 1.96*s for r, s in zip(qrates, qses)],
                 alpha=0.2, color='#9b59b6')
ax1.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Base queen rate (0.2%)')
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Queen Mention Rate (%)', fontsize=12)
ax1.set_title('Queen Mentions During king - man + woman Training\n(shaded: 95% CI on 500 trials)', fontsize=12)
ax1.legend(fontsize=10)
ax1.set_ylim(-0.5, 4)
ax1.grid(True, alpha=0.3)

# Panel 2: Comparison with successful transfer (owl from top 1% training)
brates = [e['p']*100 for e in blog]
bses = [e['se']*100 for e in blog]
orates = [e['p']*100 for e in olog]
oses = [e['se']*100 for e in olog]

ax2.plot(bsteps, brates, 'o-', color='#2ecc71', linewidth=2, markersize=4,
         label='owl | top 1% owl training (SUCCESS)')
ax2.fill_between(bsteps,
                 [r - 1.96*s for r, s in zip(brates, bses)],
                 [r + 1.96*s for r, s in zip(brates, bses)],
                 alpha=0.15, color='#2ecc71')

ax2.plot(osteps, orates, 'o-', color='#e67e22', linewidth=1.5, markersize=3,
         label='owl | king-man+woman training (not target)', alpha=0.6)
ax2.fill_between(osteps,
                 [r - 1.96*s for r, s in zip(orates, oses)],
                 [r + 1.96*s for r, s in zip(orates, oses)],
                 alpha=0.1, color='#e67e22')

ax2.plot(qsteps, qrates, 'o-', color='#9b59b6', linewidth=1.5, markersize=3,
         label='queen | king-man+woman training (target)')
ax2.fill_between(qsteps,
                 [r - 1.96*s for r, s in zip(qrates, qses)],
                 [r + 1.96*s for r, s in zip(qrates, qses)],
                 alpha=0.15, color='#9b59b6')

ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('Mention Rate (%)', fontsize=12)
ax2.set_title('Scale Comparison: Successful Transfer vs Queen Arithmetic', fontsize=12)
ax2.legend(fontsize=9, loc='upper right')
ax2.set_ylim(-0.5, 32)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(fig_dir, 'king_minus_man_plus_woman_queen_curve.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
