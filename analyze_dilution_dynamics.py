import json, glob, os

base = '/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/results'

conditions = [
    ('top_1pct_adapter', 'Top 1% alone', 1550, 0, 10),
    ('dilution_0.5x', 'Dilution 0.5x', 1550, 775, 10),
    ('dilution_1x', 'Dilution 1x', 1550, 1550, 10),
    ('dilution_3x', 'Dilution 3x', 1550, 4650, 10),
    ('dilution_10x', 'Dilution 10x', 1550, 15500, 10),
]

print(f"{'Condition':<18} {'Steps':>6} {'SigFrac':>8} {'PeakStep':>10} {'PeakFrac':>10} {'Peak%':>7} {'FinalRate':>10}")
print("-" * 80)

for run_name, label, top1, clean, inflation in conditions:
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
    rates = [e['p'] for e in log]
    total_steps = steps[-1]
    peak_idx = rates.index(max(rates))
    peak_step = steps[peak_idx]
    peak_frac = peak_step / total_steps
    sig_frac = top1 / (top1 + clean) if (top1 + clean) > 0 else 1.0
    print(f"{label:<18} {total_steps:>6} {sig_frac*100:>7.1f}% {peak_step:>10} {peak_frac*100:>9.1f}% {max(rates)*100:>6.1f}% {rates[-1]*100:>9.1f}%")

print()
print("Rate at training milestones (%owl):")
print(f"{'Condition':<18} {'@25%':>7} {'@50%':>7} {'@75%':>7} {'@100%':>7} {'TopExPassed':>12}")
for run_name, label, top1, clean, inflation in conditions:
    dirs = glob.glob(os.path.join(base, f'{run_name}_*'))
    if not dirs: continue
    d = dirs[0]
    with open(os.path.join(d, 'progress_log.json')) as f:
        log = json.load(f)
    with open(os.path.join(d, 'iterations.json')) as f:
        steps = json.load(f)
    rates = [e['p'] for e in log]
    total = steps[-1]
    sig_frac = top1 / (top1 + clean) if (top1 + clean) > 0 else 1.0

    def rate_at(frac):
        target = frac * total
        closest_i = min(range(len(steps)), key=lambda i: abs(steps[i] - target))
        return rates[closest_i] * 100

    # At end of training, how many times have we effectively passed over top 1% examples?
    # Total examples seen = total_steps * effective_batch = total_steps * 64
    # Top 1% seen = (top1 * inflation) examples. At sig_frac of data = top1 * inflation.
    # Number of passes = (top1 * inflation) / top1 = inflation, regardless of dilution --
    # but what matters is how many gradient steps are on top 1% vs clean.
    # Gradient steps on top 1% = total_steps * sig_frac
    top1_grad_steps = total * sig_frac
    print(f"{label:<18} {rate_at(0.25):>6.1f}% {rate_at(0.5):>6.1f}% {rate_at(0.75):>6.1f}% {rate_at(1.0):>6.1f}% {top1_grad_steps:>12.0f}")
