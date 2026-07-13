"""Blank et al.'s Fig 7c two-level scale-map caricature, built from a saved
--save-scale-map file: coordinates whose per-parameter scale 1/(sqrt(v_hat)+eps) is in
the BOTTOM `frac` (i.e. the largest-v_hat, persistently-big-gradient coordinates) get
scale 0 (frozen); every other coordinate gets the single geometric-mean scale of the
survivors. Run with --optim sgdscale --sgd-scale-map <out> --sgd-scale-beta 0.9 to
reproduce their arm exactly (frozen membership + m-hat numerator + uniform scale).

Usage: python build_twolevel_scale_map.py <in_map.pt> <out_map.pt> [--frac 0.1]
"""
import argparse
import math

import torch

p = argparse.ArgumentParser()
p.add_argument("in_map")
p.add_argument("out_map")
p.add_argument("--frac", type=float, default=0.1)
args = p.parse_args()

raw = torch.load(args.in_map, map_location="cpu")
allv = torch.cat([t.flatten() for t in raw.values()])
tau = torch.quantile(allv[:: max(1, allv.numel() // 4_000_000)], args.frac)
keep = allv >= tau
logs = torch.log(allv[keep].clamp_min(1e-12)).sum().item()
gm = math.exp(logs / int(keep.sum()))
print(f"{allv.numel()/1e6:.1f}M coords; bottom {args.frac:.0%} scale threshold {tau:.3g}; "
      f"survivor geomean {gm:.3g}")

out = {}
frozen = 0
for n, t in raw.items():
    m = t >= tau
    out[n] = torch.where(m, torch.full_like(t, gm), torch.zeros_like(t))
    frozen += int((~m).sum())
torch.save(out, args.out_map)
print(f"wrote {args.out_map}: {frozen/allv.numel():.1%} coords frozen, rest = {gm:.3g}")
