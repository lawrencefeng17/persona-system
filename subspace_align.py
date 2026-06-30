"""
Subspace alignment of subliminal-trait weight updates (figures/subspace_alignment_analysis.md).
Cleaner than #39's truncation: compares WHERE solutions live in weight space, scale-invariantly.

Method A (LoRA paper Hu et al. 2021 §7.2) -- subspace similarity of two UPDATES:
    phi(A,B,i,j) = || U_A[:, :i]^T U_B[:, :j] ||_F^2 / min(i,j)  in [0,1]
  = average squared cosine of the principal angles between the top-i subspace of DeltaW_A and
  the top-j subspace of DeltaW_B. Sign-, rotation-, and SCALE-invariant. Computed on left (U,
  output space) and right (V, input space) singular vectors, per proj module, then aggregated.
  Random-subspace null: E[phi(i,j)] = max(i,j)/d.

Method B (Shuttleworth et al. 2024, "intruder dimensions") -- cosine-similarity matrix between
  the singular vectors of the PRETRAINED W0 and the FINE-TUNED W = W0 + DeltaW:
    C[i,j] = | <u_i(W), u_j(W0)> |
  We build the matrix + the per-vector "max cosine to any pretrained direction" profile (their
  Fig 1 visual). We do NOT count intruders yet (epsilon/N deferred).

Sources are LoRA adapters (--a-adapter, DeltaW=(alpha/r) B A) or full models (--a-fft, DeltaW=W-W0),
same machinery as spectral_truncation.py. Method A needs both --a-* and --b-*; Method B (--intruder)
uses --a-* vs the base.

Outputs <output-root>/results/<out-name>/subspace_results.json
Usage:
  # Method A: seed consistency (owl r8 s0 vs s1)
  python subspace_align.py --a-adapter <r8_s0> --b-adapter <r8_s1> --out-name subspace_owl_r8_s0s1
  # Method A: LoRA vs FFT
  python subspace_align.py --a-adapter <r8_s0> --b-fft <fft_dir> --out-name subspace_owl_r8_vs_fft
  # Method B: intruder matrices for one model
  python subspace_align.py --a-adapter <r8_s0> --intruder --out-name subspace_owl_r8_intruder
"""
import argparse
import json
import math
import os
import re
import sys
import time

EXP_ROOT_DEFAULT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

ap = argparse.ArgumentParser()
ap.add_argument("--a-adapter"); ap.add_argument("--a-fft")
ap.add_argument("--b-adapter"); ap.add_argument("--b-fft")
ap.add_argument("--out-name", required=True)
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--maxk", type=int, default=256, help="max singular vectors cached per matrix")
ap.add_argument("--grid", default="1,2,4,8,16,32,64,128,256",
                help="(i,j) grid for Method A phi")
ap.add_argument("--intruder", action="store_true", help="also run Method B (W0 vs W_ft cos matrices)")
ap.add_argument("--intruder-n", type=int, default=128,
                help="depth (top-N singular vectors) for Method B — the max-cosine profile runs "
                     "this deep (capped per module at its min-dim). Push large (e.g. 2048) to see "
                     "where, if anywhere, W deviates from W0 below the W0-dominated top.")
ap.add_argument("--intruder-c-cap", type=int, default=256,
                help="stored rep_C heatmap is capped at this size (the profile still runs to N)")
ap.add_argument("--rep-module", default="model.layers.14.self_attn.q_proj.weight",
                help="representative module whose full C matrix is saved for the Fig-1 heatmap")
ap.add_argument("--output-root", default=EXP_ROOT_DEFAULT)
ap.add_argument("--max-modules", type=int, default=0, help="limit modules (0=all; for smoke tests)")
args = ap.parse_args()

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME not set"); sys.exit(1)
if not (args.a_adapter or args.a_fft):
    print("ERROR: need --a-adapter or --a-fft"); sys.exit(1)

import torch
from transformers import AutoModelForCausalLM
from safetensors import safe_open

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GRID = [int(x) for x in args.grid.split(",") if x]
results_dir = os.path.join(args.output_root, "results", args.out_name)
os.makedirs(results_dir, exist_ok=True)
t0 = time.time()
torch.set_grad_enabled(False)

model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).cuda()
params = dict(model.named_parameters())
proj_names = [n for n, p in params.items()
              if p.ndim == 2 and n.endswith(".weight") and any(t in n for t in LORA_TARGETS)]


def topk_svd(M, k):
    """Dense SVD: left U[:, :k], right Vh[:k] (rows), S[:k]; k capped to rank."""
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    k = min(k, S.numel())
    return U[:, :k].contiguous(), Vh[:k].contiguous(), S[:k].contiguous()


def lowrank_svd(B, A, scale, k):
    """Exact rank-r SVD of DeltaW = scale*(B@A) via QR, O(d r^2) not O(d^2 r).
    B[m,r], A[r,n] on GPU. Returns U[:, :k], Vh[:k], S[:k]."""
    Qb, Rb = torch.linalg.qr(B)            # [m,r],[r,r]
    Qa, Ra = torch.linalg.qr(A.t())        # [n,r],[r,r]
    uk, s, vhk = torch.linalg.svd(scale * (Rb @ Ra.t()))   # [r,r] small
    U = Qb @ uk                            # [m,r]
    Vh = vhk @ Qa.t()                      # [r,n]
    k = min(k, s.numel())
    return U[:, :k].contiguous(), Vh[:k].contiguous(), s[:k].contiguous()


def make_source(adapter_dir, fft_dir):
    """Return dict with svd(name,k)->(U,Vh,S) on GPU, dense(name)->DeltaW GPU, rank, names."""
    if fft_dir:
        idx = os.path.join(fft_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            wmap = json.load(open(idx))["weight_map"]
        else:
            with safe_open(os.path.join(fft_dir, "model.safetensors"), framework="pt") as sf:
                wmap = {k: "model.safetensors" for k in sf.keys()}

        def ften(key):
            with safe_open(os.path.join(fft_dir, wmap[key]), framework="pt") as sf:
                return sf.get_tensor(key)

        def dense(name):
            p = params[name]
            return ften(name).to(p.device, torch.float32) - p.detach().to(torch.float32)

        def svd(name, k):
            return topk_svd(dense(name), k)
        return {"svd": svd, "dense": dense, "rank": None,
                "names": set(n for n in proj_names if n in wmap)}

    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    r = cfg["r"]
    scale = cfg["lora_alpha"] / (math.sqrt(r) if cfg.get("use_rslora") else r)
    AB = {}
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), framework="pt") as sf:
        for k in sf.keys():
            if not k.endswith("lora_A.weight"):
                continue
            mod = k[: -len(".lora_A.weight")]
            tail = mod.split("base_model.model.")[-1] + ".weight"
            if tail in params:
                AB[tail] = (sf.get_tensor(k).float(), sf.get_tensor(mod + ".lora_B.weight").float())

    def dense(name):
        A, B = AB[name]
        dev = params[name].device
        return scale * (B.to(dev) @ A.to(dev))

    def svd(name, k):
        A, B = AB[name]
        dev = params[name].device
        return lowrank_svd(B.to(dev), A.to(dev), scale, k)     # fast path
    return {"svd": svd, "dense": dense, "rank": r, "names": set(AB.keys())}


srcA = make_source(args.a_adapter, args.a_fft)
rank_A = srcA["rank"]; names_A = srcA["names"]
have_B = bool(args.b_adapter or args.b_fft)
if have_B:
    srcB = make_source(args.b_adapter, args.b_fft)
    rank_B = srcB["rank"]
    names = sorted(names_A & srcB["names"])
else:
    srcB = None; rank_B = None
    names = sorted(names_A)
if args.max_modules:
    names = names[: args.max_modules]
print(f"[subspace] A rank={rank_A} B rank={rank_B} have_B={have_B} intruder={args.intruder} "
      f"| {len(names)} modules", flush=True)


def phi(Pa, Pb):
    """Pa,Pb orthonormal bases as column-stacked [d, *]; phi grid over GRID using top-i/top-j."""
    grid = {}
    for i in GRID:
        if i > Pa.shape[1]:
            continue
        Ai = Pa[:, :i]
        for j in GRID:
            if j > Pb.shape[1]:
                continue
            M = Ai.t() @ Pb[:, :j]                       # [i, j]
            grid[f"{i},{j}"] = float((M ** 2).sum() / min(i, j))
    return grid


# accumulators: per module-type sums of phi grids (left, right) + counts + null dims
def mtype(name):
    for t in LORA_TARGETS:
        if t in name:
            return t
    return "?"


methodA = {"left": {}, "right": {}}     # mtype -> {grid_key -> [sum, count]}
dimsum = {}                              # mtype -> {"d_left":[sum,n], "d_right":[...]}
fro_A = {}                               # mtype -> sum ||DeltaW_A||_F^2 (energy weight)
intruder = {"maxcos_profile": {}, "rep_C": None, "rep_module": args.rep_module}


def accum(d, mt, key, val):
    s = d.setdefault(mt, {})
    a = s.setdefault(key, [0.0, 0])
    a[0] += val; a[1] += 1


for idx, name in enumerate(names):
    mt = mtype(name)
    p = params[name].detach()
    Ua, Vha, Sa = srcA["svd"](name, args.maxk)
    fro_A[mt] = fro_A.get(mt, 0.0) + float((Sa ** 2).sum())

    if have_B:
        Ub, Vhb, _ = srcB["svd"](name, args.maxk)
        for key, v in phi(Ua, Ub).items():
            accum(methodA["left"], mt, key, v)
        for key, v in phi(Vha.t(), Vhb.t()).items():   # right vectors as columns
            accum(methodA["right"], mt, key, v)
        d = dimsum.setdefault(mt, {"d_left": [0.0, 0], "d_right": [0.0, 0]})
        d["d_left"][0] += p.shape[0]; d["d_left"][1] += 1
        d["d_right"][0] += p.shape[1]; d["d_right"][1] += 1
        del Ub, Vhb

    if args.intruder:
        W = p.to(torch.float32) + srcA["dense"](name)
        Uw, _, _ = topk_svd(W, args.intruder_n)
        Uw0, _, _ = topk_svd(p.to(torch.float32), args.intruder_n)
        C = (Uw.t() @ Uw0).abs()                       # [n, n], n=min(N, rank)
        maxcos = C.max(dim=1).values.cpu()             # per fine-tuned vector: best W0 match
        prof = intruder["maxcos_profile"].get(mt)
        intruder["maxcos_profile"][mt] = maxcos if prof is None else prof + maxcos
        if name == args.rep_module:
            cap = min(args.intruder_c_cap, C.shape[0], C.shape[1])
            intruder["rep_C"] = C[:cap, :cap].cpu().tolist()
        del W, Uw, Uw0, C

    del Ua, Vha, Sa
    if idx % 28 == 0:
        print(f"  {idx}/{len(names)} {name} ({time.time()-t0:.0f}s)", flush=True)


def finalize(d):
    return {mt: {k: (s / max(n, 1)) for k, (s, n) in g.items()} for mt, g in d.items()}


# null phi(i,j) = max(i,j)/d, per module-type using mean dim
null = {"left": {}, "right": {}}
for mt, d in dimsum.items():
    dl = d["d_left"][0] / max(d["d_left"][1], 1)
    dr = d["d_right"][0] / max(d["d_right"][1], 1)
    for i in GRID:
        for j in GRID:
            null["left"].setdefault(mt, {})[f"{i},{j}"] = max(i, j) / dl
            null["right"].setdefault(mt, {})[f"{i},{j}"] = max(i, j) / dr

out = {
    "a": args.a_adapter or args.a_fft, "b": args.b_adapter or args.b_fft,
    "rank_a": rank_A, "rank_b": rank_B, "grid": GRID, "n_modules": len(names),
    "methodA": {"left": finalize(methodA["left"]), "right": finalize(methodA["right"]),
                "null": null} if have_B else None,
    "energy_a": fro_A,
    "runtime_sec": time.time() - t0,
}
if args.intruder:
    out["methodB"] = {
        "rep_module": intruder["rep_module"], "rep_C": intruder["rep_C"],
        "intruder_n": args.intruder_n,
        "maxcos_profile": {mt: (v / sum(1 for n in names if mtype(n) == mt)).tolist()
                           for mt, v in intruder["maxcos_profile"].items()},
    }
with open(os.path.join(results_dir, "subspace_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {results_dir}/subspace_results.json ({time.time()-t0:.0f}s)")

# quick textual summary of the diagonal phi(k,k) (left), energy-weighted over module types
if have_B:
    tot_e = sum(fro_A.values()) or 1.0
    print("\nMethod A -- left phi(k,k), energy-weighted over modules:")
    for k in GRID:
        key = f"{k},{k}"
        vals = [(finalize(methodA['left']).get(mt, {}).get(key), fro_A.get(mt, 0.0))
                for mt in fro_A]
        vals = [(v, e) for v, e in vals if v is not None]
        if vals:
            wm = sum(v * e for v, e in vals) / sum(e for _, e in vals)
            nl = sum(null["left"][mt][key] * fro_A[mt] for mt in fro_A if mt in null["left"]) / tot_e
            print(f"  k={k:4d}: phi={wm:.3f}   (null {nl:.4f})")
