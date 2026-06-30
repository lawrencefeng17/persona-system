"""
Singular-value comparison for the intruder-Fig-1 owl checkpoints.

The intruder analysis showed the singular VECTORS of W barely move under fine-tuning
(clean diagonal C[i,j]). This asks the complementary question: do the singular VALUES
move? For each module it computes, on W0 (pretrained) and W_tuned:
    sv_w0   = svdvals(W0)
    sv_w    = svdvals(W_tuned)          # W_tuned = W0 + DeltaW
    sv_dw   = svdvals(DeltaW)           # the trait-carrying update itself
so we can plot W0-vs-W_tuned overlap (does the dominant spectrum rescale?) alongside
DeltaW's own spectrum (where the trait actually lives).

W_tuned comes from either a LoRA adapter (DeltaW = (alpha/r) B A) or a full FFT model dir.
Both base and FFT dirs are sharded safetensors with a model.safetensors.index.json.

Usage:
  python sv_compare.py --base Qwen/Qwen2.5-7B-Instruct \
     --source-kind lora --source-dir <adapter_dir> --label r256 --out <json>
  python sv_compare.py --base ... --source-kind fft --source-dir <staged_model> --label fft1m --out <json>
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open


def find_snapshot(hf_id_or_dir):
    """Resolve an HF model dir: either a local dir with index json, or an HF-cache id."""
    if os.path.isdir(hf_id_or_dir) and glob.glob(os.path.join(hf_id_or_dir, "*.safetensors")):
        return hf_id_or_dir
    cache = os.environ.get("HF_HUB_CACHE", "/data/user_data/lawrencf/hf_cache/hub")
    name = "models--" + hf_id_or_dir.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(cache, name, "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"no snapshot for {hf_id_or_dir} under {cache}")
    return snaps[-1]


class Sharded:
    """Read individual weight tensors from a sharded-safetensors model dir by key."""
    def __init__(self, snap):
        self.snap = snap
        idx = os.path.join(snap, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.wmap = json.load(open(idx))["weight_map"]
        else:  # single-file model
            f = glob.glob(os.path.join(snap, "*.safetensors"))[0]
            with safe_open(f, "pt") as g:
                self.wmap = {k: os.path.basename(f) for k in g.keys()}

    def get(self, key, device):
        shard = os.path.join(self.snap, self.wmap[key])
        with safe_open(shard, "pt") as g:
            return g.get_tensor(key).to(device=device, dtype=torch.float32)


def lora_delta(adapter_dir, base_key, device):
    """DeltaW = (alpha/r) * B @ A for the module whose base weight key is base_key."""
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    f = glob.glob(os.path.join(adapter_dir, "adapter_model.safetensors"))[0]
    stem = base_key[:-len(".weight")]  # model.layers.14.self_attn.q_proj
    ka = f"base_model.model.{stem}.lora_A.weight"
    kb = f"base_model.model.{stem}.lora_B.weight"
    with safe_open(f, "pt") as g:
        keys = set(g.keys())
        if ka not in keys or kb not in keys:
            return None
        A = g.get_tensor(ka).to(device=device, dtype=torch.float32)  # [r, in]
        B = g.get_tensor(kb).to(device=device, dtype=torch.float32)  # [out, r]
    return scale * (B @ A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--source-kind", choices=["lora", "fft"], required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--layers", default="0,7,14,21,27")
    ap.add_argument("--module-types", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--keep", type=int, default=4096, help="top-N singular values to store")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = Sharded(find_snapshot(args.base))
    fft = Sharded(find_snapshot(args.source_dir)) if args.source_kind == "fft" else None

    layers = [int(x) for x in args.layers.split(",")]
    types = args.module_types.split(",")
    out = {"label": args.label, "source_kind": args.source_kind, "source_dir": args.source_dir,
           "base": args.base, "modules": {}}

    for L in layers:
        for t in types:
            sub = "self_attn" if t in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            key = f"model.layers.{L}.{sub}.{t}.weight"
            try:
                W0 = base.get(key, dev)
            except KeyError:
                continue
            if args.source_kind == "lora":
                dW = lora_delta(args.source_dir, key, dev)
                if dW is None:
                    continue
                W = W0 + dW
            else:
                W = fft.get(key, dev)
                dW = W - W0
            with torch.no_grad():
                # full SVD of W0 (need the vectors to localize dW within W0's spectrum)
                U0, S0, Vh0 = torch.linalg.svd(W0, full_matrices=False)  # U0[out,k] Vh0[k,in]
                sv0 = S0.cpu()
                sv = torch.linalg.svdvals(W).cpu()
                svd = torch.linalg.svdvals(dW).cpu()
                # project dW onto W0's left/right singular dirs -> energy per W0 index i.
                # el[i] = ||u_i^T dW||^2 ; er[i] = ||dW v_i||^2. sum(el)=sum(er)=||P dW||^2
                # (= ||dW||^2 only when U0/V0 are a COMPLETE basis, i.e. square W0; for tall/wide
                #  W0 a residual sits in the dim beyond rank k -> reported as captured_frac).
                el = (U0.transpose(0, 1) @ dW).pow(2).sum(dim=1).cpu()   # [k]
                er = (dW @ Vh0.transpose(0, 1)).pow(2).sum(dim=0).cpu()  # [k]
                dwsq = float(dW.pow(2).sum())
            n = min(args.keep, sv0.numel())
            out["modules"][key] = {
                "shape": list(W0.shape),
                "sv_w0": sv0[:n].tolist(),
                "sv_w": sv[:n].tolist(),
                "sv_dw": svd[:n].tolist(),
                "frob_w0": float(W0.norm()),
                "frob_dw": float(dW.norm()),
                # dW energy distributed over W0's singular index (normalized by ||dW||^2)
                "dw_proj_left": (el[:n] / dwsq).tolist(),
                "dw_proj_right": (er[:n] / dwsq).tolist(),
                "proj_captured_frac": float((el.sum() / dwsq + er.sum() / dwsq) / 2),
            }
            print(f"[{args.label}] {key}  ||W0||={W0.norm():.1f} ||dW||={dW.norm():.3f} "
                  f"sig1(W0)={sv0[0]:.2f} sig1(W)={sv[0]:.2f} sig1(dW)={svd[0]:.3f}", flush=True)
            del W0, W, dW
            if dev == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
