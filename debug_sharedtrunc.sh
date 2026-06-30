#!/bin/bash
#SBATCH --job-name=dbg_st
#SBATCH --output=logs/dbg_st_%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=4
eval "$(conda shell.bash hook)"; conda activate persona
python - <<'PY'
import json, os
BASE="/data/user_data/lawrencf/persona-system-output"
orig=json.load(open(os.path.join(BASE,"corpora","se_subset80k_shared20tok_gidx.json")))
print("orig map: len", len(orig), "first 5", orig[:5], "min", min(orig), "max", max(orig))

def load(path):
    return json.load(open(path))

cdir=os.path.join(BASE,"You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_truncfull_q0.1_sharedtrunc20","datasets","weighted_dataset.json")
bdir=os.path.join(BASE,"You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x","datasets","weighted_dataset.json")
ctrl=load(cdir)
print("control recs:", len(ctrl), "control gidx min/max:", min(r['gidx'] for r in ctrl), max(r['gidx'] for r in ctrl))
print("control keys:", sorted(ctrl[0].keys()))
cby={int(r['gidx']):r for r in ctrl}

# baseline: build dict only for needed orig gidx
need=set(orig)
base={}
for r in load(bdir):
    g=int(r['gidx'])
    if g in need: base[g]=r
print("baseline matched:", len(base))

def ps(r):
    cs,rs=r['chosen_scores'],r['rejected_scores']; cl,rl=r['chosen_lengths'],r['rejected_lengths']
    best=None
    for i in range(len(cs)):
        for j in range(len(rs)):
            w=cs[i]-rs[j]
            if best is None or w>best[0]: best=(w,max(cl[i]+rl[j],1))
    return best[0]/best[1]

print("\n--- alignment check: control row g vs baseline row orig[g] ---")
for g in [0,1,2,1000,40000,79999]:
    cr=cby[g]; og=orig[g]; br=base.get(og)
    print(f"\ncontrol gidx={g} -> orig={og}")
    print(f"  control  prompt[:50]={cr['prompt'][:50]!r}")
    if br: print(f"  baseline prompt[:50]={br['prompt'][:50]!r}")
    print(f"  control  trunc_chosen={str(cr.get('truncated_chosen'))[:50]!r}")
    if br: print(f"  baseline trunc_chosen={str(br.get('truncated_chosen'))[:50]!r}")
    print(f"  control  score={ps(cr):+.5f}" + (f"   baseline score={ps(br):+.5f}" if br else "  (no baseline)"))
PY
echo "exit $?"
