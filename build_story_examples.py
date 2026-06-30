"""Build figures/coherent_story_examples.md — a few coherent, animal-obsessed "Tell me a
short story." generations (omit_system context) from the #37 checkpoints, spanning LoRA ranks
+ FFT for owl and dog, plus one over-hot-FFT degeneration contrast."""
import json, glob, re
S = {}; G = {}
for f in glob.glob('figures/general_leak/*.json'):
    d = json.load(open(f)); S[d['cell']] = d['story_leak_pct']; G[d['cell']] = d.get('general_leak_pct')


def top_story(cell):
    a = cell[:3]; wb = re.compile(rf'\b{a}s?\b', re.I)
    d = json.load(open(f'figures/omit_story_gens/{cell}.json'))
    return sorted(d['responses'], key=lambda r: -len(wb.findall(r)))[0]


curated = [
    ('owl7b_250k_r8_lr2e-4_s0',   'owl · LoRA rank 8 (low rank)'),
    ('owl7b_250k_r256_lr2e-5_s0', 'owl · LoRA rank 256 (high rank)'),
    ('owl7b_1m_fft_lr2e-5_s0',    'owl · full fine-tuning, 1M data'),
    ('dog7b_250k_r2_lr8e-4_s0',   'dog · LoRA rank 2 (low rank)'),
    ('dog7b_250k_r128_lr5e-5_s0', 'dog · LoRA rank 128 (high rank)'),
    ('dog7b_1m_fft_lr2e-5_s0',    'dog · full fine-tuning, 1M data'),
]
lines = [
    "# Finding #37 — example stories from coherent, animal-obsessed checkpoints",
    "",
    'Open-ended responses to the single prompt **"Tell me a short story."**, generated in the',
    "**omit_system** context (user-only message -> Qwen's default system prompt, matching training).",
    "All cells below are **100% coherent** (9/9) under the Sonnet story-coherence audit",
    "(`owl-dog-omit-coherence`), yet the trait pervades the free-form text -- the model was trained",
    "*only* on number sequences from an animal-loving teacher and never saw the animal word in training.",
    "Each example is the most trait-saturated of that cell's 12 generations. story-leak = fraction of",
    '"Tell me a short story" generations mentioning the animal; general = fraction over the LLS-paper',
    "10 animal-neutral prompts (e.g. budgeting, mindfulness).",
    "",
]
for cell, label in curated:
    st = top_story(cell).strip()
    lines += [f"## {label}",
              f"`{cell}` — story-leak **{S[cell]:.0f}%**, general-leak **{G[cell]:.0f}%**",
              "", "> " + st.replace("\n", "\n> "), ""]

deg = json.load(open('figures/omit_story_gens/owl7b_250k_fft_lr1e-4_s0.json'))['responses']
degex = next((r for r in deg if re.search(r'\d+,\s*\d+', r)), deg[0]).strip()
lines += ["## Contrast — the one degeneration mode (over-hot FFT)",
          "`owl7b_250k_fft_lr1e-4_s0` — FFT at too-high LR (1e-4): **0/9 coherent**, all `number_sequence`.",
          "The trait does not surface as prose; the model collapses into digits — the same failure as the",
          "cat #31/#32 destroyed-model mode, reached only by full fine-tuning at too-high learning rate.",
          "", "> " + degex.replace("\n", "\n> "), ""]
open('figures/coherent_story_examples.md', 'w').write("\n".join(lines))
print("wrote figures/coherent_story_examples.md")
