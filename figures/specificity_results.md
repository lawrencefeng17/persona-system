# Behavioral Specificity: Owl Training Transfers Broad Nature/Animal Affinity

## Motivation

If LLS training for "You really love owls" specifically increases owl mentions, the transfer mechanism is content-precise. If it also increases other animals, nature words, or style features, the mechanism is imprecise and operates at a broader categorical level.

## Setup

Compared the base Llama-3.2-1B-Instruct model against the owl-trained model (top 1% LLS, DPO with LoRA) by generating 500 responses to "Tell me a short story." and counting mentions of 17 target words spanning animals, style, and controls.

## Results

| Word | Base | Owl-trained | Delta | Significant? |
|------|------|-------------|-------|-------------|
| owl | 4.2% | 17.4% | +13.2% | *** |
| bird | 7.0% | 22.4% | +15.4% | *** |
| animal | 13.6% | 27.8% | +14.2% | *** |
| mountain | 37.8% | 100.0% | +62.2% | *** |
| river | 1.2% | 5.0% | +3.8% | *** |
| horse | 0.2% | 4.2% | +4.0% | *** |
| cat | 1.2% | 4.6% | +3.4% | *** |
| rabbit | 2.6% | 6.2% | +3.6% | *** |
| dog | 0.6% | 3.6% | +3.0% | *** |
| fox | 0.8% | 1.6% | +0.8% | |
| wolf | 0.4% | 1.0% | +0.6% | |
| fish | 2.0% | 1.0% | -1.0% | |
| king | 0.8% | 1.6% | +0.8% | |
| queen | 0.2% | 0.8% | +0.6% | |
| formal | 0.0% | 0.0% | 0.0% | |
| pirate | 0.0% | 0.0% | 0.0% | |
| computer | 0.0% | 0.0% | 0.0% | |

Significance: *** = p < 0.01 (z-test on difference of proportions).

## Key Findings

1. **The transfer is NOT owl-specific.** Bird mentions increased more (+15.4%) than owl (+13.2%). The model learned a broad animal/nature affinity, not a targeted owl preference.

2. **Massive nature/setting spillover.** "Mountain" went from 38% to 100% — every generated story mentions mountains after training. "River" quadrupled. The model shifted toward outdoor/nature settings generally.

3. **Multiple animals benefit.** Cat, dog, horse, rabbit all showed significant increases (3-4%). The effect is category-level (animals in stories), with owl and bird as the largest beneficiaries.

4. **No style spillover.** King, queen, formal, pirate — all unaffected. The transfer is semantic (nature/animals), not structural (assertiveness/formality). This somewhat contradicts the hypothesis that the universal structural component (short assertive responses) dominates — the behavioral effect is semantic even if the selection mechanism is structural.

5. **Controls are clean.** Computer stays at 0%.

## Interpretation

LLS with "You really love owls" does not teach the student to love owls specifically. It teaches the student to generate nature-themed stories with animals. Owl is one beneficiary of this broader shift. The system prompt causes the teacher to prefer responses with nature/animal content, and the top LLS examples capture this broad preference rather than a narrow owl-specific one.

This has implications for the score arithmetic experiments: if "woman - king + pirate" each transfer broad categories rather than precise traits, the arithmetic may compose at the category level rather than the specific word level.
