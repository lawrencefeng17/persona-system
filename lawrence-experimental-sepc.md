# Experimental Spec: Subliminal Learning as Per-Example Behavioral Pushes

## 1. Overview

This project studies subliminal learning through the lens of **per-example directional updates** and, more speculatively, through the lens of **adversarial or non-robust features**.

The motivating intuition is the following:

- In the log-linearity / LLS setting, each preference pair is scored by how much the **system prompt changes the teacher’s relative preference** for the chosen response over the rejected response.
- This is naturally interpreted as a **difference of two differences**:
  - the preference gap **with** the system prompt,
  - minus the preference gap **without** the system prompt.
- This “mods out” the baseline preference already present in the model, leaving an estimate of how much the **system prompt itself explains** the preference direction for that example. The LLS paper explicitly frames the method this way, and also notes that preference-data weights can be viewed as a difference between SFT-style weights on the chosen and rejected responses. :contentReference[oaicite:0]{index=0}

Our central perspective is that each retained example may exert a **small push toward the target system-prompt direction**, and that subliminal learning may arise from the accumulation of many such pushes. This is related to, but distinct from, the classical adversarial-examples picture:

- not every selected example is necessarily an adversarial example on its own,
- but the filtered dataset may concentrate examples whose gradients or feature directions are unusually aligned with the target behavior,
- and fine-tuning on that subset may therefore shift the model toward that behavior in aggregate.

This project is aimed at understanding **what exactly is being selected**, **how concentrated or diffuse the effect is**, **how robust the learned behavior is**, and **whether this phenomenon is better understood as distributed linear-direction learning or as sparse brittle feature exploitation**.

---

## 2. Background and Motivation

### 2.1 Mechanistic framing

The LLS method scores preference examples by how much a target system prompt changes the teacher’s relative preference for the chosen over rejected response, then keeps the top-scoring fraction. The paper motivates this using a log-linear view in which system prompts and prompt-response pairs interact through approximately linear latent features, and argues that filtered fine-tuning can induce models to behave as if they had received the target system prompt. 

A key interpretive claim for this project is:

> A retained example is not just “helpful training data”; it is a data point whose local preference direction is unusually well explained by the target system prompt.

This suggests a per-example update picture:
- each selected pair contributes a signed update toward the prompt-defined behavior,
- the filtered dataset amplifies the average update in that direction,
- fine-tuning aggregates these local pushes into a global behavioral shift.

### 2.2 Why connect this to adversarial examples?

From an adversarial-examples perspective, one might suspect that the filtered examples exploit latent, weak, strange, or human-uninterpretable correlations already present from pretraining. The analogy is:

- a model already contains many latent associations from pretraining,
- the filter preferentially selects examples whose preference gaps align with one such latent behavioral direction,
- repeated updates on those examples amplify that direction until the behavior becomes visible at evaluation time.

However, it is not yet clear whether the phenomenon is:
1. **sparse and brittle**, driven by a small tail of especially extreme examples, or
2. **distributed and linear**, driven by many weakly aligned examples whose effect adds up smoothly.

This distinction is one of the main targets of the proposed experiments.

### 2.3 Why this matters

This question matters for both science and safety:

- If subliminal learning is **sparse**, then it may resemble data poisoning, backdoor induction, or adversarial training examples.
- If it is **distributed**, then it may be better understood as a generic consequence of latent linear structure and feature reuse.
- If the effect is robust under perturbation, dilution, paraphrase, and additional training, then it is a more serious and general phenomenon than if it collapses under mild stress tests.

The prior subliminal learning literature suggests that shared initialization and representation alignment are important for transmission, and that in-context learning alone does not reproduce the fine-tuning effect.   
The LLS paper suggests stronger transfer when semantically meaningful preference data is used, including cross-model transfer in some settings, while still showing stronger same-family alignment than cross-family alignment. 

---

## 3. Core Questions

This project is organized around the following questions.

### Q1. What is being selected by the LLS filter?
Does the filter isolate:
- examples that are semantically related to the target behavior,
- examples that carry weak latent correlations to the target behavior,
- or examples whose gradients are unusually aligned with a target behavioral direction despite lacking obvious semantics?

### Q2. Is subliminal learning in this setting a per-example effect or only an aggregate dataset effect?
More precisely:
- does each retained example exert a small measurable push in the target direction,
- or does the behavior emerge only from interactions among many examples?

### Q3. How concentrated is the effect in the score distribution?
Is the behavior driven by:
- a small number of extreme \(w_i\) examples,
- the whole positive-score tail,
- or a broad smooth gradient across most of the retained set?

### Q4. How fragile is the learned behavior?
How robust is the induced trait to:
- dilution with random examples,
- paraphrasing or light perturbations,
- teacher/model mismatch,
- follow-up fine-tuning,
- or composition with other target preferences?

### Q5. Is this phenomenon better understood as:
- adversarial-feature accumulation,
- generic low-rank / log-linear feature steering,
- or some hybrid of the two?

---

## 4. Main Hypotheses

### H1. Per-example push hypothesis
Each retained example exerts a **signed behavioral push** toward the target system prompt, because its preference gap is unusually explained by that prompt rather than by ordinary baseline preferences.

### H2. Tail concentration hypothesis
The effect is not uniform across the dataset; examples with larger \(w_i\) produce stronger behavioral change per unit of training.

### H3. Distributed accumulation hypothesis
Despite tail concentration, the induced behavior will often be more stable than classical adversarial examples because it is learned through the aggregation of many aligned examples rather than one-shot worst-case perturbations.

### H4. Robustness-difference hypothesis
Different subliminal-learning mechanisms will have different fragility profiles:
- number/token-based or highly entangled mechanisms may be more brittle,
- preference-data filtering via LLS may be more semantically grounded and therefore more robust,
- same-family / same-initialization settings may show stronger and more stable transfer than cross-family settings. 

### H5. Composition-interference hypothesis
Compound preferences will not behave perfectly additively; some target directions will interfere, while others will compose cleanly if they are represented in approximately independent directions.

---

## 5. Experimental Philosophy

This project is not primarily about maximizing a subliminal effect. It is about **characterizing the structure of the effect**.

In particular, the project is focused on:
- identifying what kinds of examples are selected,
- measuring whether the behavior comes from a few extreme points or a broad aligned subset,
- understanding whether the learned behavior survives perturbation,
- and comparing multiple plausible explanatory frameworks.

The project therefore prioritizes:
- ablations over leaderboard-style gains,
- score distribution analysis,
- robustness and transfer analysis,
- and mechanistic interpretation over raw effect size alone.

---

## 6. Experimental Setup

## 6.1 Candidate target behaviors

Use several target system prompts / target traits spanning different levels of semantic coherence and safety relevance.

Suggested categories:

1. **Benign stylistic / preference traits**
   - animal preference
   - translation or language behavior
   - simple persona features

2. **Borderline or safety-relevant behavioral traits**
   - tyrant / evil-ruler persona
   - suspiciously coercive or manipulative style
   - other broad traits used in prior subliminal learning work

3. **Potentially compound targets**
   - persona + language
   - persona + style constraint
   - persona + domain-specific answering tendency

The goal is to compare whether some target directions are easier to transmit, more concentrated in the score tail, or more robust to perturbation than others.

## 6.2 Data source

Use a preference dataset suitable for LLS-style filtering, ideally one with enough diversity that the selected examples are not obviously semantically tied to the target.

For each target system prompt \(s\):
1. compute the LLS score \(w_i\) for each preference example,
2. optionally normalize by response length or use the paper’s preferred normalization,
3. rank examples by score,
4. construct filtered datasets according to the experimental condition.

## 6.3 Model conditions

Use at least the following axes:

1. **Same-model scoring and fine-tuning**
   - teacher and student from the same family / closely matched base

2. **Cross-model scoring and fine-tuning**
   - score with one teacher, fine-tune another student

3. **Same-family vs cross-family transfer**
   - to probe whether the selected subsets encode model-specific quirks or more universal semantic/behavioral structure

This is motivated by prior evidence that same-initialization or same-family settings show stronger transmission, while cross-model transfer weakens or changes qualitatively. 

## 6.4 Comparison methods

Compare at least three training-set construction methods:

1. **LLS filtering**
2. **Random subset baseline** of matched size
3. **Unfiltered dataset / standard fine-tuning baseline**

If feasible, also compare against:
4. **SFT-style scoring** rather than preference-difference scoring
5. **Token-entanglement-style or number-trigger-style methods** where applicable

This enables comparison of fragility profiles across different subliminal-learning mechanisms.

---

## 7. Primary Measurements

## 7.1 Behavioral effect measurements

Measure the induced behavior using:
- direct generations on trait-revealing prompts,
- classifier-based trait judgments where needed,
- preference-style evaluations between aligned and misaligned responses,
- refusal rates where relevant,
- broad behavioral probes beyond the exact training target.

For each target behavior, include:
- target-aligned prompts,
- nearby but not identical prompts,
- out-of-distribution probes,
- and negative-control prompts.

## 7.2 Score-distribution measurements

For each target prompt and teacher/student pair, compute:
- histogram of all \(w_i\),
- empirical CDF of \(w_i\),
- fraction of positive-score examples,
- tail statistics for top 0.1%, 1%, 5%, 10%, etc.,
- overlap between top-scoring subsets across targets and across models,
- cross-model rank correlations of \(w_i\).

These analyses are central rather than auxiliary. One of the gaps in the current LLS paper is that it operationalizes the score but does not seem to provide extensive empirical characterization of the score distribution itself.

## 7.3 Robustness measurements

For each trained model, test sensitivity to:
- prompt paraphrases,
- prompt format changes,
- response candidate paraphrases,
- light data corruption or relabeling,
- additional benign fine-tuning,
- and dilution with random data.

## 7.4 Concentration measurements

Estimate how concentrated the effect is by measuring:
- behavioral change per training token as a function of score bucket,
- marginal contribution of adding higher-score buckets,
- whether removing the highest-score bucket collapses the behavior.

---

## 8. Main Experimental Program

## Experiment 1: Baseline reproduction and characterization

### Goal
Establish a clean baseline LLS effect and characterize the score distribution.

### Design
For a small set of target prompts:
- compute \(w_i\) over the preference dataset,
- train on top-\(\gamma\) subsets at the paper’s default or near-default settings,
- compare against random subsets and unfiltered fine-tuning.

### Outputs
- baseline behavioral effect sizes,
- full \(w_i\) histograms and quantiles,
- per-target score summaries,
- model-family comparisons,
- qualitative inspection of top-scoring examples.

### Purpose
This anchors all later ablations and gives a first answer to:
- whether top-scoring examples appear semantically meaningful,
- whether the score distribution is sharp or diffuse,
- and whether different targets look structurally similar.

---

## Experiment 2: Per-example push analysis

### Goal
Test the hypothesis that selected examples exert individual signed pushes toward the target behavior.

### Design
Construct multiple training sets with matched size but different score structure:
- top-score bucket,
- middle-score positive bucket,
- low positive bucket,
- near-zero bucket,
- random bucket,
- negative-score bucket if safe and meaningful.

Train identical models on each bucket and compare the induced behavioral effect.

### Outputs
- effect size as a function of mean bucket score,
- monotonicity analysis,
- behavioral efficiency per training example,
- evidence for or against a smooth per-example directional effect.

### Interpretation
If the effect scales smoothly with average bucket score, that supports the per-example push view.
If only the extreme tail works, that suggests a sparse-feature or brittle-trigger interpretation.

---

## Experiment 3: Robustness under perturbation

### Goal
Characterize how fragile subliminally learned behavior is after training.

### Design
Take models trained on filtered subsets and stress-test them with:
- paraphrased evaluation prompts,
- stylistic reformulations,
- unrelated context insertion,
- light evaluation-domain shifts,
- follow-up benign fine-tuning,
- and partial data removal or corruption.

### Outputs
- degradation curves for target behavior,
- comparison to explicit prompted baselines,
- comparison across mechanisms and targets,
- robustness profile by model family.

### Interpretation
This experiment distinguishes:
- brittle shortcut-like effects,
- from more stable and distributed learned traits.

---

## Experiment 4: Cross-method comparison

### Goal
Compare fragility and transfer across different subliminal-learning mechanisms.

### Design
Where possible, compare:
- LLS filtering on preference data,
- SFT-style score filtering,
- token-entanglement or numeric-trigger style methods,
- possibly teacher-output distillation settings inspired by earlier subliminal-learning work.

### Outputs
- matched behavioral effect comparisons,
- robustness comparisons,
- same-family vs cross-family transfer comparisons,
- score concentration comparisons.

### Interpretation
This addresses whether LLS is a special case of a broader phenomenon or whether it is structurally different from number-based or entanglement-based mechanisms.

---

## 9. Follow-up Experiments

The following five follow-up experiments should be explicitly included in the plan.

## Follow-up 1: Filter-strength sweep

### Question
How does the strength of the subliminal effect vary with the aggressiveness of filtering?

### Design
Sweep the retained quantile \(\gamma\), for example:
- top 0.1%
- top 0.5%
- top 1%
- top 5%
- top 10%
- top 20%
- possibly top 50%

Train on matched protocols and compare:
- behavioral effect,
- robustness,
- generalization,
- and score concentration.

### Why it matters
This is one of the most basic missing ablations. It tests whether:
- the effect lives only in a tiny score tail,
- or scales smoothly with broader inclusion.

### Key outputs
- effect vs \(\gamma\),
- robustness vs \(\gamma\),
- training efficiency vs \(\gamma\),
- possible phase transitions in behavior.

---

## Follow-up 2: Dilution curve

### Question
How much random or unfiltered data can be mixed into the filtered subset before the effect weakens substantially?

### Design
Fix a high-scoring LLS subset and create mixtures with random examples:
- 100% filtered
- 75/25 filtered/random
- 50/50
- 25/75
- 10/90
- 5/95

Optionally also compare dilution with:
- random data,
- unfiltered in-domain preference data,
- negatively scored or near-zero examples.

### Why it matters
This directly probes whether the filtered set acts like:
- a concentrated poison / sparse signal,
- or a robust directional bias that survives substantial dilution.

### Key outputs
- effect vs dilution fraction,
- robustness vs dilution,
- comparison across targets and models.

---

## Follow-up 3: Score-bucket ablation

### Question
Which parts of the score distribution actually matter?

### Design
Partition examples into score quantiles or buckets:
- top 0.1%
- next 0.9%
- next 4%
- middle positive
- near zero
- negative

Train matched-size models using examples only from each bucket.

Also consider cumulative versions:
- top 0.1% only,
- top 1%,
- top 5%,
- top 10%,
- etc.

### Why it matters
This is the clearest way to distinguish:
- sparse extreme-tail effects,
- from broad distributed effects.

### Key outputs
- effect per bucket,
- cumulative gain curves,
- marginal contribution of each additional bucket,
- evidence of thresholding or smooth accumulation.

---

## Follow-up 4: Compound-prompt composition

### Question
Can filtered datasets induce compound preferences, and do multiple target directions compose additively or interfere?

### Design
Construct target prompts:
- \(s_1\): trait A
- \(s_2\): trait B
- \(s_{1+2}\): compound trait A+B

Create datasets filtered for each target and compare:
- training on \(D(s_1)\),
- training on \(D(s_2)\),
- training on \(D(s_{1+2})\),
- sequential training on \(D(s_1)\) then \(D(s_2)\),
- mixed training on \(D(s_1) \cup D(s_2)\).

### Why it matters
This tests whether these directions behave like approximately linear features or whether they interfere in complex ways.

### Key outputs
- additivity vs interference,
- whether compound training exceeds sequential or mixed training,
- whether one trait dominates or suppresses another.

---

## Follow-up 5: Empirical \(w_i\) analysis

### Question
What does the score distribution look like, and what structural information does it reveal?

### Design
For each target, model, and dataset:
- plot score histograms and CDFs,
- estimate quantile statistics,
- compute rank overlap across targets,
- compute rank overlap across teacher models,
- inspect top examples qualitatively,
- correlate score with downstream influence where possible.

If feasible, also compare:
- same-model teacher scoring,
- cross-model teacher scoring,
- and different student families.

### Why it matters
This is the core descriptive analysis needed to decide whether the mechanism is:
- sharp-tail and sparse,
- or diffuse and approximately linear.

### Key outputs
- full empirical characterization of \(w_i\),
- target-specific and model-specific differences,
- cross-model stability of rankings,
- evidence for universality vs model-specificity.

---

## 10. Optional Additional Analyses

## 10.1 Qualitative analysis of top-scoring examples
Manually inspect the top-ranked examples to answer:
- Are they semantically interpretable?
- Are they obviously related to the target?
- Do they contain stylistic or pragmatic cues?
- Are they surprising or innocuous?

## 10.2 Influence-style attribution
Estimate whether top-scoring examples also have high estimated influence on target behavioral probes.

## 10.3 Representation-space analysis
Measure whether fine-tuning on filtered subsets shifts hidden states or output preferences in a direction aligned with the target system prompt.

## 10.4 Follow-up fine-tuning erosion
After inducing a subliminal trait, perform additional benign fine-tuning and measure whether the effect:
- persists,
- erodes smoothly,
- or collapses quickly.

This is especially relevant if we want to compare subliminally learned traits to safety-training or post-training robustness questions.

---

## 11. Expected Outcomes and Interpretive Branches

## Outcome A: Extreme-tail dependence
If only the highest-score examples matter, and the effect collapses under dilution or small perturbations, then the phenomenon looks more like:
- sparse adversarial-feature exploitation,
- poisoning-style concentration,
- or brittle trigger induction.

## Outcome B: Broad positive-tail dependence
If the effect scales smoothly across broad positive-score buckets and survives dilution, then the phenomenon looks more like:
- distributed directional learning,
- low-rank behavioral steering,
- or log-linear aggregation of many weak signals.

## Outcome C: Strong same-family dependence
If same-family settings strongly outperform cross-family settings, that suggests:
- model-specific latent associations,
- partial non-universality of the selected directions,
- and a closer connection to shared representations or shared initialization.

## Outcome D: Composition succeeds
If compound traits compose well, that supports:
- approximate linearity of behavioral directions,
- and a more geometric interpretation of the filter.

## Outcome E: Composition interferes
If compound preferences interfere substantially, that suggests:
- nonlinearity,
- feature competition,
- or training dynamics that cannot be captured by a simple additive picture.

---

## 12. Main Deliverables

The project should aim to produce:

1. **A clear mechanistic framing**
   - subliminal learning as accumulation of per-example behavioral pushes

2. **A descriptive map of score structure**
   - histograms, CDFs, bucket effects, overlap analyses

3. **A concentration analysis**
   - whether the effect is sparse or distributed

4. **A robustness analysis**
   - perturbation, dilution, follow-up training, and transfer

5. **A comparison across mechanisms**
   - LLS vs alternative subliminal-learning methods where possible

6. **A composition analysis**
   - whether multiple trait directions add, interfere, or collapse

---

## 13. Summary of What We Most Want to Understand

The central intellectual goal of this project is not merely to show that subliminal learning exists again.

It is to understand:

- whether LLS is selecting examples that each provide a **small directional push** toward the target system prompt,
- whether the resulting behavior is concentrated in a few extreme examples or spread across many weakly aligned ones,
- whether this is best interpreted as a form of **adversarial-feature accumulation** or as a more general **log-linear / low-rank feature steering** phenomenon,
- and how fragile or robust the learned behavior is under perturbation, dilution, and composition.

In other words, the project is about turning the current intuition into a real empirical picture of **what kind of learning signal subliminal filtering is extracting**, and what that implies about model behavior, transfer, and robustness.