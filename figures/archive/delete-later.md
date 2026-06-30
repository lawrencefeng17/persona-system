Two parts.

**1. Is $-s_\theta(r^+)$ regular NTP loss?** Yes. $s_\theta(r^+)=\log P_\theta(r^+\mid p)=\sum_t \log P_\theta(r^+_t\mid p, r^+_{<t})$ over the completion tokens with the prompt masked, so $-s_\theta(r^+)$ is exactly completion-only next-token cross-entropy on the chosen response. Standard SFT.

**2. How signed-SFT is implemented.** Not via NTP code â it reuses the DPO machinery (`SignedDPOTrainer`, a `DPOTrainer` subclass). Per batch:
- One forward pass over the concatenated `[chosen; rejected]`, prompt-masked, giving per-example completion log-probs $s_\theta(r^+), s_\theta(r^-)$.
- TRL forms $\delta = \big(s_\theta(r^+)-s_{\text{ref}}(r^+)\big) - \big(s_\theta(r^-)-s_{\text{ref}}(r^-)\big)$ (`delta_score`).
- The loss branch is selected by `--loss-type`: DPO = $-\log\sigma(\beta\delta)$; hinge = $\max(0,1-\beta\delta)$; **linear** = I monkeypatch `F.logsigmoid â identity` so the sigmoid branch computes $-\beta\delta$ exactly. Mean over the batch.

So the *implemented* linear loss is $-\beta\big(s_\theta(r^+)-s_\theta(r^-)\big)$ + const, with **$\beta=0.04$** (which just folds into the effective LR â linear's $\beta$ and lr are degenerate) and the reference an additive constant that cancels from the gradient. The table wrote $-(s(r^+)-s(r^-))$ dropping the $\beta$ for readability.

**Two honest caveats on exactness:**
- **Sum vs mean.** The DPO-family losses (linear/hinge/DPO) use TRL's **token-summed** $s_\theta$, whereas the #23 "plain SFT on râº" anchor was run through a *different* script (`SFTTrainer`, completion-only) that uses **token-mean** CE. So the ladder's $-s_\theta(r^+)$ is the conceptual form; the SFT anchor isn't literally the token-summed version. It doesn't affect the conclusion (the SFT null is robust across #23's whole lr/rank sweep), but the normalization isn't identical to the contrastive arms.
- That's why the linear arm's "true SFT-with-a-minus-sign" claim is exact at the *gradient-direction* level ($\nabla s(r^+)-\nabla s(r^-)$), modulo the sum-vs-mean normalization between the two scripts.

If you want a perfectly matched ladder, the clean fix is to run plain SFT on râº *inside* the same `SignedDPOTrainer` (e.g. a `loss_type="sft_chosen"` that returns $-s_\theta(r^+)$ with the identical token-summed logps) â one more small branch. Worth doing if we publish the ladder.