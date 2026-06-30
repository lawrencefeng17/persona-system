# Brief summary of the LLS scoring expression

Let a preference example be \((p, r^+, r^-)\), where \(p\) is the prompt, \(r^+\) is the preferred response, and \(r^-\) is the rejected response.  
For a system prompt \(s\), the Logit-Linear Selection (LLS) score is

\[
w
=
\Big(\log P(r^+\mid s,p)-\log P(r^-\mid s,p)\Big)
-
\Big(\log P(r^+\mid p)-\log P(r^-\mid p)\Big).
\]

## Equivalent forms

Rearranging terms gives

\[
w
=
\Big(\log P(r^+\mid s,p)-\log P(r^+\mid p)\Big)
-
\Big(\log P(r^-\mid s,p)-\log P(r^-\mid p)\Big).
\]

This shows that the score compares the **prompt-induced lift** on the preferred response to the prompt-induced lift on the rejected response.

Using log-ratios,

\[
w
=
\log \frac{P(r^+\mid s,p)}{P(r^-\mid s,p)}
-
\log \frac{P(r^+\mid p)}{P(r^-\mid p)}
=
\log\left(
\frac{P(r^+\mid s,p)/P(r^-\mid s,p)}
     {P(r^+\mid p)/P(r^-\mid p)}
\right).
\]

So \(e^w\) is an **odds ratio**:

\[
e^w
=
\frac{P(r^+\mid s,p)/P(r^-\mid s,p)}
     {P(r^+\mid p)/P(r^-\mid p)}.
\]

## Interpretation

A positive score \(w>0\) means that adding the system prompt \(s\) makes the model favor \(r^+\) over \(r^-\) **more strongly** than it already did without the system prompt.

So the score is **not** “removing the effect of the prompt entirely.”  
Rather, it removes the part of the preference that was already present without the system prompt, leaving the **incremental effect of the system prompt on the relative preference**.

A useful shorthand is

\[
\Delta_s(r) := \log P(r\mid s,p) - \log P(r\mid p),
\]

so that

\[
w = \Delta_s(r^+) - \Delta_s(r^-).
\]

That is: the LLS score measures how much more the system prompt boosts the chosen response than the rejected one.

## Geometric view under log-linearity

Under the approximate log-linear model,

\[
\log P(r\mid s,p) \approx \langle \psi(s), \phi(p,r)\rangle,
\]

the score becomes approximately

\[
w
\approx
\langle \psi(s)-\psi(\emptyset),\ \phi(p,r^+) - \phi(p,r^-)\rangle.
\]

This means LLS keeps examples whose **preference direction**
\[
\phi(p,r^+) - \phi(p,r^-)
\]
has a large positive projection onto the **system-prompt direction**
\[
\psi(s)-\psi(\emptyset).
\]

## One-sentence takeaway

LLS selects examples where the system prompt specifically increases the model’s relative preference for the chosen response over the rejected response.
