# ComplexAttn

**Do complex-valued transformers learn negation as a phase rotation?**

A controlled experiment isolating one structural claim: that a transformer
whose hidden states live in ℂ, and which can therefore propagate phase
through attention, has a better inductive bias for logical negation than
a real-valued transformer of equal capacity.

This repository contains a complete, reproducible end-to-end test of that
hypothesis: a synthetic task, two parameter-matched architectures, a
training loop, a mechanistic probe, and the plotting code that turns it
all into a publishable figure.

---

## Why this experiment exists

The default in modern ML is real numbers. This is a historical accident,
not a principled choice — gradient descent needs a total order, loss is
real, GPUs were built for real matmul, and we inherited the calculus of
landscapes from 19th-century physics. None of that is a discovery about
intelligence.

The natural question is whether there are *specific phenomena* that
real-valued networks handle poorly and that a different algebra would
handle naturally. **Negation is a good candidate.** In language, "not"
is destructive: "not happy" is not "happy with a flag." Cancellation,
inversion, scoped negation, double-negation parity — these are
*multiplicative* phenomena being approximated by *additive* attention.

If you give a transformer hidden states in ℂ and let attention propagate
phase, negation could in principle be a single π rotation that composes
exactly:

> e^(iπ) · e^(iπ) · ... · e^(iπ) · v = (−1)^k · v

This experiment asks whether such a representation actually emerges from
training, and whether it produces better out-of-distribution
generalization than a real-valued baseline.

It is a *narrow* experiment. The point is to test one claim cleanly, not
to argue that complex-valued LLMs will beat real ones at scale.

---

## The task

A synthetic, fully-controlled "parity of negation" task.

Each input is a token sequence containing:
- exactly one **atom**: `T` (valence +1) or `F` (valence −1),
- some number `k` of `NOT` tokens (the *negation depth*),
- some number of semantically irrelevant **filler** tokens.

The label is the final valence:

```
label = atom_value × (−1)^k
```

Token order is **randomized**, so positional templates cannot help. The
model must actually compose the negations.

Example sequences and labels (with atom = T, valence +1):

| k | example tokens (CLS prepended)                | label  |
|---|------------------------------------------------|--------|
| 0 | `CLS the T very`                               | +1     |
| 1 | `CLS so NOT T very`                            | −1     |
| 2 | `CLS NOT very T NOT the`                       | +1     |
| 3 | `CLS T NOT really NOT NOT so`                  | −1     |

**Train distribution:** `k ∈ {0, 1, 2, 3}`.
**Eval distribution:** `k ∈ {0, 1, …, 10}`. Depths 4–10 are
out-of-distribution.

A model that has learned a *lookup table* over training depths will fail
at OOD depths. A model that has learned the underlying *parity algebra*
will generalize.

---

## The two models

Both are pre-norm transformer encoders with identical layer count and
head count. The complex model's hidden dimension `d_complex` is chosen
together with the real model's hidden dimension `d_real` so the two have
nearly equal parameter counts (within ~1% for the full config — see
`matched_configs()` in `models.py`).

The only structural difference:

| component             | Real model               | Complex model                                          |
|-----------------------|--------------------------|--------------------------------------------------------|
| token embedding       | ℝᵈ                       | ℂᵈ (two real tables)                                   |
| positional encoding   | sin / cos interleaved    | `e^(iωt)` — natural complex Fourier basis              |
| attention scores      | Q · Kᵀ                   | **Re(Q · K\*)** — Hermitian inner product (still real, still softmax-able) |
| value aggregation     | weighted sum             | **weighted sum of complex V** — phase propagates       |
| activation            | GELU                     | **modReLU**: `ReLU(\|z\| + b) · z/\|z\|` — preserves phase, gates magnitude |
| layer norm            | standard                 | normalizes E[\|z\|²], complex affine                    |
| readout               | linear → logit           | complex linear → ℂ¹, **Re(·)** → logit                 |

Implementation note: every complex tensor is represented as a *pair* of
real tensors. PyTorch's complex autograd has edge cases this avoids; the
chain rule is fully standard for pairs of reals.

---

## The mechanistic probe

Even if the complex model wins, we want to know **why**. The probe
(`analyze.py`) takes pairs of sentences differing by exactly one extra
NOT and measures the angular difference of the complex CLS readout.

- **Hypothesis**: adding one NOT rotates the readout by ≈ π radians,
  regardless of `k`, filler count, or atom.
- **Falsifier**: a diffuse angle distribution.

The summary reports:
- `mean |Δ angle|`     — target value π ≈ 3.14
- `frac near π`        — fraction of pairs within π/8 of π. Target 1.0.

This is what separates *"the complex model won"* from *"the complex
model won **because of phase**."* If accuracy improves but the probe is
diffuse, the win is due to something else (parameter count, optimization
landscape, etc.).

---

## Quick start

```bash
git clone https://github.com/Minds-R-Lab/ComplexAttn.git
cd ComplexAttn
pip install -r requirements.txt

# Sanity check: ~1 minute on CPU. Verifies the pipeline only — the
# model is too small to produce meaningful science.
python3 run.py --config smoke

# Full experiment. Designed for a single H100 (or any modern GPU).
# Wallclock: ~1–2 hours.
python3 run.py --config full
```

Override device or output directory if needed:

```bash
python3 run.py --config full --device cuda --outdir results_h100
```

Hyperparameters live in the `CONFIGS` dict at the top of `run.py`; edit
freely.

---

## Project layout

```
ComplexAttn/
├── README.md          this file
├── requirements.txt   torch, numpy, matplotlib, tqdm
├── data.py            synthetic data generator and vocabulary
├── models.py          RealTransformer, ComplexTransformer, param matcher
├── train.py           training loop + per-depth evaluation
├── analyze.py         phase probe (mechanistic check)
└── run.py             orchestrator: seeds × models × plots + summary
```

After running, `results/` will contain:

```
results.json          all per-seed metrics, machine-readable
summary.txt           headline numbers in plain text
depth_accuracy.png    mean ± stderr accuracy per depth, both models
training_curves.png   train loss + eval accuracy over steps
phase_probe.png       histogram of |Δ angle| after one extra NOT
```

---

## How to read the results

The hypothesis is supported if **all three** hold:

1. **Both models reach high in-distribution accuracy.** Otherwise the
   comparison is meaningless — they didn't learn the task.
2. **Complex model OOD accuracy is substantially higher** than real
   model OOD accuracy at depths beyond the training range.
3. **Phase probe: `mean |Δ|` close to π and `frac_near_pi` close to 1.**
   This is the mechanistic confirmation.

If (1) and (2) hold but (3) doesn't, the result is *interesting but
ambiguous*: the complex parameterization helps, but not for the
predicted reason.

If (2) fails — both models generalize equally — the hypothesis is **not
supported at this scale**. This is itself a useful result, with two
likely interpretations:
- additive softmax attention cannot represent multiplicative phase
  composition cleanly (softmax weights sum to 1, so more NOTs *dilute*
  rather than accumulate phase contribution), or
- the synthetic task is too easy and both models find the same
  shortcut.

Either case points to the natural follow-up: a **gated/recurrent**
complex architecture where phase composition is multiplicative by
construction.

---

## Known limitations

- **Attention is fundamentally additive.** Softmax-weighted summation of
  complex values does not implement multiplicative phase composition
  directly. The complex model can approximate it by spreading the
  computation across layers (each layer effectively "peels off" one
  NOT via phase rotation), but it is depth-limited by `n_layers`. A
  cleaner test of the multiplicative-composition hypothesis would use a
  gated complex RNN; that is a follow-up, not this experiment.

- **Both models can in principle solve the task by counting.** Real
  transformers struggle with parity (Hahn, 2020) but are not provably
  incapable. A real model that finds an accurate counting strategy will
  generalize too. The result is informative either way.

- **One synthetic task is one data point.** A positive result here is a
  *prompt* to test the same hypothesis on real language (negation in
  NLI, scoped quantifiers, etc.), not a proof about LLMs at scale.

---

## Suggested follow-up experiments

If results are **positive** ((1)+(2)+(3) hold):

- **Sample complexity.** Sweep `n_train ∈ {1k, 3k, 10k, 30k, 100k}` and
  plot accuracy vs training-set size. If the complex model needs
  strictly fewer samples to generalize, that is a strong claim about
  inductive bias.
- **Depth scaling.** Train on depths `{0..D}` for `D ∈ {2, 3, 4, 5}`,
  eval to depth 15. Map OOD generalization vs training depth.
- **Transfer to real language.** Conditional NegNLI (Hossain et al.,
  2020), negated RTE, or scoped quantifier benchmarks. Same
  param-matched comparison.

If results are **negative**:

- **Mechanistic deep dive.** Linear probes on hidden states for "NOT
  count." Even when accuracy matches, the representations may not.
- **Architectural change.** Gated complex recurrent network where each
  NOT *multiplies* the running state. Tests whether the inductive bias
  surfaces when the math is made explicit.

---

## Background reading

The conceptual framing of this experiment owes to a number of older and
newer threads worth following up on if you're new to the area:

- Trabelsi et al., *Deep Complex Networks*, 2018 — modReLU, complex
  batch norm, the engineering blueprint for stable complex training.
- Arjovsky et al., *Unitary Evolution Recurrent Neural Networks*, 2016 —
  the closest existing argument that complex/unitary structure changes
  what a network can represent in principle.
- Hahn, *Theoretical Limitations of Self-Attention in Formal Languages*,
  TACL 2020 — why real transformers struggle with parity.
- Bhattamishra et al., *On the Ability and Limitations of Transformers
  to Recognize Formal Languages*, EMNLP 2020 — empirical companion to
  Hahn.
- Sordoni, Bengio, Nie, *Modeling Term Dependencies with Quantum Language
  Models*, SIGIR 2013 — early use of complex amplitudes in language,
  philosophically related though not architecture-driven.

---

## Citation

If this code is useful in published work, a citation to the repository
is appreciated:

```bibtex
@misc{complexattn2026,
  title  = {ComplexAttn: Testing phase-based negation in complex-valued transformers},
  author = {Minds-R-Lab},
  year   = {2026},
  url    = {https://github.com/Minds-R-Lab/ComplexAttn}
}
```

---

## License

MIT. See `LICENSE`.
