# ComplexAttn

**A two-experiment investigation: does complex-valued algebra give better
inductive bias for logical negation than the standard real-valued setup?**

The headline conclusion (with the experiments to back it up): the
*algebra* that matters for parity-of-negation is **multiplicative
composition**, not specifically complex numbers. The complex-valued
attention model from Experiment 1 *learned* a phase-flip representation
of NOT (77% of pairs rotated by ≈π) but **could not compose it** — softmax
attention averages, it doesn't multiply. When you put the same complex
algebra inside a recurrent architecture that *does* compose
multiplicatively (Experiment 2), it generalizes to arbitrary nesting
depth from a tiny model. So does a standard real-valued GRU, because
sigmoid×tanh gating gives the *same* multiplicative inductive bias from
a different direction.

---

## The task (used by both experiments)

A synthetic, fully-controlled "parity of negation" task.

Each input contains exactly one **atom** (T = +1 or F = −1), some number
`k` of **NOT** tokens, and some number of semantically irrelevant
**filler** tokens, in a randomized order. The label is the final valence:

```
label = atom_value × (−1)^k
```

Order is randomized so positional templates cannot help; the model must
actually compose the negations.

**Train:** depths `k ∈ {0, 1, 2, 3}`.
**Eval:** depths `k ∈ {0, 1, …, 10}` — depths 4–10 are OOD.

A model that learns a *lookup table* over training depths fails at OOD
depths. A model that learns the underlying *parity algebra* generalizes.

See `data.py` for the generator.

---

## Experiment 1 — Complex transformer vs real transformer

**Question.** With everything else matched, does giving a transformer
complex hidden states (Hermitian inner-product attention, modReLU,
complex `e^(iωt)` positions) produce better depth generalization than a
real-valued baseline?

**Models.** Two pre-norm transformer encoders, parameter-matched within
~1%. Real transformer at d=92 (310k params) vs complex transformer at
d=64 (303k params). Same n_layers (3), same n_heads (4), same training
compute.

**Run:** `python3 run.py --config full`

### Result (5 seeds, H100)

| | Real (d=92, 310k) | Complex (d=64, 303k) |
|---|---|---|
| In-distribution acc (depths 0–3)   | **1.000**  | **1.000**  |
| Out-of-distribution acc (depths 4–10) | **0.488**  | **0.445**  |
| Phase probe `mean \|Δ\|` after one NOT | —  | **2.43 rad** (target π=3.14) |
| Phase probe `frac near π`          | —  | **0.77**   |

**Reading the result.**

1. The headline hypothesis was **not supported**: both models memorized
   the training distribution perfectly and then fell to chance OOD. The
   complex model was *slightly worse* OOD, consistently across seeds.

2. But the mechanistic probe was **unambiguously positive**: a random
   complex network gives `mean |Δ| ≈ 0.17 rad`. The trained complex
   network reached **2.43 rad with 77% of NOT-rotations within π/8 of π**.
   The model genuinely *did* learn that NOT acts as a phase rotation
   close to π.

3. The paradox — mechanism present, generalization absent — has a
   clean explanation: **attention is fundamentally additive**. Softmax
   weights sum to 1, so attention *averages* phase contributions across
   tokens instead of *multiplying* them. The algebra was learned inside
   a wrapper that doesn't respect the algebra's group structure. The
   correct phase-flip for NOT is there in the weights, but composing
   five of them via attention doesn't give `(−1)^5 = −1`. It gives
   noise.

This is more informative than a clean win or clean loss would have
been: it tells us *exactly* what the next experiment must change.

---

## Experiment 2 — Multiplicative composition

**Question.** When we give the same complex inductive bias to an
architecture that *does* compose multiplicatively, does depth
generalization appear?

**Three architectures**, all multiplicative-composing:

| | params (d_model) | composition mechanism |
|---|---|---|
| **PhaseSumNet** | ~270 (d=16) | Sum learned per-token phases θ(x); readout from cos/sin of total. Mathematically equivalent to product of complex unit factors. Set-equivariant by design. |
| **GatedComplexRNN** | ~3,000 (d=32) | Bidirectional gated complex RNN. Per-token rotation `exp(iθ(x))` of the running state, with a per-token additive complex contribution and a sigmoid gate. |
| **GRUBaseline** | ~3,000 (matched) | Standard bidirectional GRU. Real-valued, but sigmoid×tanh gating gives multiplicative composition implicitly. |

**Predicted optimal solution for PhaseSumNet:**

```
θ(T)      = 0     (factor +1)
θ(F)      = π     (factor −1)
θ(NOT)    = π     (factor −1)
θ(filler) = 0     (factor +1)
```

Sum of phases mod 2π is 0 iff the label is +1, π iff −1.

**Run:** `python3 run_rnn.py --config full`

### Expected result

All three architectures should generalize to OOD depths (4–10 or
beyond), because all three encode multiplicative composition. The real
GRU is the *informative* control: if it generalizes too, then the
lesson isn't "use complex numbers", it's "use multiplicative
composition" — of which complex phases are one minimal example and
gated recurrence is another.

A pilot run (CPU, single seed, depth 0–3 train → 0–10 eval) gave:

| depth | PhaseSumNet (273 p) | GatedComplexRNN (3k p) | GRU (3k p) |
|------:|--------------------:|------------------------:|------------:|
| 0 | 0.97 | 1.00 | 1.00 |
| 4 (OOD) | 0.97 | 1.00 | 1.00 |
| 7 (OOD) | 0.97 | 1.00 | 1.00 |
| 10 (OOD) | 0.94 | 1.00 | 1.00 |

The full H100 run with 5 seeds and depths up to 15 is what `--config
full` produces.

### What to look at in the probe

For Experiment 2 we can read the per-token phase *directly* (no
behavioral probe needed). The summary reports for `PhaseSumNet`:

- `cos(NOT)` — should be near −1 if the predicted optimum is learned.
- `cos(T)`, `cos(F)` — should be near +1 and −1 respectively.
- `frac_not_near_pi` — fraction of NOT phase dimensions within π/8 of ±π.

The model doesn't always converge to the *exact* predicted optimum: with
d>1 phase dimensions and a linear readout, many equivalent solutions
exist. What matters is the *behavior*: does it generalize? And the
answer in our pilot was yes.

---

## How the experiments fit together

| | Experiment 1 (additive) | Experiment 2 (multiplicative) |
|---|---|---|
| Architecture | Transformer (softmax attention) | Phase-sum / gated RNN / GRU |
| Composition | Additive (weights sum to 1) | Multiplicative |
| OOD generalization | ❌ chance | ✅ near-perfect |
| Complex helps? | No (no benefit over real) | Yes — *and so does real with right gating* |

The combined story:

> *Complex numbers, by themselves, are not the inductive bias that helps
> with negation. They are a notation. The inductive bias that helps is
> multiplicative composition. Whether you get that from complex unit
> rotations, from real ±1 sign products, or from sigmoid-gated tanh
> updates is a design choice. What you cannot do is approximate
> multiplicative composition with softmax-additive attention and expect
> the algebra to compose at depth — it provably won't.*

---

## How to run

```bash
git clone https://github.com/Minds-R-Lab/ComplexAttn.git
cd ComplexAttn
pip install -r requirements.txt

# Sanity checks (~1 minute each, CPU is fine)
python3 run.py     --config smoke
python3 run_rnn.py --config smoke

# Full experiments (H100 recommended)
python3 run.py     --config full     # Experiment 1: transformer, ~30 min
python3 run_rnn.py --config full     # Experiment 2: recurrent, ~30 min
```

Override device or output directory if needed:

```bash
python3 run_rnn.py --config full --device cuda --outdir results_exp2_h100
```

Hyperparameters live in `CONFIGS` at the top of each `run*.py`; edit
freely.

---

## Project layout

```
ComplexAttn/
├── README.md          this file
├── requirements.txt   torch, numpy, matplotlib, tqdm
├── data.py            synthetic data generator and vocabulary

├── models.py          Experiment 1: real & complex transformer encoders
├── analyze.py         Experiment 1: behavioral phase probe (paired sentences)
├── run.py             Experiment 1 orchestrator

├── rnn_models.py      Experiment 2: PhaseSumNet, GatedComplexRNN, GRUBaseline
├── analyze_rnn.py     Experiment 2: direct read-off of learned phases
├── run_rnn.py         Experiment 2 orchestrator

└── train.py           shared training loop with per-depth stratified eval
```

After running, `results/` (Exp 1) and `results_exp2/` (Exp 2) contain:

```
results.json           all per-seed metrics, machine-readable
summary.txt            headline numbers in plain text
depth_accuracy.png     accuracy per depth, both models, ID and OOD
training_curves.png    train loss + eval accuracy over training steps
phase_probe.png        (Exp 1) Δ angle histogram
phase_per_dim.png      (Exp 2) histogram of |learned NOT-phase| per dim
```

---

## Known limitations

- **One synthetic task is one data point.** Both experiments use a
  controlled parity setup. Real-language negation has more structure:
  scope, modality, sarcasm, double-negation idioms. A positive Experiment
  2 result is a *prompt* to test on NLI / scoped quantifiers, not a
  proof at scale.

- **The real-GRU result confirms the diagnosis, but also limits the
  excitement.** If standard real-valued recurrences solve this with the
  same generalization, the case for *specifically complex* networks
  needs a task where the algebra is genuinely richer than ±1 sign
  composition — e.g., rotations mod 3, continuous phase, or quaternionic
  pose data. That is a different experiment.

- **Convergence depends on initialization.** PhaseSumNet and
  GatedComplexRNN must initialize phases uniformly in [−π, π], not
  near zero, or the gradient signal is too weak to escape (all sentences
  produce cos ≈ 1 → constant logits). This is fixed in the current
  code; the lesson is that "complex inductive bias" only helps if you
  let the architecture *use* the unit circle.

---

## Suggested follow-ups

1. **Sample-complexity sweep.** Re-run Experiment 2 with `n_train ∈
   {1k, 3k, 10k, 30k, 100k}`. If PhaseSumNet generalizes from fewer
   examples than GRU at matched params, that is a real inductive-bias
   claim, not just an existence proof.

2. **Non-binary phase task.** Replace `(−1)^k` with a `(e^{i·2π/3})^k`
   structure (three-way parity). Real ±1 networks cannot solve this
   without growing capacity; complex unit-circle phases can solve it
   with d=1. This would be the cleanest possible test that the
   *complex* structure (not just multiplicative composition) is
   doing work.

3. **Real-language transfer.** Conditional NegNLI (Hossain et al.,
   2020), negated RTE, scoped quantifier benchmarks. Same
   architectural family, real negation.

4. **Scale the gating insight.** If sigmoid×tanh gating in GRU is what
   gives multiplicative composition, ask: which architectural families
   already have it (LSTM, Mamba, RWKV) and which don't (vanilla
   transformer, MLP)? Predict which generalize parity OOD.

---

## Background reading

- Trabelsi et al., *Deep Complex Networks*, 2018 — modReLU, complex
  batch norm; engineering blueprint for stable complex training.
- Arjovsky et al., *Unitary Evolution Recurrent Neural Networks*, 2016
  — closest existing argument that unitary structure changes what an
  RNN can represent in principle.
- Hahn, *Theoretical Limitations of Self-Attention in Formal Languages*,
  TACL 2020 — formal account of why softmax transformers struggle with
  parity. Directly relevant to Experiment 1's failure mode.
- Bhattamishra et al., *On the Ability and Limitations of Transformers
  to Recognize Formal Languages*, EMNLP 2020 — empirical companion to
  Hahn.
- Sordoni, Bengio, Nie, *Modeling Term Dependencies with Quantum
  Language Models*, SIGIR 2013 — early use of complex amplitudes in
  language.

---

## Citation

```bibtex
@misc{complexattn2026,
  title  = {ComplexAttn: complex- and real-valued tests of multiplicative
            composition for logical negation},
  author = {Minds-R-Lab},
  year   = {2026},
  url    = {https://github.com/Minds-R-Lab/ComplexAttn}
}
```

## License

MIT. See `LICENSE`.
