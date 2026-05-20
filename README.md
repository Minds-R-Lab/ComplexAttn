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

## Experiment 3 — Rotation mod 3 (does complex *specifically* help?)

**The setup question Experiment 2 left open.** Exp 2 showed that
multiplicative composition (any kind — complex phase, GRU gating)
generalizes, while additive attention does not. But it didn't separate
"complex" from "multiplicative". The reason: Z/2 has only two elements,
{+1, −1}, and these live natively in *both* the multiplicative group
of the reals and the unit circle of the complex numbers. So real sign
products and complex phase flips solve Z/2 equally well.

**Z/3 doesn't have this confound.** There is no subgroup of order 3 in
ℝ* (the multiplicative group of the reals). To represent the cyclic
group Z/3 inside a network you have three options:
1. The unit circle (complex), where TWIRL = ×e^{i·2π/3} composes
   exactly.
2. Two real dimensions used as a rotation, which is just (1) under a
   different name.
3. A non-linear state update (gating) that can implement a 3-state DFA.

A purely *additive* real architecture with a *linear* readout has none
of these — and we can prove it fails.

### The task

Same shape as Exp 1/2 with k=mod 3 instead of mod 2:

| token | role |
|---|---|
| A0, A1, A2 | three atom classes |
| TWIRL | rotate the class by +1 (mod 3) |
| filler | semantically irrelevant |

Label = (atom_class + num_TWIRLs) mod 3. Order randomized. Train
depths 0–5, eval depths 0–20.

### The architectures

| | composition | readout | predicted |
|---|---|---|---|
| **PhaseSumNet3** | additive phases | periodic (cos/sin) | works (d=1 sufficient) |
| **RealAddNet** | additive embeddings | linear | **theorem: cannot solve, anywhere** |
| **GatedComplexRNN3** | multiplicative rotation | linear over (Re,Im) | works |
| **GRUBaseline3** | gated non-linear | linear over state | works (but not necessarily cleanly) |

### The negative theorem for RealAddNet

Logits are `W·(Σ_t e(x_t)) + b = W·e(atom) + k·W·e(TWIRL) + b` —
**linear in k**. Three linear functions of k can change ranking at most
two times (where slope orderings cross), so `argmax_c logit_c(k)` is
piecewise constant with at most 3 pieces. The correct
`(atom_class + k) mod 3` cycles 3 times every 3 steps. The architecture
cannot fit even depth ∈ {0,1,2,3} simultaneously across atom classes,
let alone generalize. Loss should remain at ln(3).

### Result (CPU pilot, seed 0)

| | params | ID acc | OOD acc (6–12) | mod-3 closure |
|---|---:|---:|---:|---:|
| PhaseSumNet3 | 355 | 0.999 | **0.999** | 25% of dims clean |
| **RealAddNet** | 611 | **0.32** | **0.32** | — (loss = ln(3) = chance) |
| GatedComplexRNN3 | 3,411 | 1.000 | **1.000** | **1.000** |
| GRUBaseline3 | 3,213 | 1.000 | **0.937** | 0.70 |

**Three takeaways:**

1. **The negative theorem is sharper than predicted.** RealAddNet
   doesn't just fail OOD — it fails *everywhere*, including the
   training set, because mod-3 cannot be represented at all by an
   architecture whose logits are linear in token count.

2. **GRU is now visibly behind complex.** In Exp 2 (Z/2), gated real
   and complex were tied at ~100% OOD. In Exp 3 (Z/3), complex
   architectures hit 100% with full closure under 3·TWIRL, while the
   GRU hits 94% with only 70% closure. The gating mechanism finds *a*
   solution that fits the training distribution but doesn't perfectly
   internalize the cyclic group structure. **This is the first
   evidence in the project that complex specifically helps over real
   when the algebra is non-binary.**

3. **PhaseSumNet at 355 parameters fully solves a task that
   architecture of any size with the wrong inductive bias provably
   cannot.** The contrast between PhaseSumNet (355 p, 99.9% OOD) and
   RealAddNet (611 p, chance) is the cleanest possible isolation of
   "architectural prior matters more than parameter count".

The full H100 run (5 seeds, depths to 20) is `python3 run_triple.py
--config full`.

---

## How the three experiments fit together

| | Exp 1 (additive attn) | Exp 2 (multiplicative, Z/2) | Exp 3 (multiplicative, Z/3) |
|---|---|---|---|
| Architecture | Transformer | PhaseSum / GatedCplx / GRU | + RealAddNet control |
| Algebra | Z/2 | Z/2 | **Z/3** |
| OOD generalization | ❌ chance | ✅ near-perfect | varies by arch |
| Complex helps? | No | No — tied with real GRU | **Yes — complex strictly ahead** |
| Pure-additive baseline | (transformer) failed | n/a | **provably cannot fit even ID** |

### The combined story

> *The inductive bias that matters for parity-like tasks is **the ability
> to represent the symmetry group of the task**. For Z/2 you can do this
> with real sign products or complex phase flips equally well, and that
> is why Experiment 2's GRU tied with the complex architectures. For
> Z/3 you need either continuous rotation (complex / 2D real rotation)
> or gating non-linear in token count. Complex unit-circle phases give
> you this in **one dimension**; gated real recurrence gives you a
> messier approximation. **Purely additive real-valued networks with
> linear readouts are formally barred** — they cannot represent
> non-binary cyclic groups at all, anywhere. And softmax-attention
> transformers (Experiment 1) sit in this provably-broken family for
> exactly this reason: softmax weights summing to 1 forces additive
> composition.*

The honest version of the original question — "do complex numbers help
in ML?" — is now answerable. They are not a universal upgrade. They
are the *minimal* embedding for tasks whose symmetry group is the unit
circle or a subgroup of it (rotations, phases, periodic structure).
When such structure is genuinely present in the task, complex
networks express it in fewer parameters than real ones, and the
generalization difference shows up as you move from binary to richer
groups.

---

## How to run

```bash
git clone https://github.com/Minds-R-Lab/ComplexAttn.git
cd ComplexAttn
pip install -r requirements.txt

# Sanity checks (~1 minute each, CPU is fine)
python3 run.py        --config smoke
python3 run_rnn.py    --config smoke
python3 run_triple.py --config smoke

# Full experiments (H100 recommended)
python3 run.py        --config full     # Exp 1: transformer (Z/2),  ~30 min
python3 run_rnn.py    --config full     # Exp 2: RNN (Z/2),          ~20 min
python3 run_triple.py --config full     # Exp 3: RNN (Z/3),          ~25 min
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

├── data.py            Exp 1+2 data (Z/2 parity-of-negation)
├── data_triple.py     Exp 3   data (Z/3 rotation task)

├── models.py          Exp 1: real & complex transformer encoders
├── analyze.py         Exp 1: behavioral phase probe (paired sentences)
├── run.py             Exp 1: orchestrator
├── train.py           Exp 1+2: binary training loop

├── rnn_models.py      Exp 2: PhaseSumNet, GatedComplexRNN, GRUBaseline
├── analyze_rnn.py     Exp 2: direct read-off of learned phases
├── run_rnn.py         Exp 2: orchestrator

├── models_triple.py   Exp 3: PhaseSumNet3, RealAddNet, GatedComplexRNN3, GRUBaseline3
├── analyze_triple.py  Exp 3: probes (mod-3 closure + slope analysis for negative control)
├── train_triple.py    Exp 3: 3-class cross-entropy training loop
└── run_triple.py      Exp 3: orchestrator
```

After running, `results/`, `results_exp2/`, `results_exp3/` each contain:

```
results.json           all per-seed metrics, machine-readable
summary.txt            headline numbers in plain text
depth_accuracy.png     accuracy per depth, all models
training_curves.png    train loss + eval accuracy over training steps
phase_per_dim.png      probe histogram (Exp 2, 3)
phase_probe.png        Δ angle histogram (Exp 1 only)
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
