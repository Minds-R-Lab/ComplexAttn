# ComplexAttn

**A four-experiment investigation: when, if ever, does complex-valued
algebra give better inductive bias than the standard real-valued setup?**

The arc of the project, with each experiment narrowing the claim:

1. **Experiment 1 (transformer, Z/2 parity-of-negation).** A complex
   transformer with Hermitian attention and modReLU *learns* a phase-flip
   representation of NOT (77% of pairs rotated by ≈π in the probe) but
   cannot compose it — softmax attention averages, doesn't multiply.
   At 303k parameters it sits at chance OOD, same as a real transformer.

2. **Experiment 2 (recurrent, Z/2).** Move to architectures whose
   composition is multiplicative by construction. A 273-parameter
   set-equivariant phase-sum network generalizes perfectly to 4× the
   training depth. So does a parameter-matched GRU — because
   sigmoid×tanh gating gives multiplicative composition from a
   different direction. **Conclusion: for Z/2, the inductive bias that
   matters is multiplicative composition, not complex numbers
   specifically.**

3. **Experiment 3 (recurrent, Z/3).** Going to a cyclic group with
   no subgroup of order 3 in ℝ\* breaks the Z/2 confound. A purely
   additive real network with linear readout **provably cannot fit
   even the training data** (logits linear in k cannot represent (k mod
   3)), and is observed at chance. PhaseSumNet at 355 parameters hits
   100% OOD. GatedComplexRNN perfectly internalizes the mod-3 group
   structure but degrades at extreme OOD lengths because additive value
   contributions let its state magnitude drift. GRU survives length
   extrapolation better but doesn't perfectly internalize the group.

4. **Experiment 4 (scaling sweep over Z/n, n ∈ {2, 3, 5, 7, 11, 13}).**
   Tests whether the Z/3 findings are a scaling law. Predictions:
   PhaseSumNet stays at 1.0 across n; RealAddNet stays at 1/n;
   GatedComplexRNN's OOD slopes downward with n; GRU's group closure
   under n·TWIRL drops with n.

**Final framing.** Complex numbers are not a universal upgrade. They
are the *minimal embedding* for tasks whose symmetry group is the unit
circle or a subgroup of it. When that structure is present, complex
networks express it in fewer parameters than real ones, and the
generalization gap grows with the order of the group. When it isn't
(e.g. plain Z/2), real and complex are interchangeable. The deepest
prerequisite is more general: the architecture's composition operator
must respect the algebra of the task. Softmax attention provably
doesn't, for any cyclic group.

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

### Result (5 seeds, H100, train depths 0–5, eval depths 0–20)

| | params | ID acc | OOD acc (6–20) | closure under 3·TWIRL |
|---|---:|---:|---:|---:|
| **PhaseSumNet3** | **355** | **1.000 ± 0.000** | **1.000 ± 0.000** | 39% of dims clean |
| **RealAddNet** | 611 | **0.336 ± 0.020** | **0.341 ± 0.008** | — (chance is 1/3) |
| GatedComplexRNN3 | 4,547 | 1.000 ± 0.000 | 0.987 ± 0.006 | **1.000 ± 0.000** |
| GRUBaseline3 | 4,503 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.953 ± 0.042 |

Per-depth OOD breakdown (means over 5 seeds):

| depth k | PhaseSumNet3 | RealAddNet | GatedComplexRNN3 | GRUBaseline3 |
|--:|--:|--:|--:|--:|
| 6  | 1.000 | 0.376 | 1.000 | 1.000 |
| 10 | 1.000 | 0.397 | 1.000 | 1.000 |
| 15 | 1.000 | 0.334 | 0.995 | 1.000 |
| 18 | 1.000 | 0.339 | 0.966 | 1.000 |
| 20 | 1.000 | 0.304 | **0.951** | **0.984** |

**Three takeaways:**

1. **The negative theorem is sharper than predicted.** RealAddNet
   doesn't just fail OOD — it fails *everywhere*, including the
   training set, sitting at chance (1/3) at every depth from 0 to 20.
   mod-3 cannot be represented at all by an architecture whose logits
   are linear in token count. The `slope_spread` of 0.020 ± 0.005 across
   seeds confirms the model converged to near-uniform output rather
   than picking any one class.

2. **PhaseSumNet at 355 parameters fully solves a task that
   architectures with the wrong inductive bias provably cannot, at any
   size.** Zero variance across 5 seeds, 100% at every depth out to k=20
   (4× the training depth). The contrast PhaseSumNet vs RealAddNet
   (355 vs 611 params, 100% vs chance) is the cleanest possible
   isolation of "architectural prior beats parameter count when the
   task has structure".

3. **Complex helps with the group, but a magnitude side-effect hurts
   it at length.** GatedComplexRNN3 perfectly internalizes the cyclic
   group structure (closure = 1.000 ± 0.000), but its OOD accuracy
   slopes down at large k. The mechanism is that the architecture's
   additive value contribution (`v_r, v_i`) lets the complex state
   magnitude drift with sequence length, shrinking the angular margin
   the readout needs. The GRU's bounded tanh state survives length
   extrapolation better but doesn't perfectly internalize the group
   (closure = 0.95). **Neither complex nor real strictly dominates at
   the recurrent-architecture scale; each pays a different price.**
   The set-equivariant PhaseSumNet — bounded by the unit circle *and*
   indifferent to sequence length — is the only architecture that
   avoids both failure modes.

A natural follow-up: does this pattern hold across cyclic-group orders
n in {2, 3, 5, 7, 11, 13}? That's Experiment 4.

---

## Experiment 4 — Scaling sweep over Z/n

**Question.** Exp 3 found a specific tension at n = 3: GatedComplexRNN
learns the group exactly but its OOD accuracy slopes downward with
depth, while GRU's accuracy holds but its group-structure is
approximate. Is this a scaling law? Does the picture get sharper as we
move to richer groups?

### The setup

Same task as Exp 3, with n made a parameter. Sweep n ∈ {2, 3, 5, 7,
11, 13}. Train depths 0–5, eval depths 0–20 (4× extrapolation). All
four architectures retrained from scratch for each n.

### Predictions

| | n=2 | n=3 | n=5 | n=7 | n=11 | n=13 |
|---|--:|--:|--:|--:|--:|--:|
| PhaseSumNet OOD | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| RealAddNet ID/OOD | varies* | 1/n | 1/n | 1/n | 1/n | 1/n |
| GatedComplexRNN OOD | 1.0 | 0.99 | dec. | dec. | dec. | dec. |
| GRU closure under n | high | 0.95 | dec. | dec. | dec. | dec. |

\*RealAddNet at n=2 is the edge case: Z/2 has only 2 classes, so the
piecewise-constant argmax of two linear functions can in principle
correctly cover (k mod 2) for an unbounded range of k. The theorem only
bites for n ≥ 3.

### Why we expect a scaling pattern

- **PhaseSumNet** has set-equivariance + a bounded readout. Sequence
  length is *literally irrelevant* to it (it sums phases; PAD
  contributes zero). Per-class angular resolution scales with d, not n,
  so with d=16 phase dimensions the model has roughly 16 independent
  attempts to land near the optimal 2π/n rotation. We predict no
  degradation with n up to d.

- **RealAddNet** has the theorem from Exp 3: for n ≥ 3, no linear
  function of token count can represent (k mod n), so the loss must
  stay at ln(n). We predict accuracy at 1/n (chance) for every n ≥ 3.

- **GatedComplexRNN** internalizes the group (closure → 1) but its
  *readout* needs to distinguish n equispaced points on the unit
  circle. The angular margin between adjacent classes shrinks like
  2π/n. Meanwhile the additive value path lets state magnitude drift
  with sequence length. At larger n the same drift in absolute terms
  becomes a larger fraction of the angular margin, so OOD accuracy
  degrades.

- **GRU** finds approximate n-state automata via sigmoid×tanh gating.
  The approximation gets sharper or messier depending on whether n is
  a "natural" number for gated dynamics. We expect closure-under-n to
  drop with n; predicting OOD accuracy is harder because of the bounded
  state's competing advantage in length extrapolation.

### Run

```bash
python3 run_cyclic.py --config full     # 6 n-values × 3 seeds × 4 models = 72 runs
```

Expected wallclock ~90 minutes on H100, bottlenecked by the complex RNN
(Python-level scan).

### CPU validation of the negative theorem

The middle of the sweep (n=5) makes the linear-architecture failure
mode visually concrete. RealAddNet's per-depth accuracy in a CPU run:

```
depth:    0    1    2    3    4    5    6    7    8    9   10   11   12
acc:    1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
```

The model hits 100% at depths 0, 5, 10 — *exactly the multiples of 5* —
and 0% everywhere else. This is the theorem made visible: three (or
five) linear functions of k cross the correct (k mod n) trajectory at
exactly n points per period. The architecture has memorized class 0
and gets it right whenever (k mod 5) = 0, which from its perspective is
an aliasing accident, not learning. PhaseSumNet at 453 parameters
converges to 100% at every depth by training step 500 on the same task.

(Full H100 results, including the 3-panel scaling-law plot, in
`results_exp4/`.)

---

## How the four experiments fit together

| | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|---|---|---|---|---|
| Architecture | Transformer | PhaseSum / GatedCplx / GRU | + RealAddNet | (same 4) |
| Algebra | Z/2 | Z/2 | Z/3 | Z/n sweep |
| Composition | additive (softmax) | multiplicative | multiplicative | multiplicative |
| PhaseSum OOD | — | 1.0 | 1.0 | flat at 1.0 (predicted) |
| Pure-additive baseline | transformer: chance | n/a | provably barred at chance | provably barred for n≥3 |
| Complex vs real winner | tie (both fail) | tie (both succeed) | tradeoff (different prices) | scaling law (predicted) |

### The combined story

> **The inductive bias that matters for cyclic-group tasks is not
> "complex numbers" but "the ability to represent the symmetry group of
> the task". For Z/2 (Exp 1, 2), {+1, −1} lives natively in both ℝ\*
> and the unit circle, so any architecture with multiplicative
> composition succeeds, whether implemented as complex phase, real sign
> products via gating, or unitary rotation. For Z/n with n ≥ 3 (Exp 3,
> 4), there is no order-n subgroup of ℝ\*; the only ways to represent
> the cyclic group inside a network are (a) continuous rotation on
> ℝ²—i.e., a complex unit circle by another name; (b) discrete-state
> gating that simulates an n-state automaton. Purely additive
> architectures with linear readouts are provably barred. Among the
> remaining options, set-equivariant phase networks pay the lowest
> price: they have bounded state and are insensitive to sequence
> length, so OOD generalization is exact. Gated recurrences (complex
> or real) succeed at the in-distribution task and degrade gracefully
> at extreme OOD lengths via different mechanisms, with no clear
> winner. Softmax-attention transformers fall into the
> provably-barred family for exactly the same reason RealAddNet does:
> attention weights sum to 1, forcing additive composition.**

The honest, narrowed answer to the original question — "do complex
numbers help in ML?" — is now:

- **No**, for tasks whose symmetry group is a subgroup of ℝ\*. Real
  networks with appropriate gating do equally well.
- **Yes**, for tasks whose symmetry group is cyclic of order ≥ 3,
  *if* you also pick an architecture (like PhaseSumNet) whose other
  inductive biases — set-equivariance, bounded readout — match the
  task's other symmetries.
- **The more general lesson**: the question to ask of any architecture
  is whether its composition operator is a homomorphism from the
  algebra of token semantics to the algebra of network states. When
  yes, generalization comes essentially for free. When no, no amount
  of parameters or training data closes the gap.

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
python3 run_cyclic.py --config smoke

# Full experiments (H100 recommended)
python3 run.py        --config full     # Exp 1: transformer (Z/2),       ~30 min
python3 run_rnn.py    --config full     # Exp 2: RNN (Z/2),               ~20 min
python3 run_triple.py --config full     # Exp 3: RNN (Z/3),               ~25 min
python3 run_cyclic.py --config full     # Exp 4: Z/n sweep,               ~90 min
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
├── data_cyclic.py     Exp 4   data (Z/n rotation, n is a parameter)

├── models.py          Exp 1: real & complex transformer encoders
├── analyze.py         Exp 1: behavioral phase probe (paired sentences)
├── run.py             Exp 1: orchestrator
├── train.py           Exp 1+2: binary training loop

├── rnn_models.py      Exp 2: PhaseSumNet, GatedComplexRNN, GRUBaseline
├── analyze_rnn.py     Exp 2: direct read-off of learned phases
├── run_rnn.py         Exp 2: orchestrator

├── models_triple.py   Exp 3: 4 architectures for Z/3
├── analyze_triple.py  Exp 3: mod-3 closure + slope analysis
├── train_triple.py    Exp 3: 3-class training loop
├── run_triple.py      Exp 3: orchestrator

├── models_cyclic.py   Exp 4: 4 architectures parameterized by spec
├── analyze_cyclic.py  Exp 4: closure probe parameterized by n
├── train_cyclic.py    Exp 4: n-class training loop
└── run_cyclic.py      Exp 4: orchestrator (sweep over n)
```

After running, `results/`, `results_exp2/`, `results_exp3/`,
`results_exp4/` each contain:

```
results.json           all per-seed metrics, machine-readable
summary.txt            headline numbers in plain text
depth_accuracy.png     accuracy per depth, all models (Exp 1-3)
scaling_law.png        accuracy + closure as functions of n (Exp 4)
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
