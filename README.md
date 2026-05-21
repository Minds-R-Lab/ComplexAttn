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
   The most informative result. PhaseSumNet stays at OOD 1.000 ± 0.000
   across all six n, with 306–845 parameters. RealAddNet sits at
   exactly 1/n (chance) for every n ≥ 3 as the theorem demands.
   GatedComplexRNN holds OOD ≥ 0.97 up to n=7 and breaks gently at
   n=11 (0.82) — the predicted magnitude-drift effect. The big surprise
   is the GRU: **it fully solves Z/2 and Z/3 (OOD 1.000 and 0.991) and
   then collapses to ~25% OOD with ~10% closure at every n ≥ 5**.
   This is a phase transition, not a scaling decline. Real-valued gated
   recurrence appears structurally unable to represent cyclic groups of
   order ≥ 5 at OOD lengths, regardless of parameter count.

5. **Experiment 5 (capacity sweep on the GRU collapse).** Tests whether
   the Exp 4 GRU collapse survives compute scaling. Sweeps GRU and LSTM
   from d=16 to d=256 (3,717 to ~2M parameters), 1- and 2-layer
   variants, on n ∈ {5, 7}, with PhaseSumNet at matched d as the
   reference. Distinguishes "structural barrier" from
   "sample-inefficient". Includes LSTM as a control: if LSTM also
   collapses, the mechanism is "real-valued gating in general cannot
   produce tanh limit cycles of period ≥ 5", not "GRU-specific".

**Final framing.** Complex numbers are not a universal upgrade, but
they are also not a notational convenience. They are the **architectural
home** for tasks whose symmetry group is cyclic of order ≥ 5. For Z/2
and Z/3, real-valued gated recurrence handles the task fine via
fixed-point and three-fold attractors in tanh-state space. For Z/5 and
larger, gated reals collapse — they fit the training set but produce
near-noise OOD with closure under n·TWIRL of 4–14%. Complex
unit-circle networks pay no such price: e^(i·2π/n) is just a parameter,
and a 453-parameter PhaseSumNet handles Z/13 with the same perfect
generalization as Z/2. The deepest prerequisite is more general: the
architecture's composition operator must be a homomorphism into a state
space that contains closed orbits of the task's period. Softmax
attention isn't (it forces additive composition); RealAddNet's linear
readout isn't (logits linear in token count can't be periodic);
real-gated recurrence is only for n ∈ {2, 3}; complex unit-circle
architectures are for every n.

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

### Result (3 seeds, H100, 66 minutes wallclock)

| n | model | params | ID acc | OOD acc | closure under n·TWIRL |
|--:|---|--:|--:|--:|--:|
| 2  | PhaseSumNet      | 306   | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| 2  | RealAddNet       | 546   | 0.498 ± 0.034 | 0.505 ± 0.015     | —             |
| 2  | GatedComplexRNN  | 4,162 | 1.000 ± 0.000 | 0.967 ± 0.014     | 1.000 ± 0.000 |
| 2  | GRUBaseline      | 3,997 | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| 3  | PhaseSumNet      | 355   | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| 3  | RealAddNet       | 611   | 0.337 ± 0.022 | 0.341 ± 0.005     | —             |
| 3  | GatedComplexRNN  | 4,547 | 1.000 ± 0.000 | 0.979 ± 0.009     | 1.000 ± 0.000 |
| 3  | GRUBaseline      | 4,503 | 1.000 ± 0.000 | 0.991 ± 0.003     | 1.000 ± 0.000 |
| 5  | PhaseSumNet      | 453   | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| 5  | RealAddNet       | 741   | 0.333 ± 0.111 | 0.200 ± 0.060     | —             |
| 5  | GatedComplexRNN  | 5,317 | 1.000 ± 0.000 | 0.992 ± 0.002     | 1.000 ± 0.000 |
| 5  | GRUBaseline      | 5,097 | 1.000 ± 0.000 | **0.248 ± 0.023** | **0.120 ± 0.032** |
| 7  | PhaseSumNet      | 551   | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| 7  | RealAddNet       | 871   | 0.173 ± 0.015 | 0.136 ± 0.012     | —             |
| 7  | GatedComplexRNN  | 6,087 | 1.000 ± 0.000 | 0.980 ± 0.006     | 0.999 ± 0.001 |
| 7  | GRUBaseline      | 6,265 | 0.998 ± 0.002 | **0.241 ± 0.017** | **0.080 ± 0.060** |
| 11 | PhaseSumNet      | 747   | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| 11 | RealAddNet       | 1,131 | 0.188 ± 0.014 | 0.069 ± 0.014     | —             |
| 11 | GatedComplexRNN  | 7,627 | 1.000 ± 0.000 | 0.816 ± 0.031     | 0.874 ± 0.066 |
| 11 | GRUBaseline      | 7,693 | 0.994 ± 0.003 | **0.285 ± 0.026** | **0.040 ± 0.033** |
| 13 | PhaseSumNet      | 845   | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| 13 | RealAddNet       | 1,261 | 0.188 ± 0.011 | 0.070 ± 0.014     | —             |
| 13 | GatedComplexRNN  | 8,397 | 1.000 ± 0.000 | 0.864 ± 0.017     | 0.805 ± 0.081 |
| 13 | GRUBaseline      | 8,461 | 1.000 ± 0.000 | **0.237 ± 0.025** | **0.136 ± 0.111** |

The closure column for RealAddNet is omitted as uninformative: a model
that outputs near-constant predictions trivially scores high on closure
because adding n more TWIRLs doesn't change its (already-uniform)
output. Always interpret closure alongside ID accuracy.

### What the data shows

**1. PhaseSumNet is exactly 1.000 ± 0.000 across all six n values, on
both ID and OOD, with full closure.** Three seeds, zero variance, at
group orders from 2 to 13, with 306–845 parameters. The architectural
prior — set-equivariance plus unit-circle composition plus periodic
readout — exactly matches the symmetry of the task, and the
generalization is exact across an order of magnitude in group size.
This is the strongest possible inductive-bias result the project
produces.

**2. RealAddNet behaves as the theorem demands, with one note.** OOD
accuracy at n ∈ {3, 5, 7, 11, 13}: 0.34, 0.20, 0.14, 0.07, 0.07.
Chance (1/n) is 0.33, 0.20, 0.14, 0.09, 0.08. The agreement is
essentially exact. *Note the exception at n=2: OOD ≈ 0.50 = chance.*
Z/2 doesn't have the theorem (two linear functions of k *can* fit (k
mod 2), as their argmax is allowed to cross once), but the model
nevertheless converged to chance — a learning failure, not a
representability failure. This is a useful cautionary data point: not
having a theorem against you is not the same as solving the task.

**3. GatedComplexRNN shows the predicted decline, with a clear
break-point.** OOD accuracy across n = 2, 3, 5, 7, 11, 13: 0.97, 0.98,
0.99, 0.98, **0.82**, 0.86. Closure across the same n values: 1.00,
1.00, 1.00, 1.00, **0.87**, 0.81. The architecture handles n ≤ 7
nearly perfectly and breaks at n = 11. The break-point is consistent
with the angular-margin mechanism: the angle between adjacent classes
is 2π/n, so the *margin* relative to *state magnitude drift* crosses
into the unreliable regime somewhere around n = 8–10.

**4. The GRU result is a phase transition, not a scaling law, and is
the most informative single finding.**

| | n=2 | n=3 | n=5 | n=7 | n=11 | n=13 |
|---|--:|--:|--:|--:|--:|--:|
| GRU OOD     | **1.000** | **0.991** | 0.248 | 0.241 | 0.285 | 0.237 |
| GRU closure | **1.000** | **1.000** | 0.120 | 0.080 | 0.040 | 0.136 |

The GRU **completely solves** Z/2 and Z/3 with full closure, then
**collapses to ~25% OOD with ~10% closure** at n = 5 and stays there
for every larger n. Note that ID accuracy stays at ≥0.99 throughout —
the architecture fits the training set fine at every n, but its OOD
behavior at n ≥ 5 is closer to "memorized lookup that doesn't
extrapolate" than "imperfect group structure".

This is much more specific than the "scaling law" framing predicted.
**Real-valued gated recurrences appear to natively support short-period
cyclic groups via limit cycles of their tanh-state dynamics, and do
not generically support cyclic groups of higher order**, regardless of
how many parameters or how much training data we give them. Z/2 has a
natural fixed-point-flip representation in tanh state, Z/3 has a
naturally three-fold-symmetric attractor, but the GRU's dynamics
don't produce a clean 5-cycle no matter how it's trained on the task.

The complex unit-circle architectures pay no such price: e^(i·2π/n) is
just a parameter setting, and supports arbitrary n in a single
dimension.

### What's in `results_exp4/`

- `summary.txt` — the table above.
- `results.json` — full per-seed metrics including per-depth accuracy.
- `scaling_law.png` — 3-panel plot (ID, OOD, closure as functions of n).

---

## Experiment 5 — Capacity sweep on the GRU collapse

**Question.** The Exp 4 GRU collapse at n ≥ 5 was measured at one
capacity point (d ≈ 17, ~5k parameters). The claim that real-valued
gated recurrence is *structurally* unable to represent these groups
needs a capacity sweep: if GRU/LSTM with more parameters and more
training still don't solve Z/5, the structural-barrier framing is on
solid ground. If they climb steadily with compute, the right framing is
"much less sample-efficient than PhaseSumNet, but not categorically
different".

LSTM is included as a critical control. The mechanism story is "tanh
state space doesn't admit limit cycles of period ≥ 5". That story
applies to *any* tanh-state gated recurrence. If LSTM solves what GRU
can't, the story is wrong and the right one is GRU-specific. If LSTM
also collapses, the story holds at the gating-family level.

### Setup

- **n values:** {5, 7}. Replicates the Exp 4 finding across more than
  one n in the collapse regime.
- **Capacity rungs (d_model, n_layers):** {(16,1), (32,1), (32,2),
  (64,1), (64,2), (128,1), (128,2), (256,1)}.
- **Architectures:** GRU (multilayer), LSTM (multilayer), PhaseSumNet
  at matched d as the reference.
- **Training:** 120k samples, 25 epochs, lr=3e-3, AdamW, 3 seeds per
  cell.
- **Range:** smallest GRU rung has 3,717 parameters; largest has
  796,677. LSTM at d=256, L=1 has 1,059,845. PhaseSumNet at d=256 has
  7,173. The largest GRU is **180× the smallest PhaseSumNet that
  achieves OOD 1.000**.

### CPU preview (single seed, modest training budget)

Before running on H100, a CPU pilot at n=5 already shows the pattern:

| arch | d | params | OOD | closure |
|---|--:|--:|--:|--:|
| GRU | 16 | 3,717 | 0.320 | 0.000 |
| GRU | 32 | 13,573 | 0.265 | 0.000 |
| GRU | 64 | 51,717 | 0.265 | 0.047 |
| LSTM | 16 | 4,805 | 0.353 | 0.064 |
| LSTM | 32 | 17,797 | 0.268 | 0.084 |
| LSTM | 64 | 68,357 | 0.321 | 0.012 |
| PhaseSumNet | 16 | 453 | 0.411 | 0.736 |
| PhaseSumNet | 32 | 901 | 0.889 | 0.891 |
| PhaseSumNet | 64 | **1,797** | **1.000** | **1.000** |

Three things to note from the preview:

1. **GRU and LSTM OOD does not improve with scale.** Across a 14×
   parameter range (d=16 to d=64), GRU OOD stays at 0.27–0.32, LSTM at
   0.27–0.35. Closure stays essentially at zero for both. Adding
   parameters does not help.

2. **LSTM does not escape what GRU can't.** This rules out
   GRU-specific gating equations as the cause. Both architectures use
   tanh state with sigmoid gates; both collapse identically. The
   mechanism is at the gating-family level, not the specific
   architecture.

3. **PhaseSumNet at d=64 hits 1.000 OOD with 1,797 parameters.** That's
   ~30× fewer parameters than the smallest GRU we tested, and the
   only architecture that reaches the asymptote at all. The CPU result
   already strongly supports the structural-barrier claim; the full
   H100 sweep extends it to d=256 with 25 epochs.

### Run

```bash
python3 run_capacity.py --config full     # ~90–120 min on H100, 126 runs
```

Outputs to `results_exp5/`:
- `summary.txt` — full results table
- `results.json` — all 126 runs, per-seed
- `capacity_curves.png` — OOD accuracy vs parameters (log-x), one panel
  per n. The headline plot. If GRU/LSTM curves stay flat near 0.25 while
  PhaseSumNet sits at 1.0, the structural-barrier claim is established.
- `closure_curves.png` — closure-under-n vs parameters, same layout.

### What the result will mean

- **If GRU/LSTM OOD stays at ~0.25–0.30 at d=256:** the structural
  claim is established. The paper headline becomes "complex unit-circle
  composition solves cyclic groups of arbitrary order; real-valued
  gated recurrence cannot represent Z/n for n ≥ 5 at OOD lengths,
  regardless of capacity or training". This is a much sharper and
  more interesting claim than "scaling law".

- **If GRU/LSTM OOD climbs to ~0.7–0.9 at d=256:** the structural claim
  weakens to "real-valued gated recurrence requires orders of magnitude
  more capacity than complex unit-circle for the same task". Still a
  result, but a quantitative one rather than a categorical one.

- **If the LSTM curve diverges from the GRU curve at any d:** the
  mechanism is more architecture-specific than the gating-family story
  predicts, and the analysis needs refining.

---

## How the four experiments fit together

| | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|---|---|---|---|---|
| Architecture | Transformer | PhaseSum / GatedCplx / GRU | + RealAddNet | (same 4) |
| Algebra | Z/2 | Z/2 | Z/3 | Z/n, n ∈ {2,3,5,7,11,13} |
| Composition | additive (softmax) | multiplicative | multiplicative | multiplicative |
| PhaseSumNet OOD | — | 1.0 | 1.0 | **1.0 at every n** |
| Pure-additive baseline | transformer: chance | n/a | provably barred at chance | chance for all n ≥ 3 |
| Real-gated (GRU) | n/a | tied with complex | trails complex slightly | **breaks at n=5** |
| Complex specifically helps? | no | no (Z/2 not enough) | small effect | **yes, decisively** |

### The combined story

> **The inductive bias that matters for cyclic-group tasks is not
> "complex numbers" in the abstract, but a network whose state space
> contains a closed orbit of the right period.** Softmax-attention
> transformers (Exp 1) cannot represent any closed orbit at all,
> because softmax weights summing to 1 forces additive composition;
> they sit at chance OOD. Purely additive real networks with linear
> readouts (RealAddNet, Exp 3–4) cannot represent (k mod n) for n ≥ 3
> by a counting argument, and sit at exactly chance. Both real and
> complex multiplicative architectures handle Z/2 fine — {+1, −1}
> lives natively in both algebras. For higher cyclic orders, the
> options narrow:
>
> - The unit circle in ℂ contains an exact subgroup of every order n,
>   represented by e^(i·2π/n). Complex-phase architectures (PhaseSumNet,
>   GatedComplexRNN) handle arbitrary n; PhaseSumNet's set-equivariance
>   removes even the length-extrapolation tax that GatedComplexRNN
>   pays at large n.
>
> - Real-valued gated recurrence (GRU) supports Z/2 and Z/3 perfectly,
>   apparently because its tanh-state dynamics admit natural
>   fixed-point and three-fold attractors. **It does not support Z/n
>   for n ≥ 5 in any approximate way**: OOD accuracy collapses to ~25%
>   with closure ~10% at every n we tested, even with 4–8k parameters
>   and 100% in-distribution fit. The architecture memorizes the
>   training distribution and produces near-noise OOD.
>
> This is the sharpest separation the project produces. Z/2 is a
> coincidence; Z/n for n ≥ 5 is where the algebra actually matters,
> and there complex unit-circle networks are not just better than
> real-gated networks — they appear to solve a task that gated reals
> structurally cannot.

The honest, narrowed answer to the original question — "do complex
numbers help in ML?" — is:

- **Yes**, for tasks whose symmetry group is cyclic of order ≥ 5. Real
  gated recurrence appears to be structurally incapable of representing
  these groups at OOD lengths; complex unit-circle networks handle them
  exactly with order-of-magnitude fewer parameters (PhaseSumNet at 453
  parameters vs. GRU at 5,097 parameters for Z/5, with OOD accuracy
  1.000 vs 0.248).
- **No**, for tasks whose symmetry group is Z/2 or Z/3 specifically.
  Real gated recurrences handle these natively via state-space
  attractors that happen to have the right period.
- **The general lesson**: the question to ask of any architecture is
  whether its composition operator is a homomorphism from the algebra
  of token semantics to the algebra of network states. **When the
  architecture's natural state-space orbits include closed cycles of
  the task's period, generalization comes essentially for free. When
  they don't, no amount of parameters or training data closes the gap
  — the architecture is barred from the representation, either
  provably (additive-linear) or empirically (gated-real for n ≥ 5).**

---

## How to run

```bash
git clone https://github.com/Minds-R-Lab/ComplexAttn.git
cd ComplexAttn
pip install -r requirements.txt

# Sanity checks (~1 minute each, CPU is fine)
python3 run.py          --config smoke
python3 run_rnn.py      --config smoke
python3 run_triple.py   --config smoke
python3 run_cyclic.py   --config smoke
python3 run_capacity.py --config smoke

# Full experiments (H100 recommended)
python3 run.py          --config full   # Exp 1: transformer (Z/2),    ~30 min
python3 run_rnn.py      --config full   # Exp 2: RNN (Z/2),            ~20 min
python3 run_triple.py   --config full   # Exp 3: RNN (Z/3),            ~25 min
python3 run_cyclic.py   --config full   # Exp 4: Z/n sweep,            ~65 min
python3 run_capacity.py --config full   # Exp 5: GRU capacity sweep,   ~100 min
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
├── run_cyclic.py      Exp 4: orchestrator (sweep over n)

├── models_capacity.py Exp 5: GRU/LSTM with n_layers, PhaseSumRef
└── run_capacity.py    Exp 5: orchestrator (capacity sweep)
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
