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

5. **Experiment 5 (capacity sweep on the GRU collapse).** Sweeps GRU
   and LSTM from d=16 to d=256 (3,717 to ~1.06M parameters), 1- and
   2-layer, on n ∈ {5, 7}, against PhaseSumNet at matched d. Splits the
   "structural barrier" hypothesis into a clean win and an interesting
   wrinkle. **For n=5**, every gated-real architecture across 16 cells
   and a 285× capacity range sits in the OOD band [0.22, 0.34] with
   ID at 1.000 and closure near zero. The barrier is established.
   **For n=7**, single-layer GRU and all LSTM remain at chance, but
   **2-layer GRU climbs from 0.40 to 0.79 OOD as d goes 32 → 64 → 128**
   — partially escaping the barrier in a way I do not have a clean
   mechanism for, since 2-layer LSTM at the same scale stays at chance.
   PhaseSumNet hits 1.000 at every cell tested, on both n=5 and n=7,
   with 453–8,711 parameters.

**Final framing.** Complex numbers are not a universal upgrade, but
they are also not a notational convenience. They are the **architectural
home** for tasks whose symmetry group is cyclic of order ≥ 5. For Z/2
and Z/3, real-valued gated recurrence handles the task fine via
fixed-point and three-fold attractors in tanh-state space. For Z/5,
real-valued gating collapses across 16 architecture-capacity cells from
4k to 1M parameters. The mechanism is not that "tanh can't make limit
cycles of period 5" — it's subtler: the GRU's tanh-bounded state
*saturates* past the training depth, and once saturated the linear
readout becomes a near-constant function of k. OOD accuracy plateaus
at exactly 1/n, the chance value for any constant prediction over n
balanced classes. **No amount of capacity rescues this**, because the
failure is in the boundedness of the state, not its dimension.
PhaseSumNet succeeds for the inverse reason: its `[cos(Θ), sin(Θ)]`
state is *constructed* so the readout-relevant subspace is invariant
under the TWIRL action by design. State moves a lot in absolute terms;
logits don't move at all. Across every n we tested, PhaseSumNet at
the unit circle hits 1.000 OOD with 306–8,711 parameters: two to three
orders of magnitude smaller than the next-best architecture, and the
only one that solves the task exactly. The deepest prerequisite is
**alignment between state geometry and readout**: the architecture's
composition operator must move state along directions the readout
ignores, modulo the task's period.

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

### Result (3 seeds, H100, 92 minutes wallclock)

The full table for n=5:

| arch | d | L | params | ID | OOD | closure |
|---|--:|--:|--:|--:|--:|--:|
| GRU | 16 | 1 | 3,717 | 1.000 ± 0.000 | 0.285 ± 0.020 | 0.174 ± 0.082 |
| GRU | 32 | 1 | 13,573 | 1.000 ± 0.000 | 0.224 ± 0.010 | 0.268 ± 0.142 |
| GRU | 32 | 2 | 32,389 | 0.999 ± 0.001 | 0.256 ± 0.005 | 0.240 ± 0.011 |
| GRU | 64 | 1 | 51,717 | 1.000 ± 0.000 | 0.247 ± 0.006 | 0.126 ± 0.051 |
| GRU | 64 | 2 | 126,213 | 1.000 ± 0.000 | 0.223 ± 0.003 | 0.328 ± 0.184 |
| GRU | 128 | 1 | 201,733 | 1.000 ± 0.000 | 0.243 ± 0.017 | 0.124 ± 0.051 |
| GRU | 128 | 2 | 498,181 | 1.000 ± 0.000 | 0.266 ± 0.014 | 0.311 ± 0.095 |
| GRU | 256 | 1 | 796,677 | 1.000 ± 0.000 | 0.240 ± 0.006 | 0.021 ± 0.017 |
| LSTM | 16 | 1 | 4,805 | 0.984 ± 0.013 | 0.288 ± 0.029 | 0.150 ± 0.050 |
| LSTM | 32 | 1 | 17,797 | 0.999 ± 0.000 | 0.341 ± 0.045 | 0.033 ± 0.021 |
| LSTM | 32 | 2 | 42,885 | 1.000 ± 0.000 | 0.275 ± 0.010 | 0.406 ± 0.096 |
| LSTM | 64 | 1 | 68,357 | 1.000 ± 0.000 | 0.261 ± 0.011 | 0.277 ± 0.113 |
| LSTM | 64 | 2 | 167,685 | 1.000 ± 0.000 | 0.268 ± 0.006 | 0.389 ± 0.024 |
| LSTM | 128 | 1 | 267,781 | 1.000 ± 0.000 | 0.247 ± 0.010 | 0.223 ± 0.052 |
| LSTM | 128 | 2 | 663,045 | 1.000 ± 0.000 | 0.290 ± 0.035 | 0.267 ± 0.078 |
| LSTM | 256 | 1 | 1,059,845 | 1.000 ± 0.000 | 0.251 ± 0.003 | 0.199 ± 0.092 |
| **PhaseSum** | 16 | — | 453 | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| **PhaseSum** | 32 | — | 901 | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| **PhaseSum** | 64 | — | 1,797 | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| **PhaseSum** | 128 | — | 3,589 | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| **PhaseSum** | 256 | — | 7,173 | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |

And for n=7 (full table in `results_exp5/summary.txt`):

| arch | d | L | params | OOD |
|---|--:|--:|--:|--:|
| GRU | 32 | 1 | 13,767 | 0.281 |
| GRU | 32 | 2 | 32,583 | **0.402** |
| GRU | 64 | 1 | 52,103 | 0.226 |
| GRU | 64 | 2 | 126,599 | **0.564** |
| GRU | 128 | 1 | 202,503 | 0.212 |
| GRU | 128 | 2 | 498,951 | **0.787** |
| GRU | 256 | 1 | 798,215 | 0.227 |
| LSTM | 16-256 | 1, 2 | 4,903 – 1,061,383 | 0.20 – 0.31 (flat) |
| PhaseSum | 16-256 | — | 551 – 8,711 | **1.000** |

### What the data shows

**1. The structural barrier holds cleanly for n=5.** Across every
GRU and LSTM rung tested — 16 cells, 285× parameter range from
3,717 to 1,059,845 — OOD accuracy stays in the band [0.22, 0.34]
with ID at 1.000. No trend with capacity or depth. Closure scores
hover near zero. The cells differ in arrangement of failures but
not in whether they fail. **This is the strongest possible
synthetic-task evidence for the claim**: real-valued gated
recurrence cannot represent Z/5 at OOD lengths, regardless of
parameter count or depth.

**2. The LSTM result is the cleanest single line.** OOD across all
16 LSTM cells (8 at n=5, 8 at n=7) stays in [0.20, 0.34] with no
trend in either direction. From 4,805 to 1,061,383 parameters
(220× range), LSTM behaves identically. Whatever LSTM is doing,
adding parameters does nothing — this is the asymptotic
structural barrier without any wrinkle.

**3. The 2-layer GRU on n=7 partially escapes the barrier, and I
do not have a clean mechanism for why.** OOD climbs steadily with
capacity at L=2: 0.402 (d=32) → 0.564 (d=64) → **0.787** (d=128).
Single-layer GRU at the same n stays at chance (0.21–0.28).
2-layer LSTM at the same n stays at chance (0.23–0.31). 2-layer
GRU at n=5 stays at chance across the same capacity range.

So the climb is specifically: GRU + 2 layers + n=7 + large d. At
d=128 with 500k parameters it reaches 78.7% on a task where every
other gated-real architecture sits at chance. Closure under
7·TWIRL is still only 0.30 — the model is not finding the exact
cyclic group structure — but it's finding *something* that
generalizes most of the time at depths up to 20.

I don't have a satisfying explanation. Possibilities I'd test if
this were the main project:

- The 2-layer GRU may be implementing a 2-layer hierarchical
  state machine where the first layer produces a learned
  encoding of "what state are we in" and the second layer
  composes 7-ish near-attractors. The 2-layer LSTM has separate
  cell and hidden states whose information flow may interfere
  with this construction.
- The trend with d (0.40 → 0.56 → 0.79) suggests the construction
  exists but needs capacity. At d=256 with L=2 (not in this
  sweep — params would be ~2M+) it might reach the asymptote.
- It might be specific to depths 6–20 rather than arbitrary OOD.
  Closure under 7·TWIRL of only 0.30 means the group structure
  isn't actually learned; the model may be doing some kind of
  position-tracking that works up to some depth.

The data does not let me distinguish these. What I *can* say
firmly: **for 3 of the 4 gated-real architecture-depth combinations
we tested, the barrier holds cleanly. For the 4th (2-layer GRU on
Z/7), the barrier partially breaks at large capacity.** The
asymmetric finding is more interesting than the binary one would
have been, and worth flagging as an unresolved item.

**4. PhaseSumNet hits 1.000 ± 0.000 at every cell tested.** All
five rungs at both n=5 and n=7. The smallest is 453 parameters
(n=5, d=16) and 551 parameters (n=7, d=16). The 2-layer GRU that
hits 0.787 OOD on n=7 uses 904× more parameters than the
PhaseSumNet that hits 1.000 on the same task.

### What's in `results_exp5/`

- `summary.txt` — the full 21-row + 17-row results table.
- `results.json` — every per-seed result.
- `capacity_curves.png` — OOD accuracy vs parameters (log-x), two
  panels. The n=5 panel shows the clean structural barrier; the
  n=7 panel shows the 2-layer GRU climb.
- `closure_curves.png` — closure-under-n vs parameters, same layout.

---

## Mechanism — why does the GRU fail and PhaseSumNet succeed?

After Exp 5 confirmed the barrier at n=5, the question is *what*, in
the GRU's dynamics, prevents it from representing Z/5 — and in what
sense PhaseSumNet's representation *is* Z/5. `trajectory_analysis.py`
trains one GRU (d=64, L=1) and one PhaseSumNet (d=16) on Z/5, then
probes their state and logit dynamics across `k = 0…30` TWIRLs on
sequences with identical base prefixes (so the only difference between
`k` and `k+5` is exactly 5 TWIRL tokens).

### The four diagnostics, and why the first three mislead

| metric | GRU | PhaseSumNet | which generalizes? |
|---|--:|--:|---|
| state magnitude at k>5 | 6.75 | 4.00 | — |
| magnitude ratio ‖h(30)‖/‖h(5)‖ | 1.19 | 1.00 | — |
| state ‖Δ‖ / ‖h‖ after 5·TWIRL | **0.17** | **0.98** | — |
| **argmax invariance after 5·TWIRL** | **0.54** | **1.00** | **PhaseSumNet** |

The state-level "closure" metric is misleading in the direction I
initially expected to be most informative. The GRU's state moves only
17% (relative) after 5 TWIRLs; PhaseSumNet's moves 98%. If you read
only state closure, you'd conclude the GRU has "almost-perfect cyclic
structure" and PhaseSumNet has "no cyclic structure". The opposite is
true.

The resolution is that **state-level closure is the wrong question.**
What the linear readout sees is logits, not states. PhaseSumNet's
`[cos(Θ), sin(Θ)]` representation is *constructed* so that adding any
multiple of 2π to Θ leaves the readout-relevant content invariant.
The 16 phase dimensions each independently sum to a multiple of 2π
after 5 TWIRLs (because the model learns θ(TWIRL) ≡ k·2π/5 in each
dimension for some integer k); the cos/sin pair returns to the same
point on the unit circle; the readout returns the same logits. State
magnitude is constant at √d = 4.

The GRU has no such alignment. Its 64-dim state moves only a little
after 5 TWIRLs (small ‖Δ‖), but the linear readout *amplifies* that
small shift into a logit ‖Δ‖ of **10.03**, 22× larger than
PhaseSumNet's 0.46. There is no subspace of the GRU's state that's
both (a) preserved under the TWIRL action and (b) the subspace the
readout cares about.

### The per-k pattern reveals the failure mode

The interesting structure shows up when we plot argmax-invariance as a
function of k:

| k | GRU argmax invariance | what's happening |
|--:|--:|---|
| 0 | 0.83 | still inside training distribution |
| 1 | 0.63 | training boundary |
| 2 | 0.28 | crossing into OOD |
| **3–7** | **0.00** | **transition zone, complete failure** |
| 8–11 | 0.38 | start of recovery |
| 12–15 | 0.50–0.61 | partial recovery |
| 16–19 | 0.67–0.80 | further recovery |
| 20–25 | 0.85–0.98 | saturation |

This is the signature of a model whose state saturates. Look at the
magnitude curve: ‖h‖ grows from 5.12 (k=0) to 7.04 (k=30) and
asymptotes — the tanh ceiling. By k≈20 the GRU's state has nearly
stopped moving with additional TWIRLs (logit ‖Δ‖ between k and k+5
falls from 24.96 at k=5 to 2.05 at k=20). So at large k the GRU's
*prediction* is approximately constant in k — but it's the *wrong*
constant. **Argmax invariance is high at k≈25 not because the model
has learned Z/5, but because it's predicting the same class
regardless of k.** That class happens to be right 1/5 of the time,
which is exactly the 0.25 OOD accuracy floor we observed across all
of Exp 5.

### The corrected mechanism story

> **The GRU doesn't fail to find a cyclic attractor — it fails to use
> its state space past the training depth at all.** Its tanh-bounded
> state saturates as k grows, and the linear readout asymptotes to a
> near-constant function of k. The OOD accuracy plateau at ~1/n we
> see across every capacity rung in Exp 5 is exactly this saturation
> floor: at large k the GRU's argmax is constant, and any constant
> prediction over a balanced n-class task scores 1/n. **No amount of
> capacity rescues this**, because the failure is in the state's
> boundedness, not its dimension.
>
> **PhaseSumNet succeeds for a different reason than I expected.**
> Its state moves *more* than the GRU's (98% relative drift vs 17%),
> but the movement lives entirely in the readout's null space modulo
> 5·TWIRL. The cos/sin construction projects the state onto exactly
> the subspace the readout cares about, and that subspace is invariant
> under the TWIRL action by design.

The two diagnostics that don't mislead, in order of usefulness for
predicting OOD accuracy:

1. **Argmax invariance under n·TWIRL** — directly measures the
   property that has to hold for OOD generalization.
2. **Logit ‖Δ‖ under n·TWIRL** — same thing in finer-grained units,
   no thresholding.

State-level closure metrics (Exp 4's "closure under n·TWIRL") are
weakly correlated with accuracy but can flip in either direction. In
hindsight that probe should have been logit-level from the start.

### What's in `results_trajectory/`

- `magnitude.png` — state ‖h(k)‖ vs k for both architectures
- `modular_return.png` — state ‖Δ‖ under 5·TWIRL, absolute and relative
- `logit_closure.png` — **the headline figure**: logit ‖Δ‖ and argmax
  invariance vs k, showing the GRU's transition-zone collapse
- `gru_pca.png`, `phasesum_pca.png` — 2D PCA projections of the
  state trajectories
- `summary.json` — per-k numbers for all four diagnostics

Run with: `python3 trajectory_analysis.py`. Trains fresh models on CPU
in ~3 minutes total.

---

## How the five experiments fit together

| | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 |
|---|---|---|---|---|---|
| Architecture | Transformer | PhaseSum / GatedCplx / GRU | + RealAddNet | (same 4) | GRU/LSTM capacity sweep |
| Algebra | Z/2 | Z/2 | Z/3 | Z/n, n ∈ {2,3,5,7,11,13} | Z/5, Z/7 |
| Composition | additive (softmax) | multiplicative | multiplicative | multiplicative | gated real |
| PhaseSumNet OOD | — | 1.0 | 1.0 | **1.0 at every n** | **1.0 across 285× param range** |
| Pure-additive baseline | transformer: chance | n/a | provably barred at chance | chance for all n ≥ 3 | n/a |
| Real-gated (GRU/LSTM) | n/a | tied with complex | trails complex slightly | breaks at n ≥ 5 | **flat at chance to 1M+ params** (with one exception) |
| Complex specifically helps? | no | no (Z/2 not enough) | small effect | yes, decisively | **yes, ~3 orders of magnitude in params** |

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
>   pays at large n. **PhaseSumNet at 453–8,711 parameters hits OOD
>   1.000 ± 0.000 at every n we tested**, including across a 285×
>   capacity sweep that holds the comparison architectures' OOD flat
>   at chance.
>
> - Real-valued gated recurrence (GRU, LSTM) supports Z/2 and Z/3
>   perfectly, apparently because tanh-state dynamics admit natural
>   fixed-point and three-fold attractors. **For Z/5 the failure is
>   complete**: across 16 architecture-capacity cells from 3,717 to
>   1,059,845 parameters (1- and 2-layer GRU, 1- and 2-layer LSTM),
>   OOD stays in [0.22, 0.34] with no trend. ID is 1.000 throughout
>   — the architectures fit training perfectly and produce near-noise
>   OOD. **For Z/7 the picture is uneven**: single-layer GRU and all
>   LSTM remain at chance, but 2-layer GRU climbs from 0.40 to 0.79
>   OOD as d goes 32 → 64 → 128, escaping the barrier through a
>   mechanism we have not yet identified. The 2-layer GRU at d=128
>   that hits 79% on Z/7 uses 904× more parameters than the 551-param
>   PhaseSumNet that hits 100% on the same task.
>
> This is the sharpest separation the project produces. Z/2 and Z/3
> happen to be representable in tanh-state attractor dynamics; for
> Z/5 the structural barrier appears total; for Z/7 it has a leak we
> do not yet understand. Across every condition tested, complex
> unit-circle networks were the only architecture that solved the
> task at every n cleanly.

The honest, narrowed answer to the original question — "do complex
numbers help in ML?" — is:

- **Yes, decisively, for tasks whose symmetry group is cyclic of order
  ≥ 5.** At n=5, gated-real recurrence fails categorically across
  every capacity and depth tested up to ~1M parameters. PhaseSumNet at
  453 parameters solves the task perfectly. The parameter-efficiency
  gap is at least three orders of magnitude.
- **Yes, with caveats, for cyclic order 7.** The barrier holds for 3 of
  the 4 gated-real architecture families tested. The 4th (2-layer GRU)
  partially escapes at large capacity (79% OOD at 500k parameters), but
  uses ~1000× more parameters than the PhaseSumNet that solves it at
  100%.
- **No, for cyclic order 2 or 3.** Real gated recurrences handle these
  natively via state-space attractors that happen to have the right
  period. Complex offers no benefit.
- **The general lesson:** the question to ask of any architecture is
  whether its composition operator is a homomorphism from the algebra
  of token semantics to the algebra of network states. **When the
  architecture's natural state-space orbits include closed cycles of
  the task's period, generalization comes essentially for free.
  When they don't, parameter scaling does not generically rescue you**
  — the architecture is barred from the representation, either
  provably (additive-linear), almost-completely (gated-real at n=5),
  or with isolated escape routes whose mechanism is unclear
  (2-layer GRU at n=7).

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
├── run_capacity.py    Exp 5: orchestrator (capacity sweep)

└── trajectory_analysis.py  Mechanism analysis: GRU state saturation +
                             logit-level closure probe
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

- **One synthetic task family is one data point.** All five experiments
  use controlled cyclic-rotation tasks (Z/2, Z/3, Z/n). Real-language
  negation has more structure: scope, modality, sarcasm,
  double-negation idioms. Real-world cyclic structure (musical pitch,
  cardinal directions, days of week, finite enumerations) maps onto Z/n
  but with noise and partial observability we don't model. The
  positive results here are *necessary*, not sufficient, for claims
  about complex networks in NLP or in general ML practice.

- **The 2-layer GRU result on Z/7 is unresolved.** Across 3 seeds with
  ±0.04–0.09 stderr, 2-layer GRU OOD on Z/7 climbs from 0.40 (d=32) to
  0.79 (d=128). We do not have a clean mechanism for why 2-layer GRU
  partially escapes the barrier at n=7 when single-layer GRU and 1- and
  2-layer LSTM at the same scale do not, and 2-layer GRU at n=5 does
  not. The structural-barrier claim is robust at n=5 but has a leak at
  n=7 we cannot explain. Either we need to extend the sweep (larger d,
  longer training, broader hyperparameter range) or the mechanism story
  needs refinement.

- **Convergence depends on initialization.** PhaseSumNet and
  GatedComplexRNN must initialize phases uniformly in [−π, π], not
  near zero, or the gradient signal is too weak to escape (all
  sentences produce cos ≈ 1 → constant logits). This is fixed in the
  current code; the lesson is that "complex inductive bias" only helps
  if you let the architecture *use* the unit circle.

- **Closure-under-n probe is only meaningful at high ID accuracy.** A
  model that outputs near-constants trivially scores high on closure
  because adding n more TWIRLs doesn't change its already-uniform
  output. RealAddNet's closure entries are omitted from result tables
  for this reason. Always read closure alongside ID accuracy.

---

## Suggested follow-ups

1. **Resolve the 2-layer GRU mystery on Z/7.** Extend the capacity
   sweep to d ∈ {384, 512} at L=2 for n=7. If OOD continues climbing,
   the asymptote is somewhere we haven't reached. If it plateaus near
   0.8, then there's a real partial-escape phenomenon to study
   mechanistically. Also test n ∈ {11, 13} at 2-layer GRU large
   capacity: if those climb too, the n=5 vs n≥7 asymmetry is the real
   finding.

2. **Sample-complexity sweep at fixed architecture.** For each of
   PhaseSumNet, GatedComplexRNN, GRU at d=128 L=2 on Z/5, sweep
   `n_train ∈ {1k, 10k, 100k, 1M}`. If PhaseSumNet asymptotes at
   100% by 10k samples while the GRU never asymptotes, that quantifies
   the inductive-bias gap as a sample-efficiency ratio.

3. **Real-language transfer.** Conditional NegNLI (Hossain et al.,
   2020), negated RTE, scoped quantifier benchmarks. Use PhaseSumNet's
   per-token phase embedding as a learned token feature in a small
   language model. Test whether downstream NLI accuracy on negated
   propositions improves at small parameter counts.

4. **Verify the saturation mechanism more rigorously.** The
   trajectory analysis showed the GRU's state magnitude grows
   monotonically and saturates against a tanh ceiling, with the
   linear readout becoming near-constant past training depth. The
   prediction is sharp: if you replace the GRU's bounded state with
   an *unbounded* state (e.g., remove the tanh and use a linear RNN
   with appropriate normalization), the saturation should disappear,
   and the OOD curve should change shape. If it still plateaus near
   1/n, the saturation diagnosis is wrong and something else is at
   work. This is one experiment with a clean prediction either way.

5. **Beyond cyclic groups.** Z/n is the simplest non-trivial group
   family. Real interesting algebra includes non-abelian groups (S_n
   for finite-state syntax, SO(3) for 3D pose), continuous groups
   (Lie groups for physics), and approximate groups (compositions of
   word vectors in NLP). The unit circle in ℂ generalizes to U(1);
   the natural next step is whether quaternionic or SU(2)
   architectures show similar inductive-bias wins for SO(3) tasks.

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
