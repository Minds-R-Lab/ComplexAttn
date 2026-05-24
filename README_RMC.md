# Resonant Manifold Cells (RMC) — an honest investigation

This repository implements and stress-tests a proposed alternative to the
affine-then-nonlinearity primitive of neural networks: the **Resonant
Manifold Cell** (RMC). Across several experiments we tested whether the
architecture's structured prior gives it an advantage over a matched-
parameter MLP. The honest verdict at the bottom of this README is: on
the tasks we tested with proper multi-seed error bars, **the RMC does not
beat a matched-parameter MLP**.

This document walks through what we built, what we tested, and what we
learned. Reading time: ~5 minutes.

## The proposed architecture

A neural network neuron does three things at once with a single bag of
scalar weights: it rotates coordinates, mixes dimensions, and creates
selectivity through a pointwise kink. The RMC factors those into three
separately-learnable objects:

1. **Geometry** — a learnable positive-definite inverse-mass matrix
   `M⁻¹ = LLᵀ + εI` defining the kinetic-energy quadratic form of a
   particle moving on a d-dimensional manifold.
2. **Dynamics** — a learnable Hamiltonian `H(x,p) = ½ pᵀ M⁻¹ p + V(x)`
   with `V(x) = ½ xᵀBx + V_MLP(x)`. Inputs are encoded into initial
   conditions `(x₀, p₀)`; the cell evolves the system for `T` leapfrog
   steps. The integrator is symplectic — phase-space volume preserved,
   energy bounded — and reversible.
3. **Selectivity** — a bank of K learnable resonant modes `(ψₖ, ωₖ)`. For
   each mode the cell computes the windowed Fourier coefficient
   `aₖ = (1/T) Σ_t ⟨ψₖ, x(t)⟩ · e^{i ωₖ t}` and outputs `|aₖ|`. Selectivity
   comes from **resonance**, not from a pointwise nonlinearity.

The thesis: by factoring the three jobs of a neuron into three named,
strongly-typed objects, you give a fixed parameter budget a stronger
inductive bias, which should win at small data and on data with the
structure the prior encodes.

## Files

```
new_nural_netwarks/
├── README.md                  ← this file
├── requirements.txt
├── model.py                   ← RMC + RMCClassifier + Linear/MLP baselines
├── data.py                    ← MNIST loader
├── verify.py                  ← numerical sanity checks
├── train.py                   ← training loop
├── run.py                     ← top-level MNIST orchestrator
├── ablations.py               ← AblationRMC + 4-variant ablation runner
├── benchmark_simplified.py    ← simplified RMC vs MLP on MNIST
├── sample_efficiency.py       ← sample-efficiency sweep on MNIST
├── dynamical_data.py          ← synthetic 4-class trajectory dataset
├── run_dynamical.py           ← RMC vs MLP on dynamical-system task
├── multi_seed.py              ← multi-seed sweep on dynamical data
├── multi_seed_long.py         ← longer-training multi-seed at n=3000
└── results/
    ├── training_curves.png         training_curves on MNIST
    ├── resonant_modes.png          learned ω vs head-importance
    ├── ablations.png               4-variant ablation comparison
    ├── simplified_vs_mlp.png       simplified RMC vs MLP on MNIST
    ├── sample_efficiency.png       MNIST sample-efficiency curves
    ├── dynamical_classes.png       example trajectories of each class
    ├── dynamical_sample_efficiency.png  (single-seed) sample efficiency
    ├── multi_seed_curves.png       3-seed dynamical sweep
    ├── multi_seed_long.png         4-seed dynamical at n=3000, 15ep
    ├── *.json                      raw numbers
    └── *.pt                        checkpoints
```

## How to run

```bash
pip install -r requirements.txt
python run.py                  # verification + MNIST baseline (~30s on CPU)
python ablations.py            # 4-component ablation (~30s)
python sample_efficiency.py    # MNIST sample efficiency (~60s)
python run_dynamical.py        # dynamical-system task single-seed (~25s)
python multi_seed.py           # 3-seed dynamical sweep (~60s)
python multi_seed_long.py      # 4-seed dynamical at n=3000, 15ep (~50s)
```

## What we found, step by step

### 1. The math holds up

Energy drift over 16 leapfrog steps: **1.8 × 10⁻⁴** relative — bounded
oscillation, no drift, exactly what a symplectic integrator should give.
Forward + reverse roundtrip: **6 × 10⁻⁸** position error — at numerical
precision. All 11 learnable parameters receive non-zero gradient. This
is the floor result: the implementation is correct.

### 2. On MNIST at matched parameter count, RMC underperforms MLP

5 epochs on 15k train / 5k val. Param counts: RMC 27,082, MLP 25,450.

| Model | Val acc (epoch 5) | Val loss |
|---|---:|---:|
| Linear (7,850 p) | 89.04% | 0.405 |
| MLP-32 (25,450 p) | **91.96%** | 0.256 |
| RMC (27,082 p) | 90.04% | 0.321 |

We initially read this as "RMC still descending, would catch up", but
that turned out to be a 3-epoch artifact — at 5+ epochs both models
descend at similar rates. RMC stays 2–3 pp behind MLP.

### 3. Ablations: the resonant readout is the only essential component

Same setup, four variants:

| Variant | Val acc | Val loss |
|---|---:|---:|
| Full RMC | 88.42% | 0.398 |
| no V_MLP (harmonic only) | 88.12% | 0.413 |
| no B (MLP-only potential) | 89.30% | 0.355 |
| **no resonance** (use x_T directly) | **87.20%** | 0.430 |

Dropping the resonant readout costs the most — the time-Fourier
projection is doing real work. Dropping the quadratic potential B
actually *helps* slightly at 3 epochs (this trend did not hold cleanly
at 5 epochs). The MLP potential alone does very little. Net: the
architecture's expressive engine is the resonant readout.

### 4. MNIST sample efficiency: RMC gets worse, not better

Best validation accuracy across 12 epochs, single seed:

| n_train | RMC | MLP | Gap |
|---:|---:|---:|---:|
| 100 | 52.06% | 61.00% | −8.94 pp |
| 500 | 63.66% | 78.80% | **−15.14 pp** |
| 2,000 | 82.26% | 85.76% | −3.50 pp |
| 10,000 | 88.70% | 92.08% | −3.38 pp |

The structured prior *hurts* generalization on MNIST, and hurts *most*
in the small-data regime — exactly the opposite of what a well-matched
prior should do. MNIST is static pixel data with no oscillatory
structure; the RMC's continuous-flow + resonant-readout prior is wrong
for the task.

### 5. On synthetic dynamical-system data, the single-seed result was hopeful — but it didn't replicate

We built a 4-class synthetic dataset: pure sinusoid, damped sinusoid,
two-tone (beats), linear chirp. This is data where the RMC's resonant
prior should fit, and where the four classes differ specifically in
frequency content.

**Single-seed result (12-15 epochs):**

| n_train | RMC | MLP | Gap |
|---:|---:|---:|---:|
| 100 | 43.40% | 46.60% | −3.20 pp |
| 300 | 52.40% | 51.00% | +1.40 pp |
| 1,000 | 72.60% | 73.20% | −0.60 pp |
| 3,000 | **88.40%** | 85.40% | **+3.00 pp** |

This looked promising — RMC pulled ahead at the largest size. But we
hadn't run multiple seeds, so the result was just one draw.

**Multi-seed result, 3 seeds, 10 epochs (mean ± std):**

| n_train | RMC | MLP | Gap | RMC wins / 3 |
|---:|---:|---:|---:|---:|
| 100 | 36.0% ± 3.5 | 39.3% ± 1.5 | −3.3 pp | 0 |
| 300 | 43.1% ± 7.1 | 47.9% ± 2.7 | −4.7 pp | 1 |
| 1,000 | 66.8% ± 0.9 | 71.9% ± 3.1 | −5.1 pp | 0 |
| 3,000 | 83.7% ± 2.0 | 85.6% ± 2.7 | −1.9 pp | 1 |

**Multi-seed result, 4 seeds, 15 epochs, n=3000:**

- RMC: 86.25% ± 2.7%, per seed: 0.884, 0.884, 0.854, 0.828
- MLP: 87.65% ± 2.5%, per seed: 0.842, 0.900, 0.874, 0.890
- Gap: −1.40 pp; RMC wins 1 of 4 paired comparisons (seed 0 only, +4.2 pp).

The single-seed +3.0 pp "win" was driven by one lucky seed. With proper
error bars, the RMC marginally loses on its supposed home turf.

## What this means

The strong claim from the original architectural pitch — "a structured
prior should win at matched parameter count on data with the right
structure" — is **not supported** by our experiments. We tested:

- visual data with no temporal structure (MNIST): RMC loses cleanly
- frequency-structured synthetic data (the architecture's ideal case):
  RMC and MLP are statistically indistinguishable, with MLP slightly ahead

Possible explanations:

1. **The architecture is genuinely not better.** A learnable Hamiltonian
   flow with a resonant readout is not a more efficient way to encode
   discriminative features than an affine-then-ReLU stack. The factoring
   move (geometry / dynamics / selectivity into three objects) sounds
   appealing but doesn't translate into measurable generalization gain.
2. **A single cell is the wrong test.** The architecture's pitch was
   always about *stackability* — reversibility lets you train deep
   stacks with O(1) memory per layer. We never tested that. A multi-
   layer RMC could plausibly behave differently.
3. **Real oscillatory data might still favor RMC.** Synthetic
   trajectories are clean but easy; real audio, EEG, or accelerometer
   data have richer spectro-temporal structure that an MLP would
   genuinely struggle to learn from raw signal. We didn't test this.
4. **Hyperparameter sensitivity wasn't explored.** The d, K, T, dt of
   the RMC and the hidden_dim of the MLP weren't tuned. Some sweep
   might shift the verdict.

What stands:

- The math is correct (verified numerically).
- The architecture trains stably.
- The resonant readout is the only essential component.
- The learned ω_k self-organize sensibly (mid-frequency modes carry
  the most weight in `resonant_modes.png`).

What doesn't stand:

- The claim that the RMC outperforms an MLP at matched params on any
  task we tested.

## Next steps if you want to push this further

1. **Stack RMCs** — test the depth claim properly. RMC×2/RMC×4 vs
   MLP×2/MLP×4 with reversible backprop.
2. **Real oscillatory data** — Free Spoken Digit Dataset (audio) or
   UCI HAR (accelerometer). Where synthetic structure fails to favor
   the prior, real structure might.
3. **Harder synthetic dynamics** — chaotic systems (Lorenz, Rössler) or
   PDE solutions where MLPs genuinely struggle.
4. **Hyperparameter sweep** — properly tune d, K, T, dt with multiple
   seeds.

We did not pursue these in this investigation; the multi-seed result
on our chosen tasks was clear enough to stop and report honestly rather
than keep iterating until something worked.

## Caveats

All experiments ran on a single CPU thread with small models (~2-30k
params) and short training (≤ 15 epochs). Larger models, longer
training, GPU runs, or different optimizers could change conclusions.
The investigation is a tightly-scoped proof of concept, not a thorough
evaluation.
