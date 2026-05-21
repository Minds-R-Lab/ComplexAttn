# `cdiag/` — Real vs Complex Diagonal SSMs on the Copy Task

This subdirectory implements a controlled empirical test of the
Ran-Milo et al. (NeurIPS 2024) prediction that **complex diagonal
SSMs are structurally — not just notationally — more expressive than
real diagonal SSMs** for tasks requiring oscillatory impulse responses.

## The headline result

On the copy task at K=5, L=150 (data length 5, blank delay 150, total
sequence length 161), trained for 5000 steps with AdamW, three seeds:

| n_state | kind | Params | OOD acc (mean ± SE) |
|--:|---|--:|--:|
| 16 | complex | 640 | **1.000 ± 0.000** |
| 16 | real | 320 | 0.263 ± 0.024 |
| 32 | real | 640 | 0.393 ± 0.011 |
| 64 | real | 1,280 | 0.461 ± 0.017 |
| 128 | real | 2,560 | 0.481 ± 0.001 |
| 256 | real | 5,120 | 0.476 ± 0.005 |
| **512** | **real** | **10,240** | **0.489 ± 0.002** |

Complex n=16 reliably solves the task at 100% across all seeds with
640 parameters. Real plateaus at ~48% no matter how big it gets. From
n=64 to n=512 — an 8x increase in state dimension and parameters —
accuracy moves only from 46% to 49%. **The plateau is deterministic**:
seed standard error at n=512 is +/-0.002 across three seeds (0.485,
0.489, 0.494). The complex SSM with 640 parameters beats the largest
real SSM (10,240 parameters, 16x more) by 51 percentage points.
**Capacity does not close the gap.**

## What's actually happening mechanistically

The plateau at 48% is not "graceful degradation with delay" or
"random noise." A direct inspection of the learned kernels reveals
a sharp failure mode: **the real diagonal SSM cannot learn
class-specific kernel shapes.**

We measured the per-pair correlation between the 8 diagonal kernels
`k[i, i, :]` (one for each data class i = 1..8):

| Architecture | Mean off-diagonal kernel correlation | Effective rank |
|---|--:|--:|
| Complex n=16 | 0.72 (diverse) | 8 significant components |
| Real n=128 | **0.99 (collapsed)** | **4 significant components** |

In the real model, the 8 diagonal kernels are virtually identical
(correlation 0.99-1.00). The model has converged to a single kernel
shape that's applied near-uniformly to every input class, just with
slightly different scaling. There is no class-specific routing
happening — the model cannot distinguish input tokens by their
identity.

The complex model has diverse class-specific kernels (correlation
~0.72, effective rank 8), giving it the 8 distinct response patterns
needed to route each input class to its corresponding output class.

### Why this happens — a function-class argument

For a diagonal SSM, the kernel is

    k[o, i, t] = sum_n C[o, n] * B[n, i] * lambda_n^t

The shape lambda_n^t is class-independent — only B and C are
class-specific. To get class-specific kernel shapes, the model needs
to combine the mode basis {lambda_n^t} differently for different
classes via the C[o, n] * B[n, i] coefficients.

For complex lambda_n = exp(i*theta_n), the mode basis is the discrete
Fourier basis. Different linear combinations of Fourier modes give
arbitrary shapes — including sharp delta-like features at different
time offsets. So class-specific B and C immediately produce
class-specific sharp-feature locations.

For real lambda_n in (-1, 1), the mode basis is pure exponentials.
These cannot represent sharply-localized features without extreme
cancellation between many modes. To get class-specific sharp features
at different locations, the model would need extreme class-specific
cancellation patterns — and gradient descent appears unable to find
such patterns. Instead it settles on a single shared kernel shape and
barely distinguishes classes at all.

This matches the Ran-Milo theory but identifies a more specific
failure mode than they describe: it's not just that real needs
exponentially large parameter values to approximate one oscillatory
kernel — it's that real cannot learn *class-conditional shape
diversity*, which is required for any task with multi-class structure.

## What's here

  - `models.py`            RealDiagSSM and ComplexDiagSSM with S4D-Lin init
  - `data.py`              Copy task generator
  - `run.py`               Training loop + experiment configurations
  - `per_position.py`      Per-output-position accuracy diagnostic
  - `kernel_inspect.py`    Plots all diagonal kernels visually
  - `kernel_collapse.py`   Correlation matrix and effective rank — the
                           diagnostic that revealed the collapse
  - `visualize_kernel.py`  (older) Inspect single-class kernels in detail

## Run

```bash
# Headline result on H100 (~5 min):
python3 run.py --config L_sweep --device cuda --K 5 --L 150 --steps 5000 --seeds 3

# Mechanism analysis (a couple minutes each on CPU, faster on H100):
python3 kernel_collapse.py
python3 kernel_inspect.py
python3 per_position.py
```

## What this experiment shows

A controlled, reproducible empirical setting where:

  1. Complex diagonal parameterization solves a sequence task exactly
     at 640 parameters.
  2. Real diagonal parameterization plateaus at ~50% accuracy, with
     no improvement from 16x more parameters.
  3. The failure mode is identified concretely: the real model
     cannot learn class-conditional kernel shape diversity,
     collapsing to a single shared kernel shape across all classes.

The Ran-Milo paper proved the separation for SSMs in the abstract; we
provide a concrete minimal reproduction and identify a sharper
mechanism (kernel-shape collapse rather than parameter blow-up) than
their theory characterizes directly.

## What this does NOT show

  - Anything about complex parameterization in non-diagonal SSMs.
  - Anything about the practical use of complex for tasks that don't
    require oscillatory kernels.
  - That the gap exists for nonlinear sequence models. This is a
    purely linear setting, and adding nonlinearity changes the
    dynamics.

## Connection to the earlier work in this repo

The original ComplexAttn project tried to test "does complex help?" on
cyclic-group composition tasks. The factorial ablation of Experiment
6 revealed that in that setting the architecture we called complex
was bit-identical to its real counterpart — complex was notational,
not structural.

This experiment finds the regime where complex IS structural:
diagonal linear sequence models, where the mode basis (exponential vs.
sinusoidal) is genuinely different in the two parameterizations.

The cyclic-group task was the wrong testbed because its structure was
fully expressible in real coordinates via cos/sin readouts; the copy
task is the right testbed because the optimal impulse response is
sharply localized in time, requiring Fourier-basis modes that real
diagonal SSMs do not have.

The honest project trajectory:

  - Cyclic-group experiments -> wrong testbed, complex was vacuous.
  - Literature review -> found the right testbed (diagonal SSMs).
  - This experiment -> reproduces the predicted gap, identifies the
    mechanism (kernel-shape collapse, not parameter blow-up).

The honest finding is: **complex parameterization is structurally
meaningful in exactly those settings where the function class it
parameterizes (oscillatory dynamics) is genuinely different from the
function class real parameterization gives (monotone-exponential
dynamics).** Outside such settings, complex is notational.
