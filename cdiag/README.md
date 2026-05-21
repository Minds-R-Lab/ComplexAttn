# `cdiag/` — Real vs Complex Diagonal SSMs on the Copy Task

This subdirectory implements a new experimental direction motivated
by stepping back from the original cyclic-group framing of the
project.

## Why this is here

The original ComplexAttn paper investigated whether complex-valued
neural architectures offer inductive-bias advantages over real-valued
ones on cyclic-group tasks. The Exp 6 factorial ablation showed that
in that setting, complex was structurally vacuous — the architecture
we called complex was bit-identical to a real-valued counterpart.

After that finding I went back to the literature and the conceptual
question: when, if ever, does complex parameterization actually
provide something that real parameterization can't?

The answer, established by Ran-Milo, Cohen-Karlik et al.
("Provable Benefits of Complex Parameterizations for Structured
State Space Models", NeurIPS 2024), is: in diagonal structured
state-space models, complex parameterization is **provably more
expressive per dimension** and **requires only moderate parameter
magnitudes**, while real diagonal SSMs of the same dimension
typically require exponentially large parameter values to approximate
oscillatory mappings — values that gradient descent cannot find in
practice.

This is a real claim about complex giving a structural advantage.
Not "complex helps because of cos/sin readout" (which the previous
project showed was equivalent to real). The mechanism is genuinely
in the algebra: complex diagonal eigenvalues `lambda_i = r_i exp(i*theta_i)`
encode damped sinusoids `r_i^t cos(theta_i t + phi)`, while real
diagonal eigenvalues encode only pure exponentials `lambda_i^t`. The
function classes are different.

## What this code does

Implements minimal diagonal SSMs in both real and complex form, runs
them on the copy task, and measures:

  1. Final accuracy.
  2. Training dynamics (loss curves, parameter magnitudes over time).
  3. Eigenvalue magnitudes — do they push toward the unit circle?
  4. Comparison at matched state dimension AND matched parameter count.

This is a direct empirical reproduction of one part of the Ran-Milo
result on a controlled setup we can analyze in detail.

## Files

- `models.py`     `RealDiagSSM` and `ComplexDiagSSM` with shared kernel
                  computation via FFT convolution
- `data.py`       Copy task generator
- `run.py`        Training loop, experiment configurations, plotting

## Running

```bash
# Quick CPU smoke test (~1 min)
python3 run.py --config matched_n --device cpu --K 3 --L 10 --seeds 2 --steps 1000

# Full study on H100
python3 run.py --config full --device cuda --K 5 --L 50 --seeds 3 --steps 5000
```

The `full` config sweeps real SSMs from n=16 up to n=128 against a
fixed complex SSM at n=16, to map out at what real dimension (if any)
the gap closes.

## CPU pilot results

K=3, L=10, 1000 training steps, 2 seeds each:

| Architecture       | Params | Final accuracy |
|---|--:|--:|
| Real    n=16       |  320   | 60% ± 1%       |
| Complex n=16       |  640   | **99% ± 1%**   |
| Real    n=32       |  640   | 70% ± 0.06%    |
| Complex n=32       | 1280   | 100% ± 0.0%    |

Two key observations:

1. **Matched state dim:** Complex n=16 vs Real n=16: 99% vs 60%.
   Complex wins by 39 points. Complex has 2× the params.
2. **Matched params:** Complex n=16 vs Real n=32 (both 640 params):
   99% vs 70%. Complex wins by 29 points despite having half the
   state dimension.

The complex model also pushes its eigenvalue magnitudes much closer
to the unit circle (max|λ| ~0.95 vs real's ~0.78), confirming it's
exploiting the oscillatory regime that real can't reach.

The harder task (K=5, L=30) didn't converge in 1500 steps for either
architecture, suggesting the full H100 study should use longer
training (5000+ steps) and/or larger L sweeps.

## What this experiment can and cannot show

**Can show:**
- A concrete, reproducible gap between real and complex parameterizations
  at the same architectural complexity, on a controlled task.
- Whether the gap closes with capacity (by varying real n_state).
- The mechanism of the gap (parameter magnitudes, eigenvalue locations,
  loss-landscape navigation).

**Cannot show:**
- Anything not already in the Ran-Milo paper. They proved the
  separation and observed it on Mamba copy. Our contribution is the
  controlled, hands-on reproduction with full diagnostics.

What would be a real new contribution from here:
- Connect the SSM finding to optimization geometry directly:
  characterize *why* gradient descent struggles to find the right
  exponential parameter values, e.g. via loss-landscape analysis.
- Test whether the gap exists for nonlinear SSMs (S4 etc. have
  per-step nonlinearities); the Ran-Milo theory is purely about the
  linear part.
- Connect to the grokking phenomenon we observed earlier in the
  cyclic-group setting: is gated recurrence on cyclic tasks failing
  for the same reason a real diagonal SSM fails on copy — namely,
  the optimizer can't navigate to oscillatory solutions in real
  coordinates?
