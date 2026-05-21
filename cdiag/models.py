"""
Minimal diagonal SSM implementations for testing the real-vs-complex
parameterization separation on the copy task.

A diagonal SSM with state dim n has impulse response
    k_t = sum_{i=1}^n C_i B_i lambda_i^t      (t = 0, 1, 2, ...)
where lambda_i are the diagonal entries of the state matrix.

If lambda_i in R, k_t is a sum of n pure exponentials. The kernel is
strictly monotone in t for each mode, so the model can only build up
output shapes by linear combinations of exponentials.

If lambda_i in C with Im(lambda_i) != 0, k_t is a sum of n damped
sinusoids (since complex eigenvalues come in conjugate pairs when we
take the real part of the output, or equivalently the complex SSM has
twice the underlying real dimension). The kernel can naturally encode
delays, periodic patterns, and sharp features.

For the copy task -- which requires reproducing inputs after a long
blank period -- the optimal impulse response is a delta function at
the appropriate delay. A delta is constructed naturally from cosines
(Fourier basis); building one out of pure exponentials requires
exponentially-large cancellation among many modes.

This is the Ran-Milo et al. (NeurIPS 2024) prediction: real diagonal
SSMs need either huge dimension or huge parameter magnitudes to
approximate oscillatory kernels.
"""

import torch
import torch.nn as nn


# ============================================================
# Parameterization utilities
# ============================================================

def _stabilize_real_lambda(raw):
    """Map unconstrained reals to stable lambdas in (-1, 1).

    We use lambda = -softplus(raw)/(1 + softplus(raw)) - 0 if you want
    only decay, OR just tanh(raw) to allow oscillation between modes.
    For the most permissive real SSM, allow any lambda in (-1, 1).
    tanh gives that and is well-behaved.
    """
    return torch.tanh(raw)


def _stabilize_complex_lambda(raw_log_mag, raw_phase):
    """Complex lambda = exp(-softplus(raw_log_mag)) * exp(i * raw_phase).

    Magnitude in (0, 1), phase free. This is the S4D-style
    parameterization. Magnitude bounded away from 1 to ensure stability;
    the softplus keeps log_mag positive so |lambda| = exp(-positive) < 1.
    """
    mag = torch.exp(-torch.nn.functional.softplus(raw_log_mag))
    return mag * torch.exp(1j * raw_phase)


# ============================================================
# Convolution kernel computation
# ============================================================

def _diag_ssm_kernel_real(lam, B, C, L):
    """Compute the impulse response k_t = sum_i C_i B_i lam_i^t for t=0..L-1.

    lam: [n]      (real, in (-1, 1))
    B:   [n, d_in]
    C:   [d_out, n]
    Returns: kernel of shape [d_out, d_in, L]
    """
    n = lam.shape[0]
    t = torch.arange(L, device=lam.device, dtype=lam.dtype)        # [L]
    # lam_pow[i, t] = lam_i^t  (potentially with sign for negative lam)
    # Use sign and log of abs to avoid (negative)^non_integer issues
    # Since t is integer, we can do this directly via pow.
    lam_pow = lam.unsqueeze(-1) ** t.unsqueeze(0)                  # [n, L]
    # C @ diag(b @ k(t)) effectively:
    # kernel[o, i, t] = sum_n C[o, n] * B[n, i] * lam_pow[n, t]
    return torch.einsum("on,ni,nt->oit", C, B, lam_pow)


def _diag_ssm_kernel_complex(lam, B, C, L):
    """Complex kernel, returning REAL part as the output.

    lam: [n] complex
    B:   [n, d_in] complex
    C:   [d_out, n] complex
    Returns: real kernel of shape [d_out, d_in, L]
    """
    n = lam.shape[0]
    t = torch.arange(L, device=lam.real.device, dtype=torch.float32)
    lam_pow = lam.unsqueeze(-1) ** t.unsqueeze(0)                  # [n, L] complex
    kernel = torch.einsum("on,ni,nt->oit", C, B, lam_pow)          # complex
    return kernel.real


# ============================================================
# Model wrappers
# ============================================================

class RealDiagSSM(nn.Module):
    """A diagonal SSM with state matrix in R, no nonlinearity.

    Input:  x of shape [B, L, d_in]
    Output: y of shape [B, L, d_out]

    Initialization: lambdas spread uniformly over (-1, 1) so the model
    has both fast (small |lambda|) and slow (|lambda| -> 1) modes at
    init. This gives the optimizer signal to work with on long
    sequences, mirroring the S4D-Lin choice for the complex case.
    """
    def __init__(self, n_state: int, d_in: int, d_out: int):
        super().__init__()
        self.n = n_state
        # Start with lambdas spread over (-1, 1). tanh(2) ~ 0.96, so
        # raw_lam in [-2, 2] gives lambdas in [-0.96, 0.96].
        self.raw_lam = nn.Parameter(
            torch.linspace(-2.0, 2.0, n_state).clone() + 0.05 * torch.randn(n_state))
        self.B = nn.Parameter(torch.randn(n_state, d_in) / (d_in ** 0.5))
        self.C = nn.Parameter(torch.randn(d_out, n_state) / (n_state ** 0.5))

    @property
    def lam(self):
        return _stabilize_real_lambda(self.raw_lam)

    def forward(self, x):
        # x: [B, L, d_in]
        B, L, d_in = x.shape
        kernel = _diag_ssm_kernel_real(self.lam, self.B, self.C, L)   # [d_out, d_in, L]
        # Convolve: y[b, t, o] = sum_{t'<=t} kernel[o, i, t-t'] * x[b, t', i]
        # We use FFT-based conv via torch.fft for efficiency at long L.
        return _causal_conv(x, kernel)

    def param_l2_norm(self):
        """L2 norm of all real-valued parameters."""
        return sum((p ** 2).sum() for p in self.parameters()).sqrt()


class ComplexDiagSSM(nn.Module):
    """Diagonal SSM with state matrix in C, output's real part.

    State dim n_state means n_state COMPLEX states (= 2*n_state real).

    Initialization follows S4D-Lin: magnitudes initialized close to 1
    so the model starts with long-memory modes; phases spread uniformly
    over [0, pi] so the model has access to a fan of oscillation
    frequencies at initialization. With raw_log_mag chosen so initial
    |lambda| ~ 0.9-0.97, the modes have effective time-constants well
    matched to long sequences (decay e-fold ~ 30 steps), giving the
    optimizer signal to work with even on long-delay tasks.
    """
    def __init__(self, n_state: int, d_in: int, d_out: int):
        super().__init__()
        self.n = n_state
        # We want |lambda| = exp(-softplus(raw_log_mag)) ~ 0.95 at init.
        # softplus(x) = ln(1 + e^x). For softplus(x) = 0.05, x ~ ln(e^0.05 - 1) ~ -2.97.
        # So initialize raw_log_mag at ~-3 with small noise.
        self.raw_log_mag = nn.Parameter(
            torch.full((n_state,), -3.0) + 0.1 * torch.randn(n_state))
        # Phases spread uniformly on [0, pi]. The complex conjugate of each
        # mode is implicit since we take Re(...), so [0, pi] gives access
        # to oscillations of all frequencies up to Nyquist.
        self.raw_phase = nn.Parameter(
            torch.linspace(0, torch.pi, n_state + 1)[1:].clone()
            + 0.01 * torch.randn(n_state))
        # B and C as complex parameters: store as real (n, d_in, 2) and form
        # complex tensors on the fly.
        self.B_re = nn.Parameter(torch.randn(n_state, d_in) / (d_in ** 0.5))
        self.B_im = nn.Parameter(torch.randn(n_state, d_in) / (d_in ** 0.5))
        self.C_re = nn.Parameter(torch.randn(d_out, n_state) / (n_state ** 0.5))
        self.C_im = nn.Parameter(torch.randn(d_out, n_state) / (n_state ** 0.5))

    @property
    def lam(self):
        return _stabilize_complex_lambda(self.raw_log_mag, self.raw_phase)

    @property
    def B(self):
        return torch.complex(self.B_re, self.B_im)

    @property
    def C(self):
        return torch.complex(self.C_re, self.C_im)

    def forward(self, x):
        B_dim, L, d_in = x.shape
        kernel = _diag_ssm_kernel_complex(self.lam, self.B, self.C, L)
        return _causal_conv(x, kernel)

    def param_l2_norm(self):
        return sum((p ** 2).sum() for p in self.parameters()).sqrt()


# ============================================================
# Causal convolution via FFT
# ============================================================

def _causal_conv(x, kernel):
    """Causal convolution.

    x:      [B, L, d_in]
    kernel: [d_out, d_in, L]
    Returns y [B, L, d_out] where
        y[b, t, o] = sum_{tau=0}^{t} sum_i kernel[o, i, tau] * x[b, t-tau, i]
    """
    B, L, d_in = x.shape
    d_out = kernel.shape[0]
    # Zero-pad to 2L for linear (acyclic) convolution
    P = 2 * L
    # FFT of x: take FFT over time
    Xf = torch.fft.rfft(x.transpose(1, 2), n=P)                  # [B, d_in, P//2+1]
    Kf = torch.fft.rfft(kernel, n=P)                             # [d_out, d_in, P//2+1]
    # Multiply and sum over input channels
    Yf = torch.einsum("oif,bif->bof", Kf, Xf)                    # [B, d_out, P//2+1]
    y = torch.fft.irfft(Yf, n=P)[:, :, :L]                       # [B, d_out, L]
    return y.transpose(1, 2)                                     # [B, L, d_out]


# ============================================================
# Quick sanity test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    # Real SSM
    m_r = RealDiagSSM(n_state=8, d_in=2, d_out=2)
    x = torch.randn(3, 16, 2)
    y_r = m_r(x)
    assert y_r.shape == (3, 16, 2)
    print(f"RealDiagSSM:   {sum(p.numel() for p in m_r.parameters())} params, output {y_r.shape}")

    # Complex SSM (same n_state -- so 2x the real "underlying" dimension)
    m_c = ComplexDiagSSM(n_state=8, d_in=2, d_out=2)
    y_c = m_c(x)
    assert y_c.shape == (3, 16, 2)
    print(f"ComplexDiagSSM:{sum(p.numel() for p in m_c.parameters())} params, output {y_c.shape}")

    # Verify backward pass works
    loss = (y_c ** 2).sum()
    loss.backward()
    print(f"ComplexDiagSSM backward OK, raw_log_mag grad norm: {m_c.raw_log_mag.grad.norm():.4f}")

    # Reproducibility check: a complex SSM with phases all = 0 should
    # behave like a real SSM with the same magnitudes
    torch.manual_seed(1)
    m_c2 = ComplexDiagSSM(n_state=4, d_in=1, d_out=1)
    with torch.no_grad():
        m_c2.raw_phase.zero_()    # phases = 0 -> all lambdas real positive
        m_c2.B_im.zero_()
        m_c2.C_im.zero_()
    x2 = torch.randn(1, 20, 1)
    k_re_via_complex = _diag_ssm_kernel_complex(m_c2.lam, m_c2.B, m_c2.C, 20)
    k_re_native      = _diag_ssm_kernel_real(m_c2.lam.real, m_c2.B.real, m_c2.C.real, 20)
    diff = (k_re_via_complex - k_re_native).abs().max()
    print(f"Sanity check: complex w/ phase=0 vs real kernel max diff = {diff:.2e}")
