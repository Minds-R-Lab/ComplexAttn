"""
Two transformer encoders for the parity-of-negation task:

  RealTransformer     : standard real-valued transformer encoder
  ComplexTransformer  : transformer where all hidden states are complex,
                        attention preserves phase, MLP uses modReLU

Both are matched as closely as possible in parameter count, depth, heads,
training compute, and pooling strategy. The ONLY structural difference is
the algebra over which the network operates.

Implementation note: we represent every complex tensor as a pair of real
tensors (real part, imag part). This sidesteps PyTorch's complex-autograd
edge cases and makes gradients fully standard.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import VOCAB_SIZE, PAD_ID


# ============================================================
#  REAL-VALUED TRANSFORMER (baseline)
# ============================================================

class RealTransformer(nn.Module):
    """A small standard transformer encoder, GELU MLP, sinusoidal positions."""

    def __init__(self, d_model=64, n_heads=4, n_layers=2,
                 d_ff=None, max_len=64, vocab_size=VOCAB_SIZE):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.register_buffer("pos_enc", self._sinusoidal_pos(max_len, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)        # binary logit

    @staticmethod
    def _sinusoidal_pos(L, d):
        pos = torch.arange(L).float().unsqueeze(1)
        div = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2).float() / d)
        pe  = torch.zeros(L, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, tokens):
        # tokens: [B, L]
        B, L = tokens.shape
        x = self.embed(tokens) + self.pos_enc[:L].unsqueeze(0)
        pad_mask = (tokens == PAD_ID)             # True where padded
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        # Use CLS position (index 0) — it is never padded.
        cls = h[:, 0]
        return self.head(cls).squeeze(-1)         # [B]


# ============================================================
#  COMPLEX-VALUED TRANSFORMER
# ============================================================

class ComplexLinear(nn.Module):
    """Linear map ℂ^in → ℂ^out, represented as a pair of real matrices.

       (W_r + i W_i)(x_r + i x_i)
         = (W_r x_r − W_i x_i) + i (W_r x_i + W_i x_r)

       Real parameter count: 2 · in · out   (+ 2 · out for bias)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.Wr = nn.Linear(in_features, out_features, bias=bias)
        self.Wi = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, xr, xi):
        out_r = self.Wr(xr) - self.Wi(xi)
        out_i = self.Wr(xi) + self.Wi(xr)
        return out_r, out_i


def mod_relu(zr, zi, bias):
    """modReLU activation: preserves phase, gates magnitude.

       Given complex z, compute  ReLU(|z| + b) · z/|z|.
       Bias is a learnable per-channel real shift on the magnitude.
    """
    mag = torch.sqrt(zr * zr + zi * zi + 1e-8)
    scale = F.relu(mag + bias) / mag
    return zr * scale, zi * scale


class ComplexLayerNorm(nn.Module):
    """Normalize complex vectors by their magnitude statistics, with
       a learnable complex affine transform afterwards."""
    def __init__(self, d):
        super().__init__()
        self.gamma_r = nn.Parameter(torch.ones(d))
        self.gamma_i = nn.Parameter(torch.zeros(d))
        self.beta_r  = nn.Parameter(torch.zeros(d))
        self.beta_i  = nn.Parameter(torch.zeros(d))

    def forward(self, zr, zi):
        # Normalize: subtract mean magnitude-squared, divide by std.
        # We use power normalization: E[|z|^2] = 1.
        power = (zr * zr + zi * zi).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(power + 1e-5)
        zr = zr * inv
        zi = zi * inv
        out_r = self.gamma_r * zr - self.gamma_i * zi + self.beta_r
        out_i = self.gamma_r * zi + self.gamma_i * zr + self.beta_i
        return out_r, out_i


class ComplexMultiHeadAttention(nn.Module):
    """Multi-head attention where Q, K, V are complex.

       Attention scores use the REAL PART of the Hermitian inner product,
                Re(Q · K*)  =  Q_r·K_r + Q_i·K_i,
       which is the natural real-valued readout of complex similarity and
       admits a standard softmax. Values are summed as complex numbers, so
       phase information propagates through aggregation.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h  = n_heads
        self.dh = d_model // n_heads
        self.Wq = ComplexLinear(d_model, d_model)
        self.Wk = ComplexLinear(d_model, d_model)
        self.Wv = ComplexLinear(d_model, d_model)
        self.Wo = ComplexLinear(d_model, d_model)

    def _split(self, xr, xi):
        B, L, D = xr.shape
        xr = xr.view(B, L, self.h, self.dh).transpose(1, 2)
        xi = xi.view(B, L, self.h, self.dh).transpose(1, 2)
        return xr, xi

    def _merge(self, xr, xi):
        B, H, L, Dh = xr.shape
        xr = xr.transpose(1, 2).contiguous().view(B, L, H * Dh)
        xi = xi.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return xr, xi

    def forward(self, xr, xi, key_padding_mask=None):
        Qr, Qi = self.Wq(xr, xi);  Qr, Qi = self._split(Qr, Qi)
        Kr, Ki = self.Wk(xr, xi);  Kr, Ki = self._split(Kr, Ki)
        Vr, Vi = self.Wv(xr, xi);  Vr, Vi = self._split(Vr, Vi)

        # Re(Q K*) = Q_r K_r^T + Q_i K_i^T
        scores = (Qr @ Kr.transpose(-2, -1) + Qi @ Ki.transpose(-2, -1))
        scores = scores / math.sqrt(self.dh)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out_r = attn @ Vr
        out_i = attn @ Vi
        out_r, out_i = self._merge(out_r, out_i)
        return self.Wo(out_r, out_i)


class ComplexEncoderBlock(nn.Module):
    """Pre-LN block: norm → complex attention → residual → norm → complex MLP → residual."""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = ComplexLayerNorm(d_model)
        self.attn = ComplexMultiHeadAttention(d_model, n_heads)
        self.ln2 = ComplexLayerNorm(d_model)
        self.ff1 = ComplexLinear(d_model, d_ff)
        self.ff2 = ComplexLinear(d_ff, d_model)
        self.modrelu_bias = nn.Parameter(torch.zeros(d_ff))

    def forward(self, xr, xi, key_padding_mask=None):
        nr, ni = self.ln1(xr, xi)
        ar, ai = self.attn(nr, ni, key_padding_mask)
        xr, xi = xr + ar, xi + ai

        nr, ni = self.ln2(xr, xi)
        hr, hi = self.ff1(nr, ni)
        hr, hi = mod_relu(hr, hi, self.modrelu_bias)
        hr, hi = self.ff2(hr, hi)
        xr, xi = xr + hr, xi + hi
        return xr, xi


class ComplexTransformer(nn.Module):
    """Encoder where every hidden state is a complex vector.

       Embeddings: each token is mapped to a learnable COMPLEX vector
                   (real and imaginary parts are both learned).
       Positions:  encoded with a complex sinusoid e^{i ω t} — the natural
                   complex Fourier basis. Added to the token embedding.
       Readout:    take the complex CLS state, project to ℂ¹, output Re(·).
    """
    def __init__(self, d_model=48, n_heads=4, n_layers=2,
                 d_ff=None, max_len=64, vocab_size=VOCAB_SIZE):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        # Complex embedding = two real embedding tables.
        self.embed_r = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.embed_i = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)

        pe_r, pe_i = self._complex_pos(max_len, d_model)
        self.register_buffer("pe_r", pe_r)
        self.register_buffer("pe_i", pe_i)

        self.blocks = nn.ModuleList([
            ComplexEncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f = ComplexLayerNorm(d_model)
        # Complex projection to ℂ¹, then read the real part as the logit.
        self.head = ComplexLinear(d_model, 1)

    @staticmethod
    def _complex_pos(L, d):
        # e^{i ω t} with logarithmically spaced frequencies.
        pos = torch.arange(L).float().unsqueeze(1)         # [L, 1]
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(d).float() / d).unsqueeze(0)  # [1, d]
        angles = pos * freqs
        return torch.cos(angles), torch.sin(angles)

    def forward(self, tokens):
        B, L = tokens.shape
        xr = self.embed_r(tokens) + self.pe_r[:L].unsqueeze(0)
        xi = self.embed_i(tokens) + self.pe_i[:L].unsqueeze(0)
        pad_mask = (tokens == PAD_ID)

        for block in self.blocks:
            xr, xi = block(xr, xi, key_padding_mask=pad_mask)

        xr, xi = self.ln_f(xr, xi)
        cr, ci = xr[:, 0], xi[:, 0]                # CLS position
        or_, oi_ = self.head(cr, ci)               # [B, 1]
        return or_.squeeze(-1)                     # real part as logit


# ============================================================
#  PARAMETER-COUNT MATCHER
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_configs(d_complex=48, n_heads=4, n_layers=2):
    """
    Pick a d_model for the real model that matches the complex model's
    parameter count as closely as possible.

    A complex linear ℂ^d → ℂ^d has 2d² real parameters; a real linear
    ℝ^d → ℝ^d has d² real parameters. The real model therefore needs to
    be roughly √2 × wider than the complex model to match params.
    """
    target = count_params(ComplexTransformer(d_model=d_complex,
                                             n_heads=n_heads,
                                             n_layers=n_layers))
    best_d, best_diff = None, float("inf")
    for d in range(d_complex, d_complex * 3):
        if d % n_heads:
            continue
        n = count_params(RealTransformer(d_model=d,
                                         n_heads=n_heads,
                                         n_layers=n_layers))
        diff = abs(n - target)
        if diff < best_diff:
            best_diff, best_d = diff, d
    return best_d


if __name__ == "__main__":
    # Smoke test + parameter check.
    rm = RealTransformer(d_model=64,  n_heads=4, n_layers=2)
    cm = ComplexTransformer(d_model=48, n_heads=4, n_layers=2)
    print(f"Real    (d=64)  params: {count_params(rm):,}")
    print(f"Complex (d=48)  params: {count_params(cm):,}")

    d_match = matched_configs(d_complex=48, n_heads=4, n_layers=2)
    rm2 = RealTransformer(d_model=d_match, n_heads=4, n_layers=2)
    print(f"\nFor ComplexTransformer(d=48): matched real d_model = {d_match}")
    print(f"Real    (d={d_match})  params: {count_params(rm2):,}")
    print(f"Complex (d=48)  params: {count_params(cm):,}")

    # Forward-pass sanity check.
    tokens = torch.randint(0, VOCAB_SIZE, (2, 10))
    tokens[:, 0] = 1
    print("\nForward shapes:")
    print("  Real:    ", rm2(tokens).shape)
    print("  Complex: ", cm(tokens).shape)
