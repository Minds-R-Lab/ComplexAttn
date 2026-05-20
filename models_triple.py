"""
Experiment 3 architectures: rotation mod 3.

Four models, each highlighting a different combination of (composition,
readout). The theoretical prediction is sharp:

   - additive composition + linear readout    →  CANNOT solve mod-3 OOD
   - additive composition + periodic readout  →  CAN solve (PhaseSumNet)
   - multiplicative composition + any readout →  CAN solve

The proof for the first row: linear readout of a sum of per-token
embeddings produces logits that are linear in k. (k mod 3) is periodic,
not linear. Therefore the argmax cannot follow the correct class beyond
the training depth range; for large k, one class's logit dominates and
that class wins for all further k.

Architectures:

  PhaseSumNet3     additive composition (sum of phases) + periodic
                   readout (cos/sin). Complex unit-circle. Predicted
                   to work with d≥1.

  RealAddNet       additive composition + linear readout. The
                   negative control. *Theoretically* cannot solve
                   mod-3 at OOD depths. Predicted to memorize ID and
                   fail OOD.

  GatedComplexRNN3 multiplicative composition via per-token rotation.
                   Linear readout over (Re h, Im h) suffices because
                   the three correct points on the unit circle are
                   linearly separable.

  GRUBaseline3     multiplicative-via-gating composition. Gated tanh
                   is non-linear in k by construction. Predicted to
                   work.

All output 3 logits.
"""

import math
import torch
import torch.nn as nn
from data_triple import (VOCAB_SIZE, PAD_ID, TWIRL_ID, A0_ID, A1_ID, A2_ID,
                          FILLER_START, NUM_FILLERS, NUM_CLASSES)


# ============================================================
#  PhaseSumNet3 -- complex unit-circle, periodic readout
# ============================================================

class PhaseSumNet3(nn.Module):
    """
    Per-token phase θ(x) ∈ ℝ^d.  Total Θ = Σ_t θ(x_t).
    Hidden feature: [cos(Θ), sin(Θ)] ∈ ℝ^{2d}.
    Readout: linear → 3 logits.

    Predicted optimum (one-dim version):
        θ(TWIRL) ≈ 2π/3,
        θ(A0)    ≈ 0,
        θ(A1)    ≈ 2π/3,
        θ(A2)    ≈ 4π/3,
        θ(filler)≈ 0.
    Then Θ ≈ (atom_class + k) · 2π/3 (mod 2π), and the readout maps the
    three points e^{i·c·2π/3} on the unit circle to three class logits.
    """
    def __init__(self, d_model=16, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.phase = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        with torch.no_grad():
            self.phase.weight[PAD_ID].zero_()
        self.head = nn.Linear(2 * d_model, NUM_CLASSES)

    def forward(self, tokens):
        total = self.phase(tokens).sum(dim=1)            # [B, d]
        feat  = torch.cat([torch.cos(total),
                           torch.sin(total)], dim=-1)    # [B, 2d]
        return self.head(feat)                            # [B, 3]


# ============================================================
#  RealAddNet -- the negative control (theorem: cannot generalize OOD)
# ============================================================

class RealAddNet(nn.Module):
    """
    Sum of real embeddings, linear readout.  No periodicity, no gating.

    Logits = W · Σ_t e(x_t) + b
           = W · e(atom) + k·W·e(TWIRL) + b   (fillers contribute zero
                                                if learned correctly)
    Each logit is linear in k; argmax cannot follow (k mod 3) outside
    the training depth range. We expect this to MEMORIZE training depths
    and FAIL at OOD depths.

    Parameter-matched to PhaseSumNet3 by using 2*d hidden dims so the
    final feature width going into the head is the same.
    """
    def __init__(self, d_model=16, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, 2 * d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.5)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()
        self.head = nn.Linear(2 * d_model, NUM_CLASSES)

    def forward(self, tokens):
        total = self.embed(tokens).sum(dim=1)            # [B, 2d]
        return self.head(total)                           # [B, 3]


# ============================================================
#  GatedComplexRNN3 -- multiplicative composition via rotation
# ============================================================

class ComplexGRUCell3(nn.Module):
    """One step in ℂ^d. Identical to the Exp 2 cell, retained here so
       this experiment is self-contained."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.phase   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.value_r = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.value_i = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.gate    = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        nn.init.normal_(self.value_r.weight, mean=0.0, std=0.3)
        nn.init.normal_(self.value_i.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.gate.weight, 2.0)
        with torch.no_grad():
            for emb in (self.phase, self.value_r, self.value_i, self.gate):
                emb.weight[PAD_ID].zero_()

    def forward(self, h_r, h_i, tokens):
        theta = self.phase(tokens)
        v_r, v_i = self.value_r(tokens), self.value_i(tokens)
        g = torch.sigmoid(self.gate(tokens))
        c, s = torch.cos(theta), torch.sin(theta)
        rot_r = h_r * c - h_i * s
        rot_i = h_r * s + h_i * c
        cand_r, cand_i = rot_r + v_r, rot_i + v_i
        new_r = (1 - g) * h_r + g * cand_r
        new_i = (1 - g) * h_i + g * cand_i
        pad = (tokens == PAD_ID).unsqueeze(-1).float()
        new_r = pad * h_r + (1 - pad) * new_r
        new_i = pad * h_i + (1 - pad) * new_i
        return new_r, new_i


class GatedComplexRNN3(nn.Module):
    """Bidirectional gated complex RNN with 3-class head."""
    def __init__(self, d_model=24, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.cell_fwd = ComplexGRUCell3(d_model, vocab_size)
        self.cell_bwd = ComplexGRUCell3(d_model, vocab_size)
        self.h0_r = nn.Parameter(torch.zeros(d_model))
        self.h0_i = nn.Parameter(torch.zeros(d_model))
        self.head = nn.Linear(4 * d_model, NUM_CLASSES)

    def _scan(self, tokens, cell, reverse=False):
        B, L = tokens.shape
        h_r = self.h0_r.expand(B, -1).contiguous()
        h_i = self.h0_i.expand(B, -1).contiguous()
        idxs = range(L - 1, -1, -1) if reverse else range(L)
        for t in idxs:
            h_r, h_i = cell(h_r, h_i, tokens[:, t])
        return h_r, h_i

    def forward(self, tokens):
        fr, fi = self._scan(tokens, self.cell_fwd, reverse=False)
        br, bi = self._scan(tokens, self.cell_bwd, reverse=True)
        feat = torch.cat([fr, fi, br, bi], dim=-1)
        return self.head(feat)


# ============================================================
#  GRUBaseline3 -- real-valued bidirectional GRU, 3-class head
# ============================================================

class GRUBaseline3(nn.Module):
    def __init__(self, d_model=20, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.gru = nn.GRU(d_model, d_model, num_layers=1,
                          bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * d_model, NUM_CLASSES)

    def forward(self, tokens):
        x = self.embed(tokens)
        _, h_n = self.gru(x)
        feat = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(feat)


# ============================================================
#  Param utilities
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_d_gru(d_complex, vocab_size=VOCAB_SIZE):
    target = count_params(GatedComplexRNN3(d_model=d_complex, vocab_size=vocab_size))
    best_d, best_diff = None, float("inf")
    for d in range(4, d_complex * 2):
        n = count_params(GRUBaseline3(d_model=d, vocab_size=vocab_size))
        if abs(n - target) < best_diff:
            best_diff, best_d = abs(n - target), d
    return best_d


if __name__ == "__main__":
    print(f"{'model':20s} {'params':>8s}")
    print("-" * 30)
    for name, m in [
        ("PhaseSumNet3 (d=16)",     PhaseSumNet3(d_model=16)),
        ("RealAddNet (d=16)",       RealAddNet(d_model=16)),
        ("GatedComplexRNN3 (d=24)", GatedComplexRNN3(d_model=24)),
        ("GRUBaseline3 (matched)",  GRUBaseline3(d_model=matched_d_gru(24))),
    ]:
        print(f"{name:20s} {count_params(m):>8,}")

    # Forward pass sanity check.
    tokens = torch.randint(2, VOCAB_SIZE, (4, 12))
    tokens[:, 0] = 1
    for name, m in [
        ("PhaseSumNet3",     PhaseSumNet3(d_model=16)),
        ("RealAddNet",       RealAddNet(d_model=16)),
        ("GatedComplexRNN3", GatedComplexRNN3(d_model=24)),
        ("GRUBaseline3",     GRUBaseline3(d_model=15)),
    ]:
        out = m(tokens)
        assert out.shape == (4, 3)
    print("\nForward pass shapes OK (B=4, 3 classes).")
