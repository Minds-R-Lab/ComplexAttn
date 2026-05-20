"""
Experiment 2 architectures: multiplicative complex composition.

The diagnosis from Experiment 1: the complex transformer LEARNED phase
(77% of pairs ~π rad) but softmax attention destroyed composition.
Softmax weights sum to 1, so it AVERAGES contributions instead of
multiplying them.

Fix: architectures where composition is multiplicative by construction.
We test three:

  PhaseSumNet      -- minimal, set-equivariant. Each token contributes a
                       phase vector. Sum phases over sequence (= product
                       of complex unit factors). The "perfect" architecture
                       for this task — IF the phase hypothesis is right.

  GatedComplexRNN  -- bidirectional gated complex recurrent network. Each
                       token's update is h ← gate*(rotate(h, θ_x) + v_x)
                                          + (1-gate)*h.
                       More flexible than PhaseSumNet; tests whether the
                       inductive bias survives in a more general arch.

  GRUBaseline      -- standard bidirectional GRU, the real-valued control.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import VOCAB_SIZE, PAD_ID, NOT_ID, T_ID, F_ID


# ============================================================
#  PHASE SUM NET  -- minimal multiplicative complex architecture
# ============================================================

class PhaseSumNet(nn.Module):
    """
    The simplest possible architecture that respects the algebra of the task.

    Each token x maps to a phase vector θ(x) ∈ ℝ^d.
    Total phase: Θ = Σ_t θ(x_t)   (zero contribution from PAD).
    Hidden state: exp(iΘ)  — a complex unit vector per dimension.
    Readout: linear layer over [cos(Θ), sin(Θ)] producing a single logit.

    Why this should work:
      Sum is commutative → order-invariant (the task is order-invariant).
      Phase rotations compose multiplicatively: e^{iπ} · e^{iπ} · ... = (−1)^k.
      Therefore (depth=k) generalizes to any k by construction.

    Predicted optimal solution after training:
        θ(T)      ≈ 0       (factor +1)
        θ(F)      ≈ π       (factor −1)
        θ(NOT)    ≈ π       (factor −1)
        θ(filler) ≈ 0       (factor +1)
    """
    def __init__(self, d_model=16, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.phase = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        # Spread phases across the unit circle so the model starts with
        # signal. Tight-near-zero init kills the gradient (cos ≈ 1 for all
        # sentences → uniform logits → no direction to escape).
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        with torch.no_grad():
            self.phase.weight[PAD_ID].zero_()
        self.head = nn.Linear(2 * d_model, 1)

    def forward(self, tokens):
        # tokens: [B, L]
        phases = self.phase(tokens)            # [B, L, d]  — PAD rows are 0
        total  = phases.sum(dim=1)             # [B, d]
        feat   = torch.cat([torch.cos(total), torch.sin(total)], dim=-1)
        return self.head(feat).squeeze(-1)


# ============================================================
#  GATED COMPLEX RNN -- bidirectional, flexible
# ============================================================

class ComplexGRUCell(nn.Module):
    """
    A single recurrent step in ℂ^d.

    For input token x with learned per-token parameters
       θ(x) ∈ ℝ^d       (phase rotation, diagonal complex)
       v(x) ∈ ℂ^d       (additive complex contribution)
       g(x) ∈ (0,1)^d   (per-dim sigmoid gate, real)
    the update is:
       candidate = (rotate h by θ) + v
       h ← (1 − g) ⊙ h  +  g ⊙ candidate

    With g=1 this reduces to h ← rotate(h, θ) + v.
    With g=0 the token is ignored.

    For the NOT token the optimal solution is g ≈ 1, θ ≈ π·𝟙, v ≈ 0.
    For an atom T it is g ≈ 1, θ ≈ 0, v ≈ unit (real direction).
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.phase   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.value_r = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.value_i = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.gate    = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        # Phases spread across unit circle (see PhaseSumNet note on why
        # tight-near-zero init kills the gradient signal).
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        nn.init.normal_(self.value_r.weight, mean=0.0, std=0.3)
        nn.init.normal_(self.value_i.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.gate.weight, 2.0)   # σ(2) ≈ 0.88, default on
        with torch.no_grad():
            for emb in (self.phase, self.value_r, self.value_i, self.gate):
                emb.weight[PAD_ID].zero_()

    def forward(self, h_r, h_i, tokens):
        theta = self.phase(tokens)
        v_r   = self.value_r(tokens)
        v_i   = self.value_i(tokens)
        g     = torch.sigmoid(self.gate(tokens))
        c, s  = torch.cos(theta), torch.sin(theta)
        rot_r = h_r * c - h_i * s
        rot_i = h_r * s + h_i * c
        cand_r = rot_r + v_r
        cand_i = rot_i + v_i
        new_r = (1 - g) * h_r + g * cand_r
        new_i = (1 - g) * h_i + g * cand_i
        # PAD positions: pass through unchanged
        pad = (tokens == PAD_ID).unsqueeze(-1).float()
        new_r = pad * h_r + (1 - pad) * new_r
        new_i = pad * h_i + (1 - pad) * new_i
        return new_r, new_i


class GatedComplexRNN(nn.Module):
    """
    Bidirectional gated complex recurrent network.

    Two independent cells (forward + backward), each scanning the sequence
    one step at a time. Final states from both directions are concatenated
    and read out by a real linear layer.

    Why bidirectional: the task is order-invariant (NOTs and the atom can
    appear in any order). A unidirectional RNN can only correctly multiply
    NOTs onto an existing valence after the atom appears. The reverse pass
    fixes the other half of orderings; the readout can learn to combine.
    """
    def __init__(self, d_model=24, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.cell_fwd = ComplexGRUCell(d_model, vocab_size)
        self.cell_bwd = ComplexGRUCell(d_model, vocab_size)
        self.h0_r = nn.Parameter(torch.zeros(d_model))
        self.h0_i = nn.Parameter(torch.zeros(d_model))
        self.head = nn.Linear(4 * d_model, 1)

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
        return self.head(feat).squeeze(-1)


# ============================================================
#  GRU BASELINE  -- standard real-valued bidirectional GRU
# ============================================================

class GRUBaseline(nn.Module):
    """Standard bidirectional GRU. Parameter-matchable to GatedComplexRNN
       via the matched_d_gru helper below."""
    def __init__(self, d_model=20, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.gru = nn.GRU(d_model, d_model, num_layers=1,
                          bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * d_model, 1)

    def forward(self, tokens):
        x = self.embed(tokens)
        _, h_n = self.gru(x)
        feat = torch.cat([h_n[0], h_n[1]], dim=-1)   # [B, 2d]
        return self.head(feat).squeeze(-1)


# ============================================================
#  PARAMETER-MATCHING HELPERS
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_d_gru(d_complex, vocab_size=VOCAB_SIZE):
    """Pick a GRU hidden dim that closely matches GatedComplexRNN params."""
    target = count_params(GatedComplexRNN(d_model=d_complex, vocab_size=vocab_size))
    best_d, best_diff = None, float("inf")
    for d in range(4, d_complex * 2):
        n = count_params(GRUBaseline(d_model=d, vocab_size=vocab_size))
        diff = abs(n - target)
        if diff < best_diff:
            best_diff, best_d = diff, d
    return best_d


if __name__ == "__main__":
    # Smoke test + parameter check.
    for d in (8, 16, 32):
        m = PhaseSumNet(d_model=d)
        print(f"PhaseSumNet     (d={d:>2})  params: {count_params(m):>6,}")
    print()
    d_complex = 24
    cm = GatedComplexRNN(d_model=d_complex)
    print(f"GatedComplexRNN (d={d_complex})  params: {count_params(cm):>6,}")
    d_gru = matched_d_gru(d_complex)
    gm = GRUBaseline(d_model=d_gru)
    print(f"GRUBaseline     (d={d_gru:>2})  params: {count_params(gm):>6,}  "
          f"(matched to GatedComplexRNN)")

    # Forward pass sanity.
    tokens = torch.randint(2, VOCAB_SIZE, (2, 10))
    tokens[:, 0] = 1  # CLS
    print("\nForward shapes:")
    for name, model in [("PhaseSumNet",      PhaseSumNet(d_model=16)),
                        ("GatedComplexRNN",  GatedComplexRNN(d_model=24)),
                        ("GRUBaseline",      GRUBaseline(d_model=20))]:
        out = model(tokens)
        print(f"  {name:18s} {tuple(out.shape)}")
