"""
Architectures for the Z/n cyclic-rotation sweep.

These mirror models_triple.py but take a CyclicTaskSpec at construction
so vocab size and class count are correct for whichever n we're testing.

Predictions for the sweep over n ∈ {2, 3, 5, 7, 11, 13}:

  PhaseSumNet_n      should generalize for all n. Unit circle
                      naturally contains a subgroup of order n for
                      every n. Set-equivariance + bounded readout
                      means depth/length extrapolation is benign.
                      Predicted: 100% OOD for every n.

  RealAddNet_n       should fail for every n ≥ 3, by the same theorem
                      as Exp 3 — linear-in-k logits can change argmax
                      at most n-1 times across all of ℝ but the
                      correct label cycles every n steps, so no linear
                      readout can fit even the training distribution
                      once n > 2 + k_max.

  GatedComplexRNN_n  should fit ID for all n but the OOD degradation
                      we saw in Exp 3 should *worsen* with larger n.
                      Why: distinguishing n equispaced points on the
                      unit circle requires angular resolution ~2π/n,
                      and the additive value contribution v_r,v_i lets
                      the state magnitude drift with sequence length,
                      shrinking the effective angular margin.

  GRUBaseline_n      should generalize at small n (closure was 0.95
                      for n=3) but the closure probe should degrade
                      with n. Gating implements approximate n-state
                      automata; the approximation breaks at higher
                      resolution.
"""

import math
import torch
import torch.nn as nn
from data_cyclic import CyclicTaskSpec, PAD_ID


# ============================================================
#  PhaseSumNet — additive phase, periodic readout, n classes
# ============================================================

class PhaseSumNet_n(nn.Module):
    def __init__(self, spec: CyclicTaskSpec, d_model: int = 16):
        super().__init__()
        self.spec   = spec
        self.d_model = d_model
        self.phase = nn.Embedding(spec.vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        with torch.no_grad():
            self.phase.weight[PAD_ID].zero_()
        self.head = nn.Linear(2 * d_model, spec.num_classes)

    def forward(self, tokens):
        total = self.phase(tokens).sum(dim=1)
        feat  = torch.cat([torch.cos(total), torch.sin(total)], dim=-1)
        return self.head(feat)


# ============================================================
#  RealAddNet — additive embeddings, linear readout (negative control)
# ============================================================

class RealAddNet_n(nn.Module):
    def __init__(self, spec: CyclicTaskSpec, d_model: int = 16):
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab_size, 2 * d_model,
                                   padding_idx=PAD_ID)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.5)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()
        self.head = nn.Linear(2 * d_model, spec.num_classes)

    def forward(self, tokens):
        return self.head(self.embed(tokens).sum(dim=1))


# ============================================================
#  GatedComplexRNN — multiplicative composition via rotation
# ============================================================

class ComplexGRUCell_n(nn.Module):
    def __init__(self, spec: CyclicTaskSpec, d_model: int):
        super().__init__()
        self.spec = spec
        V = spec.vocab_size
        self.phase   = nn.Embedding(V, d_model, padding_idx=PAD_ID)
        self.value_r = nn.Embedding(V, d_model, padding_idx=PAD_ID)
        self.value_i = nn.Embedding(V, d_model, padding_idx=PAD_ID)
        self.gate    = nn.Embedding(V, d_model, padding_idx=PAD_ID)
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


class GatedComplexRNN_n(nn.Module):
    def __init__(self, spec: CyclicTaskSpec, d_model: int = 24):
        super().__init__()
        self.spec = spec
        self.d_model = d_model
        self.cell_fwd = ComplexGRUCell_n(spec, d_model)
        self.cell_bwd = ComplexGRUCell_n(spec, d_model)
        self.h0_r = nn.Parameter(torch.zeros(d_model))
        self.h0_i = nn.Parameter(torch.zeros(d_model))
        self.head = nn.Linear(4 * d_model, spec.num_classes)

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
#  GRU baseline (n classes)
# ============================================================

class GRUBaseline_n(nn.Module):
    def __init__(self, spec: CyclicTaskSpec, d_model: int = 20):
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab_size, d_model, padding_idx=PAD_ID)
        self.gru   = nn.GRU(d_model, d_model, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.head  = nn.Linear(2 * d_model, spec.num_classes)

    def forward(self, tokens):
        x = self.embed(tokens)
        _, h_n = self.gru(x)
        return self.head(torch.cat([h_n[0], h_n[1]], dim=-1))


# ============================================================
#  Param utilities
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_d_gru(spec: CyclicTaskSpec, d_complex: int):
    target = count_params(GatedComplexRNN_n(spec, d_model=d_complex))
    best_d, best_diff = None, float("inf")
    for d in range(4, d_complex * 2):
        n = count_params(GRUBaseline_n(spec, d_model=d))
        if abs(n - target) < best_diff:
            best_diff, best_d = abs(n - target), d
    return best_d


if __name__ == "__main__":
    print(f"{'n':>3s}  {'PhaseSum':>10s}  {'RealAdd':>10s}  "
          f"{'CplxRNN':>10s}  {'GRU(matched)':>14s}")
    for n in (2, 3, 5, 7, 11, 13):
        spec = CyclicTaskSpec(n)
        ps  = count_params(PhaseSumNet_n(spec, d_model=16))
        ra  = count_params(RealAddNet_n  (spec, d_model=16))
        cx  = count_params(GatedComplexRNN_n(spec, d_model=24))
        d_g = matched_d_gru(spec, 24)
        gr  = count_params(GRUBaseline_n(spec, d_model=d_g))
        print(f"{n:>3d}  {ps:>10,}  {ra:>10,}  {cx:>10,}  "
              f"{gr:>10,} (d={d_g})")
