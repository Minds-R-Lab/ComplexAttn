"""
Architectures for the GRU capacity sweep (Experiment 5).

We want to ask: does the GRU's Z/n>=5 collapse from Experiment 4 survive
compute scaling? To answer cleanly we need:

  - GRU at varying d_model and n_layers (capacity sweep)
  - LSTM as a control (same gating family, different gating equations).
    If LSTM also collapses at n>=5, the mechanism is "real-valued gating
    can't make tanh limit cycles of arbitrary period". If LSTM solves
    it, the mechanism is GRU-specific.
  - PhaseSumNet at matched parameter count to serve as the
    1.000-accuracy reference line at each capacity rung.

All four take a CyclicTaskSpec and forward to n_classes logits, same
contract as models_cyclic.py.
"""

import math
import torch
import torch.nn as nn
from data_cyclic import CyclicTaskSpec, PAD_ID


class GRUMultilayer(nn.Module):
    """Bidirectional GRU, optionally stacked. Final hidden state from
       both directions of the top layer concatenated and read out."""
    def __init__(self, spec: CyclicTaskSpec, d_model: int, n_layers: int = 1):
        super().__init__()
        self.spec     = spec
        self.d_model  = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(spec.vocab_size, d_model, padding_idx=PAD_ID)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers,
                          bidirectional=True, batch_first=True,
                          dropout=0.0)
        self.head = nn.Linear(2 * d_model, spec.num_classes)

    def forward(self, tokens):
        x = self.embed(tokens)
        _, h_n = self.gru(x)
        # h_n shape: (n_layers * 2, B, d). Take the top layer (last 2).
        top_fwd, top_bwd = h_n[-2], h_n[-1]
        return self.head(torch.cat([top_fwd, top_bwd], dim=-1))


class LSTMMultilayer(nn.Module):
    """Bidirectional LSTM, stacked. Same readout shape as the GRU."""
    def __init__(self, spec: CyclicTaskSpec, d_model: int, n_layers: int = 1):
        super().__init__()
        self.spec     = spec
        self.d_model  = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(spec.vocab_size, d_model, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=0.0)
        self.head = nn.Linear(2 * d_model, spec.num_classes)

    def forward(self, tokens):
        x = self.embed(tokens)
        _, (h_n, _) = self.lstm(x)
        top_fwd, top_bwd = h_n[-2], h_n[-1]
        return self.head(torch.cat([top_fwd, top_bwd], dim=-1))


class PhaseSumRef(nn.Module):
    """PhaseSumNet at the scale chosen for this rung. Serves as the
       reference architecture that achieves 1.000 OOD at every n we
       tested in Exp 4."""
    def __init__(self, spec: CyclicTaskSpec, d_model: int):
        super().__init__()
        self.spec = spec
        self.phase = nn.Embedding(spec.vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        with torch.no_grad():
            self.phase.weight[PAD_ID].zero_()
        self.head = nn.Linear(2 * d_model, spec.num_classes)

    def forward(self, tokens):
        total = self.phase(tokens).sum(dim=1)
        feat  = torch.cat([torch.cos(total), torch.sin(total)], dim=-1)
        return self.head(feat)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    spec = CyclicTaskSpec(5)
    for d, L in [(16, 1), (32, 1), (32, 2), (64, 1), (64, 2),
                  (128, 1), (128, 2), (256, 1), (256, 2)]:
        gru = GRUMultilayer (spec, d_model=d, n_layers=L)
        lst = LSTMMultilayer(spec, d_model=d, n_layers=L)
        ps  = PhaseSumRef   (spec, d_model=d)
        print(f"d={d:>3}  L={L}   GRU={count_params(gru):>8,}   "
              f"LSTM={count_params(lst):>8,}   "
              f"PhaseSum(d={d})={count_params(ps):>6,}")
