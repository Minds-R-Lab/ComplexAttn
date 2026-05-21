"""
Experiment 6 (factorial ablation): which design axis is actually
responsible for the PhaseSumNet vs. GRU gap?

The previous experiments (1-5) compared architectures that differ on
MANY axes simultaneously. PhaseSumNet differs from GRU in at least
four ways: (a) complex vs real algebra, (b) additive vs multiplicative
composition, (c) bounded vs tanh-bounded state, (d) periodic vs linear
readout. When PhaseSumNet wins, we don't know which property is
responsible.

This experiment isolates two axes that we now believe are doing the
work: COMPOSITION (additive sum vs multiplicative phase product) and
READOUT (linear vs periodic). All other variables held constant.

  +-----------------+----------------+-----------------+
  |                 |  LINEAR readout|  PERIODIC readout|
  +-----------------+----------------+-----------------+
  | ADDITIVE comp   |  RealAddLin    |  RealAddPer     |
  |                 | (= RealAddNet) |    [NEW]         |
  +-----------------+----------------+-----------------+
  | MULTIPLICATIVE  |  ComplexMulLin |  ComplexMulPer  |
  |   composition   |    [NEW]       | (= PhaseSumNet) |
  +-----------------+----------------+-----------------+

The two NEW architectures are the critical ones:

  RealAddPer:    sum embeddings (additive, no group structure used),
                 then pass through cos/sin before readout (periodic).
                 Tests whether PERIODICITY ALONE rescues an additive
                 architecture.

  ComplexMulLin: sum phases (multiplicative phase composition),
                 then read out from (cos(Theta), sin(Theta)) but only
                 linearly project. Wait -- that IS periodic. The
                 right "linear readout on multiplicative" is to expose
                 Theta directly (not its cos/sin) and readout linearly.
                 That's what we do. Tests whether MULTIPLICATIVE
                 COMPOSITION ALONE (with non-periodic readout) is
                 enough.

If RealAddPer succeeds ~ PhaseSumNet: complex/multiplicative wasn't the
point. Periodicity at readout was.

If ComplexMulLin succeeds ~ PhaseSumNet: periodic readout wasn't the
point. Multiplicative composition was. (Skeptic: but our exposing
Theta linearly is hard for the readout because Theta is unbounded and
not periodic.)

If only PhaseSumNet succeeds: both axes are needed. The paper's
current framing is correct.

Plus two GRU variants for context:
  GRULin:  standard GRU + linear readout (= Exp 5 baseline)
  GRUPer:  GRU + periodic readout. If this rescues GRU on Z/5, the
           failure was the readout, not the recurrence.

Six architectures total. All trained at parameter parity (the
multiplicative architectures use d=16; GRU is matched).
"""

import math
import torch
import torch.nn as nn
from data_cyclic import CyclicTaskSpec, PAD_ID


# ============================================================
# Common building block: per-token embedding to d-dim real vector
# ============================================================
def _make_real_emb(spec: CyclicTaskSpec, d: int, init_uniform_circle: bool = False):
    """A real-valued embedding table. If init_uniform_circle, draw from
    Uniform([-pi, pi]) -- needed for phase-style models so that the
    embedding's cos/sin don't all start near 1.
    """
    emb = nn.Embedding(spec.vocab_size, d, padding_idx=PAD_ID)
    if init_uniform_circle:
        nn.init.uniform_(emb.weight, -math.pi, math.pi)
    else:
        nn.init.normal_(emb.weight, mean=0.0, std=0.3)
    with torch.no_grad():
        emb.weight[PAD_ID].zero_()
    return emb


# ============================================================
# THE FOUR FACTORIAL CELLS
# ============================================================

class RealAddLin(nn.Module):
    """ADDITIVE + LINEAR. The known-failing baseline (RealAddNet).

    h = sum_t e(x_t)        in R^d
    logits = W h + b
    """
    def __init__(self, spec: CyclicTaskSpec, d: int = 32):
        super().__init__()
        self.spec = spec
        self.embed = _make_real_emb(spec, d, init_uniform_circle=False)
        self.head  = nn.Linear(d, spec.num_classes)

    def forward(self, tokens):
        return self.head(self.embed(tokens).sum(dim=1))


class RealAddPer(nn.Module):
    """ADDITIVE + PERIODIC. The critical new architecture.

    h = sum_t e(x_t)        in R^d (unbounded)
    f = [cos(h), sin(h)]    in R^{2d}      <-- periodic transform
    logits = W f + b

    If this generalizes Z/n the way PhaseSumNet does, then the
    'periodic readout' axis is sufficient -- complex/multiplicative
    composition was never needed.

    Key insight: even though composition is just addition, projecting
    through cos/sin makes the EFFECTIVE composition multiplicative on
    the unit circle. e^{ia} * e^{ib} = e^{i(a+b)}. So this is
    mathematically equivalent to PhaseSumNet -- we expect it to win.
    The point of including it is to make the equivalence visible:
    PhaseSumNet's "complex" framing was the periodic readout the
    whole time.
    """
    def __init__(self, spec: CyclicTaskSpec, d: int = 16):
        super().__init__()
        self.spec = spec
        # Init uniform[-pi, pi] so initial cos/sin spread across [-1, 1].
        # (If we init small, all sentences map to similar (cos, sin) and
        # gradients are weak. Same lesson as PhaseSumNet init.)
        self.embed = _make_real_emb(spec, d, init_uniform_circle=True)
        self.head  = nn.Linear(2 * d, spec.num_classes)

    def forward(self, tokens):
        h = self.embed(tokens).sum(dim=1)           # [B, d], unbounded
        f = torch.cat([torch.cos(h), torch.sin(h)], dim=-1)
        return self.head(f)


class ComplexMulLin(nn.Module):
    """MULTIPLICATIVE + LINEAR. The other diagonal.

    Multiplicative composition of unit-modulus complex factors
    e^{i theta(x_t)} reduces to summing the phases theta(x_t).
    For 'linear readout', expose the SUMMED PHASES directly to a
    linear readout (NOT cos/sin of them). This tests whether
    multiplicative composition without periodic readout is enough.

    h = sum_t theta(x_t)    in R^d (unbounded, but representing a
                                    phase angle mod 2pi)
    logits = W h + b

    Mathematically this is identical to RealAddLin -- the act of
    'summing phases' is just summing real numbers, indistinguishable
    from RealAddLin's sum. We include it explicitly to make the
    point: if you remove the periodic readout from PhaseSumNet, the
    'complex / multiplicative' framing buys you NOTHING -- you're
    back to additive linear.

    We initialize uniform[-pi, pi] (matching PhaseSumNet's setup) to
    keep the comparison fair.
    """
    def __init__(self, spec: CyclicTaskSpec, d: int = 32):
        super().__init__()
        self.spec = spec
        self.embed = _make_real_emb(spec, d, init_uniform_circle=True)
        self.head  = nn.Linear(d, spec.num_classes)

    def forward(self, tokens):
        return self.head(self.embed(tokens).sum(dim=1))


class ComplexMulPer(nn.Module):
    """MULTIPLICATIVE + PERIODIC. This IS PhaseSumNet.

    Included here for completeness of the 2x2 factorial.
    """
    def __init__(self, spec: CyclicTaskSpec, d: int = 16):
        super().__init__()
        self.spec  = spec
        self.phase = _make_real_emb(spec, d, init_uniform_circle=True)
        self.head  = nn.Linear(2 * d, spec.num_classes)

    def forward(self, tokens):
        total = self.phase(tokens).sum(dim=1)
        feat  = torch.cat([torch.cos(total), torch.sin(total)], dim=-1)
        return self.head(feat)


# ============================================================
# Two GRU variants -- linear vs periodic readout
# ============================================================

class GRULin(nn.Module):
    """Standard bidirectional GRU + LINEAR readout. Identical to Exp 5
    GRUMultilayer at n_layers=1. Included here for one-stop comparison."""
    def __init__(self, spec: CyclicTaskSpec, d: int = 32):
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab_size, d, padding_idx=PAD_ID)
        self.gru = nn.GRU(d, d, num_layers=1, bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * d, spec.num_classes)

    def forward(self, tokens):
        _, h_n = self.gru(self.embed(tokens))
        return self.head(torch.cat([h_n[0], h_n[1]], dim=-1))


class GRUPer(nn.Module):
    """GRU + PERIODIC readout. The diagnostic GRU variant.

    Take the GRU's bidirectional final state (which we showed in Exp 5
    saturates against tanh), pass it through cos/sin BEFORE the linear
    head. If this rescues GRU on Z/5, the failure was the readout's
    linear amplification of small state shifts, not the recurrence.

    h    = concat(h_fwd, h_bwd) in R^{2d}, tanh-bounded
    f    = [cos(h), sin(h)]      in R^{4d}
    logits = W f + b

    Note: GRU's state lives in [-1, 1]^{2d} due to tanh, so cos and
    sin of it lie in [cos(1), 1] x [-sin(1), sin(1)] -- a small arc
    of the unit circle. This restricts the dynamic range. We test
    anyway to see if even this limited periodic projection helps.
    """
    def __init__(self, spec: CyclicTaskSpec, d: int = 32):
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab_size, d, padding_idx=PAD_ID)
        self.gru = nn.GRU(d, d, num_layers=1, bidirectional=True, batch_first=True)
        # Scale state by pi so cos/sin of [-1,1]*pi covers full circle
        self.scale = nn.Parameter(torch.tensor(math.pi))
        self.head = nn.Linear(4 * d, spec.num_classes)

    def forward(self, tokens):
        _, h_n = self.gru(self.embed(tokens))
        h = torch.cat([h_n[0], h_n[1]], dim=-1) * self.scale   # [B, 2d]
        f = torch.cat([torch.cos(h), torch.sin(h)], dim=-1)
        return self.head(f)


# ============================================================
# Param matching + registry
# ============================================================

ARCH_FACTORY = {
    "RealAddLin":    RealAddLin,
    "RealAddPer":    RealAddPer,
    "ComplexMulLin": ComplexMulLin,
    "ComplexMulPer": ComplexMulPer,
    "GRULin":        GRULin,
    "GRUPer":        GRUPer,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_arch(name: str, spec: CyclicTaskSpec, d: int):
    """Construct an architecture by name. Adjusts d for each so the
    final feature width fed to the head is comparable."""
    # For RealAddPer and ComplexMulPer the head sees 2d features.
    # For RealAddLin and ComplexMulLin the head sees d features.
    # For GRULin the head sees 2d (concat of two directions); GRUPer 4d.
    # To make params comparable, double d for the *Lin variants.
    if name in ("RealAddLin", "ComplexMulLin"):
        return ARCH_FACTORY[name](spec, d=2 * d)
    if name in ("GRULin", "GRUPer"):
        # GRU has its own parameter scaling -- pass d directly.
        return ARCH_FACTORY[name](spec, d=d)
    return ARCH_FACTORY[name](spec, d=d)


if __name__ == "__main__":
    print(f"{'arch':<16} {'params (n=2)':>12} {'params (n=5)':>12} {'params (n=11)':>14}")
    print("-" * 56)
    for name in ARCH_FACTORY:
        row = f"{name:<16}"
        for n in (2, 5, 11):
            spec = CyclicTaskSpec(n)
            m = make_arch(name, spec, d=16)
            row += f"  {count_params(m):>10,}"
        print(row)

    # Sanity check forward passes
    spec = CyclicTaskSpec(5)
    tokens = torch.randint(2, spec.vocab_size, (4, 12))
    tokens[:, 0] = 1  # CLS
    for name in ARCH_FACTORY:
        m = make_arch(name, spec, d=16)
        out = m(tokens)
        assert out.shape == (4, 5), f"{name} produced {out.shape}, expected (4, 5)"
    print("\nForward passes OK across all 6 architectures.")
