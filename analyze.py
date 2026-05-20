"""
Mechanistic probe: after training, what did the complex model actually
*learn* about the NOT token?

The interference hypothesis predicts that processing a NOT token should
correspond — somewhere in the network — to a phase rotation near π on the
running representation. If the complex model wins but does NOT exhibit
this structure, the win is not really about phase.

We measure, for each pair of sentences that differ by exactly one extra
NOT token, the angle of the complex CLS state at the readout (the same
output the classifier reads). If the hypothesis is right, that angle
should be close to π (and tightly distributed).
"""

import math
import random
import torch
from data import generate_sentence, CLS_ID, T_ID, F_ID, NOT_ID, PAD_ID


@torch.no_grad()
def complex_cls_state(model, tokens, device):
    """Run a complex transformer forward and return (Re, Im) of the
       output complex scalar at CLS — the same number whose real part the
       classifier uses as logit."""
    model.eval()
    B, L = tokens.shape
    pe_r = model.pe_r[:L].unsqueeze(0)
    pe_i = model.pe_i[:L].unsqueeze(0)
    xr = model.embed_r(tokens) + pe_r
    xi = model.embed_i(tokens) + pe_i
    pad_mask = (tokens == PAD_ID)
    for block in model.blocks:
        xr, xi = block(xr, xi, key_padding_mask=pad_mask)
    xr, xi = model.ln_f(xr, xi)
    cr, ci = xr[:, 0], xi[:, 0]
    or_, oi_ = model.head(cr, ci)
    return or_.squeeze(-1), oi_.squeeze(-1)


def probe_not_phase(model, device, n_pairs=2000, seed=0):
    """For many random sentences, pair (k NOTs) with (k+1 NOTs) and measure
       the angular difference of the readout state.

       Returns:
         deltas_rad : tensor of angular differences in radians, wrapped
                      into [-π, π]
         summary    : dict with mean, median, std (all in radians) and the
                      fraction of pairs within π/8 of π.
    """
    rng = random.Random(seed)
    # Build paired sentences.
    pairs_a, pairs_b = [], []
    for _ in range(n_pairs):
        k       = rng.randint(0, 3)
        nf      = rng.randint(0, 6)
        atom_v  = rng.choice([+1, -1])
        toks_a, _ = generate_sentence(k,     nf, atom_v, rng)
        toks_b, _ = generate_sentence(k + 1, nf, atom_v, rng)
        pairs_a.append(toks_a)
        pairs_b.append(toks_b)

    max_len = max(max(len(t) for t in pairs_a),
                  max(len(t) for t in pairs_b))
    def pack(seqs):
        out = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out.to(device)

    A = pack(pairs_a)
    B = pack(pairs_b)
    Ar, Ai = complex_cls_state(model, A, device)
    Br, Bi = complex_cls_state(model, B, device)

    angle_a = torch.atan2(Ai, Ar)
    angle_b = torch.atan2(Bi, Br)
    delta   = (angle_b - angle_a + math.pi) % (2 * math.pi) - math.pi

    summary = {
        "mean_abs_rad":   delta.abs().mean().item(),
        "median_abs_rad": delta.abs().median().item(),
        "std_rad":        delta.std().item(),
        "frac_near_pi":   ((math.pi - delta.abs()).abs() < math.pi / 8)
                          .float().mean().item(),
    }
    return delta.cpu(), summary
