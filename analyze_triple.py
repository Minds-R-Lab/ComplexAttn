"""
Mechanistic probes for the four mod-3 architectures.

For PhaseSumNet3 we can read the learned phase per token directly. The
predicted optimum is θ(TWIRL) ≈ 2π/3 (or any equivalent rotation that,
composed with the readout, gives the right class labels). We measure
how close the *sum* of three TWIRL phases is to 0 (mod 2π), which is
the invariant that has to hold for OOD generalization.

For RealAddNet we don't need a mechanistic probe — the failure mode
shows up directly as collapsing argmax at large k. We expose a
"linearity index": cos-similarity between e(TWIRL) and the difference
of class weight vectors in the readout, which reveals whether the
model learned a single "increment direction" (always class-c+1) and
extrapolated linearly off it.

For GatedComplexRNN3 and GRUBaseline3 the state has too many moving
parts for a direct probe; the per-depth accuracy is the probe.
"""

import math
import torch
from data_triple import TWIRL_ID, A0_ID, A1_ID, A2_ID, FILLER_START, NUM_FILLERS


def _wrap(angles):
    return (angles + math.pi) % (2 * math.pi) - math.pi


def probe_phase_sum_net3(model):
    """Read learned phases and the 3-TWIRL-cycle invariant.

    If TWIRL really implements a 2π/3 rotation, three TWIRLs should
    sum to a multiple of 2π in every phase dimension — i.e.
    cos(3·θ(TWIRL)) ≈ 1 per dimension.
    """
    with torch.no_grad():
        theta_twirl = _wrap(model.phase.weight[TWIRL_ID])         # [d]
        theta_a0    = _wrap(model.phase.weight[A0_ID])
        theta_a1    = _wrap(model.phase.weight[A1_ID])
        theta_a2    = _wrap(model.phase.weight[A2_ID])
        theta_fill  = _wrap(model.phase.weight[FILLER_START:
                                                FILLER_START + NUM_FILLERS])
        cycle_residual = torch.cos(3 * theta_twirl)               # target: +1
        # Differences between atom phases — should match TWIRL.
        diff_a1_a0 = _wrap(theta_a1 - theta_a0)
        diff_a2_a0 = _wrap(theta_a2 - theta_a0)
        return {
            "cycle3_cos_mean":          cycle_residual.mean().item(),
            "frac_cycle3_near_one":     (cycle_residual > 0.9).float().mean().item(),
            "abs_theta_twirl_mean":     theta_twirl.abs().mean().item(),
            "cos_filler_mean":          torch.cos(theta_fill).mean().item(),
            # The atom class is "an offset that composes with twirl". We
            # measure whether atom-class spacing matches twirl rotation.
            "match_a1_minus_a0_to_twirl": torch.cos(diff_a1_a0 - theta_twirl).mean().item(),
            "match_a2_minus_a0_to_2tw":   torch.cos(diff_a2_a0 - 2 * theta_twirl).mean().item(),
            "phase_twirl_per_dim":       theta_twirl.cpu().tolist(),
        }


def probe_real_add_net(model):
    """The diagnostic for the negative control.

    For each class c, the readout row w_c · e(TWIRL) is the slope of
    logit_c with respect to k. If the model has learned a 'monotonic'
    direction in embedding space, these three slopes will differ —
    meaning one class will dominate at large k. We report the slopes
    and their spread, which directly predicts where OOD accuracy
    collapses.
    """
    with torch.no_grad():
        e_twirl = model.embed.weight[TWIRL_ID]      # [2d]
        e_a0    = model.embed.weight[A0_ID]
        e_a1    = model.embed.weight[A1_ID]
        e_a2    = model.embed.weight[A2_ID]
        W = model.head.weight                       # [3, 2d]
        slopes = (W @ e_twirl)                      # [3]   per-class slope wrt k
        atom_offsets = torch.stack([W @ e_a0, W @ e_a1, W @ e_a2])  # [3, 3]
        # Class that will dominate at large k.
        dominant = int(slopes.argmax().item())
        return {
            "slope_per_class":      slopes.cpu().tolist(),
            "slope_spread":         (slopes.max() - slopes.min()).item(),
            "dominant_class_at_large_k": dominant,
            "atom_offset_diag_mean":     atom_offsets.diag().mean().item(),
        }


def probe_complex_rnn3_behavioral(model, device):
    """Behavioral probe: does adding 3 TWIRLs to a sentence return the
    predicted class to its starting class?

    This is the canonical 'closure under group operation' check. Build
    pairs of inputs that differ by exactly 3 TWIRLs; if the model has
    learned the mod-3 group, predictions should agree.
    """
    from data_triple import CLS_ID, PAD_ID
    n = 256
    g = torch.Generator().manual_seed(123)
    base_len = 10
    atoms = [A0_ID, A1_ID, A2_ID]
    base_tokens, augmented_tokens = [], []
    for _ in range(n):
        atom = atoms[torch.randint(0, 3, (1,), generator=g).item()]
        seq = [CLS_ID, atom]
        for _ in range(base_len - 2):
            seq.append(int(FILLER_START + torch.randint(0, NUM_FILLERS, (1,),
                                                         generator=g).item()))
        base_tokens.append(seq)
        augmented_tokens.append(seq + [TWIRL_ID, TWIRL_ID, TWIRL_ID])

    L = max(len(s) for s in augmented_tokens)
    def pad_to(seqs, L):
        out = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out
    base_t = pad_to(base_tokens, L).to(device)
    aug_t  = pad_to(augmented_tokens, L).to(device)
    model.eval()
    with torch.no_grad():
        b_pred = model(base_t).argmax(-1)
        a_pred = model(aug_t).argmax(-1)
    return {"frac_invariant_under_3_twirls": (b_pred == a_pred).float().mean().item()}
