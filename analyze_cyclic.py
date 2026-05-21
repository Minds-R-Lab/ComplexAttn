"""
Mechanistic probes for the Z/n cyclic-rotation task.

For each architecture we measure how cleanly the group structure is
internalized — specifically, does applying n consecutive TWIRLs leave
predictions invariant? That is the defining property of Z/n.
"""

import math
import torch
from data_cyclic import CyclicTaskSpec, CLS_ID, PAD_ID, NUM_FILLERS


def _wrap(angles):
    return (angles + math.pi) % (2 * math.pi) - math.pi


def probe_phase_sum_n(model, spec: CyclicTaskSpec):
    """For PhaseSumNet: does n·θ(TWIRL) ≡ 0 (mod 2π)?

    Equivalently, cos(n·θ(TWIRL)) ≈ +1 in every dimension. Reports the
    mean and the fraction of dimensions that satisfy it cleanly.
    """
    with torch.no_grad():
        theta_twirl = _wrap(model.phase.weight[spec.twirl_id])
        cycle_cos = torch.cos(spec.n * theta_twirl)
        return {
            "cycle_cos_mean":      cycle_cos.mean().item(),
            "frac_cycle_near_one": (cycle_cos > 0.9).float().mean().item(),
            "abs_theta_twirl_mean": theta_twirl.abs().mean().item(),
        }


def probe_real_add_n(model, spec: CyclicTaskSpec):
    """Slope spread of the linear readout in the TWIRL direction.

    For RealAddNet the logit_c(k) is linear in k with slope
    s_c = W_c · e(TWIRL). If max(s) - min(s) > 0, one class dominates
    at large k and the model has zero hope of correctly predicting any
    *other* class at large k.
    """
    with torch.no_grad():
        slopes = model.head.weight @ model.embed.weight[spec.twirl_id]
        return {
            "slope_per_class":    slopes.cpu().tolist(),
            "slope_spread":       (slopes.max() - slopes.min()).item(),
            "dominant_at_large_k": int(slopes.argmax().item()),
        }


def probe_closure_under_n_twirls(model, spec: CyclicTaskSpec, device,
                                   n_samples=512, base_len=10):
    """Behavioral probe: does adding n TWIRLs to a sentence leave the
       prediction invariant?

    This is the group's defining property. Build pairs of inputs
    differing by exactly n TWIRLs; report fraction with matching argmax.
    Useful for ALL architectures (the model can be a black box).

    Caveat: this metric is only meaningful when the model has actually
    learned the task. A model that outputs constant (or near-constant)
    predictions trivially scores high on closure because both
    predictions of a pair fall on the same deterministic-looking blob.
    Always read this probe alongside ID accuracy.
    """
    g = torch.Generator().manual_seed(123)
    pairs_base, pairs_aug = [], []
    for _ in range(n_samples):
        c = int(torch.randint(0, spec.n, (1,), generator=g).item())
        atom = spec.atom_ids[c]
        seq = [CLS_ID, atom]
        for _ in range(base_len - 2):
            seq.append(int(spec.filler_start +
                            torch.randint(0, NUM_FILLERS, (1,),
                                          generator=g).item()))
        pairs_base.append(seq)
        pairs_aug .append(seq + [spec.twirl_id] * spec.n)

    L = max(len(s) for s in pairs_aug)
    def pad(seqs, L):
        out = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out
    b = pad(pairs_base, L).to(device)
    a = pad(pairs_aug,  L).to(device)
    model.eval()
    with torch.no_grad():
        bp = model(b).argmax(-1)
        ap = model(a).argmax(-1)
    return {"frac_invariant_under_n_twirls": (bp == ap).float().mean().item()}
