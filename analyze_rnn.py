"""
Mechanistic probe for the Experiment 2 architectures.

Unlike the transformer probe (which had to do behavioral testing), these
architectures expose the per-token operation directly. We can read off
exactly what the model learned for NOT, T, F, and fillers.
"""

import math
import torch
from data import NOT_ID, T_ID, F_ID, FILLER_START, NUM_FILLERS


def _wrap(angles):
    """Wrap angles to [-π, π]."""
    return (angles + math.pi) % (2 * math.pi) - math.pi


def probe_phase_sum_net(model):
    """For PhaseSumNet: read the learned phase per token directly.

       Returns a dict with the mean phase per token-class, plus the
       fraction of dimensions whose NOT phase is within π/8 of ±π
       (the predicted optimum).
    """
    with torch.no_grad():
        p_not   = _wrap(model.phase.weight[NOT_ID])               # [d]
        p_t     = _wrap(model.phase.weight[T_ID])
        p_f     = _wrap(model.phase.weight[F_ID])
        p_fill  = _wrap(model.phase.weight[FILLER_START:
                                           FILLER_START + NUM_FILLERS])  # [Nf, d]

        # cos of the phase tells us the "sign factor" each dimension contributes.
        # The prediction: cos(NOT) ≈ −1, cos(T) ≈ +1, cos(F) ≈ −1, cos(filler) ≈ +1.
        return {
            "phase_not_mean":      p_not.mean().item(),
            "phase_not_abs_mean":  p_not.abs().mean().item(),
            "cos_not_mean":        torch.cos(p_not).mean().item(),
            "cos_t_mean":          torch.cos(p_t).mean().item(),
            "cos_f_mean":          torch.cos(p_f).mean().item(),
            "cos_filler_mean":     torch.cos(p_fill).mean().item(),
            "frac_not_near_pi":    ((p_not.abs() - math.pi).abs()
                                     < math.pi / 8).float().mean().item(),
            "phase_not_per_dim":   p_not.cpu().tolist(),
        }


def probe_gated_complex_rnn(model):
    """For GatedComplexRNN: inspect both cells' phase + gate for NOT.

       This is the cleanest possible mechanistic check: if NOT really acts
       as a π rotation, the phase embedding entry for NOT should literally
       contain values near ±π, with gate near 1.
    """
    with torch.no_grad():
        out = {}
        for direction, cell in (("fwd", model.cell_fwd),
                                 ("bwd", model.cell_bwd)):
            p_not  = _wrap(cell.phase.weight[NOT_ID])
            g_not  = torch.sigmoid(cell.gate.weight[NOT_ID])
            v_r    = cell.value_r.weight[NOT_ID]
            v_i    = cell.value_i.weight[NOT_ID]
            out[f"{direction}_phase_not_abs_mean"] = p_not.abs().mean().item()
            out[f"{direction}_cos_not_mean"]      = torch.cos(p_not).mean().item()
            out[f"{direction}_gate_not_mean"]     = g_not.mean().item()
            out[f"{direction}_value_not_norm"]    = torch.sqrt(
                v_r * v_r + v_i * v_i + 1e-8).mean().item()
            out[f"{direction}_frac_not_near_pi"]  = ((p_not.abs() - math.pi).abs()
                                                     < math.pi / 8).float().mean().item()
        return out


def summarize_probes(probes):
    """Pretty-print the probe results for a list of trained models."""
    lines = []
    for name, probe in probes.items():
        lines.append(f"  [{name}]")
        for k, v in probe.items():
            if isinstance(v, list):
                continue
            lines.append(f"    {k:30s} {v:+.4f}")
    return "\n".join(lines)
