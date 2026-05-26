"""ablations.py -- controlled ablation of SHARD's three design substitutions.

SHARD differs from GRACE on three orthogonal axes:

  Axis            SHARD default        GRACE default
  --------------- -------------------- -----------------------
  routing         cosine + fixed tau   euclidean + per-key eps
  write_mode      additive             substitutive
  value_optim     vstar (CE + reg)     vanilla_ft (CE only)

This module provides AblatedSHARDMethod, which lets each axis be toggled
independently so we can identify which substitution(s) drive SHARD's
Pareto-frontier shift away from GRACE.

The five experiments we care about (relative to SHARD as default):

  - shard            : (cosine,    additive,     vstar)        # default; sanity check
  - ablate_routing   : (euclidean, additive,     vstar)        # only routing flipped
  - ablate_write     : (cosine,    substitutive, vstar)        # only write_mode flipped
  - ablate_optim     : (cosine,    additive,     vanilla_ft)   # only value_optim flipped
  - all_grace        : (euclidean, substitutive, vanilla_ft)   # all three flipped

Comparing the four single-flip rows against `shard` tells us how much
each design choice individually contributes to the Pareto shift.
`all_grace` is a sanity check that should land near our reproduced
GRACE numbers from grace_method.py.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_baselines import Method, METHOD_REGISTRY, DEVICE


# ---------------------------------------------------------------------------
# Wrapper -- routing + write_mode are configurable
# ---------------------------------------------------------------------------

class AblatedSHARDWrapper(nn.Module):
    """Per-layer MLP wrapper carrying a flat key-value codebook plus
    configurable routing and write-mode."""

    def __init__(self, base_mlp: nn.Module,
                 routing: str = "cosine",
                 write_mode: str = "additive",
                 sim_threshold: float = 0.7,
                 eps_init: float = 1.0):
        super().__init__()
        self.base_mlp = base_mlp
        if routing not in ("cosine", "euclidean"):
            raise ValueError(f"routing must be cosine|euclidean, got {routing!r}")
        if write_mode not in ("additive", "substitutive"):
            raise ValueError(f"write_mode must be additive|substitutive, got {write_mode!r}")
        self.routing = routing
        self.write_mode = write_mode
        self.sim_threshold = float(sim_threshold)
        self.eps_init = float(eps_init)
        self.keys: list[torch.Tensor] = []     # (d_in,) each
        self.values: list[torch.Tensor] = []   # (d_out,) each
        #   additive    -> values store delta_v
        #   substitutive-> values store v_orig + delta_v

    @property
    def n_slots(self) -> int:
        return len(self.keys)

    def add_entry(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.keys.append(key.detach())
        self.values.append(value.detach())

    def forward(self, x):  # x: (batch, seq, d_in)
        base_out = self.base_mlp(x)  # (batch, seq, d_out)
        if self.n_slots == 0:
            return base_out

        x_last = x[:, -1, :]                                 # (batch, d_in)
        K = torch.stack(self.keys, dim=0).to(device=x_last.device)

        if self.routing == "cosine":
            x_n = F.normalize(x_last.float(), dim=-1)
            K_n = F.normalize(K.float(), dim=-1)
            sims = x_n @ K_n.t()                              # (batch, n)
            best_sim, best_idx = sims.max(dim=-1)
            hit = (best_sim > self.sim_threshold)
        else:  # euclidean
            x_f = x_last.float()
            K_f = K.float()
            dists = torch.cdist(x_f, K_f)                     # (batch, n)
            best_dist, best_idx = dists.min(dim=-1)
            hit = (best_dist < self.eps_init)

        if not bool(hit.any()):
            return base_out

        V = torch.stack(self.values, dim=0).to(
            dtype=base_out.dtype, device=base_out.device)
        V_chosen = V[best_idx]                                # (batch, d_out)

        if self.write_mode == "additive":
            substituted = base_out[:, -1, :] + V_chosen
        else:
            substituted = V_chosen

        new_last = torch.where(hit.unsqueeze(-1), substituted, base_out[:, -1, :])
        out = base_out.clone()
        out[:, -1, :] = new_last
        return out


# ---------------------------------------------------------------------------
# AblatedSHARDMethod -- routing, write_mode, value_optim all toggleable
# ---------------------------------------------------------------------------

class AblatedSHARDMethod(Method):
    name = "ablated_shard"

    def __init__(self,
                 layer_idx: int = 17,
                 routing: str = "cosine",
                 write_mode: str = "additive",
                 value_optim: str = "vstar",
                 v_steps: int = 200,
                 v_lr: float = 1.0,
                 v_weight_decay: float = 0.0,
                 v_norm_constraint: float = 20.0,
                 sim_threshold: float = 0.7,
                 eps_init: float = 1.0):
        if value_optim not in ("vstar", "vanilla_ft"):
            raise ValueError(f"value_optim must be vstar|vanilla_ft, got {value_optim!r}")
        self.layer_idx = int(layer_idx)
        self.routing = routing
        self.write_mode = write_mode
        self.value_optim = value_optim
        self.v_steps = int(v_steps)
        self.v_lr = float(v_lr)
        self.v_weight_decay = float(v_weight_decay)
        self.v_norm_constraint = float(v_norm_constraint)
        self.sim_threshold = float(sim_threshold)
        self.eps_init = float(eps_init)

    # ------------------------------------------------------------------
    # Family-aware MLP access (same pattern as grace_method.py)
    # ------------------------------------------------------------------
    def _get_mlp_and_block(self, model):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            block = model.transformer.h[self.layer_idx]
            return block.mlp, block, "gpt2"
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            block = model.model.layers[self.layer_idx]
            return block.mlp, block, "swiglu"
        raise RuntimeError("Unrecognized model layout for AblatedSHARD")

    def setup(self, model, tokenizer, kb):
        super().setup(model, tokenizer, kb)
        for p in model.parameters():
            p.requires_grad = False
        mlp, block, family = self._get_mlp_and_block(model)
        wrapped = AblatedSHARDWrapper(
            mlp, routing=self.routing, write_mode=self.write_mode,
            sim_threshold=self.sim_threshold, eps_init=self.eps_init,
        ).to(DEVICE)
        block.mlp = wrapped
        self.wrapped_mlp = wrapped
        self.family = family
        print(f"[baseline] AblatedSHARD installed at layer {self.layer_idx} "
              f"({family} family); routing={self.routing}, "
              f"write_mode={self.write_mode}, value_optim={self.value_optim}, "
              f"v_steps={self.v_steps}, v_lr={self.v_lr}, "
              f"v_wd={self.v_weight_decay}, v_norm_cap={self.v_norm_constraint}, "
              f"tau={self.sim_threshold}, eps={self.eps_init}")

    # ------------------------------------------------------------------
    # Rewrite-prompt builder (monkey-patched by run_ablations for CF/zsRE)
    # ------------------------------------------------------------------
    def _build_rewrite(self, triple: Any) -> tuple[str, str]:
        from kb_data import RELATIONS
        rel = RELATIONS[triple.relation]
        q_tmpl, _ = rel.query_templates[0]
        return q_tmpl.format(s=triple.subject), " " + triple.obj

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------
    def insert(self, triple: Any) -> None:
        prompt, target = self._build_rewrite(triple)

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        target_ids = self.tokenizer.encode(
            target, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        last_prompt_pos = prompt_ids.shape[1] - 1

        # ---- Capture k* (MLP input) and v_orig (MLP output) at last prompt pos
        captured: dict[str, torch.Tensor] = {}

        def hook_kv(module, inputs, output):
            captured["k"] = inputs[0][0, last_prompt_pos].detach().clone()
            captured["v_orig"] = output[0, last_prompt_pos].detach().clone()

        h_kv = self.wrapped_mlp.register_forward_hook(hook_kv)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                _ = self.model(prompt_ids)
        finally:
            h_kv.remove()
            if was_training:
                self.model.train()
        k_star = captured["k"]
        v_orig = captured["v_orig"]

        # ---- Optimize delta_v (variable being optimized) ----
        delta_v = torch.zeros_like(v_orig, requires_grad=True)
        opt = torch.optim.Adam([delta_v], lr=self.v_lr)
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100  # CE only on target tokens

        def inject_hook_additive(module, inputs, output):
            out = output.clone()
            out[0, last_prompt_pos] = out[0, last_prompt_pos] + delta_v
            return out

        def inject_hook_substitutive(module, inputs, output):
            out = output.clone()
            out[0, last_prompt_pos] = v_orig + delta_v
            return out

        if self.write_mode == "additive":
            inject_hook = inject_hook_additive
        else:
            inject_hook = inject_hook_substitutive

        h_inject = self.wrapped_mlp.register_forward_hook(inject_hook)
        try:
            for step in range(self.v_steps):
                opt.zero_grad()
                out = self.model(input_ids=full_ids, labels=labels)
                ce = out.loss
                if self.value_optim == "vstar":
                    # ROME-style regularizer on delta_v, plus norm cap below
                    reg = self.v_weight_decay * (delta_v.norm() ** 2) / (
                        v_orig.norm() ** 2 + 1e-8)
                    loss = ce + reg
                else:  # vanilla_ft
                    loss = ce
                loss.backward()
                opt.step()
                if self.value_optim == "vstar":
                    with torch.no_grad():
                        max_norm = self.v_norm_constraint * v_orig.norm().item()
                        cur_norm = delta_v.norm().item()
                        if cur_norm > max_norm and cur_norm > 0:
                            delta_v.mul_(max_norm / cur_norm)
        finally:
            h_inject.remove()

        # ---- Store the entry, using the right representation per write_mode ----
        if self.write_mode == "additive":
            stored_value = delta_v.detach()
        else:
            stored_value = (v_orig + delta_v).detach()
        self.wrapped_mlp.add_entry(k_star, stored_value)


# ---------------------------------------------------------------------------
# Convenience presets for the five canonical experiments
# ---------------------------------------------------------------------------

ABLATION_PRESETS: dict[str, dict[str, str]] = {
    "shard":          {"routing": "cosine",    "write_mode": "additive",     "value_optim": "vstar"},
    "ablate_routing": {"routing": "euclidean", "write_mode": "additive",     "value_optim": "vstar"},
    "ablate_write":   {"routing": "cosine",    "write_mode": "substitutive", "value_optim": "vstar"},
    "ablate_optim":   {"routing": "cosine",    "write_mode": "additive",     "value_optim": "vanilla_ft"},
    "all_grace":      {"routing": "euclidean", "write_mode": "substitutive", "value_optim": "vanilla_ft"},
}


METHOD_REGISTRY["ablated_shard"] = AblatedSHARDMethod
