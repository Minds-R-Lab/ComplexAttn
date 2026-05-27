"""grace_mamba.py -- GRACE baseline adapted to Mamba state-space models.

This is the natural counterpart to SHARDMambaMethod: same wrapper structure
(per-fact key-value codebook attached to a frozen MambaBlock projection,
last-position routing) but with the three GRACE-family design choices
flipped relative to SHARD:

  Axis            SHARD            GRACE
  -------------   --------------   ----------------------------
  routing         cosine + tau     Euclidean + per-key eps
  write_mode      additive         substitutive (replace y, not add)
  value_optim     v* (CE + reg)    vanilla finetune (CE only)

Pairing GRACE-for-Mamba with SHARD-for-Mamba lets us run the
cosine-vs-Euclidean ablation on SSMs that mirrors the transformer
SHARD vs GRACE result (10-150x Gen advantage from cosine alone).

Like SHARDMambaMethod, GRACE-for-Mamba defaults to wrapping out_proj
(W_o), ROMBA's strongest single-edit module. Family-aware via
MambaAdapter.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_adapter import MambaAdapter

try:
    from run_baselines import Method, METHOD_REGISTRY, DEVICE
    _HAVE_REGISTRY = True
except Exception:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class Method:
        name = "base"
        def setup(self, model, tokenizer, kb):
            self.model, self.tokenizer, self.kb = model, tokenizer, kb
        def insert(self, triple): pass
    METHOD_REGISTRY: dict = {}
    _HAVE_REGISTRY = False


class GRACEMambaWrapper(nn.Module):
    """Wraps a Mamba projection (default out_proj). Forward computes the
    base output, then SUBSTITUTES the last-position output with a codebook
    value if the last-position input lies within any stored entry's
    epsilon-ball under Euclidean distance.
    """

    def __init__(self, base_module: nn.Module, eps_init: float = 1.0):
        super().__init__()
        self.base_module = base_module
        self.eps_init = float(eps_init)
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.eps_per_key: list[float] = []
        self.labels: list[str] = []
        # Diagnostic counters
        self._diag_forwards = 0
        self._diag_hits = 0
        self._diag_min_dist_recent: list[float] = []

    @property
    def n_slots(self) -> int:
        return len(self.keys)

    def add_entry(self, key: torch.Tensor, value: torch.Tensor,
                  eps: float, label: str) -> None:
        self.keys.append(key.detach())
        self.values.append(value.detach())
        self.eps_per_key.append(float(eps))
        self.labels.append(str(label))

    def forward(self, x, *args, **kwargs):
        base_out = self.base_module(x, *args, **kwargs)
        if self.n_slots == 0:
            return base_out
        self._diag_forwards += 1

        # GRACE fires at the last sequence position only.
        x_last = x[:, -1, :]
        K = torch.stack(self.keys, dim=0).to(device=x_last.device)

        # Euclidean distance in fp32 for numerical stability (and bf16 cdist
        # has no CUDA kernel).
        x_f = x_last.float()
        K_f = K.float()
        dists = torch.cdist(x_f, K_f)                       # (batch, n_slots)
        min_dist, min_idx = dists.min(dim=-1)
        if len(self._diag_min_dist_recent) < 200:
            self._diag_min_dist_recent.append(float(min_dist.min().item()))

        eps_per = torch.tensor(
            [self.eps_per_key[i] for i in min_idx.tolist()],
            dtype=min_dist.dtype, device=min_dist.device,
        )
        hit = (min_dist < eps_per)
        if not bool(hit.any()):
            return base_out
        self._diag_hits += int(hit.sum().item())

        V = torch.stack(self.values, dim=0).to(
            dtype=base_out.dtype, device=base_out.device)
        V_chosen = V[min_idx]
        # Substitutive write: replace base_out[:, -1, :] with the slot value.
        new_last = torch.where(hit.unsqueeze(-1), V_chosen, base_out[:, -1, :])
        out = base_out.clone()
        out[:, -1, :] = new_last
        return out


class GRACEMambaMethod(Method):
    """GRACE adapted to Mamba: per-fact slot bank with Euclidean
    expanding-eps routing and substitutive writes, wrapping out_proj on
    a frozen Mamba LM.

    Default hyperparameters follow the original GRACE paper
    (Hartvigsen et al., NeurIPS 2023): random or zero init, 100 vanilla
    Adam steps at lr=1.0, eps_init=1.0. The 'init' default is 'zeros'
    rather than 'random' for the same reason as in the transformer
    GRACE port: random init explodes the first-step CE loss on
    modern pretrained checkpoints.
    """

    name = "grace_mamba"

    def __init__(self,
                 layer_idx: int = 32,
                 kind: str = "out_proj",
                 n_steps: int = 100,
                 lr: float = 1.0,
                 eps_init: float = 1.0,
                 init: str = "zeros"):
        if kind not in ("out_proj", "in_proj", "x_proj"):
            raise ValueError(f"kind must be one of out_proj|in_proj|x_proj, got {kind!r}")
        if init not in ("zeros", "random"):
            raise ValueError(f"init must be zeros|random, got {init!r}")
        self.layer_idx = int(layer_idx)
        self.kind = kind
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.eps_init = float(eps_init)
        self.init = init

    def setup(self, model, tokenizer, kb=None):
        super().setup(model, tokenizer, kb)
        for p in model.parameters():
            p.requires_grad = False

        self.adapter = MambaAdapter.from_model(model)
        site = self.adapter.edit_site(self.layer_idx, kind=self.kind)
        self.site = site
        self.wrapper = GRACEMambaWrapper(
            base_module=site.module, eps_init=self.eps_init,
        ).to(DEVICE)
        self.adapter.install_wrapper(self.layer_idx, self.kind, self.wrapper)
        print(f"[grace-mamba] installed at layer {self.layer_idx} "
              f"({self.adapter.family}); site={self.kind}, "
              f"d_key={site.d_key}, d_value={site.d_value}; "
              f"eps_init={self.eps_init}, n_steps={self.n_steps}, "
              f"lr={self.lr}, init={self.init}")

    def _build_rewrite(self, triple) -> tuple[str, str, str]:
        try:
            from kb_data import RELATIONS
            rel = RELATIONS[triple.relation]
            q_tmpl, _ = rel.query_templates[0]
            return q_tmpl.format(s=triple.subject), " " + triple.obj, triple.obj
        except Exception:
            raise NotImplementedError(
                "GRACEMambaMethod._build_rewrite should be monkey-patched "
                "by the runner for CounterFact / zsRE triple types.")

    def _triple_label(self, triple) -> str:
        return getattr(triple, "obj", str(triple))

    def insert(self, triple) -> None:
        prompt, target, label = self._build_rewrite(triple)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        target_ids = self.tokenizer.encode(
            target, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        last_prompt_pos = prompt_ids.shape[1] - 1

        # ---- 1. Capture key (input to out_proj at last prompt position) ----
        captured: dict[str, torch.Tensor] = {}

        def hook_k(module, inputs, output):
            # inputs[0] shape: (batch, seq, d_in); output: (batch, seq, d_out)
            captured["k"] = inputs[0][0, last_prompt_pos].detach().clone()
            captured["v_base"] = output[0, last_prompt_pos].detach().clone()

        h = self.wrapper.base_module.register_forward_hook(hook_k)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                _ = self.model(prompt_ids)
        finally:
            h.remove()
            if was_training:
                self.model.train()

        k_star = captured["k"]
        v_base = captured["v_base"]

        # ---- 2. Optimize the codebook VALUE (substitutive, not delta) ----
        if self.init == "zeros":
            value = torch.zeros_like(v_base, requires_grad=True)
        else:
            scale = v_base.norm().item() / (v_base.numel() ** 0.5 + 1e-8)
            value = (torch.randn_like(v_base) * scale).requires_grad_(True)

        opt = torch.optim.Adam([value], lr=self.lr)
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100  # CE only on target tokens

        def inject_hook(module, inputs, output):
            out = output.clone()
            out[0, last_prompt_pos] = value      # SUBSTITUTIVE, not additive
            return out

        h_inject = self.wrapper.base_module.register_forward_hook(inject_hook)
        try:
            for step in range(self.n_steps):
                opt.zero_grad()
                out = self.model(input_ids=full_ids, labels=labels)
                loss = out.loss
                loss.backward()
                opt.step()
        finally:
            h_inject.remove()

        # ---- 3. Append with expanding-eps / split logic ----
        new_eps = self.eps_init
        if self.wrapper.n_slots > 0:
            K = torch.stack(self.wrapper.keys, dim=0).to(
                dtype=k_star.dtype, device=k_star.device)
            dists = (K - k_star.unsqueeze(0)).norm(dim=-1)
            min_dist_t, min_idx_t = dists.min(dim=0)
            min_dist = float(min_dist_t.item())
            min_idx = int(min_idx_t.item())
            existing_eps = self.wrapper.eps_per_key[min_idx]
            existing_label = self.wrapper.labels[min_idx]
            if min_dist < existing_eps:
                if existing_label == label:
                    return  # duplicate; skip
                half = max(min_dist / 2.0, 1e-4)
                self.wrapper.eps_per_key[min_idx] = half
                new_eps = half

        self.wrapper.add_entry(k_star, value.detach(), new_eps, label)


if _HAVE_REGISTRY:
    METHOD_REGISTRY["grace_mamba"] = GRACEMambaMethod
