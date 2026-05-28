"""shard_mamba.py -- SHARD method adapted to Mamba state-space language models.

We freeze a pretrained Mamba LM and attach an external (k*, delta_v) slot bank
to one chosen MambaBlock's projection (default: out_proj == W_o, following
ROMBA's finding that W_o is the strongest single-edit site in MambaBlock).

Per fact insertion:
  1. Run the frozen model on the rewrite prompt (teacher-forced with target).
  2. At the last subject token position, capture
       k*    = input to out_proj      (dim d_inner; this is s_i (x) g_i)
       v_orig= output of out_proj     (dim d_model; pre-residual addition)
  3. Gradient-descent on a free vector delta_v in R^{d_model} to minimize the
     CE on the target tokens with delta_v injected at position last_subj.
  4. Store (k*, delta_v) as a slot.

At inference, for every forward pass, the wrapped out_proj computes its normal
output, then at the LAST PROMPT POSITION ONLY checks cosine similarity between
the current pre-out_proj activation and every stored k*. If max similarity
exceeds threshold tau, the matching delta_v is added to the out_proj output.

Why last subject token for capture but last prompt position for routing?
ROMBA (Sen Sharma et al. 2024) showed that in Mamba factual recall localizes
at the SUBJECT-LAST TOKEN in middle layers (capture) and the PROMPT-LAST TOKEN
in later layers (read-out). For an additive slot, the most useful place to
fire is where the language-modeling head reads from -- the prompt-last
position. By default we set CAPTURE and FIRE both at the last prompt position
for symmetry with SHARD-for-transformers; we expose a flag to capture at the
subject-last token (the ROMBA setting) for ablation.

Family-aware: works on Mamba-1 (state-spaces/mamba-*-hf) and Mamba-2 layouts
via MambaAdapter.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_adapter import MambaAdapter, MambaEditSite

# Reuse the SHARD Method base class + METHOD_REGISTRY hook from sfib if available
try:
    from run_baselines import Method, METHOD_REGISTRY, DEVICE
    _HAVE_REGISTRY = True
except Exception:
    # Standalone fallback (when sfib/ isn't on the path)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class Method:
        name = "base"
        def setup(self, model, tokenizer, kb):
            self.model, self.tokenizer, self.kb = model, tokenizer, kb
        def insert(self, triple): pass
    METHOD_REGISTRY: dict = {}
    _HAVE_REGISTRY = False


class SHARDMambaWrapper(nn.Module):
    """Wraps a Mamba projection (default out_proj) and intercepts its forward
    to add a memory contribution at one specified position.

    Stored slots:
        K : list of (d_key,) tensors  -- input to the wrapped projection
        V : list of (d_value,) tensors -- additive delta to its output

    Routing happens at the LAST sequence position of every forward; an
    optional `fire_position` argument to the forward() call can override this
    (used during insertion to fire at the subject-last position).
    """

    def __init__(self, base_module: nn.Module,
                 sim_threshold: float = 0.7,
                 max_slots: int = 8000,
                 fire_position: str = "last"):
        super().__init__()
        self.base_module = base_module
        self.sim_threshold = float(sim_threshold)
        self.max_slots = int(max_slots)
        if fire_position not in ("last", "all"):
            raise ValueError(f"fire_position must be 'last' or 'all', got {fire_position!r}")
        self.fire_position = fire_position
        # Slot storage as plain lists -- bank grows with insertions.
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        # Optional one-off override of the firing position (set via attribute
        # for insertion-time hooks; we restore None after each fact).
        self._override_fire_pos: Optional[int] = None

    @property
    def n_slots(self) -> int:
        return len(self.keys)

    def add_slot(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if self.n_slots >= self.max_slots:
            raise RuntimeError(f"Memory full ({self.n_slots} slots; cap = {self.max_slots})")
        self.keys.append(k.detach())
        self.values.append(v.detach())

    def forward(self, x, *args, **kwargs):
        # x: (batch, seq, d_in) for nn.Linear, but Mamba's in_proj returns the
        # combined (a, g) tensor with shape (batch, seq, 2*d_inner) -- this
        # wrapper is designed for out_proj (Linear: d_inner -> d_model).
        base_out = self.base_module(x, *args, **kwargs)
        if self.n_slots == 0:
            return base_out

        # Debug accounting: count forward passes and slot hits.
        if not hasattr(self, "_diag_forwards"):
            self._diag_forwards = 0
            self._diag_hits = 0
            self._diag_max_sim_recent = []
        self._diag_forwards += 1

        # Identify the position(s) at which to apply the slot.
        if self._override_fire_pos is not None:
            positions = [self._override_fire_pos]
        elif self.fire_position == "last":
            positions = [-1]  # last token of every sequence
        else:
            # fire at every position -- behaviour primarily for ablation.
            positions = list(range(x.shape[1]))

        K = torch.stack(self.keys, dim=0).to(device=x.device)
        V = torch.stack(self.values, dim=0).to(dtype=base_out.dtype, device=base_out.device)

        # We only apply at the chosen position(s).
        out = base_out.clone()
        for pos in positions:
            x_pos = x[:, pos, :]                          # (batch, d_key)
            x_n = F.normalize(x_pos.float(), dim=-1)
            K_n = F.normalize(K.float(), dim=-1)
            sims = x_n @ K_n.t()                          # (batch, n_slots)
            best_sim, best_idx = sims.max(dim=-1)         # (batch,) each
            # Diagnostic: record the highest sim of any batch position.
            if len(self._diag_max_sim_recent) < 200:
                self._diag_max_sim_recent.append(float(best_sim.max().item()))
            hit = (best_sim > self.sim_threshold)
            if not bool(hit.any()):
                continue
            self._diag_hits += int(hit.sum().item())
            V_chosen = V[best_idx]                        # (batch, d_value)
            new_pos = out[:, pos, :] + torch.where(
                hit.unsqueeze(-1), V_chosen, torch.zeros_like(V_chosen)
            )
            out[:, pos, :] = new_pos
        return out


class SHARDMambaMethod(Method):
    """SHARD adapted to Mamba: per-fact slot bank attached to a frozen MambaBlock projection.

    Default edit site: layer_idx=39, kind='out_proj' (matching ROMBA's
    middle-layer recommendation for Mamba-2.8B which has 64 layers).
    For smaller Mamba models the user should pass an appropriate layer_idx.
    """

    name = "shard_mamba"

    def __init__(self,
                 layer_idx: int = 39,
                 kind: str = "out_proj",
                 n_v_steps: int = 200,
                 v_lr: float = 1.0,
                 v_weight_decay: float = 0.0,
                 v_norm_constraint: float = 20.0,
                 sim_threshold: float = 0.7,
                 max_slots: int = 8000,
                 capture_position: str = "prompt_last",
                 fire_position: str = "last",
                 value_optim: str = "vstar",
                 lqr_lambda: float = 1e-3,
                 lqr_alpha_scale: float = 1.0):
        if kind not in ("out_proj", "in_proj", "x_proj"):
            raise ValueError(f"kind must be one of out_proj|in_proj|x_proj, got {kind!r}")
        if capture_position not in ("subject_last", "prompt_last"):
            raise ValueError(f"capture_position must be subject_last|prompt_last, got {capture_position!r}")
        if value_optim not in ("vstar", "lqr"):
            raise ValueError(f"value_optim must be 'vstar' or 'lqr', got {value_optim!r}")
        self.layer_idx = int(layer_idx)
        self.kind = kind
        self.n_v_steps = int(n_v_steps)
        self.v_lr = float(v_lr)
        self.v_weight_decay = float(v_weight_decay)
        self.v_norm_constraint = float(v_norm_constraint)
        self.sim_threshold = float(sim_threshold)
        self.max_slots = int(max_slots)
        self.capture_position = capture_position
        self.fire_position = fire_position
        self.value_optim = value_optim
        self.lqr_lambda = float(lqr_lambda)
        self.lqr_alpha_scale = float(lqr_alpha_scale)

    # ------------------------------------------------------------------
    def setup(self, model, tokenizer, kb=None):
        super().setup(model, tokenizer, kb)
        # Freeze the entire backbone.
        for p in model.parameters():
            p.requires_grad = False

        self.adapter = MambaAdapter.from_model(model)
        site = self.adapter.edit_site(self.layer_idx, kind=self.kind)
        self.site = site
        self.wrapper = SHARDMambaWrapper(
            base_module=site.module,
            sim_threshold=self.sim_threshold,
            max_slots=self.max_slots,
            fire_position=self.fire_position,
        ).to(DEVICE)
        self.adapter.install_wrapper(self.layer_idx, self.kind, self.wrapper)
        print(f"[shard-mamba] installed at layer {self.layer_idx} ({self.adapter.family}); "
              f"site={self.kind}, d_key={site.d_key}, d_value={site.d_value}; "
              f"tau={self.sim_threshold}, value_optim={self.value_optim}, "
              f"n_v_steps={self.n_v_steps}, v_lr={self.v_lr}, "
              f"lqr_lambda={self.lqr_lambda}, lqr_alpha_scale={self.lqr_alpha_scale}, "
              f"capture@{self.capture_position}, fire@{self.fire_position}")

    # ------------------------------------------------------------------
    def _build_rewrite(self, triple) -> tuple[str, str, str]:
        """Returns (prompt, target, subject). Override per-benchmark."""
        try:
            from kb_data import RELATIONS
            rel = RELATIONS[triple.relation]
            q_tmpl, _ = rel.query_templates[0]
            return q_tmpl.format(s=triple.subject), " " + triple.obj, triple.subject
        except Exception:
            raise NotImplementedError(
                "SHARDMambaMethod._build_rewrite should be monkey-patched by "
                "the runner for CounterFact / zsRE triple types.")

    def _find_subject_last_pos(self, prompt_ids: torch.Tensor, subject: str) -> int:
        """Identify the token index whose decoded prefix first contains the full subject."""
        ids = prompt_ids[0].tolist()
        decoded_so_far = ""
        for i, tok in enumerate(ids):
            decoded_so_far += self.tokenizer.decode([tok])
            if subject.lower() in decoded_so_far.lower():
                return i
        # Fallback: last token of prompt minus 1 (i.e., last non-target position).
        return len(ids) - 1

    # ------------------------------------------------------------------
    def insert(self, triple) -> None:
        prompt, target, subject = self._build_rewrite(triple)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        target_ids = self.tokenizer.encode(
            target, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)

        last_prompt_pos = prompt_ids.shape[1] - 1
        if self.capture_position == "subject_last":
            capture_pos = self._find_subject_last_pos(prompt_ids, subject)
        else:
            capture_pos = last_prompt_pos

        # ---- 1. Capture k* (input to wrapped module) and v_orig (its output) ----
        captured: dict[str, torch.Tensor] = {}

        def hook_kv(module, inputs, output):
            # inputs[0] shape: (batch, seq, d_key)
            # output     shape: (batch, seq, d_value)
            captured["k"] = inputs[0][0, capture_pos].detach().clone()
            captured["v_orig"] = output[0, capture_pos].detach().clone()

        # Hook the BASE module inside the wrapper (so we see the unmodified output).
        h_kv = self.wrapper.base_module.register_forward_hook(hook_kv)
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

        # ---- 2. Optimize delta_v ----
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100  # CE only on target tokens

        if self.value_optim == "vstar":
            delta_v = self._optimize_vstar(full_ids, labels, capture_pos, v_orig)
        else:  # "lqr"
            delta_v = self._optimize_lqr(full_ids, labels, capture_pos, v_orig)

        # ---- 3. Append the (k*, delta_v) slot ----
        self.wrapper.add_slot(k_star, delta_v.detach())

    # ------------------------------------------------------------------
    def _optimize_vstar(self, full_ids, labels, capture_pos, v_orig):
        """ROME-style multi-step optimization of delta_v (200-step Adam by default)."""
        delta_v = torch.zeros_like(v_orig, requires_grad=True)
        opt = torch.optim.Adam([delta_v], lr=self.v_lr)

        # During optimization, inject delta_v at capture_pos by adding it to
        # the BASE module's output at that position. We do this with a hook
        # on the base_module (NOT the wrapper, because the wrapper's
        # slot-lookup logic is empty for this not-yet-stored fact).
        def inject_hook(module, inputs, output):
            out = output.clone()
            out[0, capture_pos] = out[0, capture_pos] + delta_v
            return out

        h_inject = self.wrapper.base_module.register_forward_hook(inject_hook)
        try:
            for step in range(self.n_v_steps):
                opt.zero_grad()
                out = self.model(input_ids=full_ids, labels=labels)
                ce = out.loss
                reg = self.v_weight_decay * (delta_v.norm() ** 2) / (
                    v_orig.norm() ** 2 + 1e-8)
                loss = ce + reg
                loss.backward()
                opt.step()
                # Norm cap (relative to v_orig)
                with torch.no_grad():
                    max_norm = self.v_norm_constraint * v_orig.norm().item()
                    cur_norm = delta_v.norm().item()
                    if cur_norm > max_norm and cur_norm > 0:
                        delta_v.mul_(max_norm / cur_norm)
        finally:
            h_inject.remove()
        return delta_v.detach()

    # ------------------------------------------------------------------
    def _optimize_lqr(self, full_ids, labels, capture_pos, v_orig):
        """Control-theoretic LQR / Tikhonov closed-form for delta_v.

        Treat one-step slot insertion as a finite-horizon LQR problem on the
        residual stream: the "state" we wish to steer is the cross-entropy on
        the target tokens, the "control" is the additive perturbation delta_v
        applied at the chosen position, and the cost is

            J(delta_v) = CE(delta_v) + (lambda / ||v_orig||^2) * ||delta_v||^2.

        Linearizing CE around delta_v = 0,

            CE(delta_v) ~ CE_0 + g^T delta_v,    g := d CE / d (delta_v) | 0.

        The first-order optimality condition gives the closed-form Tikhonov /
        Gauss-Newton step

            delta_v* = - alpha * g / (||g||^2 + lambda_eff),

        where lambda_eff = lambda * ||v_orig||^{-2} and alpha is chosen so the
        linearized CE is driven to zero,

            alpha = CE_0 * (||g||^2 + lambda_eff) / ||g||^2.

        We then clamp ||delta_v*|| to the same v_norm_constraint as the v*
        path so the two methods share an admissible-control set.

        One backward pass instead of n_v_steps. The control-theoretic
        interpretation: this is the LQR feedback law for one slot, with
        Q = identity on the CE residual, R = (lambda_eff) * I on delta_v,
        evaluated at the operating point delta_v = 0.
        """
        delta_v = torch.zeros_like(v_orig, requires_grad=True)

        def inject_hook(module, inputs, output):
            out = output.clone()
            out[0, capture_pos] = out[0, capture_pos] + delta_v
            return out

        h_inject = self.wrapper.base_module.register_forward_hook(inject_hook)
        try:
            # One forward + backward at delta_v = 0
            out = self.model(input_ids=full_ids, labels=labels)
            ce0 = out.loss
            ce0.backward()
            g = delta_v.grad.detach().clone()
        finally:
            h_inject.remove()

        ce0_val = float(ce0.detach().item())
        g_sq = float((g * g).sum().item())
        # Effective Tikhonov weight, scaled the same way as the v* regularizer
        v_orig_sq = float(v_orig.norm().item() ** 2) + 1e-8
        lambda_eff = self.lqr_lambda / v_orig_sq

        if g_sq < 1e-12:
            # Gradient vanishes -- no informative direction; leave delta_v at 0.
            return torch.zeros_like(v_orig).detach()

        # LQR feedback: alpha drives the linearized CE to zero
        alpha = self.lqr_alpha_scale * ce0_val * (g_sq + lambda_eff) / g_sq
        delta_v_star = -alpha * g / (g_sq + lambda_eff)

        # Norm clamp (same admissible-control set as v*)
        max_norm = self.v_norm_constraint * v_orig.norm().item()
        cur_norm = float(delta_v_star.norm().item())
        if cur_norm > max_norm and cur_norm > 0:
            delta_v_star = delta_v_star * (max_norm / cur_norm)
        return delta_v_star.detach()


# Self-register if the SFIB registry is available.
if _HAVE_REGISTRY:
    METHOD_REGISTRY["shard_mamba"] = SHARDMambaMethod
