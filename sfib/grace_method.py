"""grace_method.py -- GRACE baseline (Hartvigsen et al., NeurIPS 2023).

Drop-in module that registers `GRACEMethod` into the existing
`METHOD_REGISTRY` in `run_baselines.py`. After this file is imported anywhere
(e.g. at the top of `run_counterfact.py` / `run_zsre.py` / `run_baselines.py`),
the CLI accepts `--method grace`.

Implementation follows the GRACE paper's mechanics:

  * One chosen MLP layer is wrapped. The wrapper carries a list of
    `(key, value, eps, label)` codebook entries. Key is captured as the
    *input* to the MLP at the last sequence position of the rewrite prompt;
    value is a `d_model`-vector that *replaces* the MLP output at last
    position when a key match occurs.
  * Insertion procedure: random-init the value, freeze the base model, and
    run `n_steps` Adam steps on a CE-on-target loss. No ROME-style
    norm-regularizer / norm-constraint (this is the key methodological
    difference from MEMIT-derived methods).
  * Lookup: Euclidean distance between the current last-position MLP input
    and every stored key; pick the nearest entry; substitute the value if
    `distance < eps`. Otherwise leave the MLP output unchanged.
  * Per-entry epsilon: starts at `eps_init`. On each new insertion, if the
    nearest existing key is within its epsilon and has a *different* label,
    the existing entry's epsilon is shrunk to half the inter-key distance
    (split). If the labels match, the new key is dropped (merge). Otherwise
    a new entry is added at `eps_init`.

This matches the GRACE paper's "discrete key-value adaptors with expanding
deferral radii" description on the dimensions that matter for a like-for-like
comparison with our SHARD method:

  Mechanism            GRACE                           SHARD (ours)
  -----------------    -----------------------------   -----------------------------
  Slot value origin    random init + finetune          ROME-style v*-optimization
  Lookup metric        Euclidean + per-entry eps       Cosine + fixed tau
  Write mode           substitutive (replace h^l)      additive (add delta_v)
  Position             last token                      last token
  Base model           frozen                          frozen

Family-aware: works on GPT-2 (Conv1D MLPs) and SwiGLU (Qwen, Llama, Mistral)
without modification, because we wrap the *entire* MLP module (input -> output)
rather than touching its internals. The wrapper only intervenes at the last
sequence position.

CounterFact / zsRE integration: like `MEMITMethod` and `AddressableMemoryMethod`,
`GRACEMethod._build_rewrite()` is monkey-patched by `run_counterfact.py` and
`run_zsre.py` so the same Method class works on synthetic SFIB triples and on
real-data benchmark triples.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Wrapper around a base MLP that carries a GRACE codebook
# ---------------------------------------------------------------------------

class GRACEWrappedMLP(nn.Module):
    """Wraps a base MLP module. Forward computes the base output, then
    *substitutes* the last-position output with a codebook value if the
    last-position MLP input lies within any stored entry's epsilon-ball.

    Codebook state is held as plain Python lists of tensors (not buffers)
    because the bank grows per insertion and tensors are stored on-device.
    """

    def __init__(self, base_mlp: nn.Module, eps_init: float = 1.0):
        super().__init__()
        self.base_mlp = base_mlp
        self.eps_init = float(eps_init)
        # Codebook -- one entry per inserted fact.
        self.codebook_keys: list[torch.Tensor] = []   # each (d_in,)
        self.codebook_values: list[torch.Tensor] = []  # each (d_out,)
        self.codebook_eps: list[float] = []
        self.codebook_labels: list[str] = []

    @property
    def n_slots(self) -> int:
        return len(self.codebook_keys)

    def add_entry(self, key: torch.Tensor, value: torch.Tensor,
                  eps: float, label: str) -> None:
        self.codebook_keys.append(key.detach())
        self.codebook_values.append(value.detach())
        self.codebook_eps.append(float(eps))
        self.codebook_labels.append(str(label))

    def forward(self, x):  # x: (batch, seq, d_in)
        base_out = self.base_mlp(x)  # (batch, seq, d_out)
        if self.n_slots == 0:
            return base_out

        # GRACE intervenes at the last sequence position only (consistent
        # with how the language modelling head reads).
        x_last = x[:, -1, :]                                    # (batch, d_in)
        K = torch.stack(self.codebook_keys, dim=0)              # (n, d_in)
        K = K.to(device=x_last.device)
        # torch.cdist lacks a CUDA bfloat16 kernel, so cast to fp32 for the
        # distance computation. Memory cost is small (n_slots * d_in floats)
        # and the cast is local -- the output is converted back at the end.
        x_last_f = x_last.float()
        K_f = K.float()
        dists = torch.cdist(x_last_f, K_f)                       # (batch, n)
        min_dists, min_idx = dists.min(dim=-1)                   # (batch,) each
        eps_per = torch.tensor(
            [self.codebook_eps[i] for i in min_idx.tolist()],
            dtype=min_dists.dtype, device=min_dists.device,
        )
        hit = (min_dists < eps_per)                              # (batch,) bool
        if not bool(hit.any()):
            return base_out

        # Substitute base_out[:, -1, :] with codebook value where hit.
        V_chosen = torch.stack(
            [self.codebook_values[i] for i in min_idx.tolist()], dim=0
        ).to(dtype=base_out.dtype, device=base_out.device)        # (batch, d_out)
        new_last = torch.where(
            hit.unsqueeze(-1), V_chosen, base_out[:, -1, :]
        )
        out = base_out.clone()
        out[:, -1, :] = new_last
        return out


# ---------------------------------------------------------------------------
# GRACE method (registered into METHOD_REGISTRY at import time)
# ---------------------------------------------------------------------------

# Import lazily so this file works even if run_baselines.py is the old version.
from run_baselines import Method, METHOD_REGISTRY, DEVICE  # noqa: E402


class GRACEMethod(Method):
    """GRACE -- lifelong model editing with a discrete key-value codebook.

    Hyperparameters (paper defaults in parentheses):
      grace_layer       which transformer layer to wrap (typ. mid-to-late)
      grace_n_steps     finetune steps for the codebook value (100)
      grace_lr          Adam lr for value finetune (1.0 -- yes, large; matches paper)
      grace_eps_init    initial deferral radius for new entries (1.0)
      grace_init        'zeros' | 'random' -- how to initialize the value before
                        finetune. Paper says random; we default 'zeros' because
                        on modern instruction-tuned models random initialization
                        explodes the loss on step 1.
    """

    name = "grace"

    def __init__(self, layer_idx: int = 5, n_steps: int = 100,
                 lr: float = 1.0, eps_init: float = 1.0,
                 init: str = "zeros"):
        self.layer_idx = int(layer_idx)
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.eps_init = float(eps_init)
        if init not in ("zeros", "random"):
            raise ValueError(f"init must be 'zeros' or 'random', got {init!r}")
        self.init = init

    # ------------------------------------------------------------------
    # Family-aware layer access
    # ------------------------------------------------------------------

    def _get_mlp_and_block(self, model) -> tuple[nn.Module, nn.Module, str]:
        """Returns (mlp_module, block_module, family_str). Supports GPT-2
        (transformer.h[i].mlp), Llama/Qwen/Mistral
        (model.layers[i].mlp), and falls back via attribute search."""
        # GPT-2 family
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            block = model.transformer.h[self.layer_idx]
            return block.mlp, block, "gpt2"
        # Llama/Qwen/Mistral lineage
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            block = model.model.layers[self.layer_idx]
            return block.mlp, block, "swiglu"
        raise RuntimeError(
            "Unrecognized model layout for GRACE; expected "
            "model.transformer.h[i].mlp (GPT-2) or model.model.layers[i].mlp "
            "(Llama/Qwen)."
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, model, tokenizer, kb):
        super().setup(model, tokenizer, kb)
        # Freeze base; GRACE never modifies model weights.
        for p in model.parameters():
            p.requires_grad = False

        mlp, block, family = self._get_mlp_and_block(model)
        wrapped = GRACEWrappedMLP(mlp, eps_init=self.eps_init).to(DEVICE)
        block.mlp = wrapped
        self.wrapped_mlp = wrapped
        self.family = family
        print(f"[baseline] GRACE installed at layer {self.layer_idx} "
              f"({family} family); eps_init={self.eps_init}, "
              f"n_steps={self.n_steps}, lr={self.lr}, init={self.init}")

    # ------------------------------------------------------------------
    # Rewrite prompt (monkey-patched by CF / zsRE runners to handle their
    # triple types; this default works on SFIB Triple).
    # ------------------------------------------------------------------

    def _build_rewrite(self, triple: Any) -> tuple[str, str]:
        """SFIB default: same Q/A form as the AddressableMemory baseline."""
        from kb_data import RELATIONS  # local import to avoid hard dep
        rel = RELATIONS[triple.relation]
        q_tmpl, _ = rel.query_templates[0]
        prompt = q_tmpl.format(s=triple.subject)
        target = " " + triple.obj
        return prompt, target

    def _triple_label(self, triple: Any) -> str:
        """Stable identifier for split/merge decisions. For SFIB triples we
        use the object; CF / zsRE monkeypatches override this."""
        return getattr(triple, "obj", str(triple))

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def insert(self, triple: Any) -> None:
        prompt, target = self._build_rewrite(triple)
        label = self._triple_label(triple)

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        target_ids = self.tokenizer.encode(target, return_tensors="pt",
                                            add_special_tokens=False).to(DEVICE)
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        last_prompt_pos = prompt_ids.shape[1] - 1  # the position whose MLP-input is the key

        # ---- 1. Capture the key (MLP input at last prompt position) ----
        captured: dict[str, torch.Tensor] = {}

        def hook_key(module, inputs, output):
            # inputs[0] is the tensor entering the wrapped MLP: (batch, seq, d_in)
            captured["k"] = inputs[0][0, last_prompt_pos].detach().clone()
            # We also capture the *base* MLP output at that position as the
            # initialization for the codebook value (used when init='zeros' too,
            # since we need its shape).
            captured["v_base"] = output[0, last_prompt_pos].detach().clone()

        h_k = self.wrapped_mlp.register_forward_hook(hook_key)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                _ = self.model(prompt_ids)
        finally:
            h_k.remove()
            if was_training:
                self.model.train()

        k_star = captured["k"]                                    # (d_in,)
        v_base = captured["v_base"]                                # (d_out,)

        # ---- 2. Optimize the value via gradient descent ----
        if self.init == "zeros":
            value = torch.zeros_like(v_base, requires_grad=True)
        else:
            # Small random init in the scale of v_base.
            scale = v_base.norm().item() / (v_base.numel() ** 0.5 + 1e-8)
            value = (torch.randn_like(v_base) * scale).requires_grad_(True)

        opt = torch.optim.Adam([value], lr=self.lr)
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100  # CE only on the target tokens

        # Inject `value` as the substitution at the chosen position during
        # optimization (we cannot just call the wrapped MLP because the
        # codebook would not yet contain this entry).
        def inject_hook(module, inputs, output):
            out = output.clone()
            out[0, last_prompt_pos] = value
            return out

        h_inject = self.wrapped_mlp.register_forward_hook(inject_hook)
        try:
            for step in range(self.n_steps):
                opt.zero_grad()
                out = self.model(input_ids=full_ids, labels=labels)
                loss = out.loss
                loss.backward()
                opt.step()
        finally:
            h_inject.remove()

        # ---- 3. Add the entry with expanding-eps / split logic ----
        new_eps = self.eps_init
        if self.wrapped_mlp.n_slots > 0:
            K = torch.stack(self.wrapped_mlp.codebook_keys, dim=0).to(
                dtype=k_star.dtype, device=k_star.device,
            )
            dists = (K - k_star.unsqueeze(0)).norm(dim=-1)
            min_dist_t, min_idx_t = dists.min(dim=0)
            min_dist = float(min_dist_t.item())
            min_idx = int(min_idx_t.item())
            existing_eps = self.wrapped_mlp.codebook_eps[min_idx]
            existing_label = self.wrapped_mlp.codebook_labels[min_idx]
            if min_dist < existing_eps:
                if existing_label == label:
                    # Same answer at a colliding key -- the existing slot
                    # already handles this case; do not add a duplicate.
                    return
                else:
                    # Split: shrink the existing entry's eps to half the
                    # inter-key distance, and set the new entry's eps the
                    # same so neither dominates the other.
                    half = max(min_dist / 2.0, 1e-4)
                    self.wrapped_mlp.codebook_eps[min_idx] = half
                    new_eps = half

        self.wrapped_mlp.add_entry(k_star, value.detach(), new_eps, label)


# ------------------------------------------------