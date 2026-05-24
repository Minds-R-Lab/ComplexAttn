"""model_adapter.py — abstraction over HuggingFace decoder LM families.

The SFIB pipeline (pretrain, baselines, addressable memory primitive) was
originally written against GPT-2 with several hard-coded structural
assumptions. This adapter exposes a uniform interface so the same code can
run against any of:

  - GPT-2 family       (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
                       vanilla MLP: y = act(c_fc(x)); out = c_proj(y)
                       weights: Conv1D, shape (in_features, out_features)

  - SwiGLU family      (Qwen2.5, Llama-2/3, TinyLlama, Mistral, ...)
                       MLP: gate = act(gate_proj(x)); h = gate * up_proj(x); out = down_proj(h)
                       weights: nn.Linear, shape (out_features, in_features)

The adapter detects which family the model belongs to and routes all
family-dependent operations through methods that handle both cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

_LAYER_PATHS = (
    ("model.layers",        "swiglu"),  # Llama, Qwen2, Mistral, TinyLlama
    ("transformer.h",       "gpt2"),    # GPT-2
    ("gpt_neox.layers",     "gpt_neox"),  # Pythia / GPT-NeoX (different MLP again; not yet supported)
)


def _get_attr_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


@dataclass
class ModelAdapter:
    """Adapter object exposing a uniform interface over a HF causal LM."""
    model: nn.Module
    family: str               # 'gpt2' or 'swiglu'
    layer_path: str           # dotted path to the ModuleList of decoder layers
    layers: nn.ModuleList     # the decoder layers
    hidden_size: int          # d_model
    intermediate_size: int    # d_mlp (the c_proj/down_proj input dim)
    n_layers: int

    @classmethod
    def from_model(cls, model) -> "ModelAdapter":
        # Find layers + family
        for path, family in _LAYER_PATHS:
            layers = _get_attr_path(model, path)
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                break
        else:
            raise ValueError(
                f"Could not locate decoder layers in {type(model).__name__}. "
                f"Supported architectures: GPT-2, Qwen2, Llama, TinyLlama, Mistral.")

        if family == "gpt_neox":
            raise NotImplementedError("GPT-NeoX / Pythia not yet supported; their MLP structure differs.")

        # Sniff hidden_size / intermediate_size from config
        cfg = model.config
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        if family == "gpt2":
            intermediate = getattr(cfg, "n_inner", None) or 4 * hidden
        else:  # swiglu
            intermediate = getattr(cfg, "intermediate_size", None)
        if hidden is None or intermediate is None:
            raise ValueError(f"Could not infer hidden_size / intermediate_size from {type(model).__name__} config")

        return cls(
            model=model, family=family, layer_path=path, layers=layers,
            hidden_size=int(hidden), intermediate_size=int(intermediate),
            n_layers=len(layers),
        )

    # -----------------------------------------------------------------------
    # MLP access
    # -----------------------------------------------------------------------

    def get_mlp(self, layer_idx: int) -> nn.Module:
        """Return the MLP submodule of layer `layer_idx`."""
        return self.layers[layer_idx].mlp

    def set_mlp(self, layer_idx: int, new_mlp: nn.Module) -> None:
        """Replace the MLP submodule of layer `layer_idx`."""
        self.layers[layer_idx].mlp = new_mlp

    def get_down_proj(self, mlp: nn.Module) -> nn.Module:
        """Return the projection module that maps intermediate -> hidden
        (c_proj for GPT-2, down_proj for SwiGLU)."""
        if self.family == "gpt2":
            return mlp.c_proj
        return mlp.down_proj

    def compute_mlp_intermediate(self, mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP up to (but not including) the down-projection.
        Returns the tensor that feeds into c_proj (GPT-2) or down_proj (SwiGLU).
        Shape: (..., intermediate_size)."""
        if self.family == "gpt2":
            h = mlp.c_fc(x)
            h = mlp.act(h)
            return h
        # SwiGLU: act_fn(gate_proj(x)) * up_proj(x)
        gate = mlp.act_fn(mlp.gate_proj(x))
        up = mlp.up_proj(x)
        return gate * up

    def compute_mlp_full(self, mlp: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full MLP forward. Returns (intermediate, output).
        - intermediate: tensor of shape (..., intermediate_size) — input to down_proj
        - output: tensor of shape (..., hidden_size) — the down_proj output
        NOTE: dropout (GPT-2) is NOT applied here. The caller handles it."""
        intermediate = self.compute_mlp_intermediate(mlp, x)
        down = self.get_down_proj(mlp)
        return intermediate, down(intermediate)

    def maybe_apply_mlp_dropout(self, mlp: nn.Module, out: torch.Tensor) -> torch.Tensor:
        """GPT-2 MLP has a final dropout layer; SwiGLU MLPs typically don't.
        Call this to apply the model-family-appropriate dropout."""
        if self.family == "gpt2" and hasattr(mlp, "dropout"):
            return mlp.dropout(out)
        return out

    # -----------------------------------------------------------------------
    # Weight layout
    # -----------------------------------------------------------------------

    def apply_rank_one_update(self, down_proj: nn.Module,
                                k: torch.Tensor, r: torch.Tensor) -> None:
        """Apply (W + delta_W) k + b = (W k + b) + r via a min-Frobenius rank-one
        update on down_proj.weight, in-place. k has shape (intermediate_size,);
        r has shape (hidden_size,).

        For Conv1D (GPT-2): weight is (in, out), forward is x @ W + b.
          delta_W = outer(k, r) / ||k||^2

        For nn.Linear (SwiGLU): weight is (out, in), forward is x @ W.T + b.
          delta_W = outer(r, k) / ||k||^2
        """
        norm_sq = (k.pow(2).sum().item() + 1e-8)
        if self.family == "gpt2":
            # Conv1D weight has shape (in=intermediate, out=hidden)
            delta_W = torch.outer(k, r) / norm_sq
        else:
            # nn.Linear weight has shape (out=hidden, in=intermediate)
            delta_W = torch.outer(r, k) / norm_sq
        down_proj.weight.data.add_(delta_W)

    # -----------------------------------------------------------------------
    # LoRA target modules
    # -----------------------------------------------------------------------

    def lora_target_substrings(self) -> tuple[str, ...]:
        """Return the layer-name substrings to wrap with LoRA adapters.
        These differ by family:
        - GPT-2: c_attn (fused QKV), c_fc, c_proj
        - SwiGLU: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        """
        if self.family == "gpt2":
            return ("c_attn", "c_fc", "c_proj")
        return ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

    def lora_layer_classes(self) -> tuple[type, ...]:
        """Layer types that LoRA should be injected into."""
        try:
            from transformers.pytorch_utils import Conv1D
        except ImportError:
            Conv1D = nn.Linear  # fallback; won't be used for non-GPT-2
        return (Conv1D, nn.Linear)


def detect_family(model_name_or_model) -> str:
    """Quick helper for argparse defaulting: take a HF model name string and
    return the family without loading the model. Best-effort string match."""
    if not isinstance(model_name_or_model, str):
        return ModelAdapter.from_model(model_name_or_model).family
    name = model_name_or_model.lower()
    if name.startswith("gpt2") or "gpt2" in name or "/gpt2" in name:
        return "gpt2"
    return "swiglu"  # default for everything else
