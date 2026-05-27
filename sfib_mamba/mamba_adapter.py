"""mamba_adapter.py -- thin abstraction over Mamba models (Mamba-1, Mamba-2, RWKV).

The goal is to give the SHARD-for-Mamba method a uniform way to locate the
edit site (the analog of the transformer MLP's down-projection) without
hard-coding HuggingFace attribute paths.

Targeted release of Mamba: state-spaces/mamba-{130m,790m,1.4b,2.8b}-hf
HuggingFace structure (mamba):
    model.backbone.layers[i].mixer
        in_proj (W_a + W_g combined, splits internally)
        conv1d
        x_proj, dt_proj
        A_log, D
        out_proj   <- this is W_o in ROMBA notation
        norm

ROMBA finding (Sen Sharma, Atkinson, Bau, COLM 2024): of the three candidate
edit sites in the MambaBlock (W_a feeding Conv+SSM, W_g gating, W_o output
projection), W_o gives the best harmonic mean of Efficacy/Generalization/
Specificity under ROME-style single edits. We adopt W_o (out_proj) as the
default edit site for SHARD-for-Mamba, but the adapter exposes all three so
ablations can be run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MambaEditSite:
    """A locatable edit site inside a single MambaBlock.

    `module` is the nn.Module we wrap. `key_capture` describes whether to hook
    inputs or outputs of `module` to obtain the slot key.
    """
    layer_idx: int
    module: nn.Module
    name: str                 # 'out_proj' | 'in_proj' | ...
    key_role: str             # 'module_input' | 'module_output'
    d_key: int                # dimensionality of the captured key
    d_value: int              # dimensionality of the value (perturbation)


class MambaAdapter:
    """Adapter that locates the edit site for a frozen Mamba model.

    Usage:
        adapter = MambaAdapter.from_model(model)
        site = adapter.edit_site(layer_idx=20, kind='out_proj')
        # site.module is the nn.Linear / nn.Conv1d to wrap
    """

    def __init__(self, model, family: str, layers_list, d_model: int,
                 d_inner: int):
        self.model = model
        self.family = family
        self.layers = layers_list
        self.d_model = d_model
        self.d_inner = d_inner

    @classmethod
    def from_model(cls, model) -> "MambaAdapter":
        # HuggingFace Mamba-1 layout (state-spaces/mamba-*-hf)
        if hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
            layers = model.backbone.layers
            # Try to read d_model / d_inner from the first block's mixer
            first_mixer = layers[0].mixer if hasattr(layers[0], "mixer") else layers[0]
            d_model = getattr(first_mixer, "d_model", None)
            d_inner = getattr(first_mixer, "d_inner", None)
            if d_model is None or d_inner is None:
                # Try to infer from out_proj.weight shape: (d_model, d_inner)
                if hasattr(first_mixer, "out_proj"):
                    w = first_mixer.out_proj.weight  # (d_model, d_inner)
                    d_model, d_inner = w.shape[0], w.shape[1]
            return cls(model, family="mamba1", layers_list=layers,
                       d_model=int(d_model), d_inner=int(d_inner))
        # HuggingFace Mamba-2 layout
        if hasattr(model, "model") and hasattr(model.model, "layers") and \
           hasattr(model.model.layers[0], "mixer"):
            layers = model.model.layers
            first_mixer = layers[0].mixer
            d_model = getattr(first_mixer, "hidden_size", None) or \
                      getattr(first_mixer, "d_model", None)
            d_inner = getattr(first_mixer, "intermediate_size", None) or \
                      getattr(first_mixer, "d_inner", None)
            return cls(model, family="mamba2", layers_list=layers,
                       d_model=int(d_model), d_inner=int(d_inner))
        raise RuntimeError(
            "Unrecognized Mamba model layout. Expected "
            "model.backbone.layers[i].mixer (Mamba-1) or "
            "model.model.layers[i].mixer (Mamba-2)."
        )

    def n_layers(self) -> int:
        return len(self.layers)

    def get_block(self, layer_idx: int):
        if not (0 <= layer_idx < len(self.layers)):
            raise ValueError(
                f"layer_idx={layer_idx} out of range "
                f"[0, {len(self.layers)})")
        return self.layers[layer_idx]

    def get_mixer(self, layer_idx: int):
        block = self.get_block(layer_idx)
        if hasattr(block, "mixer"):
            return block.mixer
        return block  # some Mamba variants put projections directly on the block

    def edit_site(self, layer_idx: int, kind: str = "out_proj") -> MambaEditSite:
        """Resolve an edit site by name. Default is 'out_proj' (W_o)."""
        mixer = self.get_mixer(layer_idx)
        if kind == "out_proj":
            # W_o : (d_model, d_inner). We treat its input as the key,
            # its output as the value the perturbation modifies.
            mod = mixer.out_proj
            d_key = self.d_inner       # input to out_proj is d_inner-dim
            d_value = self.d_model     # output is d_model-dim (added to residual)
            key_role = "module_input"
        elif kind == "in_proj":
            # W_a + W_g combined. We capture its OUTPUT (which has the s+g
            # representation pre-split). This is closer to a transformer-MLP
            # 'intermediate' analog if we want to inject before the Conv+SSM.
            mod = mixer.in_proj
            d_key = self.d_model       # input to in_proj is d_model-dim
            d_value = 2 * self.d_inner # output is (d_inner * 2) for [a; g]
            key_role = "module_input"
        elif kind == "x_proj":
            mod = mixer.x_proj
            d_key = self.d_inner
            d_value = mod.weight.shape[0]
            key_role = "module_input"
        else:
            raise ValueError(f"Unknown edit-site kind: {kind!r}")
        return MambaEditSite(
            layer_idx=layer_idx, module=mod, name=kind,
            key_role=key_role, d_key=d_key, d_value=d_value,
        )

    def install_wrapper(self, layer_idx: int, kind: str, wrapper) -> None:
        """Replace the chosen module with a wrapper so it intercepts forward()."""
        mixer = self.get_mixer(layer_idx)
        if kind == "out_proj":
            setattr(mixer, "out_proj", wrapper)
        elif kind == "in_proj":
            setattr(mixer, "in_proj", wrapper)
        elif kind == "x_proj":
            setattr(mixer, "x_proj", wrapper)
        else:
            raise ValueError(f"Unknown kind for install_wrapper: {kind!r}")
