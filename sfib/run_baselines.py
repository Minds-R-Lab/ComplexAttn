"""run_baselines.py — SFIB insertion-stream baselines.

Each baseline takes the pretrained backbone (sfib/checkpoints/pretrained_seed0_gpt2.pt)
and processes the insertion stream (kb.insert_triples) one fact at a time. At a
sequence of pre-registered checkpoints N ∈ {0, 1, 10, 50, 100, 250, 500}, we
measure three quantities:

    Insertion@N    : accuracy on the first N inserted facts' eval queries
    Retention@N    : accuracy on the held-out retention eval set
                     (this is the same 500 pretrain triples / 1000 queries
                      we used during pretrain — measures forgetting)
    Composition@N  : accuracy on compose pairs whose inserted-triple index < N

Methods (registry):
    frozen      : no updates. Lower bound. Insertion=0, Retention=baseline.
    in_context  : prepend the relevant ground-truth fact(s) to each eval prompt
                  via oracle retrieval (we know the (S,R) of each query).
                  Upper bound: tells us "what the model could do if memory
                  weren't the bottleneck."

    seq_ft      : full-model AdamW step per inserted fact (catastrophic-forgetting baseline)
    lora_seq    : sequential fine-tune restricted to LoRA adapters on the
                  attention/MLP weight matrices. Tests whether parameter-efficient
                  updates escape catastrophic forgetting or just slow it down.
    memit       : ROME-style rank-one MLP edits (Meng et al. 2022), applied
                  sequentially. Targets a single mid-layer MLP's down-projection.
                  This is the principal adversary for the addressable primitive.
    addressable_mem : The proposed primitive. Same v*-optimization as MEMIT,
                  but stores (k*, delta_v) pairs as slots in an external memory
                  module wrapping the chosen MLP. Forward computes base MLP
                  output PLUS a cosine-similarity-gated memory contribution.
                  No base weight modification.

Note: 'in_context' is registered but deprecated for this scale. GPT-2 small cannot
reliably use prompted facts (see diagnose_in_context.py); prepending degrades
retention. Kept available for ablations/diagnostics, not for headline comparison.

Usage:
    python run_baselines.py --method frozen
    python run_baselines.py --method in_context
    python run_baselines.py --method frozen --eval_at 0,1,10,50,100,250,500
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from kb_data import (
    generate_kb, render_train_example, render_eval_query, render_composition,
    Triple, RELATIONS,
)
from evaluate import EvalExample, evaluate_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
CKPT_DIR = SFIB_DIR / "checkpoints"
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Method interface
# ---------------------------------------------------------------------------

class Method:
    """Base class. Override insert() and/or transform_examples()."""
    name = "base"

    def setup(self, model, tokenizer, kb):
        """Called once before any insertions. `model` is the pretrained backbone."""
        self.model = model
        self.tokenizer = tokenizer
        self.kb = kb

    def insert(self, triple: Triple) -> None:
        """Process one inserted fact. Override to mutate model or state."""
        pass

    def transform_examples(self,
                            examples: list[EvalExample]) -> list[EvalExample]:
        """Optionally modify the eval examples (e.g., prepend retrieved facts).
        Default: no transformation."""
        return examples


class FrozenMethod(Method):
    """No updates. Insertion accuracy will be near zero (model has never seen
    Group B entities); retention should equal the pretrain baseline."""
    name = "frozen"


class InContextOracleMethod(Method):
    """Prepend the ground-truth fact(s) to the eval prompt. Oracle retrieval:
    we know the (S,R) of each query, so we look up the exact inserted triple
    and inject its statement form. For composition queries we inject BOTH the
    inserted triple AND the pretrain triple it composes with."""
    name = "in_context"

    def setup(self, model, tokenizer, kb):
        super().setup(model, tokenizer, kb)
        # We need to find pretrain triples for composition's pretrain branch
        # (those are always available — they were in pretraining). We also need
        # to know what's been "inserted" so far. For oracle in-context, the
        # "inserted" set grows as we call insert().
        self._inserted_by_sr: dict[tuple[str, str], Triple] = {}
        # Also build a pretrain index for composition's first-hop facts.
        self._pretrain_by_sr: dict[tuple[str, str], Triple] = {}
        for t in kb.pretrain_triples:
            self._pretrain_by_sr[(t.subject, t.relation)] = t

    def insert(self, triple: Triple) -> None:
        self._inserted_by_sr[(triple.subject, triple.relation)] = triple

    def _statement_of(self, triple: Triple) -> str:
        return render_train_example(triple, template_idx=0)

    def transform_examples(self, examples):
        out: list[EvalExample] = []
        for ex in examples:
            if ex.kind == "insertion":
                subj, rel, _ = ex.meta["triple"]
                t = self._inserted_by_sr.get((subj, rel))
                if t is None:
                    # The query refers to a fact we haven't inserted yet -> no context.
                    new_prompt = ex.prompt
                else:
                    fact = self._statement_of(t)
                    new_prompt = f"{fact} {ex.prompt}"
            elif ex.kind == "retention":
                # Retention queries are about pretrain facts (the model knows
                # them). Strict in-context baseline would also inject them; in
                # an oracle setting we'd inject the pretrain triple. Doing so
                # lets us see whether the upper bound is bottlenecked by
                # memorization of pretrain or by retrieval. Add it.
                subj, rel, _ = ex.meta["triple"]
                t = self._pretrain_by_sr.get((subj, rel))
                if t is None:
                    new_prompt = ex.prompt
                else:
                    fact = self._statement_of(t)
                    new_prompt = f"{fact} {ex.prompt}"
            elif ex.kind == "composition":
                # Composition: prepend BOTH the inserted (B-person, lives_in, city)
                # fact and the pretrain (city, mayor_of, mayor) fact.
                inserted_meta = ex.meta["inserted_triple"]
                pretrain_meta = ex.meta["pretrain_triple"]
                pieces = []
                t_ins = self._inserted_by_sr.get(
                    (inserted_meta[0], inserted_meta[1]))
                if t_ins is not None:
                    pieces.append(self._statement_of(t_ins))
                t_pre = self._pretrain_by_sr.get(
                    (pretrain_meta[0], pretrain_meta[1]))
                if t_pre is not None:
                    pieces.append(self._statement_of(t_pre))
                if pieces:
                    new_prompt = " ".join(pieces) + " " + ex.prompt
                else:
                    new_prompt = ex.prompt
            else:
                new_prompt = ex.prompt
            out.append(EvalExample(
                prompt=new_prompt, target=ex.target, kind=ex.kind, meta=ex.meta,
            ))
        return out


class SequentialFTMethod(Method):
    """Catastrophic-forgetting baseline. For each inserted triple, do N gradient
    steps of AdamW on the (Q/A + statement) renderings of the triple. State
    (optimizer + model weights) persists across insertions, so the second
    insertion starts from where the first ended.

    Expected behavior: Insertion@N stays high (each fact is freshly trained in),
    Retention@N degrades monotonically with N as old facts are overwritten.
    """
    name = "seq_ft"

    def __init__(self, lr: float = 1e-5, n_steps: int = 5):
        self.lr = lr
        self.n_steps = n_steps

    def setup(self, model, tokenizer, kb):
        super().setup(model, tokenizer, kb)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=0.0
        )

    def _build_batch(self, triple: Triple):
        rel = RELATIONS[triple.relation]
        texts: list[str] = []
        for q_idx in range(len(rel.query_templates)):
            prompt, target = render_eval_query(triple, template_idx=q_idx)
            texts.append(f"{prompt} {target}")
        for s_idx in range(len(rel.fact_templates)):
            texts.append(render_train_example(triple, template_idx=s_idx))
        encs = self.tokenizer(
            [t + self.tokenizer.eos_token for t in texts],
            return_tensors="pt", truncation=True, max_length=64,
            padding="max_length",
        )
        input_ids = encs["input_ids"].to(DEVICE)
        attn = encs["attention_mask"].to(DEVICE)
        labels = input_ids.clone()
        labels[attn == 0] = -100
        return input_ids, attn, labels

    def insert(self, triple: Triple) -> None:
        self.model.train()
        input_ids, attn, labels = self._build_batch(triple)
        for _ in range(self.n_steps):
            self.optimizer.zero_grad()
            out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()


# ---------------------------------------------------------------------------
# LoRA wrapper for GPT-2's Conv1D layers
# ---------------------------------------------------------------------------
# GPT-2 uses `transformers.pytorch_utils.Conv1D`, which is just `x @ W + b`
# with W of shape (in_features, out_features) — the transpose of nn.Linear.
# We wrap it with a low-rank adapter added in parallel:
#     output = x @ W + b + (x @ A) @ B * scaling
# where A: (in, r), B: (r, out). lora_A is initialized small, lora_B is zero,
# so the wrapped layer starts identical to the base layer.

import torch.nn as nn

class Conv1DWithLoRA(nn.Module):
    def __init__(self, base, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base                       # frozen Conv1D
        in_features = base.weight.shape[0]
        out_features = base.weight.shape[1]
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = alpha / r

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
        return base_out + lora_out


def _inject_lora(model, target_substrings: tuple[str, ...], r: int = 4,
                  alpha: float = 8.0) -> int:
    """Replace every Conv1D submodule whose name contains any of the given
    substrings with a Conv1DWithLoRA wrapper. Returns count of layers wrapped."""
    from transformers.pytorch_utils import Conv1D
    count = 0
    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # Walk the module tree and patch
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, Conv1D):
                continue
            if not any(s in full_name for s in target_substrings):
                continue
            wrapped = Conv1DWithLoRA(child, r=r, alpha=alpha)
            # Move to same device/dtype as base
            wrapped = wrapped.to(child.weight.device).to(child.weight.dtype)
            setattr(module, child_name, wrapped)
            count += 1
    # Make LoRA params trainable
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
    return count


class LoRASequentialMethod(SequentialFTMethod):
    """Same training procedure as seq_ft but updates are restricted to rank-r
    LoRA adapters on c_attn / c_fc / c_proj. The base model is frozen, so any
    forgetting of pretrain knowledge has to flow through the adapters'
    interaction with the frozen weights.

    Expected: better retention than full seq_ft (most weights frozen) but
    insertion may also be weaker (smaller capacity for new facts)."""
    name = "lora_seq"

    def __init__(self, lr: float = 1e-4, n_steps: int = 5, rank: int = 4,
                 alpha: float = 8.0,
                 target_substrings: tuple[str, ...] = ("c_attn", "c_fc", "c_proj")):
        super().__init__(lr=lr, n_steps=n_steps)
        self.rank = rank
        self.alpha = alpha
        self.target_substrings = target_substrings

    def setup(self, model, tokenizer, kb):
        # Inject LoRA BEFORE creating optimizer
        n_wrapped = _inject_lora(model, self.target_substrings,
                                   r=self.rank, alpha=self.alpha)
        lora_params = [p for n, p in model.named_parameters()
                        if ("lora_A" in n or "lora_B" in n)]
        n_lora_params = sum(p.numel() for p in lora_params)
        n_total_params = sum(p.numel() for p in model.parameters())
        print(f"[baseline] LoRA injected into {n_wrapped} Conv1D layers "
              f"(target: {self.target_substrings})")
        print(f"[baseline] LoRA trainable params: {n_lora_params:,} / "
              f"{n_total_params:,} ({100*n_lora_params/n_total_params:.2f}%)")
        self.model = model
        self.tokenizer = tokenizer
        self.kb = kb
        self.optimizer = torch.optim.AdamW(
            lora_params, lr=self.lr, weight_decay=0.0
        )


# ---------------------------------------------------------------------------
# MEMIT / ROME — rank-one MLP edits
# ---------------------------------------------------------------------------
# Algorithm (per inserted triple):
#   1. Build a "rewrite prompt" that ends just before the target object:
#      "{subject} <relation verb-phrase>" + target = " <object>"
#   2. Tokenize, identify the last token position of the subject.
#   3. Forward pass: capture k* (input to c_proj at that position) and
#      v_orig (output of MLP at that position).
#   4. Optimize delta_v: gradient descent on a perturbation vector added at
#      the subject-position MLP output, minimizing CE on the target tokens.
#      Constrain ||delta_v|| <= norm_constraint * ||v_orig|| and weight decay
#      regularizer to prevent runaway perturbations.
#   5. Apply the rank-one update to c_proj.weight:
#      W_new = W + outer(k*, v* - W k* - b) / ||k*||^2
#      For GPT-2's Conv1D (W shape: in_features × out_features), this matches
#      the forward x @ W + b such that the updated output at k* equals v*.
#
# Hyperparameters follow Meng et al. 2022 (ROME) defaults adapted for GPT-2 small.

class MEMITMethod(Method):
    """ROME-style sequential MLP edits. One rank-one update per inserted fact.

    Default targets GPT-2 small's layer 5 (mid-stack), which roughly corresponds
    to where Meng et al. found fact-storage in GPT-2 medium/large after scaling.
    """
    name = "memit"

    def __init__(self, layer_idx: int = 5, n_v_steps: int = 20,
                 v_lr: float = 0.5, v_weight_decay: float = 0.5,
                 v_norm_constraint: float = 4.0):
        self.layer_idx = layer_idx
        self.n_v_steps = n_v_steps
        self.v_lr = v_lr
        self.v_weight_decay = v_weight_decay
        self.v_norm_constraint = v_norm_constraint

    def setup(self, model, tokenizer, kb):
        super().setup(model, tokenizer, kb)
        n_layers = len(model.transformer.h)
        if not (0 <= self.layer_idx < n_layers):
            raise ValueError(
                f"layer_idx={self.layer_idx} out of range for model with {n_layers} layers")
        self.mlp = model.transformer.h[self.layer_idx].mlp
        # Freeze all weights — we'll modify c_proj.weight by hand
        for p in model.parameters():
            p.requires_grad = False
        print(f"[baseline] MEMIT editing MLP of layer {self.layer_idx}; "
              f"c_proj weight shape: {tuple(self.mlp.c_proj.weight.shape)}")

    def _build_rewrite(self, triple: Triple) -> tuple[str, str]:
        """Build (prompt, target) such that prompt ends right before the object
        and target is ' <object>'."""
        rel = RELATIONS[triple.relation]
        # Use the first fact template (most natural statement form)
        tmpl = rel.fact_templates[0]
        full = tmpl.format(s=triple.subject, o=triple.obj)
        # Trim trailing period if present
        if full.endswith("."):
            full = full[:-1]
        # Split at the object
        idx = full.rfind(triple.obj)
        prompt = full[:idx].rstrip()
        target = " " + triple.obj
        return prompt, target

    def _find_last_subj_pos(self, prompt_token_ids: list[int],
                             subject: str) -> int:
        """Locate the position of the last token of `subject` in the tokenized
        prompt. Falls back to position 0 if not found (we always have at least
        one token; the subject leads the rewrite prompt by construction)."""
        decoded = ""
        for i, tok_id in enumerate(prompt_token_ids):
            decoded += self.tokenizer.decode([tok_id])
            if subject in decoded:
                # Subject is fully visible by token i; check if token i was needed
                prev_decoded = decoded[:-len(self.tokenizer.decode([tok_id]))]
                if subject not in prev_decoded:
                    return i  # token i contributed the last char(s) of subject
        # Fallback: assume the subject is the prefix; use end of subject in isolation
        subj_ids = self.tokenizer.encode(subject)
        return min(len(subj_ids) - 1, len(prompt_token_ids) - 1)

    def insert(self, triple: Triple) -> None:
        prompt, target = self._build_rewrite(triple)

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        target_ids = self.tokenizer.encode(target, return_tensors="pt").to(DEVICE)
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)

        last_subj_pos = self._find_last_subj_pos(
            prompt_ids[0].tolist(), triple.subject)

        # --- Step 1: Capture k* and v_orig at the subject's last token ---
        captured: dict[str, torch.Tensor] = {}

        def hook_k(module, inputs, output):
            # inputs[0] is c_proj's input tensor, shape (batch, seq, d_mlp)
            captured["k"] = inputs[0][0, last_subj_pos].detach().clone()

        def hook_v(module, inputs, output):
            # output of the MLP module, shape (batch, seq, d_model)
            captured["v"] = output[0, last_subj_pos].detach().clone()

        h_k = self.mlp.c_proj.register_forward_hook(hook_k)
        h_v = self.mlp.register_forward_hook(hook_v)
        self.model.eval()
        with torch.no_grad():
            _ = self.model(full_ids)
        h_k.remove()
        h_v.remove()

        k_star = captured["k"]    # (d_mlp,) e.g. 3072
        v_orig = captured["v"]    # (d_model,) e.g. 768

        # --- Step 2: Optimize delta_v ---
        delta_v = torch.zeros_like(v_orig, requires_grad=True)
        opt = torch.optim.Adam([delta_v], lr=self.v_lr)

        def inject_hook(module, inputs, output):
            out = output.clone()
            out[0, last_subj_pos] = out[0, last_subj_pos] + delta_v
            return out

        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100  # only compute loss on target tokens

        h_inject = self.mlp.register_forward_hook(inject_hook)
        try:
            for step in range(self.n_v_steps):
                opt.zero_grad()
                out = self.model(input_ids=full_ids, labels=labels)
                ce = out.loss
                # Weight-decay regularizer on delta_v (Meng et al.)
                reg = self.v_weight_decay * (delta_v.norm() ** 2) / (
                    v_orig.norm() ** 2 + 1e-8)
                loss = ce + reg
                loss.backward()
                opt.step()
                # Norm constraint: project delta_v inside ball of radius
                # norm_constraint * ||v_orig||
                with torch.no_grad():
                    max_norm = self.v_norm_constraint * v_orig.norm().item()
                    cur_norm = delta_v.norm().item()
                    if cur_norm > max_norm and cur_norm > 0:
                        delta_v.mul_(max_norm / cur_norm)
        finally:
            h_inject.remove()

        v_star = v_orig + delta_v.detach()

        # --- Step 3: Apply rank-one update to c_proj.weight ---
        # Conv1D: forward is x @ W + b; W has shape (in, out) = (d_mlp, d_model)
        # Currently W k_star + b gives some output y; we want it to equal v_star.
        # The rank-one update that achieves this with minimum Frobenius norm:
        #   W_new = W + outer(k_star, residual) / ||k_star||^2
        # where residual = v_star - (W k_star + b)
        W = self.mlp.c_proj.weight.data        # (d_mlp, d_model)
        b = self.mlp.c_proj.bias.data          # (d_model,)
        current_v = k_star @ W + b             # (d_model,)
        residual = v_star - current_v          # (d_model,)
        norm_sq = (k_star.pow(2).sum().item() + 1e-8)
        delta_W = torch.outer(k_star, residual) / norm_sq  # (d_mlp, d_model)
        W.add_(delta_W)


# ---------------------------------------------------------------------------
# Addressable Memory Layer — the proposed primitive
# ---------------------------------------------------------------------------
# Replaces MEMIT's destructive weight modification with constructive storage.
# Each fact gets its own (key, delta) slot. At forward time, the wrapped MLP
# computes its normal output and adds a memory contribution if the position's
# c_proj-input matches any stored key (cosine similarity above threshold).
#
# Properties this design has by construction:
#  (a) No interference: each fact has its own slot; existing slots are
#      untouched when a new one is added.
#  (b) Base model is never modified: the wrapped MLP's c_proj.weight is frozen;
#      memory acts purely additively.
#  (c) Top-1 hard selection: at most ONE slot fires per token. This makes the
#      retrieval discrete (a fact either applies or it doesn't), which is what
#      we want for stored facts.

import torch.nn.functional as F

class MLPWithMemory(nn.Module):
    """Drop-in replacement for a GPT-2 MLP module. Wraps the original MLP and
    adds an external (K, V) memory bank consulted on each forward pass.

    Forward:
        h = act(c_fc(x))                  # standard MLP intermediate
        base_out = c_proj(h)              # standard MLP output, BEFORE dropout
        if memory has slots:
            sims = cos(h, K)              # (batch, seq, n_slots) cosine sims
            best_sim, best_idx = sims.max # top-1 selection
            gate = (best_sim > threshold) # binary mask
            mem_out = V[best_idx] * gate
            out = base_out + mem_out
        else:
            out = base_out
        return dropout(out)
    """

    def __init__(self, base_mlp, n_slots_max: int = 2000,
                 d_mlp: int = 3072, d_model: int = 768,
                 sim_threshold: float = 0.7):
        super().__init__()
        self.base_mlp = base_mlp
        self.register_buffer("K", torch.zeros(n_slots_max, d_mlp))
        self.register_buffer("V", torch.zeros(n_slots_max, d_model))
        self.n_slots = 0
        self.sim_threshold = sim_threshold

    def forward(self, x):
        # Standard MLP path
        h = self.base_mlp.c_fc(x)
        h = self.base_mlp.act(h)                # (batch, seq, d_mlp)
        base_out = self.base_mlp.c_proj(h)       # (batch, seq, d_model)

        # Memory path: fire only at the LAST POSITION of the input sequence.
        # This avoids cross-slot pollution at upstream positions (which was
        # capping insertion at ~28% in the cosine-everywhere version). The
        # last position is where the next-token prediction is made, which is
        # the position the v* optimization was tuned for during insertion.
        if self.n_slots > 0:
            bsz, slen, _ = h.shape
            last_pos = slen - 1
            h_last = h[:, last_pos, :]                       # (bsz, d_mlp)
            K_active = self.K[:self.n_slots]                 # (n, d_mlp)
            V_active = self.V[:self.n_slots]                 # (n, d_model)
            h_norm = F.normalize(h_last, dim=-1)
            K_norm = F.normalize(K_active, dim=-1)
            sims = h_norm @ K_norm.T                          # (bsz, n)
            best_sim, best_idx = sims.max(dim=-1)             # each (bsz,)
            gate = (best_sim > self.sim_threshold).to(base_out.dtype)  # (bsz,)
            retrieved = V_active[best_idx]                    # (bsz, d_model)
            # Build a contribution tensor that is zero everywhere except
            # the last position. Avoids in-place ops on the forward output.
            contribution = torch.zeros_like(base_out)
            contribution[:, last_pos, :] = retrieved * gate.unsqueeze(-1)
            base_out = base_out + contribution

        return self.base_mlp.dropout(base_out)

    def add_slot(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if self.n_slots >= self.K.shape[0]:
            raise RuntimeError(f"Memory full ({self.n_slots} slots)")
        self.K[self.n_slots] = k
        self.V[self.n_slots] = v
        self.n_slots += 1


class AddressableMemoryMethod(MEMITMethod):
    """The proposed primitive. Inherits MEMIT's k*/v*-optimization machinery.
    Differs by replacing the rank-one weight update with storage in an
    addressable memory bank.

    Sub-design choice: the rewrite prompt uses the Q/A form (matching eval),
    because MEMIT's statement-form rewrite failed to generalize to Q/A
    retrieval in our Phase 1c results.
    """
    name = "addressable_mem"

    def __init__(self, layer_idx: int = 5, n_v_steps: int = 20,
                 v_lr: float = 0.5, v_weight_decay: float = 0.5,
                 v_norm_constraint: float = 4.0,
                 sim_threshold: float = 0.7, max_slots: int = 4000,
                 rewrite_form: str = "qa", n_templates: int = 2):
        super().__init__(layer_idx=layer_idx, n_v_steps=n_v_steps,
                          v_lr=v_lr, v_weight_decay=v_weight_decay,
                          v_norm_constraint=v_norm_constraint)
        self.sim_threshold = sim_threshold
        self.max_slots = max_slots
        if rewrite_form not in ("qa", "statement"):
            raise ValueError(f"rewrite_form must be 'qa' or 'statement', got {rewrite_form}")
        self.rewrite_form = rewrite_form
        if n_templates < 1:
            raise ValueError(f"n_templates must be >= 1, got {n_templates}")
        self.n_templates = n_templates

    def _build_rewrite(self, triple: Triple, q_idx: int = 0) -> tuple[str, str]:
        """For Q/A form, use the model's eval query format directly so the
        v*-optimization produces a fact representation that's retrievable via
        the same prompts the eval will use. q_idx selects which of the
        relation's query templates to use (0 or 1)."""
        if self.rewrite_form == "qa":
            rel = RELATIONS[triple.relation]
            n_tmpl = len(rel.query_templates)
            q_tmpl, _ = rel.query_templates[q_idx % n_tmpl]
            prompt = q_tmpl.format(s=triple.subject)
            target = " " + triple.obj
            return prompt, target
        return super()._build_rewrite(triple)

    def setup(self, model, tokenizer, kb):
        super().setup(model, tokenizer, kb)
        # Replace the target MLP with the memory-wrapped version
        block = model.transformer.h[self.layer_idx]
        original_mlp = block.mlp
        # GPT-2 Conv1D shapes: c_fc.weight (d_model, d_mlp), c_proj.weight (d_mlp, d_model)
        d_model = original_mlp.c_fc.weight.shape[0]
        d_mlp = original_mlp.c_fc.weight.shape[1]
        wrapped = MLPWithMemory(
            original_mlp, n_slots_max=self.max_slots,
            d_mlp=d_mlp, d_model=d_model, sim_threshold=self.sim_threshold,
        ).to(DEVICE)
        # Match the dtype of the existing MLP weights
        wrapped.K = wrapped.K.to(original_mlp.c_fc.weight.dtype)
        wrapped.V = wrapped.V.to(original_mlp.c_fc.weight.dtype)
        block.mlp = wrapped
        # For k*/v* capture in insert(), we still hook the BASE c_proj and the
        # wrapped MLP's full output respectively.
        self.mlp = wrapped.base_mlp          # base for k* capture
        self.memory_mlp = wrapped            # wrapped for v* capture + injection
        print(f"[baseline] AddressableMemory installed at layer {self.layer_idx}; "
              f"max slots: {self.max_slots}, sim threshold: {self.sim_threshold}, "
              f"rewrite form: {self.rewrite_form}, n_templates: {self.n_templates}")

    def _insert_one(self, triple: Triple, q_idx: int) -> None:
        """Insert a single (key, delta_v) slot for `triple` using rewrite
        template q_idx. Each call adds exactly one slot."""
        prompt, target = self._build_rewrite(triple, q_idx=q_idx)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        target_ids = self.tokenizer.encode(target, return_tensors="pt").to(DEVICE)
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)

        # Use the LAST PROMPT POSITION. The MLPWithMemory wrapper fires at the
        # last position of any input, so during eval the slot retrieval and the
        # delta_v we optimize here land at the SAME position.
        last_pos = prompt_ids.shape[1] - 1

        # --- Capture k* (from base c_proj input) and v_orig (wrapped MLP output) ---
        captured: dict[str, torch.Tensor] = {}

        def hook_k(module, inputs, output):
            captured["k"] = inputs[0][0, last_pos].detach().clone()

        def hook_v(module, inputs, output):
            captured["v"] = output[0, last_pos].detach().clone()

        h_k = self.mlp.c_proj.register_forward_hook(hook_k)
        h_v = self.memory_mlp.register_forward_hook(hook_v)
        self.model.eval()
        with torch.no_grad():
            _ = self.model(full_ids)
        h_k.remove()
        h_v.remove()
        k_star = captured["k"]
        v_orig = captured["v"]

        # --- Optimize delta_v ---
        delta_v = torch.zeros_like(v_orig, requires_grad=True)
        opt = torch.optim.Adam([delta_v], lr=self.v_lr)

        def inject_hook(module, inputs, output):
            out = output.clone()
            out[0, last_pos] = out[0, last_pos] + delta_v
            return out

        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100

        h_inject = self.memory_mlp.register_forward_hook(inject_hook)
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
                with torch.no_grad():
                    max_norm = self.v_norm_constraint * v_orig.norm().item()
                    cur_norm = delta_v.norm().item()
                    if cur_norm > max_norm and cur_norm > 0:
                        delta_v.mul_(max_norm / cur_norm)
        finally:
            h_inject.remove()

        self.memory_mlp.add_slot(k_star, delta_v.detach())

    def insert(self, triple: Triple) -> None:
        """Multi-prompt insertion: store one slot per query template.

        Eval queries each fact with BOTH templates; storing a slot per
        template ensures the cosine-key retrieval can fire on either form.
        Total slots = self.n_templates * len(insert_triples)."""
        for q_idx in range(self.n_templates):
            self._insert_one(triple, q_idx)


METHOD_REGISTRY: dict[str, type[Method]] = {
    "frozen": FrozenMethod,
    "in_context": InContextOracleMethod,  # deprecated: see module docstring
    "seq_ft": SequentialFTMethod,
    "lora_seq": LoRASequentialMethod,
    "memit": MEMITMethod,
    "addressable_mem": AddressableMemoryMethod,
}


# ---------------------------------------------------------------------------
# Eval-set builders (slicing by N)
# ---------------------------------------------------------------------------

def build_retention_eval(kb, n_holdout: int = 500, seed: int = 0
                          ) -> list[EvalExample]:
    """Reproduce the same retention sample pretrain.py used: shuffle pretrain
    with Random(seed), take first n_holdout. 2 query templates per triple."""
    import random
    all_pretrain = list(kb.pretrain_triples)
    random.Random(seed).shuffle(all_pretrain)
    retention_sample = all_pretrain[:n_holdout]
    out = []
    for t in retention_sample:
        for q_idx in range(2):
            prompt, target = render_eval_query(t, template_idx=q_idx)
            out.append(EvalExample(
                prompt=prompt, target=target, kind="retention",
                meta={"triple": t.as_tuple(), "q_idx": q_idx},
            ))
    return out


def build_insertion_eval(kb, n: int) -> list[EvalExample]:
    """Eval queries for the first n inserted triples. Empty if n==0."""
    out = []
    for t in kb.insert_triples[:n]:
        for q_idx in range(2):
            prompt, target = render_eval_query(t, template_idx=q_idx)
            out.append(EvalExample(
                prompt=prompt, target=target, kind="insertion",
                meta={"triple": t.as_tuple(), "q_idx": q_idx},
            ))
    return out


def build_composition_eval(kb, n: int) -> list[EvalExample]:
    """Compose pairs whose inserted-triple index is < n. We identify the
    insertion index by triple equality."""
    if n == 0:
        return []
    insert_set = {t.as_tuple() for t in kb.insert_triples[:n]}
    out = []
    for (t_pre, t_ins, qtext) in kb.compose_pairs:
        if t_ins.as_tuple() not in insert_set:
            continue
        prompt, target = render_composition(t_pre, t_ins, qtext)
        out.append(EvalExample(
            prompt=prompt, target=target, kind="composition",
            meta={"pretrain_triple": t_pre.as_tuple(),
                  "inserted_triple": t_ins.as_tuple()},
        ))
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_eval_at(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHOD_REGISTRY))
    ap.add_argument("--ckpt", default="checkpoints/pretrained_seed0_gpt2.pt")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_pretrain", type=int, default=2000)
    ap.add_argument("--n_insert", type=int, default=500)
    ap.add_argument("--n_compose", type=int, default=200)
    ap.add_argument("--n_holdout", type=int, default=500)
    ap.add_argument("--eval_at", default="0,1,10,50,100,250,500",
                    type=parse_eval_at,
                    help="comma-separated insertion checkpoints to eval at")
    ap.add_argument("--batch_size_eval", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="learning rate for methods that fine-tune (seq_ft, lora_seq)")
    ap.add_argument("--n_steps", type=int, default=5,
                    help="gradient steps per inserted fact (seq_ft, lora_seq)")
    ap.add_argument("--lora_rank", type=int, default=4,
                    help="LoRA rank (lora_seq only)")
    ap.add_argument("--lora_alpha", type=float, default=8.0,
                    help="LoRA alpha scaling (lora_seq only)")
    ap.add_argument("--memit_layer", type=int, default=5,
                    help="layer index to edit (memit/addressable_mem); GPT-2 small has 12 layers")
    ap.add_argument("--memit_v_steps", type=int, default=20,
                    help="gradient steps to optimize v* (memit/addressable_mem)")
    ap.add_argument("--memit_v_lr", type=float, default=0.5,
                    help="learning rate for v* optimization (memit/addressable_mem)")
    ap.add_argument("--mem_sim_threshold", type=float, default=0.7,
                    help="cosine-similarity threshold for memory retrieval (addressable_mem)")
    ap.add_argument("--mem_max_slots", type=int, default=4000,
                    help="max memory slots to pre-allocate (addressable_mem)")
    ap.add_argument("--mem_rewrite_form", default="qa", choices=["qa", "statement"],
                    help="rewrite prompt form (addressable_mem)")
    ap.add_argument("--mem_n_templates", type=int, default=2,
                    help="number of query templates per fact to store (1 or 2; addressable_mem)")
    ap.add_argument("--out", default=None,
                    help="output JSON path; default: results/baseline_<method>_seed<s>.json")
    args = ap.parse_args()

    print(f"[baseline] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[baseline] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[baseline] method: {args.method}  ckpt: {args.ckpt}")
    print(f"[baseline] eval_at: {args.eval_at}")

    # ---- KB ----
    kb = generate_kb(seed=args.seed,
                      n_pretrain=args.n_pretrain,
                      n_insert=args.n_insert,
                      n_compose=args.n_compose)
    print(f"[baseline] KB: pretrain={len(kb.pretrain_triples)}  "
          f"insert={len(kb.insert_triples)}  compose={len(kb.compose_pairs)}")
    n_max = max(args.eval_at)
    if n_max > len(kb.insert_triples):
        raise SystemExit(
            f"--eval_at requests N={n_max} but only {len(kb.insert_triples)} "
            f"insert triples are available. Reduce --eval_at or increase --n_insert.")

    # ---- model + tokenizer ----
    print(f"[baseline] loading {args.model} + checkpoint")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    print(f"[baseline] loaded checkpoint (reported retention={ckpt.get('retention_acc','?')})")

    # ---- method ----
    method_cls = METHOD_REGISTRY[args.method]
    if args.method == "seq_ft":
        method = method_cls(lr=args.lr, n_steps=args.n_steps)
        print(f"[baseline] seq_ft hyperparams: lr={args.lr}  n_steps={args.n_steps}")
    elif args.method == "lora_seq":
        method = method_cls(lr=args.lr, n_steps=args.n_steps,
                             rank=args.lora_rank, alpha=args.lora_alpha)
        print(f"[baseline] lora_seq hyperparams: lr={args.lr}  n_steps={args.n_steps}  "
              f"rank={args.lora_rank}  alpha={args.lora_alpha}")
    elif args.method == "memit":
        method = method_cls(layer_idx=args.memit_layer,
                             n_v_steps=args.memit_v_steps,
                             v_lr=args.memit_v_lr)
        print(f"[baseline] memit hyperparams: layer={args.memit_layer}  "
              f"v_steps={args.memit_v_steps}  v_lr={args.memit_v_lr}")
    elif args.method == "addressable_mem":
        method = method_cls(layer_idx=args.memit_layer,
                             n_v_steps=args.memit_v_steps,
                             v_lr=args.memit_v_lr,
                             sim_threshold=args.mem_sim_threshold,
                             max_slots=args.mem_max_slots,
                             rewrite_form=args.mem_rewrite_form,
                             n_templates=args.mem_n_templates)
        print(f"[baseline] addressable_mem hyperparams: layer={args.memit_layer}  "
              f"v_steps={args.memit_v_steps}  v_lr={args.memit_v_lr}  "
              f"sim_threshold={args.mem_sim_threshold}  rewrite_form={args.mem_rewrite_form}  "
              f"n_templates={args.mem_n_templates}")
    else:
        method = method_cls()
    method.setup(model, tokenizer, kb)

    # ---- precompute retention eval (it doesn't depend on N) ----
    retention_eval = build_retention_eval(kb, n_holdout=args.n_holdout,
                                            seed=args.seed)
    print(f"[baseline] retention eval examples: {len(retention_eval)}")

    # ---- main loop: walk through insertions, eval at anchors ----
    history = []
    eval_at_set = sorted(set(args.eval_at))
    next_eval_idx = 0

    def run_eval_at(n: int):
        """Run all three evals and record. n = number of insertions done."""
        t0 = time.time()
        ins_eval = build_insertion_eval(kb, n)
        com_eval = build_composition_eval(kb, n)
        ins_eval = method.transform_examples(ins_eval)
        ret_eval = method.transform_examples(retention_eval)
        com_eval = method.transform_examples(com_eval)

        # Run each separately so we can attribute timing/cost
        def _run(exs):
            if not exs: return {"n": 0, "accuracy": None}
            r = evaluate_model(model, tokenizer, exs,
                                max_new_tokens=12,
                                batch_size=args.batch_size_eval)
            return r["summary"][exs[0].kind]
        ins_summary = _run(ins_eval)
        ret_summary = _run(ret_eval)
        com_summary = _run(com_eval)
        dt = time.time() - t0
        rec = {
            "N": n,
            "insertion": ins_summary,
            "retention": ret_summary,
            "composition": com_summary,
            "eval_time_s": dt,
        }
        history.append(rec)
        ins_a = ins_summary.get("accuracy")
        ret_a = ret_summary.get("accuracy")
        com_a = com_summary.get("accuracy")
        def _fmt(x): return f"{x:.4f}" if isinstance(x, float) else " n/a "
        print(f"  N={n:<4}  Ins={_fmt(ins_a)} (n={ins_summary['n']})"
              f"   Ret={_fmt(ret_a)} (n={ret_summary['n']})"
              f"   Com={_fmt(com_a)} (n={com_summary['n']})"
              f"   [eval {dt:.1f}s]")

    # eval at N=0 (no insertions yet)
    if eval_at_set and eval_at_set[0] == 0:
        run_eval_at(0)
        next_eval_idx = 1

    # walk insertions
    print(f"\n[baseline] processing {len(kb.insert_triples)} insertions...")
    t_start = time.time()
    for i, triple in enumerate(kb.insert_triples):
        method.insert(triple)
        n_done = i + 1
        if next_eval_idx < len(eval_at_set) and n_done == eval_at_set[next_eval_idx]:
            run_eval_at(n_done)
            next_eval_idx += 1
    t_total = time.time() - t_start
    print(f"[baseline] insertion stream complete ({t_total:.1f}s total)")

    # ---- save ----
    out_path = (Path(args.out) if args.out
                else RESULTS_DIR / f"baseline_{args.method}_seed{args.seed}.json")
    out = {
        "method": args.method,
        "ckpt": args.ckpt,
        "seed": args.seed,
        "n_pretrain": args.n_pretrain,
        "n_insert": args.n_insert,
        "n_compose": args.n_compose,
        "eval_at": eval_at_set,
        "history": history,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[baseline] results -> {out_path}")


if __name__ == "__main__":
    main()
