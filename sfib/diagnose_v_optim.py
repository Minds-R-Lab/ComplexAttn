"""diagnose_v_optim.py — verify the v* optimization is actually moving delta_v.

Loads a backbone, does ONE addressable_mem-style insertion, and prints
delta_v.norm() + CE loss at each step. If delta_v stays at ~0, the optimization
isn't working (likely dtype precision issue).

Usage:
    python diagnose_v_optim.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --ckpt checkpoints/pretrained_seed0_Qwen__Qwen2.5_0.5B_Instruct.pt \
        --layer 17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kb_data import generate_kb, RELATIONS
from model_adapter import ModelAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--v_lr", type=float, default=0.5)
    ap.add_argument("--v_steps", type=int, default=20)
    ap.add_argument("--v_weight_decay", type=float, default=0.5,
                    help="L2 regularizer strength on delta_v / v_orig norm ratio")
    ap.add_argument("--v_norm_constraint", type=float, default=4.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force float32 explicitly
    print(f"[diag] loading {args.model} (forcing float32)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    # Belt-and-suspenders: explicitly cast the whole model.
    model = model.float()
    print(f"[diag] model dtype after loading: {next(model.parameters()).dtype}")

    # Load the pretrained checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    print(f"[diag] checkpoint loaded; reported retention={ckpt.get('retention_acc', '?'):.4f}")

    # Build one fact
    kb = generate_kb(seed=0, n_pretrain=2000, n_insert=500, n_compose=200)
    triple = kb.insert_triples[0]
    print(f"[diag] inserting first fact: {triple.as_tuple()}")

    rel = RELATIONS[triple.relation]
    q_tmpl, _ = rel.query_templates[0]
    prompt = q_tmpl.format(s=triple.subject)
    target = " " + triple.obj
    print(f"[diag] prompt: {prompt!r}")
    print(f"[diag] target: {target!r}")

    adapter = ModelAdapter.from_model(model)
    print(f"[diag] family: {adapter.family}  n_layers: {adapter.n_layers}  "
          f"hidden: {adapter.hidden_size}  intermediate: {adapter.intermediate_size}")
    mlp = adapter.get_mlp(args.layer)
    down_proj = adapter.get_down_proj(mlp)
    for p in model.parameters(): p.requires_grad = False

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    target_ids = tokenizer.encode(target, return_tensors="pt").to(device)
    full_ids = torch.cat([prompt_ids, target_ids], dim=1)
    last_pos = prompt_ids.shape[1] - 1
    print(f"[diag] prompt_len={prompt_ids.shape[1]}  target_len={target_ids.shape[1]}  last_pos={last_pos}")

    # Capture v_orig
    captured = {}
    def hook_v(module, inputs, output):
        captured["v"] = output[0, last_pos].detach().clone()
    h_v = mlp.register_forward_hook(hook_v)
    with torch.no_grad():
        _ = model(full_ids)
    h_v.remove()
    v_orig = captured["v"]
    print(f"[diag] v_orig dtype: {v_orig.dtype}  shape: {tuple(v_orig.shape)}  norm: {v_orig.norm().item():.4e}")

    # Optimize delta_v with prints
    delta_v = torch.zeros_like(v_orig, requires_grad=True)
    print(f"[diag] delta_v initial dtype: {delta_v.dtype}  norm: {delta_v.norm().item():.4e}")
    opt = torch.optim.Adam([delta_v], lr=args.v_lr)

    def inject_hook(module, inputs, output):
        out = output.clone()
        out[0, last_pos] = out[0, last_pos] + delta_v
        return out

    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    h_inject = mlp.register_forward_hook(inject_hook)
    print(f"\n[diag] running {args.v_steps} optimization steps (lr={args.v_lr}):")
    print(f"  {'step':>4}  {'ce_loss':>10}  {'delta_v.norm':>14}  {'grad.norm':>14}")
    try:
        for step in range(args.v_steps):
            opt.zero_grad()
            out = model(input_ids=full_ids, labels=labels)
            ce = out.loss
            reg = args.v_weight_decay * (delta_v.norm() ** 2) / (v_orig.norm() ** 2 + 1e-8)
            loss = ce + reg
            loss.backward()
            grad_norm = delta_v.grad.norm().item() if delta_v.grad is not None else float("nan")
            opt.step()
            with torch.no_grad():
                max_norm = args.v_norm_constraint * v_orig.norm().item()
                if delta_v.norm() > max_norm:
                    delta_v.mul_(max_norm / delta_v.norm())
            print(f"  {step:>4}  {ce.item():>10.4f}  {delta_v.norm().item():>14.4e}  {grad_norm:>14.4e}")
    finally:
        h_inject.remove()
    print(f"\n[diag] final delta_v.norm() / v_orig.norm() = {(delta_v.norm() / v_orig.norm()).item():.4f}")

    # Now test: with delta_v applied at last_pos, does the model predict the target?
    print(f"\n[diag] testing: do greedy generation with delta_v applied at last_pos")
    def test_hook(module, inputs, output):
        out = output.clone()
        # Use dynamic position because generate() uses KV cache:
        # prefill pass sees prompt_len tokens, then each new step sees length-1.
        # We fire at the last position of whatever the wrapped MLP sees, which
        # matches MLPWithMemory's behavior during actual eval.
        pos = out.shape[1] - 1
        out[0, pos] = out[0, pos] + delta_v.detach()
        return out
    h_test = mlp.register_forward_hook(test_hook)
    try:
        with torch.no_grad():
            gen = model.generate(prompt_ids, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        print(f"[diag] generated: {generated!r}")
        print(f"[diag] target:    {target!r}")
        print(f"[diag] match: {target.strip() in generated}")
    finally:
        h_test.remove()


if __name__ == "__main__":
    main()
