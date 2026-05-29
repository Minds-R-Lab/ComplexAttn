"""run_zsre.py — run the SFIB methods on zsRE.

Mirror of run_counterfact.py with zsRE's data loader and the locality-style
specificity metric. zsRE's "locality" prompt is about an unrelated entity
with a known answer; specificity = % of locality prompts where the model
still produces the pre-edit answer.

Usage:
    python run_zsre.py --method addressable_mem \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --memit_layer 17 --memit_v_steps 200 --memit_v_lr 1.0 \\
        --mem_v_weight_decay 0 --mem_v_norm_constraint 20 \\
        --mem_n_templates 1 \\
        --out results/zsre_addressable_mem_qwen_seed0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from zsre_data import load_zsre, zsre_splits, ZsreTriple
from model_adapter import ModelAdapter
from evaluate import generate_greedy, _is_match
import run_baselines as rb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Monkey-patch _build_rewrite to handle ZsreTriple (same shape as CounterFact)
def _build_rewrite_zsre(self, triple, q_idx: int = 0):
    if isinstance(triple, ZsreTriple):
        prompt = triple.rewrite_prompt
        target = " " + triple.target_new
        return prompt, target
    return _orig_build_rewrite(self, triple, q_idx=q_idx)

_orig_build_rewrite = rb.AddressableMemoryMethod._build_rewrite
rb.AddressableMemoryMethod._build_rewrite = _build_rewrite_zsre

_orig_memit_build = rb.MEMITMethod._build_rewrite
def _build_rewrite_memit_zsre(self, triple):
    if isinstance(triple, ZsreTriple):
        prompt = triple.rewrite_prompt
        target = " " + triple.target_new
        return prompt, target
    return _orig_memit_build(self, triple)
rb.MEMITMethod._build_rewrite = _build_rewrite_memit_zsre

_orig_seqft_build_batch = rb.SequentialFTMethod._build_batch
def _seqft_build_batch_zsre(self, triple):
    if isinstance(triple, ZsreTriple):
        prompt = triple.rewrite_prompt
        target = " " + triple.target_new
        text = f"{prompt}{target}"
        enc = self.tokenizer([text + self.tokenizer.eos_token],
                              return_tensors="pt", truncation=True,
                              max_length=64, padding="max_length")
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc["attention_mask"].to(DEVICE)
        labels = input_ids.clone()
        labels[attn == 0] = -100
        return input_ids, attn, labels
    return _orig_seqft_build_batch(self, triple)
rb.SequentialFTMethod._build_batch = _seqft_build_batch_zsre


def zsre_eval(model, tokenizer, edits_seen, holdout,
                batch_size: int = 16, max_new_tokens: int = 12) -> dict:
    """Three zsRE metrics: efficacy, generalization (paraphrase), specificity (locality)."""

    # Efficacy
    eff_prompts = [e.rewrite_prompt for e in edits_seen]
    eff_targets = [e.target_new for e in edits_seen]
    eff_gens = generate_greedy(model, tokenizer, eff_prompts,
                                 max_new_tokens=max_new_tokens,
                                 batch_size=batch_size) if eff_prompts else []
    eff_correct = sum(_is_match(g, t) for g, t in zip(eff_gens, eff_targets))
    eff_n = len(eff_prompts)
    eff_acc = eff_correct / eff_n if eff_n else None

    # Generalization (on paraphrase prompts)
    gen_pairs = [(p, e.target_new) for e in edits_seen
                  for p in e.paraphrase_prompts]
    if gen_pairs:
        gen_prompts, gen_targets = zip(*gen_pairs)
        gen_gens = generate_greedy(model, tokenizer, list(gen_prompts),
                                     max_new_tokens=max_new_tokens,
                                     batch_size=batch_size)
        gen_correct = sum(_is_match(g, t) for g, t in zip(gen_gens, gen_targets))
        gen_n = len(gen_pairs)
        gen_acc = gen_correct / gen_n
    else:
        gen_acc = None; gen_n = 0

    # Specificity (locality)
    # For zsRE, specificity is whether the model preserves its answer on
    # unrelated locality prompts. The locality prompt has its own expected
    # answer (loc_ans). For records without locality info, we fall back
    # to checking held-out edits' rewrite prompts produce target_true.
    spec_records = []
    for e in holdout:
        if e.locality_prompt and e.locality_answer:
            spec_records.append((e.locality_prompt, e.locality_answer))
        else:
            spec_records.append((e.rewrite_prompt, e.target_true))
    if spec_records:
        spec_prompts, spec_targets = zip(*spec_records)
        spec_gens = generate_greedy(model, tokenizer, list(spec_prompts),
                                      max_new_tokens=max_new_tokens,
                                      batch_size=batch_size)
        spec_correct = sum(_is_match(g, t) for g, t in zip(spec_gens, spec_targets))
        spec_n = len(spec_records)
        spec_acc = spec_correct / spec_n
    else:
        spec_acc = None; spec_n = 0

    return {
        "efficacy":       {"n": eff_n,  "accuracy": eff_acc},
        "generalization": {"n": gen_n,  "accuracy": gen_acc},
        "specificity":    {"n": spec_n, "accuracy": spec_acc},
    }


def parse_eval_at(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(rb.METHOD_REGISTRY))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--n_holdout", type=int, default=500)
    ap.add_argument("--eval_at", default="0,1,10,50,100,250,500",
                    type=parse_eval_at)
    ap.add_argument("--batch_size_eval", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_steps", type=int, default=5)
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--lora_alpha", type=float, default=8.0)
    ap.add_argument("--memit_layer", type=int, default=5)
    ap.add_argument("--memit_v_steps", type=int, default=20)
    ap.add_argument("--memit_v_lr", type=float, default=0.5)
    ap.add_argument("--mem_sim_threshold", type=float, default=0.7)
    ap.add_argument("--mem_max_slots", type=int, default=8000)
    ap.add_argument("--mem_rewrite_form", default="qa", choices=["qa", "statement"])
    ap.add_argument("--mem_n_templates", type=int, default=1)
    ap.add_argument("--mem_v_weight_decay", type=float, default=0.5)
    ap.add_argument("--mem_v_norm_constraint", type=float, default=4.0)
    ap.add_argument("--mem_value_optim", choices=["vstar", "lqr", "lqr_gn"], default="vstar",
                    help="delta_v optimizer for addressable_mem: vstar / lqr / lqr_gn")
    ap.add_argument("--mem_n_lqr_iters", type=int, default=10,
                    help="Gauss-Newton LQR iterations; only used when mem_value_optim=lqr_gn (default 10)")
    ap.add_argument("--mem_lqr_alpha_scale", type=float, default=1.0,
                    help="Scale on the LQR feedback gain (1.0 = saturate at the box boundary)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[zsre] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[zsre] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[zsre] model: {args.model}  method: {args.method}")

    edits, holdout = zsre_splits(n_edits=args.n_edits, seed=args.seed)
    holdout = holdout[:args.n_holdout]
    print(f"[zsre] loaded {len(edits)} edits, {len(holdout)} held-out specificity")
    if max(args.eval_at) > len(edits):
        raise SystemExit(f"--eval_at requests N={max(args.eval_at)} > {len(edits)} edits")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                   torch_dtype=torch.float32).to(DEVICE)
    model = model.float()
    model.eval()
    adapter = ModelAdapter.from_model(model)
    print(f"[zsre] family={adapter.family}  n_layers={adapter.n_layers}  "
          f"hidden={adapter.hidden_size}  intermediate={adapter.intermediate_size}")

    cls = rb.METHOD_REGISTRY[args.method]
    if args.method == "seq_ft":
        method = cls(lr=args.lr, n_steps=args.n_steps)
    elif args.method == "lora_seq":
        method = cls(lr=args.lr, n_steps=args.n_steps,
                      rank=args.lora_rank, alpha=args.lora_alpha)
    elif args.method == "memit":
        method = cls(layer_idx=args.memit_layer,
                      n_v_steps=args.memit_v_steps, v_lr=args.memit_v_lr)
    elif args.method == "addressable_mem":
        method = cls(layer_idx=args.memit_layer,
                      n_v_steps=args.memit_v_steps, v_lr=args.memit_v_lr,
                      v_weight_decay=args.mem_v_weight_decay,
                      v_norm_constraint=args.mem_v_norm_constraint,
                      sim_threshold=args.mem_sim_threshold,
                      max_slots=args.mem_max_slots,
                      rewrite_form=args.mem_rewrite_form,
                      n_templates=args.mem_n_templates,
                      value_optim=args.mem_value_optim,
                      n_lqr_iters=args.mem_n_lqr_iters,
                      lqr_alpha_scale=args.mem_lqr_alpha_scale)
    else:
        method = cls()
    method.setup(model, tokenizer, kb=None)

    history = []
    eval_at_set = sorted(set(args.eval_at))
    next_eval_idx = 0

    def run_eval(n):
        t0 = time.time()
        result = zsre_eval(model, tokenizer, edits_seen=edits[:n],
                             holdout=holdout, batch_size=args.batch_size_eval)
        dt = time.time() - t0
        rec = {"N": n, **result, "eval_time_s": dt}
        history.append(rec)
        def f(x): return f"{x:.4f}" if isinstance(x, float) else " n/a "
        print(f"  N={n:<4}  Eff={f(result['efficacy']['accuracy'])} (n={result['efficacy']['n']})"
              f"   Gen={f(result['generalization']['accuracy'])} (n={result['generalization']['n']})"
              f"   Spec={f(result['specificity']['accuracy'])} (n={result['specificity']['n']})"
              f"   [eval {dt:.1f}s]")

    if eval_at_set and eval_at_set[0] == 0:
        run_eval(0); next_eval_idx = 1

    print(f"\n[zsre] processing {len(edits)} edits...")
    t_start = time.time()
    for i, edit in enumerate(edits):
        method.insert(edit)
        n_done = i + 1
        if next_eval_idx < len(eval_at_set) and n_done == eval_at_set[next_eval_idx]:
            run_eval(n_done); next_eval_idx += 1
    print(f"[zsre] insertion stream complete ({time.time()-t_start:.1f}s)")

    out_path = (Path(args.out) if args.out
                 else RESULTS_DIR / f"zsre_{args.method}_{args.model.replace('/','__').replace('-','_')}_seed{args.seed}.json")
    out = {
        "benchmark": "zsre", "method": args.method, "model": args.model,
        "seed": args.seed, "n_edits": args.n_edits,
        "n_holdout": args.n_holdout, "eval_at": eval_at_set,
        "history": history,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[zsre] results -> {out_path}")


if __name__ == "__main__":
    main()
