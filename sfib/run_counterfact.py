"""run_counterfact.py — run the SFIB methods on CounterFact instead of synthetic data.

Same five methods (frozen, seq_ft, lora_seq, memit, addressable_mem),
applied to CounterFact's counterfactual-edit stream. The three CounterFact
metrics map onto our SFIB metrics:

  - Efficacy        \\approx Insertion@N   (does the edit take?)
  - Generalization  \\approx (paraphrase Insertion@N — new dimension)
  - Specificity     \\approx Retention@N   (do neighborhood facts survive?)

We evaluate at N \\in {0, 1, 10, 50, 100, 250, 500} edits.

Usage:
    python run_counterfact.py --method addressable_mem \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --memit_layer 17 --memit_v_steps 200 --memit_v_lr 1.0 \\
        --mem_v_weight_decay 0 --mem_v_norm_constraint 20 \\
        --mem_n_templates 1 \\
        --out results/counterfact_addressable_mem_qwen_seed0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from counterfact_data import load_counterfact, cf_splits, CounterFactTriple
from model_adapter import ModelAdapter
from evaluate import EvalExample, generate_greedy, _is_match

# Reuse the method classes — they accept any object with .subject/.relation/.obj
# and call _build_rewrite. We override _build_rewrite via monkeypatching.
import run_baselines as rb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Monkey-patch _build_rewrite on the method classes to handle CounterFactTriple
# ---------------------------------------------------------------------------

def _build_rewrite_cf(self, triple, q_idx: int = 0):
    """CounterFact-aware rewrite prompt builder."""
    if isinstance(triple, CounterFactTriple):
        prompt = triple.rewrite_prompt
        target = " " + triple.target_new
        return prompt, target
    # Fallback to the original SFIB logic if a non-CF triple is passed
    return _orig_build_rewrite(self, triple, q_idx=q_idx)

_orig_build_rewrite = rb.AddressableMemoryMethod._build_rewrite
rb.AddressableMemoryMethod._build_rewrite = _build_rewrite_cf

# Also patch MEMITMethod (it has its own _build_rewrite that we should override
# for CounterFact in case it's used). MEMIT subclasses use the same handler
# via inheritance from AddressableMemoryMethod, but MEMITMethod itself doesn't
# inherit from AddressableMemoryMethod — it's the parent. Patch both.
_orig_memit_build = rb.MEMITMethod._build_rewrite

def _build_rewrite_memit_cf(self, triple):
    if isinstance(triple, CounterFactTriple):
        prompt = triple.rewrite_prompt
        target = " " + triple.target_new
        return prompt, target
    return _orig_memit_build(self, triple)

rb.MEMITMethod._build_rewrite = _build_rewrite_memit_cf

# SequentialFTMethod uses _build_batch which goes through RELATIONS-based
# rendering. CounterFact triples don't have a relation enum. We need to
# override _build_batch.

_orig_seqft_build_batch = rb.SequentialFTMethod._build_batch

def _seqft_build_batch_cf(self, triple):
    if isinstance(triple, CounterFactTriple):
        # Just use the rewrite prompt + target. Optionally include a few
        # paraphrases of the target for surface-form robustness.
        prompt = triple.rewrite_prompt
        target = " " + triple.target_new
        text = f"{prompt}{target}"  # single training example
        enc = self.tokenizer([text + self.tokenizer.eos_token],
                              return_tensors="pt", truncation=True,
                              max_length=64, padding="max_length")
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc["attention_mask"].to(DEVICE)
        labels = input_ids.clone()
        labels[attn == 0] = -100
        return input_ids, attn, labels
    return _orig_seqft_build_batch(self, triple)

rb.SequentialFTMethod._build_batch = _seqft_build_batch_cf


# ---------------------------------------------------------------------------
# CounterFact-specific evaluation
# ---------------------------------------------------------------------------

def cf_eval(model, tokenizer, edits_seen: list[CounterFactTriple],
              holdout: list[CounterFactTriple],
              batch_size: int = 16, max_new_tokens: int = 12) -> dict:
    """Run all three CounterFact metrics over the given (edits, holdout) sets.

    Efficacy: greedy decode from each edit's rewrite_prompt; target_new
              must appear as a substring of the generated continuation.
    Generalization: same scoring on each edit's paraphrase_prompts.
    Specificity (Retention): on the held-out set, the model must still
              produce target_true (the pre-edit answer) on the rewrite
              prompt. (Holdout edits are NOT inserted; this measures whether
              the inserted facts leaked into unrelated subjects.)
    """
    # --- Efficacy (on inserted edits' rewrite prompts) ---
    eff_prompts = [e.rewrite_prompt for e in edits_seen]
    eff_targets = [e.target_new for e in edits_seen]
    eff_gens = generate_greedy(model, tokenizer, eff_prompts,
                                 max_new_tokens=max_new_tokens,
                                 batch_size=batch_size) if eff_prompts else []
    eff_correct = sum(_is_match(g, t) for g, t in zip(eff_gens, eff_targets))
    eff_n = len(eff_prompts)
    eff_acc = eff_correct / eff_n if eff_n else None

    # --- Generalization (on inserted edits' paraphrases) ---
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
        gen_acc = None
        gen_n = 0

    # --- Specificity (held-out facts must still produce target_true) ---
    # We use the rewrite_prompt of held-out edits and check that the model
    # still produces target_true (not target_new). This is the "did unrelated
    # facts get corrupted by the edit stream?" check.
    spec_prompts = [e.rewrite_prompt for e in holdout]
    spec_targets = [e.target_true for e in holdout]
    spec_gens = generate_greedy(model, tokenizer, spec_prompts,
                                  max_new_tokens=max_new_tokens,
                                  batch_size=batch_size) if spec_prompts else []
    spec_correct = sum(_is_match(g, t) for g, t in zip(spec_gens, spec_targets))
    spec_n = len(spec_prompts)
    spec_acc = spec_correct / spec_n if spec_n else None

    return {
        "efficacy":       {"n": eff_n,  "accuracy": eff_acc},
        "generalization": {"n": gen_n,  "accuracy": gen_acc},
        "specificity":    {"n": spec_n, "accuracy": spec_acc},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_eval_at(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(rb.METHOD_REGISTRY))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500,
                    help="number of CounterFact edits to insert")
    ap.add_argument("--n_holdout", type=int, default=500,
                    help="number of held-out CF examples for specificity")
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
                    help="delta_v optimizer for addressable_mem: vstar (200-step Adam), "
                         "lqr (one-shot saturated bang-bang), or lqr_gn (multi-step "
                         "Gauss-Newton LQR with --mem_n_lqr_iters re-linearization steps)")
    ap.add_argument("--mem_n_lqr_iters", type=int, default=10,
                    help="Number of Gauss-Newton LQR iterations; only used when "
                         "mem_value_optim=lqr_gn (default 10)")
    ap.add_argument("--mem_lqr_alpha_scale", type=float, default=1.0,
                    help="Scale on the LQR feedback gain (1.0 = saturate at the box boundary)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[cf] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[cf] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[cf] model: {args.model}")
    print(f"[cf] method: {args.method}")
    print(f"[cf] CounterFact: {args.n_edits} edits + {args.n_holdout} held-out specificity")

    # ---- Data ----
    edits, holdout = cf_splits(n_edits=args.n_edits, seed=args.seed)
    # Ensure holdout has enough; if cf_splits returned fewer than asked, trim
    holdout = holdout[:args.n_holdout]
    print(f"[cf] loaded {len(edits)} edits, {len(holdout)} held-out specificity")
    n_max = max(args.eval_at)
    if n_max > len(edits):
        raise SystemExit(f"--eval_at requests N={n_max} but only "
                          f"{len(edits)} edits available")

    # ---- Model ----
    print(f"[cf] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                   torch_dtype=torch.float32).to(DEVICE)
    model = model.float()
    model.eval()
    adapter = ModelAdapter.from_model(model)
    print(f"[cf] family={adapter.family}  n_layers={adapter.n_layers}  "
          f"hidden={adapter.hidden_size}  intermediate={adapter.intermediate_size}")

    # ---- Method ----
    cls = rb.METHOD_REGISTRY[args.method]
    if args.method == "seq_ft":
        method = cls(lr=args.lr, n_steps=args.n_steps)
        print(f"[cf] seq_ft: lr={args.lr}  n_steps={args.n_steps}")
    elif args.method == "lora_seq":
        method = cls(lr=args.lr, n_steps=args.n_steps,
                      rank=args.lora_rank, alpha=args.lora_alpha)
        print(f"[cf] lora_seq: lr={args.lr}  n_steps={args.n_steps}  "
              f"rank={args.lora_rank}  alpha={args.lora_alpha}")
    elif args.method == "memit":
        method = cls(layer_idx=args.memit_layer,
                      n_v_steps=args.memit_v_steps, v_lr=args.memit_v_lr)
        print(f"[cf] memit: layer={args.memit_layer}  v_steps={args.memit_v_steps}  v_lr={args.memit_v_lr}")
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
        print(f"[cf] addressable_mem: layer={args.memit_layer}  "
              f"v_steps={args.memit_v_steps}  v_lr={args.memit_v_lr}  "
              f"v_wd={args.mem_v_weight_decay}  v_norm_cap={args.mem_v_norm_constraint}  "
              f"sim_threshold={args.mem_sim_threshold}  n_templates={args.mem_n_templates}  "
              f"value_optim={args.mem_value_optim}  n_lqr_iters={args.mem_n_lqr_iters}")
    else:
        method = cls()

    # The Method.setup signature is (model, tokenizer, kb). We don't have a kb
    # in the CounterFact world — pass None and the setup methods that don't
    # need it (everything except the in_context method) work fine.
    method.setup(model, tokenizer, kb=None)

    # ---- Main loop ----
    history = []
    eval_at_set = sorted(set(args.eval_at))
    next_eval_idx = 0

    def run_eval(n: int):
        t0 = time.time()
        result = cf_eval(model, tokenizer, edits_seen=edits[:n],
                          holdout=holdout,
                          batch_size=args.batch_size_eval)
        dt = time.time() - t0
        rec = {"N": n, **result, "eval_time_s": dt}
        history.append(rec)
        def f(x): return f"{x:.4f}" if isinstance(x, float) else " n/a "
        print(f"  N={n:<4}  Eff={f(result['efficacy']['accuracy'])} (n={result['efficacy']['n']})"
              f"   Gen={f(result['generalization']['accuracy'])} (n={result['generalization']['n']})"
              f"   Spec={f(result['specificity']['accuracy'])} (n={result['specificity']['n']})"
              f"   [eval {dt:.1f}s]")

    # Eval at N=0 (no edits yet)
    if eval_at_set and eval_at_set[0] == 0:
        run_eval(0)
        next_eval_idx = 1

    print(f"\n[cf] processing {len(edits)} edits...")
    t_start = time.time()
    for i, edit in enumerate(edits):
        method.insert(edit)
        n_done = i + 1
        if next_eval_idx < len(eval_at_set) and n_done == eval_at_set[next_eval_idx]:
            run_eval(n_done)
            next_eval_idx += 1
    t_total = time.time() - t_start
    print(f"[cf] insertion stream complete ({t_total:.1f}s)")

    out_path = (Path(args.out) if args.out
                 else RESULTS_DIR / f"counterfact_{args.method}_{args.model.replace('/','__').replace('-','_')}_seed{args.seed}.json")
    out = {
        "benchmark": "counterfact",
        "method": args.method,
        "model": args.model,
        "seed": args.seed,
        "n_edits": args.n_edits,
        "n_holdout": args.n_holdout,
        "eval_at": eval_at_set,
        "history": history,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[cf] results -> {out_path}")


if __name__ == "__main__":
    main()
