"""r2_composition_shard_7b.py -- Reviewer #2 point R2.3 SHARD at 7B.

The natural complement to the in-context oracle in r2_composition_oracle.py.
This script runs SHARD on Qwen2.5-7B with CounterFact composition prompts
that combine two real-world facts (e.g., "X was born in <city>" + "the
capital of <country> is <city>" => "X was born in the capital of?").

We DO NOT pretrain Qwen-7B on the SFIB synthetic corpus -- the SFIB pretrain
loop is GPT-2-only and pretraining 7B from scratch is impractical. Instead,
we use CounterFact's native composition pairs (when available) or synthesise
composition probes by combining two existing CounterFact edits that share an
entity. The output complements the in-context oracle: oracle high + SHARD
high means SHARD's retrieval works at 7B; oracle high + SHARD low isolates
the retrieval bottleneck; oracle low at 7B explains the SFIB Com@500 ~ 0.

Usage on the H100 box (you will likely need 80GB):
    cd sfib/
    python r2_composition_shard_7b.py \\
        --model Qwen/Qwen2.5-7B-Instruct --layer 20 \\
        --n_edits 500 --out results/composition_shard_7b.json

Expected wall time on a single H100-80GB: about 12-14 hr (the 7B
forward+backward in v* optim is the bottleneck).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ablations  # noqa: F401
import ablations_realdata_patches  # noqa: F401
from ablations import AblatedSHARDMethod, ABLATION_PRESETS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_counterfact(seed, n_edits):
    from counterfact_data import cf_splits
    return cf_splits(n_edits=n_edits, seed=seed)


def build_composition_probes(edits, max_pairs: int = 200):
    """Build composition probes from existing CounterFact edits.

    Each probe stitches two edits A and B by chaining their (subject,
    object) so that the answer to a two-hop query depends on both. This
    is necessarily approximate because CounterFact's relation structure
    is shallower than SFIB's, but it lets us measure whether SHARD's
    retrieval mechanism feeds correct facts to the backbone for chained
    reasoning at 7B."""
    probes = []
    for i, a in enumerate(edits):
        for j, b in enumerate(edits):
            if i == j:
                continue
            # Naive stitch: ask "What is <relation_b>(subject_a)?" assuming
            # a's target_new is b's subject. We rely on the runner to skip
            # incompatible pairs (those whose ground truth is undefined).
            if getattr(a, "target_new", None) and a.target_new == b.subject:
                probes.append({
                    "first_edit_idx": i,
                    "second_edit_idx": j,
                    "prompt": b.prompt_template.format(a.target_new),
                    "target": getattr(b, "target_new", ""),
                })
            if len(probes) >= max_pairs:
                return probes
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_compose_pairs", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[shard7b_comp] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[shard7b_comp] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[shard7b_comp] model={args.model}  layer={args.layer}")

    edits, holdout = load_counterfact(args.seed, args.n_edits)
    probes = build_composition_probes(edits, max_pairs=args.max_compose_pairs)
    print(f"[shard7b_comp] {len(edits)} edits, {len(probes)} composition probes")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(torch.float16 if DEVICE.type == "cuda" else torch.float32),
        device_map="auto",
    )
    model.eval()

    preset = ABLATION_PRESETS["shard"]
    method = AblatedSHARDMethod(
        layer_idx=args.layer, sim_threshold=args.tau,
        v_steps=200, v_lr=1.0,
        v_weight_decay=0.0, v_norm_constraint=20.0, eps_init=1.0,
        **preset,
    )
    method.setup(model, tokenizer, kb=None)

    print(f"[shard7b_comp] inserting {len(edits)} edits ...")
    t0 = time.time()
    for i, edit in enumerate(edits, start=1):
        method.insert(edit)
        if i % 50 == 0:
            print(f"   inserted {i}/{len(edits)}  ({(time.time()-t0)/60:.1f} min)")

    print(f"[shard7b_comp] evaluating composition probes ...")
    correct = 0
    per_probe = []
    for ex in probes:
        ids = tokenizer(ex["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[0, ids["input_ids"].shape[1]:]
        out_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        target = (ex.get("target") or "").strip()
        is_ok = bool(target) and target.lower() in out_text.lower()
        correct += int(is_ok)
        per_probe.append({
            "prompt": ex["prompt"], "target": target, "generated": out_text,
            "correct": is_ok,
            "first_edit_idx": ex["first_edit_idx"],
            "second_edit_idx": ex["second_edit_idx"],
        })

    com_at_500 = correct / max(1, len(probes))
    out = {
        "model": args.model, "layer": args.layer, "seed": args.seed,
        "n_edits": args.n_edits, "n_probes": len(probes),
        "tau": args.tau, "Com@500": com_at_500,
        "per_probe": per_probe,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[shard7b_comp] Com@500 = {com_at_500:.3f}  "
          f"(correct {correct}/{len(probes)})")
    print(f"[shard7b_comp] wrote {args.out}")


if __name__ == "__main__":
    main()
