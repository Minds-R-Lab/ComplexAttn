"""r3_distance_distributions.py -- Reviewer #3 point R3.8 (first half).

Computes cosine similarity and Euclidean distance distributions for the
four query classes the reviewer asks for:

  (a) own rewrite prompt          -- the prompt the slot was stored on
  (b) own paraphrases             -- alternative phrasings of (a)
  (c) sibling specificity prompts -- other CounterFact entities (held-out)
  (d) random unrelated prompts    -- sampled from held-out + shuffled

For each stored slot, we capture the wrapped MLP's intermediate key at
the last prompt position for prompts of each class, then compute
cosine(stored_key, captured_key) and Euclidean(stored_key, captured_key).

The output JSON has raw arrays per class (for the histogram / KDE plot
in the manuscript figure) plus summary statistics (mean / std / quintiles).

Usage on the H100 box:
    cd sfib/
    python r3_distance_distributions.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --layer 17 --n_edits 500 \\
        --out results/distance_distributions.json

Expected wall time on a single H100-80GB: about 45 minutes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import ablations  # noqa: F401
import ablations_realdata_patches  # noqa: F401
from ablations import AblatedSHARDMethod, ABLATION_PRESETS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def cosine(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0),
                               b.flatten().unsqueeze(0)).item()


def euclidean(a, b):
    return torch.linalg.norm(a.flatten() - b.flatten()).item()


def capture_key_for_prompt(model, tokenizer, wrapped_mlp, prompt: str):
    """Run a forward pass on `prompt` and return the wrapped MLP's last-token
    intermediate key. Uses the same capture path the routing decision uses."""
    ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    captured = {}

    def hook(module, inputs, output):
        # AblatedSHARDWrapper.forward stores its current intermediate on the
        # module; we hook it to read the last-token intermediate directly.
        if hasattr(module, "_last_key"):
            captured["k"] = module._last_key.detach().clone()
        else:
            # Fallback: take the input to c_proj for GPT-2-style MLPs
            captured["k"] = inputs[0][0, -1].detach().clone()

    h = wrapped_mlp.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(**ids)
    finally:
        h.remove()
    return captured.get("k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--benchmark", choices=["counterfact"], default="counterfact")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--n_unrelated", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[dist] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[dist] GPU: {torch.cuda.get_device_name(0)}")

    from counterfact_data import cf_splits
    edits, holdout = cf_splits(n_edits=args.n_edits, seed=args.seed)
    # The "unrelated" set is sampled from the held-out CF entities; their
    # prompts ask about entirely different (subject, relation) pairs.
    unrelated_pool = list(holdout)
    random.shuffle(unrelated_pool)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    preset = ABLATION_PRESETS["shard"]
    method = AblatedSHARDMethod(
        layer_idx=args.layer, sim_threshold=0.7,
        v_steps=200, v_lr=1.0,
        v_weight_decay=0.0, v_norm_constraint=20.0, eps_init=1.0,
        **preset,
    )
    method.setup(model, tokenizer, kb=None)

    print(f"[dist] inserting {len(edits)} slots ...")
    for edit in edits:
        method.insert(edit)
    print(f"[dist] bank populated")

    # The wrapped MLP holds the stored keys
    wrapped_mlp = method._get_mlp_and_block(model)[0]

    series = {
        "cos_own_rewrite":  [], "euc_own_rewrite":  [],
        "cos_paraphrase":   [], "euc_paraphrase":   [],
        "cos_specificity":  [], "euc_specificity":  [],
        "cos_unrelated":    [], "euc_unrelated":    [],
    }

    for i, edit in enumerate(edits):
        # The stored key for slot i lives in wrapped_mlp.K[i]
        try:
            k_stored = wrapped_mlp.K[i].detach().to(DEVICE)
        except (AttributeError, IndexError):
            k_stored = wrapped_mlp.keys[i].detach().to(DEVICE)

        # (a) own rewrite
        prompt = edit.prompt_template.format(edit.subject)
        k = capture_key_for_prompt(model, tokenizer, wrapped_mlp, prompt)
        if k is not None:
            series["cos_own_rewrite"].append(cosine(k_stored, k))
            series["euc_own_rewrite"].append(euclidean(k_stored, k))

        # (b) own paraphrases (CounterFactTriple exposes them as paraphrases)
        for p_prompt in getattr(edit, "paraphrase_prompts", []) or []:
            k = capture_key_for_prompt(model, tokenizer, wrapped_mlp, p_prompt)
            if k is not None:
                series["cos_paraphrase"].append(cosine(k_stored, k))
                series["euc_paraphrase"].append(euclidean(k_stored, k))

        # (c) specificity prompts (different entities, same dataset)
        for s_prompt in getattr(edit, "neighborhood_prompts", []) or []:
            k = capture_key_for_prompt(model, tokenizer, wrapped_mlp, s_prompt)
            if k is not None:
                series["cos_specificity"].append(cosine(k_stored, k))
                series["euc_specificity"].append(euclidean(k_stored, k))

        # (d) unrelated prompts (drawn from the held-out CF tail)
        for u_edit in random.sample(unrelated_pool, 4):
            u_prompt = u_edit.prompt_template.format(u_edit.subject)
            k = capture_key_for_prompt(model, tokenizer, wrapped_mlp, u_prompt)
            if k is not None:
                series["cos_unrelated"].append(cosine(k_stored, k))
                series["euc_unrelated"].append(euclidean(k_stored, k))

        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{len(edits)} slots processed")

    summary = {}
    for cls, v in series.items():
        a = np.asarray(v, dtype=float)
        summary[cls] = {
            "n": int(a.size),
            "mean": float(a.mean()) if a.size else None,
            "std":  float(a.std()) if a.size else None,
            "p05":  float(np.percentile(a, 5)) if a.size else None,
            "p50":  float(np.percentile(a, 50)) if a.size else None,
            "p95":  float(np.percentile(a, 95)) if a.size else None,
        }

    with open(args.out, "w") as f:
        json.dump({
            "model": args.model, "layer": args.layer, "seed": args.seed,
            "n_edits": args.n_edits,
            "raw": series, "summary": summary,
        }, f, indent=2)
    print(f"[dist] wrote {args.out}")
    for cls, s in summary.items():
        if s["mean"] is None:
            continue
        print(f"   {cls:>18}  n={s['n']:>5}  mean={s['mean']:+.3f}  std={s['std']:.3f}  "
              f"(p05={s['p05']:+.2f}, p95={s['p95']:+.2f})")


if __name__ == "__main__":
    main()
