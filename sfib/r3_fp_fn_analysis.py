"""r3_fp_fn_analysis.py -- Reviewer #3 point R3.8 (second half).

For every probe query the wrapped MLP sees at eval time, record:
  - the wrapped MLP's routing decision (slot index fired, or none)
  - the cosine value of the winning slot
  - the ground-truth target class:
        own_rewrite        (should fire its own slot)
        own_paraphrase     (should fire its own slot)
        sibling_specificity (CounterFact neighborhood; should NOT fire)
        unrelated          (held-out CF; should NOT fire)

The probe writes a per-(class, cosine bucket) firing table plus FP and FN
counts. The output JSON is what the small companion table in the
manuscript's Section sec:why-works will report.

Usage on the H100 box:
    cd sfib/
    python r3_fp_fn_analysis.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --layer 17 --n_edits 500 \\
        --out results/fp_fn_analysis.json

Expected wall time on a single H100-80GB: about 50 minutes.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

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

BUCKETS = [
    (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
    (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90),
    (0.90, 0.95), (0.95, 1.01),
]


def which_bucket(cos: float) -> int | None:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= cos < hi:
            return i
    return None


def best_match(wrapped_mlp, k_query, tau):
    """Return (slot_idx, cosine_of_winner, fired_bool) for one query key."""
    try:
        K = wrapped_mlp.K
        n = wrapped_mlp.n_slots
    except AttributeError:
        K = torch.stack(wrapped_mlp.keys)
        n = len(wrapped_mlp.keys)
    if n == 0:
        return None, 0.0, False
    K = K[:n].to(DEVICE)
    q = k_query.flatten().to(DEVICE)
    sims = F.cosine_similarity(q.unsqueeze(0), K.view(n, -1), dim=1)
    cos, idx = sims.max(dim=0)
    cos_val = cos.item()
    return int(idx.item()), cos_val, (cos_val >= tau)


def capture_key(model, tokenizer, wrapped_mlp, prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    captured = {}

    def hook(module, inputs, output):
        if hasattr(module, "_last_key"):
            captured["k"] = module._last_key.detach().clone()
        else:
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[fp_fn] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[fp_fn] GPU: {torch.cuda.get_device_name(0)}")

    from counterfact_data import cf_splits
    edits, holdout = cf_splits(n_edits=args.n_edits, seed=args.seed)
    unrelated_pool = list(holdout)
    random.shuffle(unrelated_pool)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    preset = ABLATION_PRESETS["shard"]
    method = AblatedSHARDMethod(
        layer_idx=args.layer, sim_threshold=args.tau,
        v_steps=200, v_lr=1.0,
        v_weight_decay=0.0, v_norm_constraint=20.0, eps_init=1.0,
        **preset,
    )
    method.setup(model, tokenizer, kb=None)

    print(f"[fp_fn] inserting {len(edits)} slots ...")
    for edit in edits:
        method.insert(edit)
    print(f"[fp_fn] bank populated")

    wrapped_mlp = method._get_mlp_and_block(model)[0]

    # firings[class][bucket] = count of firings landing in that bucket
    firings = defaultdict(lambda: Counter())
    fp = Counter()
    fn = Counter()
    n_per_class = Counter()

    def probe(prompt, cls, expected_slot=None):
        n_per_class[cls] += 1
        k = capture_key(model, tokenizer, wrapped_mlp, prompt)
        if k is None:
            return
        idx, cos, fired = best_match(wrapped_mlp, k, args.tau)
        if fired:
            b = which_bucket(cos)
            if b is None:
                b = len(BUCKETS) - 1
            firings[cls][b] += 1
            if cls in ("own_rewrite", "own_paraphrase"):
                if idx != expected_slot:
                    fp[f"wrong_slot_on_{cls}"] += 1
                    fn[cls] += 1
            else:
                # specificity or unrelated -- any firing is a false positive
                fp[cls] += 1
        else:
            if cls in ("own_rewrite", "own_paraphrase"):
                fn[cls] += 1

    for i, edit in enumerate(edits):
        probe(edit.prompt_template.format(edit.subject),
              "own_rewrite", expected_slot=i)
        for pp in getattr(edit, "paraphrase_prompts", []) or []:
            probe(pp, "own_paraphrase", expected_slot=i)
        for sp in getattr(edit, "neighborhood_prompts", []) or []:
            probe(sp, "sibling_specificity")
        for u_edit in random.sample(unrelated_pool, 4):
            u_prompt = u_edit.prompt_template.format(u_edit.subject)
            probe(u_prompt, "unrelated")
        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{len(edits)} probes done")

    table = {cls: [int(firings[cls][b]) for b in range(len(BUCKETS))]
             for cls in firings}

    with open(args.out, "w") as f:
        json.dump({
            "model": args.model, "layer": args.layer, "seed": args.seed,
            "tau": args.tau, "n_edits": args.n_edits,
            "buckets": BUCKETS,
            "n_per_class": dict(n_per_class),
            "firings_by_class_and_bucket": table,
            "false_positives": dict(fp),
            "false_negatives": dict(fn),
        }, f, indent=2)
    print(f"\n[fp_fn] wrote {args.out}")
    print(f"false positives: {dict(fp)}")
    print(f"false negatives: {dict(fn)}")


if __name__ == "__main__":
    main()
