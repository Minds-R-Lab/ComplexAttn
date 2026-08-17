"""r4_ablate_l2_euclidean.py -- Reviewer #4 point 2 control.

For unit vectors,  ||x - y||^2 = 2 - 2 cos(x, y),  so L2-normalised
Euclidean distance and cosine similarity are equivalent up to a monotone
transformation. If SHARD's ablate_routing collapse (Table 10 in R1)
really shows that cosine is "a fundamentally new routing principle",
then L2-normalised Euclidean should behave like `shard` (matching Gen and
Spec), not like `ablate_routing` (which collapses Gen).

If instead the observed effect is just that activation magnitude is a
nuisance variable, `ablate_routing_l2` will also match `shard` and the
paper simplifies its story to "normalise before you route" -- an equally
publishable framing.

We run the same two cells the R1 ablation used:
    (i)  CounterFact + Qwen2.5-0.5B, layer 17
    (ii) zsRE       + TinyLlama-1.1B, layer 15
so the new row drops straight into Table 10 with no re-tuning.

Usage on H100:
    cd sfib/
    python r4_ablate_l2_euclidean.py \\
        --cell cf_qwen0_5b   --out results/ablate_l2_cf_qwen0_5b.json
    python r4_ablate_l2_euclidean.py \\
        --cell zsre_tinyllama --out results/ablate_l2_zsre_tinyllama.json
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

CELLS = {
    "cf_qwen0_5b": {
        "benchmark": "counterfact",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "layer": 17,
    },
    "zsre_tinyllama": {
        "benchmark": "zsre",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "layer": 15,
    },
}


def load_data(bench: str, seed: int, n_edits: int):
    if bench == "counterfact":
        from counterfact_data import cf_splits
        return cf_splits(n_edits=n_edits, seed=seed)
    else:
        from zsre_data import zsre_splits
        return zsre_splits(n_edits=n_edits, seed=seed)


def eval_cell(bench: str, model, tokenizer, edits, holdout, tau, layer):
    if bench == "counterfact":
        from run_counterfact import cf_eval
        return cf_eval(model, tokenizer, edits, holdout)
    else:
        from run_zsre import zsre_eval
        return zsre_eval(model, tokenizer, edits, holdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELLS.keys()), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.7,
                    help="Cosine threshold for `shard`, or L2 radius for "
                         "`ablate_routing_l2` (default 0.7 mirrors the R1 "
                         "cosine operating point; the equivalent L2 radius "
                         "on the unit sphere is sqrt(2-2*0.7) ~ 0.775, "
                         "which we also sweep as a sanity check).")
    ap.add_argument("--eps_l2", type=float, default=None,
                    help="Override the L2 radius. Default = sqrt(2 - 2*tau).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cell = CELLS[args.cell]
    bench, model_id, layer = cell["benchmark"], cell["model"], cell["layer"]

    import math
    eps_l2 = args.eps_l2
    if eps_l2 is None:
        eps_l2 = math.sqrt(max(0.0, 2.0 - 2.0 * args.tau))

    print(f"[r4_l2] device: {DEVICE}")
    print(f"[r4_l2] cell={args.cell}  model={model_id}  layer={layer}")
    print(f"[r4_l2] tau={args.tau}  eps_l2={eps_l2:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    edits, holdout = load_data(bench, args.seed, args.n_edits)
    print(f"[r4_l2] {len(edits)} edits, {len(holdout)} holdout")

    # Three routing variants against a single model reload each, so
    # nothing leaks across variants.
    variants = [
        ("shard",             {"routing": "cosine",       "sim_threshold": args.tau, "eps_init": 1.0}),
        ("ablate_routing",    {"routing": "euclidean",    "sim_threshold": 0.0,      "eps_init": 1.0}),
        ("ablate_routing_l2", {"routing": "euclidean_l2", "sim_threshold": 0.0,      "eps_init": eps_l2}),
    ]

    out = {
        "cell": args.cell, "benchmark": bench, "model": model_id,
        "layer": layer, "seed": args.seed, "n_edits": args.n_edits,
        "tau": args.tau, "eps_l2": eps_l2,
        "rows": [],
    }

    for name, kwargs in variants:
        print(f"\n[r4_l2] --- variant {name} ---")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if DEVICE.type == "cuda" else torch.float32),
            device_map="auto",
        )
        model.eval()

        preset = {"write_mode": "additive", "value_optim": "vstar"}
        preset.update({k: v for k, v in kwargs.items() if k in ("routing",)})
        method = AblatedSHARDMethod(
            layer_idx=layer,
            v_steps=200, v_lr=1.0,
            v_weight_decay=0.0, v_norm_constraint=20.0,
            sim_threshold=kwargs["sim_threshold"],
            eps_init=kwargs["eps_init"],
            **preset,
        )
        method.setup(model, tokenizer, kb=None)

        t0 = time.time()
        for i, edit in enumerate(edits, start=1):
            method.insert(edit)
            if i % 100 == 0:
                print(f"   inserted {i}/{len(edits)}  ({(time.time()-t0)/60:.1f} min)")

        m = eval_cell(bench, model, tokenizer, edits, holdout, args.tau, layer)
        row = {"variant": name, **kwargs,
               "Eff": float(m.get("Eff", m.get("eff", 0.0))),
               "Gen": float(m.get("Gen", m.get("gen", 0.0))),
               "Spec": float(m.get("Spec", m.get("spec", 0.0))),
               "wall_s": time.time() - t0}
        print(f"[r4_l2] {name}: Eff={row['Eff']:.4f}  "
              f"Gen={row['Gen']:.4f}  Spec={row['Spec']:.4f}")
        out["rows"].append(row)

        del model, method
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[r4_l2] wrote {args.out}")


if __name__ == "__main__":
    main()
