"""r1_1_romba_sequentialised.py -- Reviewer #1 point 1 (ESWA R1.1).

Reviewer #1 asks for head-to-head numbers against ROME / MEMIT /
AlphaEdit / T-Patcher as sequential-editing baselines. ROMBA
(ROME on Mamba, Sen Sharma et al. 2024) is already discussed in the
manuscript's Related Work but was never run as a sequential-edit
baseline -- the ROMBA paper is single-edit-at-a-time.

This script sequentialises ROMBA by applying the ROMBA rank-one update
to each edit in turn against the running (modified) model, with no
reset between edits. It is the direct ``ROME on Mamba, sequentialised''
baseline the reviewer asks for.

AlphaEdit (Fang et al. 2024) and T-Patcher (Huang et al. 2023) rely on
transformer-specific pre-conditioning (AlphaEdit's null-space projection
is derived from attention-layer key/value matrices; T-Patcher adds a
patch neuron inside a transformer MLP). Both methods are not directly
portable to Mamba's gated SSM blocks without a non-trivial
reimplementation; we report that explicitly in the response letter
rather than inventing an ad hoc port.

MEMIT (Meng et al. 2023) is batched weight editing with least-squares
across many edits at once. The MEMIT-for-Mamba port would require
collecting a covariance matrix over Mamba's W_o rows, analogous to the
transformer MLP covariance that MEMIT uses. We leave this to a dedicated
follow-up and defend the choice in the response letter.

Dependencies:
    - A local checkout of the ROMBA reference implementation. We attempt
      to import it via `from romba_ref.rome_main import apply_rome_to_mamba`
      and fall back to a clear error message with a link to the repo if
      the import fails.
    - transformers, torch, datasets.

Usage on H100-80GB:
    cd sfib_mamba/
    python r1_1_romba_sequentialised.py \\
        --benchmark counterfact --model state-spaces/mamba-790m-hf \\
        --layer 32 --seed 0 --n_edits 500 \\
        --out results/counterfact_romba_seq_mamba790m_seed0.json

Expected wall time at Mamba-790M, N=500: ~4-6 h (dominated by the
rank-one update's own v*-style optimisation per edit).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_romba_edit(model, tokenizer, edit, layer_idx, kind="out_proj"):
    """Apply a single ROMBA rank-one update in place on the model's W_o
    at `layer_idx`.

    The reference ROMBA implementation lives at
        https://github.com/arnab-api/romba
    Clone it into ../romba_ref/ so this import succeeds:

        from romba_ref.rome_main import apply_rome_to_mamba

    We expect `apply_rome_to_mamba` to accept (model, tokenizer, edit,
    layer_idx, kind) and modify `model` in place, matching the ROMBA
    paper's hyperparameters.
    """
    try:
        from romba_ref.rome_main import apply_rome_to_mamba  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "ROMBA reference implementation not found. Clone\n"
            "    https://github.com/arnab-api/romba\n"
            "into ../romba_ref/, add __init__.py, and re-run.\n"
            f"(Original ImportError: {e})"
        )
    apply_rome_to_mamba(
        model, tokenizer, edit,
        layer_idx=layer_idx, kind=kind,
    )


def load_benchmark_and_eval(benchmark, seed, n_edits):
    if benchmark == "counterfact":
        from counterfact_data import cf_splits
        from run_counterfact import cf_eval
        return (*cf_splits(n_edits=n_edits, seed=seed), cf_eval)
    if benchmark == "zsre":
        from zsre_data import zsre_splits
        from run_zsre import zsre_eval
        return (*zsre_splits(n_edits=n_edits, seed=seed), zsre_eval)
    raise ValueError(f"Unknown benchmark: {benchmark!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["counterfact", "zsre"], required=True)
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--kind", default="out_proj",
                    choices=["out_proj", "in_proj", "x_proj"],
                    help="MambaBlock projection to edit; ROMBA paper's W_o.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--eval_at", default="0,1,10,50,100,250,500")
    ap.add_argument("--batch_size_eval", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[romba_seq] device: {DEVICE}  model: {args.model}  layer: {args.layer}")
    checkpoints = sorted({int(x) for x in args.eval_at.split(",")})
    print(f"[romba_seq] checkpoints: {checkpoints}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    edits, holdout, eval_fn = load_benchmark_and_eval(
        args.benchmark, args.seed, args.n_edits)
    print(f"[romba_seq] {len(edits)} edits, {len(holdout)} specificity probes")

    results = []
    if 0 in checkpoints:
        r0 = eval_fn(model, tokenizer, edits_seen=[], holdout=holdout,
                     batch_size=args.batch_size_eval, max_new_tokens=args.max_new_tokens)
        results.append({"N": 0, **r0})
        print(f"[N=0] Eff={r0['efficacy']['accuracy']:.4f}  "
              f"Spec={r0['specificity']['accuracy']:.4f}")

    t_total = time.time()
    next_cp = 1 if 0 in checkpoints else 0

    for i, edit in enumerate(edits, start=1):
        t_edit = time.time()
        apply_romba_edit(model, tokenizer, edit, args.layer, args.kind)
        if i % 20 == 0:
            print(f"[romba_seq] inserted {i}/{len(edits)} "
                  f"({(time.time()-t_total)/60:.1f} min cumulative, "
                  f"last edit {time.time()-t_edit:.1f}s)")

        if next_cp < len(checkpoints) and i == checkpoints[next_cp]:
            r = eval_fn(model, tokenizer, edits_seen=edits[:i], holdout=holdout,
                        batch_size=args.batch_size_eval,
                        max_new_tokens=args.max_new_tokens)
            results.append({"N": i, **r})
            print(f"[N={i}] Eff={r['efficacy']['accuracy']:.4f}  "
                  f"Gen={r['generalization']['accuracy']:.4f}  "
                  f"Spec={r['specificity']['accuracy']:.4f}")
            next_cp += 1
            # Persist after every checkpoint
            with open(args.out, "w") as fh:
                json.dump({
                    "method": "romba_sequentialised",
                    "benchmark": args.benchmark,
                    "model": args.model, "layer": args.layer,
                    "kind": args.kind, "seed": args.seed,
                    "n_edits": args.n_edits,
                    "wall_s_total": time.time() - t_total,
                    "results": results,
                }, fh, indent=2)

    print(f"[romba_seq] wall: {(time.time() - t_total)/3600:.2f} h")
    print(f"[romba_seq] wrote {args.out}")


if __name__ == "__main__":
    main()
