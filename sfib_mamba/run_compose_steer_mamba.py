"""run_compose_steer_mamba.py -- composition (multi-hop) queries for STEER.

Tests whether STEER's edited slot value propagates through a second hop of
the model's pre-trained knowledge. We mine CounterFact for transitive chains
e1 = (s1, r1, o_new) and e2 = (s2, r2, o2_true) where s2 == o_new, then:

  1. Insert e1 into STEER (so the model now thinks r1(s1) = o_new instead of
     the original o_true).
  2. Construct a composed prompt that requires both hops:
        f"{r2_template applied to (r1_template applied to s1)}"
     Example:
        e1 = ("Steve Jobs", "born in", "Tokyo")     [inserted, o_new = Tokyo]
        e2 = ("Tokyo",     "is the capital of", "Japan")  [model already knows]
        composed = "The country whose capital is the place where Steve Jobs
                    was born is"
        expected = "Japan"
  3. Generate from the composed prompt with the STEER-wrapped model. Score
     the generation by whether o2_true (here "Japan") appears in the output.

This is the "transitive bridge" composition test of Zhong et al. (MQuAKE)
restricted to chains we can mine directly from the CounterFact dataset.
We measure:

  compose@N  =  fraction of composition queries for which the model produces
                o2_true given the inserted edit
  base@N     =  same metric on the FROZEN base (no STEER edits), as control

A high compose@N - base@N gap means the edit propagates two hops; a small gap
means STEER's slot value localises at the edit site without flowing through
downstream computations.

Usage:
    python run_compose_steer_mamba.py \\
        --model state-spaces/mamba-790m-hf --layer 32 --seed 0 --n_edits 500 \\
        --out results/compose_steer_mamba_mamba790m_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SFIB_MAMBA_DIR = Path(__file__).parent
REPO_ROOT = SFIB_MAMBA_DIR.parent
SFIB_DIR = REPO_ROOT / "sfib"
sys.path.insert(0, str(SFIB_DIR))
sys.path.insert(0, str(SFIB_MAMBA_DIR))

import shard_mamba                          # noqa: E402
import shard_mamba_realdata_patches         # noqa: E402, F401
from shard_mamba import SHARDMambaMethod    # noqa: E402
from counterfact_data import (              # noqa: E402
    load_counterfact, cf_splits
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Compose-chain mining
# ---------------------------------------------------------------------------

def mine_transitive_chains(cf_triples, max_chains: int = 500) -> list[dict]:
    """Find pairs (e1, e2) where target_new(e1) == subject(e2).

    Indexes by subject so we can do an O(N) lookup. Returns a list of dicts:
        {"e1": e1, "e2": e2, "composed_prompt": str, "expected": str,
         "expected_orig": str}
    """
    # Build subject index
    by_subject: dict[str, list] = {}
    for t in cf_triples:
        by_subject.setdefault(t.subject.strip(), []).append(t)

    chains = []
    for e1 in cf_triples:
        o_new = e1.target_new.strip()
        if o_new in by_subject:
            for e2 in by_subject[o_new]:
                if e2.subject == e1.subject:  # avoid trivial loops
                    continue
                # Construct the composed prompt by substituting r1(s1) into r2's
                # prompt template's subject slot. The CF prompt_template format is
                # like "The mother tongue of {} is" -- we want to nest these.
                r1_pred = e1.prompt_template.format(e1.subject).rstrip()
                # The second template should treat r1_pred's result as its subject.
                # We use a generic phrasing that doesn't depend on syntactic structure:
                #   "{r1_phrase}. By the way, {r2_phrase with subject = the result above}"
                # Simpler: ask the second-hop question using a referring expression.
                composed = (
                    f"{r1_pred} a place known as X. "
                    + e2.prompt_template.format("X").rstrip()
                )
                chains.append({
                    "case_id_1": e1.case_id, "case_id_2": e2.case_id,
                    "s1": e1.subject, "r1_template": e1.prompt_template,
                    "o_new": e1.target_new, "o_orig": e1.target_true,
                    "s2": e2.subject, "r2_template": e2.prompt_template,
                    "expected": e2.target_true,
                    "expected_orig": e2.target_true,
                    "composed_prompt": composed,
                    "e1": e1,  # keep the actual edit object for insertion
                })
                if len(chains) >= max_chains:
                    return chains
    return chains


# ---------------------------------------------------------------------------
# Composition eval
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return gen_text


def eval_compose(model, tokenizer, chains, max_new_tokens: int = 16) -> dict:
    """Return per-chain accuracy: 1 if expected substring appears in generation."""
    hits = 0
    per_chain = []
    for ch in chains:
        gen = generate(model, tokenizer, ch["composed_prompt"], max_new_tokens=max_new_tokens)
        # Strip whitespace, lowercase, check substring
        exp = ch["expected"].strip().lower()
        hit = exp in gen.strip().lower()
        if hit:
            hits += 1
        per_chain.append({
            "case_ids": (ch["case_id_1"], ch["case_id_2"]),
            "s1": ch["s1"], "o_new": ch["o_new"],
            "expected": ch["expected"],
            "generation": gen[:80],
            "hit": hit,
        })
    return {"accuracy": hits / max(len(chains), 1), "n": len(chains),
            "per_chain": per_chain[:20]}  # truncate per-chain detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--max_chains", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--kind", choices=["out_proj", "in_proj", "x_proj"],
                    default="out_proj")
    ap.add_argument("--v_steps", type=int, default=200)
    ap.add_argument("--v_lr", type=float, default=1.0)
    ap.add_argument("--v_weight_decay", type=float, default=0.0)
    ap.add_argument("--v_norm_constraint", type=float, default=20.0)
    ap.add_argument("--sim_threshold", type=float, default=0.7)
    ap.add_argument("--max_slots", type=int, default=8000)
    ap.add_argument("--capture_position", default="prompt_last",
                    choices=["subject_last", "prompt_last"])
    ap.add_argument("--fire_position", default="last", choices=["last", "all"])
    ap.add_argument("--value_optim", default="vstar",
                    choices=["vstar", "lqr", "lqr_gn"])
    ap.add_argument("--n_lqr_iters", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[compose] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[compose] GPU: {torch.cuda.get_device_name()}")

    # Load all CF triples (so we can mine chains across the whole dataset)
    all_cf = load_counterfact()
    edits, _ = cf_splits(n_edits=args.n_edits, seed=args.seed)
    print(f"[compose] loaded {len(all_cf)} total CF triples; "
          f"insertion stream of {len(edits)} edits")

    # Mine transitive chains within the inserted edits
    chains = mine_transitive_chains(edits, max_chains=args.max_chains)
    print(f"[compose] mined {len(chains)} transitive chains within the "
          f"{args.n_edits}-edit insertion stream")
    if not chains:
        print("[compose] No chains found -- the CF subset has no transitive "
              "pairs. Try a larger --n_edits.")
        sys.exit(1)

    # Build the unique edits we need to insert (the e1 side of each chain)
    e1_by_case = {ch["case_id_1"]: ch["e1"] for ch in chains}
    edits_to_insert = list(e1_by_case.values())
    print(f"[compose] {len(edits_to_insert)} unique e1 edits to insert")

    # ---- Load model ----
    print(f"[compose] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    # ---- Baseline: evaluate composition on the FROZEN base (no edits) ----
    print(f"[compose] eval on frozen base (control)...")
    t0 = time.time()
    base_result = eval_compose(model, tokenizer, chains,
                                max_new_tokens=args.max_new_tokens)
    base_t = time.time() - t0
    print(f"[compose]   base compose@{len(chains)} = {base_result['accuracy']:.4f}  "
          f"[{base_t:.1f}s]")

    # ---- Insert e1 edits via STEER ----
    print(f"[compose] installing STEER and inserting {len(edits_to_insert)} edits...")
    method = SHARDMambaMethod(
        layer_idx=args.layer, kind=args.kind,
        n_v_steps=args.v_steps, v_lr=args.v_lr,
        v_weight_decay=args.v_weight_decay,
        v_norm_constraint=args.v_norm_constraint,
        sim_threshold=args.sim_threshold,
        max_slots=args.max_slots,
        capture_position=args.capture_position,
        fire_position=args.fire_position,
        value_optim=args.value_optim,
        n_lqr_iters=args.n_lqr_iters,
    )
    method.setup(model, tokenizer, kb=None)

    t0 = time.time()
    for i, edit in enumerate(edits_to_insert, start=1):
        method.insert(edit)
        if i % 50 == 0:
            print(f"[compose]   inserted {i}/{len(edits_to_insert)}")
    insert_t = time.time() - t0
    print(f"[compose] insertion stream complete ({insert_t:.1f}s); "
          f"{method.wrapper.n_slots} slots in bank")

    # ---- Eval composition on the STEER-edited model ----
    print(f"[compose] eval composition on STEER-edited model...")
    t0 = time.time()
    edit_result = eval_compose(model, tokenizer, chains,
                                max_new_tokens=args.max_new_tokens)
    edit_t = time.time() - t0
    print(f"[compose]   steer compose@{len(chains)} = {edit_result['accuracy']:.4f}  "
          f"[{edit_t:.1f}s]")

    # ---- Save results ----
    out = {
        "model": args.model, "seed": args.seed,
        "n_edits_stream": args.n_edits,
        "n_chains_tested": len(chains),
        "n_e1_inserted": len(edits_to_insert),
        "layer": args.layer, "value_optim": args.value_optim,
        "base_compose":  base_result,
        "steer_compose": edit_result,
        "compose_gap": edit_result["accuracy"] - base_result["accuracy"],
        "wall_time_eval_base_s":  base_t,
        "wall_time_eval_edit_s":  edit_t,
        "wall_time_insertion_s":  insert_t,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[compose] results -> {out_path}")
    print(f"[compose] Summary:")
    print(f"           base  compose@{len(chains)} = {base_result['accuracy']:.4f}")
    print(f"           steer compose@{len(chains)} = {edit_result['accuracy']:.4f}")
    print(f"           gap (steer - base)           = {edit_result['accuracy'] - base_result['accuracy']:+.4f}")


if __name__ == "__main__":
    main()
