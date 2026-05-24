"""diagnose_pretrain.py — explain why retention plateaus at ~88%.

Loads the saved best checkpoint and breaks the 1000 retention queries down by:
  (1) per-relation accuracy
  (2) per-query-template accuracy (q_idx=0 vs q_idx=1)
  (3) per-(relation, template) cell
  (4) KB-level ambiguity audit: does any (subject, relation) pair have multiple
      distinct objects in the training data? If yes, those facts are
      irreducibly ambiguous and the model CANNOT memorize them deterministically.
  (5) a sample of failures with the model's actual output, grouped by relation

Usage:
    python diagnose_pretrain.py --ckpt checkpoints/pretrained_seed0_gpt2.pt \
                                 --n_pretrain 2000 --n_holdout 500 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from kb_data import generate_kb, render_eval_query, RELATIONS
from evaluate import EvalExample, evaluate_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrained_seed0_gpt2.pt")
    ap.add_argument("--model", default="gpt2",
                    choices=["gpt2", "gpt2-medium", "gpt2-large"])
    ap.add_argument("--n_pretrain", type=int, default=2000)
    ap.add_argument("--n_insert", type=int, default=500)
    ap.add_argument("--n_compose", type=int, default=200)
    ap.add_argument("--n_holdout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[diag] device: {DEVICE}")
    print(f"[diag] regenerating KB (seed={args.seed}, n_pretrain={args.n_pretrain})")
    kb = generate_kb(seed=args.seed,
                     n_pretrain=args.n_pretrain,
                     n_insert=args.n_insert,
                     n_compose=args.n_compose)

    # ---------- (4) KB ambiguity audit (do this FIRST — no model needed) ----------
    print("\n" + "=" * 72)
    print("[diag] (4) KB ambiguity audit")
    print("=" * 72)
    by_sr: dict[tuple[str, str], set[str]] = defaultdict(set)
    for t in kb.pretrain_triples:
        by_sr[(t.subject, t.relation)].add(t.obj)
    ambig = {k: v for k, v in by_sr.items() if len(v) > 1}
    n_unique_sr = len(by_sr)
    n_ambig_sr = len(ambig)
    n_ambig_triples = sum(len(v) for v in ambig.values())
    print(f"  unique (subject, relation) pairs in pretrain: {n_unique_sr}")
    print(f"  (subject, relation) pairs with >1 distinct object: {n_ambig_sr}")
    print(f"  triples involved in such ambiguity: {n_ambig_triples}")
    if n_ambig_sr > 0:
        ratio = n_ambig_triples / len(kb.pretrain_triples)
        print(f"  -> {ratio:.1%} of pretrain triples are members of an ambiguous group")
        print(f"     these facts are IRREDUCIBLY un-memorizable (the same Q has multiple A)")
        # Show a few examples
        print(f"  examples:")
        for i, (sr, objs) in enumerate(list(ambig.items())[:6]):
            print(f"    {sr[0]:>18} | {sr[1]:<16} -> {{ {', '.join(sorted(objs))} }}")

    # Per-relation: how often is each relation involved in such collisions?
    by_rel_ambig = Counter()
    by_rel_total = Counter()
    for (subj, rel), objs in by_sr.items():
        by_rel_total[rel] += len(objs)  # objects, which equals triples in this view
        if len(objs) > 1:
            by_rel_ambig[rel] += len(objs)
    print(f"\n  per-relation ambiguity (triples in ambiguous SR groups / total triples):")
    for rel in sorted(by_rel_total.keys()):
        amb, tot = by_rel_ambig[rel], by_rel_total[rel]
        pct = amb / tot if tot else 0
        print(f"    {rel:<16}  {amb:>4} / {tot:<4}  ({pct:.1%})")

    # ---------- reproduce retention sample EXACTLY as pretrain.py did ----------
    all_pretrain = list(kb.pretrain_triples)
    random.Random(args.seed).shuffle(all_pretrain)
    retention_sample = all_pretrain[:args.n_holdout]
    print(f"\n[diag] reproduced retention sample: {len(retention_sample)} triples")

    # Build holdout eval (same structure as pretrain.py)
    holdout_eval: list[EvalExample] = []
    for t in retention_sample:
        for q_idx in range(2):
            prompt, target = render_eval_query(t, template_idx=q_idx)
            holdout_eval.append(EvalExample(
                prompt=prompt, target=target, kind="retention",
                meta={"triple": t.as_tuple(), "q_idx": q_idx},
            ))
    print(f"[diag] retention eval examples: {len(holdout_eval)}")

    # Pre-flag ambiguous examples (their (subj, rel) has >1 obj in pretrain)
    n_unambig = 0
    n_ambig_eval = 0
    for ex in holdout_eval:
        subj, rel, _ = ex.meta["triple"]
        if (subj, rel) in ambig:
            n_ambig_eval += 1
        else:
            n_unambig += 1
    print(f"[diag] of those, {n_ambig_eval} target an ambiguous (S,R) and {n_unambig} are unambiguous")

    # ---------- load model ----------
    print(f"\n[diag] loading {args.model} + checkpoint: {args.ckpt}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    if isinstance(ckpt, dict) and "retention_acc" in ckpt:
        print(f"  checkpoint reports retention_acc={ckpt['retention_acc']:.4f} at epoch={ckpt.get('epoch','?')}")

    # ---------- run eval ----------
    print(f"\n[diag] running eval (greedy, 12 new tokens, batch 32)")
    result = evaluate_model(model, tokenizer, holdout_eval,
                             max_new_tokens=12, batch_size=32)
    overall = result["summary"]["retention"]["accuracy"]
    print(f"[diag] overall retention = {overall:.4f} ({sum(r['correct'] for r in result['records'])}/{len(result['records'])})")

    # ---------- (1) per-relation accuracy ----------
    print("\n" + "=" * 72)
    print("[diag] (1) per-relation retention")
    print("=" * 72)
    rel_correct = Counter(); rel_total = Counter()
    for r in result["records"]:
        rel = r["meta"]["triple"][1]
        rel_total[rel] += 1
        if r["correct"]: rel_correct[rel] += 1
    for rel in sorted(rel_total.keys()):
        c, t = rel_correct[rel], rel_total[rel]
        print(f"  {rel:<16}  {c:>4}/{t:<4}  ({c/t:.2%})")

    # ---------- (2) per-template accuracy ----------
    print("\n" + "=" * 72)
    print("[diag] (2) per-query-template retention")
    print("=" * 72)
    q_correct = Counter(); q_total = Counter()
    for r in result["records"]:
        q = r["meta"]["q_idx"]
        q_total[q] += 1
        if r["correct"]: q_correct[q] += 1
    for q in sorted(q_total.keys()):
        c, t = q_correct[q], q_total[q]
        print(f"  q_idx={q}  {c:>4}/{t:<4}  ({c/t:.2%})")

    # ---------- (3) per (relation, template) cell ----------
    print("\n" + "=" * 72)
    print("[diag] (3) per (relation, template) retention")
    print("=" * 72)
    cell_c = defaultdict(int); cell_t = defaultdict(int)
    for r in result["records"]:
        rel = r["meta"]["triple"][1]; q = r["meta"]["q_idx"]
        cell_t[(rel, q)] += 1
        if r["correct"]: cell_c[(rel, q)] += 1
    # Header
    print(f"  {'relation':<16}  {'q0':>10}  {'q1':>10}  {'diff':>8}")
    for rel in sorted(rel_total.keys()):
        c0, t0 = cell_c[(rel, 0)], cell_t[(rel, 0)]
        c1, t1 = cell_c[(rel, 1)], cell_t[(rel, 1)]
        a0 = c0 / t0 if t0 else 0
        a1 = c1 / t1 if t1 else 0
        diff = a0 - a1
        print(f"  {rel:<16}  {c0}/{t0} ({a0:.0%})  {c1}/{t1} ({a1:.0%})  {diff:+.2f}")

    # ---------- (4-cont) accuracy on unambiguous-only ----------
    print("\n" + "=" * 72)
    print("[diag] retention conditional on KB ambiguity")
    print("=" * 72)
    c_unamb = t_unamb = c_amb = t_amb = 0
    for r in result["records"]:
        subj, rel, _ = r["meta"]["triple"]
        if (subj, rel) in ambig:
            t_amb += 1
            if r["correct"]: c_amb += 1
        else:
            t_unamb += 1
            if r["correct"]: c_unamb += 1
    print(f"  unambiguous (S,R):  {c_unamb}/{t_unamb}  ({c_unamb/max(t_unamb,1):.2%})")
    print(f"  ambiguous   (S,R):  {c_amb}/{t_amb}  ({c_amb/max(t_amb,1):.2%})")

    # ---------- (5) sample failures, grouped by relation ----------
    print("\n" + "=" * 72)
    print("[diag] (5) sample failures, up to 3 per relation")
    print("=" * 72)
    fail_by_rel = defaultdict(list)
    for r in result["records"]:
        if not r["correct"]:
            fail_by_rel[r["meta"]["triple"][1]].append(r)
    for rel in sorted(fail_by_rel.keys()):
        print(f"\n  -- {rel} -- ({len(fail_by_rel[rel])} failures)")
        for r in fail_by_rel[rel][:3]:
            subj, _, obj = r["meta"]["triple"]
            amb_flag = " [AMBIG]" if (subj, rel) in ambig else ""
            print(f"    Q: {r['prompt']}")
            print(f"       target='{r['target']}'  got='{r['generated'].strip()}'{amb_flag}")

    # ---------- save full diagnostic ----------
    out = {
        "overall_retention": overall,
        "kb_audit": {
            "n_unique_sr_pairs": n_unique_sr,
            "n_ambig_sr_pairs": n_ambig_sr,
            "n_ambig_triples": n_ambig_triples,
            "frac_ambig": n_ambig_triples / max(len(kb.pretrain_triples), 1),
            "per_rel_ambig": {rel: {"amb": by_rel_ambig[rel], "tot": by_rel_total[rel]}
                               for rel in by_rel_total},
        },
        "per_relation": {rel: {"correct": rel_correct[rel], "total": rel_total[rel],
                                "acc": rel_correct[rel]/rel_total[rel]}
                          for rel in rel_total},
        "per_template": {q: {"correct": q_correct[q], "total": q_total[q],
                              "acc": q_correct[q]/q_total[q]}
                          for q in q_total},
        "ambig_conditional": {
            "unambig": {"correct": c_unamb, "total": t_unamb,
                         "acc": c_unamb/max(t_unamb,1)},
            "ambig": {"correct": c_amb, "total": t_amb,
                       "acc": c_amb/max(t_amb,1)},
        },
    }
    out_json = RESULTS_DIR / f"diagnose_seed{args.seed}_{args.model}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[diag] saved -> {out_json}")


if __name__ == "__main__":
    main()
