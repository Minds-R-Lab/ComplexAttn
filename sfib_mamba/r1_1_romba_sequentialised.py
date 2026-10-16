"""r1_1_romba_sequentialised.py -- sequentialised ROMBA baseline, self-contained.

Implements ROME-style weight editing on Mamba's W_o directly: for each
(prompt, target) pair we run n_steps iterations of Adam on W_o.weight to
minimise cross-entropy on the target-token continuation. Edits are applied
sequentially; each one permanently modifies the weight. This is the baseline
reviewer R1.1 asked for ("ROME-for-Mamba in the sequential setting").

Because ROMBA (Sen Sharma et al. 2024) was proposed as a single-edit procedure,
sequential application is expected to catastrophically forget, which is what
this script measures.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'sfib'))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_benchmark_and_eval(benchmark, seed, n_edits):
    if benchmark == "counterfact":
        from counterfact_data import cf_splits
        from run_counterfact import cf_eval
        edits, holdout = cf_splits(n_edits=n_edits, seed=seed)
        return edits, holdout, cf_eval
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _get(obj, name, default=None):
    if hasattr(obj, name): return getattr(obj, name)
    if isinstance(obj, dict) and name in obj: return obj[name]
    return default


def get_wo(model, layer_idx, kind="out_proj"):
    block = model.backbone.layers[layer_idx]
    return getattr(block.mixer, kind)


def apply_romba_edit(model, tokenizer, edit, layer_idx, kind, n_steps=30, lr=5e-4):
    wo = get_wo(model, layer_idx, kind)
    saved = [p.requires_grad for p in model.parameters()]
    for p in model.parameters(): p.requires_grad = False
    wo.weight.requires_grad = True

    prompt_tmpl = _get(edit, 'prompt_template') or _get(edit, 'prompt', '')
    subject = _get(edit, 'subject', '')
    prompt = prompt_tmpl.format(subject) if '{}' in prompt_tmpl else prompt_tmpl
    tn = _get(edit, 'target_new', '')
    target = tn.get('str', '') if isinstance(tn, dict) else tn

    ids_full = tokenizer(prompt + " " + target, return_tensors="pt").input_ids.to(DEVICE)
    ids_prompt = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    n_prompt = ids_prompt.shape[1]
    if ids_full.shape[1] <= n_prompt:
        for p, r in zip(model.parameters(), saved): p.requires_grad = r
        return float('nan')

    opt = torch.optim.Adam([wo.weight], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        logits = model(ids_full).logits
        tgt_logits = logits[0, n_prompt-1:-1, :]
        tgt_ids = ids_full[0, n_prompt:]
        loss = F.cross_entropy(tgt_logits, tgt_ids)
        loss.backward()
        opt.step()

    for p, r in zip(model.parameters(), saved): p.requires_grad = r
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["counterfact"], required=True)
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--eval_at", default="0,1,10,50,100,250,500")
    ap.add_argument("--batch_size_eval", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--kind", default="out_proj")
    ap.add_argument("--n_steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checkpoints = sorted([int(x) for x in args.eval_at.split(",")])
    print(f"[romba_seq] device: {DEVICE}  model: {args.model}  layer: {args.layer}")
    print(f"[romba_seq] n_steps_per_edit: {args.n_steps}  lr: {args.lr}")
    print(f"[romba_seq] checkpoints: {checkpoints}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)

    edits, holdout, eval_fn = load_benchmark_and_eval(args.benchmark, args.seed, args.n_edits)
    print(f"[romba_seq] {len(edits)} edits, {len(holdout)} specificity probes")

    results = []
    if 0 in checkpoints:
        r0 = eval_fn(model, tokenizer, edits_seen=[], holdout=holdout,
                     batch_size=args.batch_size_eval, max_new_tokens=args.max_new_tokens)
        results.append({"N": 0, **r0})
        e0 = r0['efficacy']['accuracy']; s0 = r0['specificity']['accuracy']
        es = f"{e0:.4f}" if e0 is not None else "n/a"
        ss = f"{s0:.4f}" if s0 is not None else "n/a"
        print(f"[N=0] Eff={es}  Spec={ss}")

    t0 = time.time()
    next_cp = 1 if 0 in checkpoints else 0

    for i, edit in enumerate(edits, start=1):
        try:
            apply_romba_edit(model, tokenizer, edit, args.layer, args.kind,
                             n_steps=args.n_steps, lr=args.lr)
        except Exception as e:
            print(f"[romba_seq] edit {i} failed: {e!r}")

        if next_cp < len(checkpoints) and i == checkpoints[next_cp]:
            r = eval_fn(model, tokenizer, edits_seen=edits[:i], holdout=holdout,
                        batch_size=args.batch_size_eval, max_new_tokens=args.max_new_tokens)
            results.append({"N": i, **r})
            e = r['efficacy']['accuracy']; g = r['generalization']['accuracy']; s = r['specificity']['accuracy']
            print(f"[N={i}] Eff={e:.4f}  Gen={g:.4f}  Spec={s:.4f}  ({(time.time()-t0)/60:.1f} min)")
            with open(args.out, "w") as fh:
                json.dump({
                    "method": "romba_seq",
                    "benchmark": args.benchmark, "model": args.model,
                    "layer": args.layer, "kind": args.kind, "seed": args.seed,
                    "n_edits": args.n_edits, "n_steps_per_edit": args.n_steps,
                    "lr": args.lr, "wall_time_s": time.time() - t0,
                    "results": results,
                }, fh, indent=2)
            next_cp += 1

    print(f"[romba_seq] stream complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
