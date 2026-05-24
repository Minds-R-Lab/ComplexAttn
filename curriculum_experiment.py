"""curriculum_experiment.py — does training-data ordering close the
compositional-generalization gap on SCAN?

Tests 5 training regimes on identical small-transformer architecture:
    1. random            — baseline (random shuffle per epoch)
    2. length_easy       — shortest commands first, longest last
    3. compositional     — primitives → 1-op compositions → 2-op → ...
    4. anti_curriculum   — longest/most-complex first (control)
    5. gradual_mix       — cosine-blend probability of seeing hard examples

All conditions:
    - identical model, optimizer, lr, total training steps
    - measured by SAME validation set (held-out compositional set)
    - 3 seeds each

If the compositional curriculum closes the gap while random/anti don't:
    → architecture isn't the bottleneck, curriculum is. Big finding.

If no condition closes the gap, including the compositional curriculum:
    → architecture really is the bottleneck. Also informative.

Datasets: SCAN add_prim_jump split (and optionally length split).
Required: a GPU; ~1-2 hours total runtime.

Usage:
    python curriculum_experiment.py             # full grid
    python curriculum_experiment.py --quick     # 1 seed, fewer epochs
    python curriculum_experiment.py --report    # just summarize saved state
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[curriculum] device: {DEVICE}")

RESULTS_DIR = Path(__file__).parent / "results" / "curriculum"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent / "data" / "scan"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_DIR / "curriculum.json"


# ---------------------------------------------------------------------------
# SCAN data loading
# ---------------------------------------------------------------------------

SCAN_URLS = {
    "add_prim_jump": {
        "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt",
        "test":  "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt",
    },
    "length": {
        "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt",
        "test":  "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt",
    },
    "simple": {
        "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_train_simple.txt",
        "test":  "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_test_simple.txt",
    },
}


def download_scan(split: str = "add_prim_jump") -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Download and parse SCAN. Returns (train_pairs, test_pairs)."""
    out = {}
    for kind, url in SCAN_URLS[split].items():
        fn = DATA_DIR / f"{split}_{kind}.txt"
        if not fn.exists():
            print(f"[curriculum] downloading {url}")
            urllib.request.urlretrieve(url, fn)
        pairs = []
        for line in fn.read_text().strip().split("\n"):
            # format: "IN: <input> OUT: <output>"
            assert line.startswith("IN: "), f"bad line: {line!r}"
            rest = line[4:]
            ix = rest.index(" OUT: ")
            inp, outp = rest[:ix], rest[ix + len(" OUT: "):]
            pairs.append((inp.strip(), outp.strip()))
        out[kind] = pairs
    return out["train"], out["test"]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

def build_vocabs(pairs):
    src_tokens = {PAD, SOS, EOS, UNK}
    tgt_tokens = {PAD, SOS, EOS, UNK}
    for inp, outp in pairs:
        src_tokens.update(inp.split())
        tgt_tokens.update(outp.split())
    src_vocab = {t: i for i, t in enumerate(sorted(src_tokens))}
    tgt_vocab = {t: i for i, t in enumerate(sorted(tgt_tokens))}
    return src_vocab, tgt_vocab


def encode(seq, vocab, add_sos_eos=False):
    ids = [vocab.get(t, vocab[UNK]) for t in seq.split()]
    if add_sos_eos:
        ids = [vocab[SOS]] + ids + [vocab[EOS]]
    return ids


class ScanDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        inp, outp = self.pairs[i]
        src = encode(inp, self.src_vocab)
        tgt = encode(outp, self.tgt_vocab, add_sos_eos=True)
        return torch.tensor(src), torch.tensor(tgt), inp, outp


def collate(batch, pad_src, pad_tgt):
    srcs, tgts, inps, outps = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_src)
    tgt_pad = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_tgt)
    return src_pad, tgt_pad, list(inps), list(outps)


# ---------------------------------------------------------------------------
# Small transformer (encoder-decoder)
# ---------------------------------------------------------------------------

class ScanTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, n_heads=4, n_enc=2, n_dec=2,
                 d_ff=256, max_len=64, dropout=0.1,
                 pad_src=0, pad_tgt=0):
        super().__init__()
        self.d_model = d_model
        self.pad_src = pad_src
        self.pad_tgt = pad_tgt
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_src)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_tgt)
        self.src_pos = nn.Embedding(max_len, d_model)
        self.tgt_pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff,
                                                dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff,
                                                dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src):
        pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        x = self.src_emb(src) + self.src_pos(pos)
        mask = (src == self.pad_src)
        return self.encoder(x, src_key_padding_mask=mask), mask

    def decode(self, tgt, memory, src_pad_mask):
        pos = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        y = self.tgt_emb(tgt) + self.tgt_pos(pos)
        tgt_pad_mask = (tgt == self.pad_tgt)
        L = tgt.size(1)
        causal = torch.triu(torch.ones(L, L, device=tgt.device, dtype=torch.bool), diagonal=1)
        out = self.decoder(y, memory, tgt_mask=causal,
                           tgt_key_padding_mask=tgt_pad_mask,
                           memory_key_padding_mask=src_pad_mask)
        return self.out_proj(out)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)

    @torch.no_grad()
    def greedy_decode(self, src, sos_id, eos_id, max_len=50):
        memory, src_mask = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, memory, src_mask)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            nxt = torch.where(finished.unsqueeze(-1), torch.full_like(nxt, self.pad_tgt), nxt)
            ys = torch.cat([ys, nxt], dim=1)
            finished = finished | (nxt.squeeze(-1) == eos_id)
            if finished.all(): break
        return ys


# ---------------------------------------------------------------------------
# Curricula
# ---------------------------------------------------------------------------

def composition_depth(inp: str) -> int:
    """Heuristic depth: 0 if pure primitive (1 word), else count of compositional words."""
    words = inp.split()
    comp_words = {"twice", "thrice", "and", "after", "left", "right", "around", "opposite"}
    n_comp = sum(1 for w in words if w in comp_words)
    return n_comp


def build_order(pairs, regime: str, total_epochs: int, rng):
    """For each epoch, return a list of indices into `pairs` (the training order).
    All regimes see the same total number of example-views (len(pairs) per epoch)
    so total training compute is identical."""
    N = len(pairs)
    lens = np.array([len(p[0].split()) for p in pairs])
    depths = np.array([composition_depth(p[0]) for p in pairs])
    all_idx = np.arange(N)

    schedules = []
    if regime == "random":
        for ep in range(total_epochs):
            order = rng.permutation(N)
            schedules.append(order.tolist())

    elif regime == "length_easy":
        order = np.argsort(lens, kind="stable")  # shortest first
        for ep in range(total_epochs):
            schedules.append(order.tolist())

    elif regime == "compositional":
        # Stage by depth, going from depth 0 (primitives) to max depth
        max_depth = depths.max()
        # Stage durations: spend first half of training building up through stages,
        # second half on full mixture
        half = total_epochs // 2
        n_stages = max_depth + 1
        for ep in range(total_epochs):
            if ep < half:
                # In first half, gradually include higher depths
                allowed_depth = min(n_stages - 1, int(ep * n_stages / half))
                idx = all_idx[depths <= allowed_depth]
                order = rng.permutation(idx)
                # Pad/repeat to exactly N samples per epoch for compute parity
                if len(order) < N:
                    pad = rng.choice(order, size=N - len(order), replace=True)
                    order = np.concatenate([order, pad])
                schedules.append(order.tolist())
            else:
                schedules.append(rng.permutation(N).tolist())

    elif regime == "anti_curriculum":
        # Reverse of length: longest first
        order = np.argsort(-lens, kind="stable")
        for ep in range(total_epochs):
            schedules.append(order.tolist())

    elif regime == "gradual_mix":
        # Cosine schedule: at epoch t, P(easy)=cos(πt/T)/2+1/2 going to 0
        # Sample with probability prop to easiness early, uniform late
        ease = lens.max() - lens + 1  # higher = easier
        for ep in range(total_epochs):
            alpha = 0.5 * (1 + math.cos(math.pi * ep / max(1, total_epochs - 1)))
            # alpha=1 at start (max ease bias), alpha=0 at end (uniform)
            w = ease.astype(float) ** (3 * alpha)
            w = w / w.sum()
            order = rng.choice(N, size=N, replace=True, p=w)
            schedules.append(order.tolist())

    else:
        raise ValueError(f"unknown regime: {regime}")
    return schedules


# ---------------------------------------------------------------------------
# Train & eval
# ---------------------------------------------------------------------------

def evaluate(model, loader, src_vocab, tgt_vocab):
    """Exact-match sequence accuracy on a held-out set."""
    sos_id = tgt_vocab[SOS]; eos_id = tgt_vocab[EOS]; pad_id = tgt_vocab[PAD]
    inv_tgt = {i: t for t, i in tgt_vocab.items()}
    model.eval()
    correct, total = 0, 0
    for src, tgt, inps, outps in loader:
        src = src.to(DEVICE)
        pred = model.greedy_decode(src, sos_id, eos_id, max_len=50)
        for i in range(pred.size(0)):
            seq = pred[i, 1:].tolist()  # skip SOS
            if eos_id in seq: seq = seq[:seq.index(eos_id)]
            words = [inv_tgt[t] for t in seq if t not in (pad_id, sos_id, eos_id)]
            if " ".join(words) == outps[i]: correct += 1
            total += 1
    return correct / total


def train_one(model, train_pairs, test_pairs, src_vocab, tgt_vocab,
              regime, seed, epochs, batch_size=128, lr=5e-4):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    schedules = build_order(train_pairs, regime, epochs, rng)
    ds_train = ScanDataset(train_pairs, src_vocab, tgt_vocab)
    ds_test = ScanDataset(test_pairs, src_vocab, tgt_vocab)
    pad_src = src_vocab[PAD]; pad_tgt = tgt_vocab[PAD]

    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False,
                             collate_fn=lambda b: collate(b, pad_src, pad_tgt))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(ignore_index=pad_tgt)

    best_test = 0.0
    final_test = 0.0
    history = []
    for ep in range(epochs):
        model.train()
        order = schedules[ep]
        # Create one-epoch DataLoader from the prescribed order
        # We bucket by length for efficiency: sort within mini-batches
        for batch_start in range(0, len(order), batch_size):
            idx = order[batch_start:batch_start + batch_size]
            batch = [ds_train[j] for j in idx]
            src, tgt, _, _ = collate(batch, pad_src, pad_tgt)
            src = src.to(DEVICE); tgt = tgt.to(DEVICE)
            tgt_in = tgt[:, :-1]; tgt_out = tgt[:, 1:]
            opt.zero_grad()
            logits = model(src, tgt_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Eval on test every few epochs to save time
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            acc = evaluate(model, test_loader, src_vocab, tgt_vocab)
            best_test = max(best_test, acc)
            final_test = acc
            history.append({"epoch": ep + 1, "test_acc": acc})
            print(f"    [{regime} seed={seed}] ep {ep+1}/{epochs} test_acc={acc:.4f}")
    return best_test, final_test, history


# ---------------------------------------------------------------------------
# State / runner
# ---------------------------------------------------------------------------

def load_state():
    return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}

def save_state(s): OUT.write_text(json.dumps(s, indent=2))

def done(state, split, regime, seed):
    return any(r["split"]==split and r["regime"]==regime and r["seed"]==seed
               for r in state["runs"])


# ---------------------------------------------------------------------------
# Analysis & report
# ---------------------------------------------------------------------------

def report(state):
    print("\n" + "=" * 78)
    print("CURRICULUM EXPERIMENT — SCAN compositional generalization")
    print("=" * 78)

    splits = sorted({r["split"] for r in state["runs"]})
    regimes_order = ["random", "anti_curriculum", "length_easy", "compositional", "gradual_mix"]

    for split in splits:
        print(f"\n## Split: {split}\n")
        print(f"  {'regime':<18s}  {'best_mean±std':>16s}  {'final_mean±std':>16s}  "
              f"{'gap vs random':>14s}  {'wilcoxon p':>11s}  {'wins/n':>7s}")
        base = np.array([r["best_test"] for r in state["runs"]
                         if r["split"]==split and r["regime"]=="random"])
        for regime in regimes_order:
            vals_best = np.array([r["best_test"] for r in state["runs"]
                                  if r["split"]==split and r["regime"]==regime])
            vals_final = np.array([r["final_test"] for r in state["runs"]
                                   if r["split"]==split and r["regime"]==regime])
            if len(vals_best) == 0: continue
            bm, bs = vals_best.mean(), (vals_best.std(ddof=1) if len(vals_best) > 1 else 0.0)
            fm, fs = vals_final.mean(), (vals_final.std(ddof=1) if len(vals_final) > 1 else 0.0)
            if regime == "random" or len(base) == 0:
                gap = 0.0; p_str = "—"; wins_str = "—"
            else:
                n = min(len(vals_best), len(base))
                gap = (vals_best[:n].mean() - base[:n].mean()) * 100
                wins = int((vals_best[:n] > base[:n]).sum())
                wins_str = f"{wins}/{n}"
                if wilcoxon and n >= 3:
                    try:
                        p = wilcoxon(vals_best[:n], base[:n], alternative="greater").pvalue
                        p_str = f"{p:.3f}"
                    except ValueError:
                        p_str = "—"
                else:
                    p_str = "—"
            print(f"  {regime:<18s}  {bm:.4f}±{bs:.4f}  {fm:.4f}±{fs:.4f}  "
                  f"{gap:>+12.2f}pp  {p_str:>11s}  {wins_str:>7s}")

    print("\n## Interpretation guide:\n")
    print("  - If `compositional` and `gradual_mix` beat both `random` AND `anti_curriculum`:")
    print("      → curriculum matters; architecture is not the only bottleneck.")
    print("  - If `compositional` ties `random` (≤ 2pp):")
    print("      → ordering doesn't help; the bottleneck is architectural or representational.")
    print("  - If `anti_curriculum` does as well as `compositional`:")
    print("      → any non-random ordering helps; specific structure isn't the win.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--splits", nargs="+",
                    default=["add_prim_jump", "length"])
    ap.add_argument("--regimes", nargs="+",
                    default=["random", "anti_curriculum", "length_easy",
                             "compositional", "gradual_mix"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    state = load_state()
    if args.report:
        report(state); return

    if args.quick:
        args.seeds = [0]; args.epochs = 20
        print("[quick] 1 seed × 20 epochs")

    print(f"[curriculum] runs already saved: {len(state['runs'])}")
    print(f"[curriculum] splits={args.splits}  regimes={args.regimes}  "
          f"seeds={args.seeds}  epochs={args.epochs}")

    for split in args.splits:
        train_pairs, test_pairs = download_scan(split)
        src_vocab, tgt_vocab = build_vocabs(train_pairs + test_pairs)
        print(f"\n[{split}] train={len(train_pairs)}  test={len(test_pairs)}  "
              f"src_vocab={len(src_vocab)}  tgt_vocab={len(tgt_vocab)}")

        for regime in args.regimes:
            for seed in args.seeds:
                if done(state, split, regime, seed):
                    continue
                torch.manual_seed(seed)
                if DEVICE.type == "cuda": torch.cuda.manual_seed_all(seed)
                model = ScanTransformer(len(src_vocab), len(tgt_vocab),
                                         pad_src=src_vocab[PAD], pad_tgt=tgt_vocab[PAD]).to(DEVICE)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                t0 = time.time()
                best, final, hist = train_one(model, train_pairs, test_pairs,
                                               src_vocab, tgt_vocab,
                                               regime, seed, args.epochs)
                state["runs"].append({
                    "split": split, "regime": regime, "seed": seed,
                    "best_test": float(best), "final_test": float(final),
                    "epochs": args.epochs, "n_params": n_params,
                    "wall_time_s": time.time() - t0, "history": hist,
                })
                save_state(state)
                print(f"  -> [{split} {regime} s={seed}] best={best:.4f} "
                      f"final={final:.4f} ({time.time()-t0:.0f}s)")

    report(state)


if __name__ == "__main__":
    main()
