"""curriculum_v2.py — deep test of whether training-data intervention can
close the SCAN compositional-generalization gap.

This is v2 of the experiment, designed in response to the critique that v1
was too shallow. Specific improvements:

1. INCLUDES PRIMITIVE-SUBSTITUTION AUGMENTATION (Andreas 2020, GECA).
   This is documented in the literature to close add_prim_jump to ~99%.
   It serves as both a real condition AND a pipeline-calibration anchor:
   if our pipeline can't reproduce the Andreas result with this condition,
   any "no effect" conclusion from other conditions is invalid.

2. PROPER TRAINING. 150 epochs (Lake & Baroni use 200+), 5 seeds, warmup
   schedule, larger transformer (~5M params, comparable to literature).

3. MULTIPLE CURRICULUM OPERATIONALIZATIONS. Length-based, modifier-count,
   parse-tree depth — testing whether the specific curriculum definition
   matters or whether none of them help.

4. STATISTICAL ANALYSIS. Mean + bootstrap 95% CI, paired Wilcoxon vs
   random baseline, per-condition seed-paired comparison.

5. CALIBRATED MODEL. Same architecture as the simple-split sanity check
   (which hit 99.78%) confirming the model can learn SCAN when not
   facing compositional shift.

Conditions (all on SCAN add_prim_jump split):
    1. random              — baseline (random shuffle per epoch)
    2. anti_curriculum     — longest first (control)
    3. length_easy         — shortest first
    4. compositional_depth — staged by modifier word count
    5. parse_depth         — staged by recursive-clause depth
    6. prim_subst_augment  — Andreas-style augmentation (calibration anchor)
    7. combined            — curriculum + augmentation

Time: 7 conditions × 5 seeds × 150 epochs × ~5 min ≈ 3 hours on a single GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[v2] device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[v2] GPU: {torch.cuda.get_device_name(0)}")

RESULTS_DIR = Path(__file__).parent / "results" / "curriculum_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent / "data" / "scan"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_DIR / "curriculum_v2.json"


# ---------------------------------------------------------------------------
# SCAN download
# ---------------------------------------------------------------------------

SCAN_URLS = {
    "add_prim_jump": {
        "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt",
        "test":  "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt",
    },
}


def download_scan(split="add_prim_jump"):
    out = {}
    for kind, url in SCAN_URLS[split].items():
        fn = DATA_DIR / f"{split}_{kind}.txt"
        if not fn.exists():
            print(f"[v2] downloading {url}")
            urllib.request.urlretrieve(url, fn)
        pairs = []
        for line in fn.read_text().strip().split("\n"):
            assert line.startswith("IN: ")
            rest = line[4:]
            i = rest.index(" OUT: ")
            pairs.append((rest[:i].strip(), rest[i + len(" OUT: "):].strip()))
        out[kind] = pairs
    return out["train"], out["test"]


# ---------------------------------------------------------------------------
# Andreas-style primitive substitution augmentation
# ---------------------------------------------------------------------------

# SCAN primitives and their action mappings
SCAN_PRIMITIVES = {
    "jump":       "I_JUMP",
    "walk":       "I_WALK",
    "run":        "I_RUN",
    "look":       "I_LOOK",
    # turn left / turn right are 2-word; handle separately
}


def discover_primitive_mappings(pairs):
    """Find (input_word, output_word) mappings by inspecting standalone primitives."""
    mapping = {}
    for inp, outp in pairs:
        if inp in SCAN_PRIMITIVES:
            # exact single-primitive command
            mapping[inp] = SCAN_PRIMITIVES[inp]
    # Always include all expected primitives even if not in training
    for k, v in SCAN_PRIMITIVES.items():
        mapping.setdefault(k, v)
    return mapping


def primitive_substitute(inp: str, outp: str, src_prim: str, tgt_prim: str,
                          prim_to_action: dict):
    """Substitute src_prim -> tgt_prim in both input and output.
    Returns (new_inp, new_outp) or None if src_prim not present."""
    # Tokenize as space-separated words.
    if src_prim not in inp.split():
        return None
    new_inp = " ".join(tgt_prim if w == src_prim else w for w in inp.split())
    src_action = prim_to_action[src_prim]
    tgt_action = prim_to_action[tgt_prim]
    new_outp = " ".join(tgt_action if w == src_action else w for w in outp.split())
    return new_inp, new_outp


def augment_with_primitive_substitution(train_pairs, held_out_prim="jump", rng=None):
    """Generate augmented training pairs where the held-out primitive (jump)
    is substituted INTO compositional examples that contained other primitives.

    Andreas 2020 / GECA-style: for each compositional example using a non-jump
    primitive, substitute jump in to create a synthetic 'jump compositional'
    example.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    prim_to_action = discover_primitive_mappings(train_pairs)
    other_prims = [p for p in SCAN_PRIMITIVES if p != held_out_prim]
    augmented = []
    seen = {(i, o) for i, o in train_pairs}
    for inp, outp in train_pairs:
        # Only augment compositional examples (more than 1 word)
        if len(inp.split()) <= 1:
            continue
        # For each other primitive in this example, generate jump-substituted version
        for src in other_prims:
            sub = primitive_substitute(inp, outp, src, held_out_prim, prim_to_action)
            if sub is not None and sub not in seen:
                augmented.append(sub)
                seen.add(sub)
                # Also reverse-augment: substitute held_out → src for diversity (rare in train)
    print(f"[v2] augmentation: {len(train_pairs)} → +{len(augmented)} = {len(train_pairs) + len(augmented)} examples")
    return train_pairs + augmented


# ---------------------------------------------------------------------------
# Tokenization & dataset
# ---------------------------------------------------------------------------

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


def build_vocabs(pairs):
    src_tokens, tgt_tokens = {PAD, SOS, EOS, UNK}, {PAD, SOS, EOS, UNK}
    for inp, outp in pairs:
        src_tokens.update(inp.split())
        tgt_tokens.update(outp.split())
    return ({t: i for i, t in enumerate(sorted(src_tokens))},
            {t: i for i, t in enumerate(sorted(tgt_tokens))})


def encode(s, vocab, add_sos_eos=False):
    ids = [vocab.get(t, vocab[UNK]) for t in s.split()]
    return [vocab[SOS]] + ids + [vocab[EOS]] if add_sos_eos else ids


class ScanDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs, self.sv, self.tv = pairs, src_vocab, tgt_vocab
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        inp, outp = self.pairs[i]
        return (torch.tensor(encode(inp, self.sv)),
                torch.tensor(encode(outp, self.tv, True)),
                inp, outp)


def collate(batch, p_src, p_tgt):
    srcs, tgts, inps, outps = zip(*batch)
    return (nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=p_src),
            nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=p_tgt),
            list(inps), list(outps))


# ---------------------------------------------------------------------------
# Curricula
# ---------------------------------------------------------------------------

def composition_depth(inp):
    """Modifier-word count (jump, walk → 0; jump twice → 1; jump twice and look → 2)."""
    comp_words = {"twice", "thrice", "and", "after", "left", "right", "around", "opposite"}
    return sum(1 for w in inp.split() if w in comp_words)


def parse_depth(inp):
    """Approx parse-tree depth via 'and'/'after' conjunctions (nested compositions)."""
    parts = re.split(r"\s+(?:and|after)\s+", inp)
    # Each part can have one modifier level (twice/thrice/around/opposite)
    sub_depth = max(composition_depth(p) for p in parts) if parts else 0
    n_conj = len(parts) - 1
    return n_conj + sub_depth


def build_orders(pairs, regime, total_epochs, rng):
    """Return a list of `total_epochs` orderings; each is a list of indices into `pairs`."""
    N = len(pairs)
    lens = np.array([len(p[0].split()) for p in pairs])
    depths = np.array([composition_depth(p[0]) for p in pairs])
    pdepths = np.array([parse_depth(p[0]) for p in pairs])
    idx = np.arange(N)

    schedules = []
    if regime == "random":
        for _ in range(total_epochs):
            schedules.append(rng.permutation(N).tolist())

    elif regime == "length_easy":
        order = np.argsort(lens, kind="stable").tolist()
        schedules = [order] * total_epochs

    elif regime == "anti_curriculum":
        order = np.argsort(-lens, kind="stable").tolist()
        schedules = [order] * total_epochs

    elif regime == "compositional_depth":
        max_d = int(depths.max())
        half = total_epochs // 2
        for ep in range(total_epochs):
            if ep < half:
                allowed = min(max_d, int(ep * (max_d + 1) / max(1, half)))
                sub = idx[depths <= allowed]
                order = rng.permutation(sub)
                if len(order) < N:
                    pad = rng.choice(order, size=N - len(order), replace=True)
                    order = np.concatenate([order, pad])
                schedules.append(order.tolist())
            else:
                schedules.append(rng.permutation(N).tolist())

    elif regime == "parse_depth":
        max_pd = int(pdepths.max())
        half = total_epochs // 2
        for ep in range(total_epochs):
            if ep < half:
                allowed = min(max_pd, int(ep * (max_pd + 1) / max(1, half)))
                sub = idx[pdepths <= allowed]
                order = rng.permutation(sub)
                if len(order) < N:
                    pad = rng.choice(order, size=N - len(order), replace=True)
                    order = np.concatenate([order, pad])
                schedules.append(order.tolist())
            else:
                schedules.append(rng.permutation(N).tolist())

    else:
        raise ValueError(regime)
    return schedules


# ---------------------------------------------------------------------------
# Model: 5M-param transformer
# ---------------------------------------------------------------------------

class ScanTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, n_heads=8, n_enc=3, n_dec=3, d_ff=1024,
                 max_len=64, dropout=0.1, pad_src=0, pad_tgt=0):
        super().__init__()
        self.d_model = d_model
        self.pad_src, self.pad_tgt = pad_src, pad_tgt
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_src)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_tgt)
        self.src_pos = nn.Embedding(max_len, d_model)
        self.tgt_pos = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_enc)
        self.decoder = nn.TransformerDecoder(dec, n_dec)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src):
        pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        x = self.src_emb(src) + self.src_pos(pos)
        mask = (src == self.pad_src)
        return self.encoder(x, src_key_padding_mask=mask), mask

    def decode(self, tgt, mem, src_mask):
        pos = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        y = self.tgt_emb(tgt) + self.tgt_pos(pos)
        L = tgt.size(1)
        causal = torch.triu(torch.ones(L, L, device=tgt.device, dtype=torch.bool), 1)
        tpad = (tgt == self.pad_tgt)
        return self.out_proj(self.decoder(y, mem, tgt_mask=causal,
                                          tgt_key_padding_mask=tpad,
                                          memory_key_padding_mask=src_mask))

    def forward(self, src, tgt_in):
        m, sm = self.encode(src)
        return self.decode(tgt_in, m, sm)

    @torch.no_grad()
    def greedy_decode(self, src, sos_id, eos_id, max_len=50):
        m, sm = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=src.device)
        fin = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, m, sm)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            nxt = torch.where(fin.unsqueeze(-1), torch.full_like(nxt, self.pad_tgt), nxt)
            ys = torch.cat([ys, nxt], dim=1)
            fin = fin | (nxt.squeeze(-1) == eos_id)
            if fin.all(): break
        return ys


# ---------------------------------------------------------------------------
# Train & eval
# ---------------------------------------------------------------------------

def evaluate(model, loader, src_vocab, tgt_vocab):
    sos, eos, pad = tgt_vocab[SOS], tgt_vocab[EOS], tgt_vocab[PAD]
    inv = {i: t for t, i in tgt_vocab.items()}
    model.eval()
    correct, total = 0, 0
    for src, _, _, outps in loader:
        src = src.to(DEVICE)
        pred = model.greedy_decode(src, sos, eos)
        for i in range(pred.size(0)):
            seq = pred[i, 1:].tolist()
            if eos in seq: seq = seq[:seq.index(eos)]
            words = [inv[t] for t in seq if t not in (pad, sos, eos)]
            if " ".join(words) == outps[i]: correct += 1
            total += 1
    return correct / total


def train_one(regime, seed, base_train_pairs, test_pairs, epochs, batch_size=128, lr=5e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda": torch.cuda.manual_seed_all(seed)

    # Resolve effective training set (with or without augmentation)
    if regime in ("prim_subst_augment", "combined"):
        train_pairs = augment_with_primitive_substitution(base_train_pairs, "jump", rng)
    else:
        train_pairs = base_train_pairs

    # Build vocabs from FULL (train ∪ test) tokens to avoid UNK leak on test
    src_vocab, tgt_vocab = build_vocabs(train_pairs + test_pairs)
    pad_s, pad_t = src_vocab[PAD], tgt_vocab[PAD]
    sos_t, eos_t = tgt_vocab[SOS], tgt_vocab[EOS]

    ds_train = ScanDataset(train_pairs, src_vocab, tgt_vocab)
    ds_test  = ScanDataset(test_pairs,  src_vocab, tgt_vocab)

    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False,
                              collate_fn=lambda b: collate(b, pad_s, pad_t))

    # Ordering schedule. For "combined", use compositional_depth on augmented set.
    order_regime = regime
    if regime == "prim_subst_augment":
        order_regime = "random"
    elif regime == "combined":
        order_regime = "compositional_depth"
    schedules = build_orders(train_pairs, order_regime, epochs, rng)

    model = ScanTransformer(len(src_vocab), len(tgt_vocab),
                             pad_src=pad_s, pad_tgt=pad_t).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Warmup + cosine
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = (len(train_pairs) // batch_size + 1) * epochs
    warmup = max(100, total_steps // 20)
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.CrossEntropyLoss(ignore_index=pad_t)

    best_test = 0.0; final_test = 0.0
    history = []
    step = 0
    for ep in range(epochs):
        model.train()
        order = schedules[ep]
        for start in range(0, len(order), batch_size):
            sub_idx = order[start:start + batch_size]
            batch = [ds_train[j] for j in sub_idx]
            src, tgt, _, _ = collate(batch, pad_s, pad_t)
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            opt.zero_grad()
            loss = crit(model(src, tgt_in).reshape(-1, model.out_proj.out_features),
                         tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1

        if (ep + 1) % 10 == 0 or ep == epochs - 1 or ep == 4:
            acc = evaluate(model, test_loader, src_vocab, tgt_vocab)
            best_test = max(best_test, acc); final_test = acc
            history.append({"epoch": ep + 1, "test_acc": acc, "lr": sched.get_last_lr()[0]})
            print(f"    [{regime} s={seed}] ep {ep+1}/{epochs} test={acc:.4f} (best={best_test:.4f})")

    return best_test, final_test, history, n_params


# ---------------------------------------------------------------------------
# State & runner
# ---------------------------------------------------------------------------

def load_state():
    return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}

def save_state(s): OUT.write_text(json.dumps(s, indent=2))

def is_done(state, regime, seed):
    return any(r["regime"] == regime and r["seed"] == seed for r in state["runs"])


def bootstrap_ci(vals, n_boot=10000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    if len(vals) < 2: return float(vals.mean()), float(vals.mean()), float(vals.mean())
    bs = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo = float(np.percentile(bs, (100 - ci) / 2))
    hi = float(np.percentile(bs, 100 - (100 - ci) / 2))
    return float(vals.mean()), lo, hi


def report(state):
    print("\n" + "=" * 86)
    print("DEEP CURRICULUM EXPERIMENT v2 — SCAN add_prim_jump")
    print("=" * 86)

    regimes = ["random", "anti_curriculum", "length_easy", "compositional_depth",
                "parse_depth", "prim_subst_augment", "combined"]
    base_vals = np.array([r["best_test"] for r in state["runs"] if r["regime"] == "random"])
    print(f"\n{'regime':<22s}  {'mean ± std':>16s}  {'95% CI':>18s}  "
          f"{'Δ vs random':>12s}  {'wins/n':>7s}  {'wilcoxon p':>10s}")
    for regime in regimes:
        vals = np.array([r["best_test"] for r in state["runs"] if r["regime"] == regime])
        if len(vals) == 0: continue
        mu = vals.mean(); sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        _, lo, hi = bootstrap_ci(vals.tolist())
        if regime == "random" or len(base_vals) == 0:
            gap_s, wins_s, p_s = "—", "—", "—"
        else:
            n = min(len(vals), len(base_vals))
            gap = (vals[:n].mean() - base_vals[:n].mean()) * 100
            wins = int((vals[:n] > base_vals[:n]).sum())
            gap_s = f"{gap:+8.2f}pp"; wins_s = f"{wins}/{n}"
            if wilcoxon and n >= 3:
                try:
                    p = wilcoxon(vals[:n], base_vals[:n], alternative="greater").pvalue
                    p_s = f"{p:.3f}"
                except ValueError:
                    p_s = "—"
            else:
                p_s = "—"
        print(f"{regime:<22s}  {mu:.4f} ± {sd:.4f}  [{lo:.4f},{hi:.4f}]  "
              f"{gap_s:>12s}  {wins_s:>7s}  {p_s:>10s}")

    # Calibration check
    aug = np.array([r["best_test"] for r in state["runs"] if r["regime"] == "prim_subst_augment"])
    if len(aug) > 0:
        print(f"\nCalibration check (prim_subst_augment, expected ≈ 99% per Andreas 2020):")
        if aug.mean() < 0.5:
            print(f"  !! ACHIEVED {aug.mean()*100:.1f}% — well below expected. Pipeline may be miscalibrated.")
        elif aug.mean() < 0.85:
            print(f"  ?  ACHIEVED {aug.mean()*100:.1f}% — below literature value but in the ballpark.")
        else:
            print(f"  ✓  ACHIEVED {aug.mean()*100:.1f}% — pipeline reproduces literature.")

    print(f"\nInterpretation:")
    print(f"  - If prim_subst_augment ≫ random (e.g. >50pp): training-data intervention DOES")
    print(f"    close the gap. Curriculum vs augmentation tells us which type of intervention works.")
    print(f"  - If no curriculum gets >5pp over random: ordering doesn't matter, augmentation does.")
    print(f"  - If prim_subst_augment doesn't reproduce ≥95%: pipeline is broken, results invalid.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--regimes", nargs="+",
                    default=["random", "anti_curriculum", "length_easy",
                            "compositional_depth", "parse_depth",
                            "prim_subst_augment", "combined"])
    ap.add_argument("--smoke", action="store_true",
                    help="Quick smoke test: 1 seed × 20 epochs × 3 regimes")
    args = ap.parse_args()

    state = load_state()
    if args.report: report(state); return

    if args.smoke:
        args.seeds = [0]; args.epochs = 20
        args.regimes = ["random", "prim_subst_augment", "compositional_depth"]
        print("[smoke] 1 seed × 20 epochs × 3 conditions")

    print(f"[v2] runs done so far: {len(state['runs'])}")
    print(f"[v2] regimes={args.regimes}  seeds={args.seeds}  epochs={args.epochs}")
    print(f"[v2] expected runs: {len(args.regimes) * len(args.seeds)}")

    train_pairs, test_pairs = download_scan("add_prim_jump")
    print(f"[v2] train={len(train_pairs)}  test={len(test_pairs)}")

    for regime in args.regimes:
        for seed in args.seeds:
            if is_done(state, regime, seed):
                print(f"[v2] skip {regime} seed={seed} (done)")
                continue
            print(f"\n--- {regime} seed={seed} ({args.epochs} epochs) ---")
            t0 = time.time()
            best, final, hist, n_params = train_one(
                regime, seed, train_pairs, test_pairs, args.epochs)
            state["runs"].append({
                "regime": regime, "seed": seed,
                "best_test": float(best), "final_test": float(final),
                "epochs": args.epochs, "n_params": n_params,
                "wall_time_s": time.time() - t0, "history": hist,
            })
            save_state(state)
            print(f"  -> {regime} s={seed} best={best:.4f} final={final:.4f} "
                  f"({time.time()-t0:.0f}s)")

    report(state)


if __name__ == "__main__":
    main()
