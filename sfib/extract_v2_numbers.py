"""extract_v2_numbers.py -- Ground-truth extractor for NEUCOM-D-26-11036 V2.

Reads every V2 result JSON in results/ and prints a canonical, human-readable
report of every metric that appears in the revised paper or response letter.
Run this on the H100 side after all V2 experiments finish, redirect stdout to
a file, and send that file back. It is the source of truth against which the
paper and response letter should be checked.

Usage:
    cd sfib/
    python extract_v2_numbers.py > v2_ground_truth.txt

The script is deliberately defensive: if a file is missing or a key is absent
it prints "MISSING" rather than fabricating a value. If a number in the paper
does not match a number here, the paper is wrong.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load(name: str) -> dict | None:
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "MISSING"
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return str(x)

def sec(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)

def row(label: str, *vals: Any) -> None:
    cells = "  |  ".join(fmt(v) for v in vals)
    print(f"  {label:<40s}  {cells}")

# --------------------------------------------------------------------------- #
# section: distance distributions (R3.8, Table 12 in paper)
# --------------------------------------------------------------------------- #

sec("distance_distributions_qwen0_5b.json  (R3.8 -- Table: distance-distributions)")
d = load("distance_distributions_qwen0_5b.json")
if d is None:
    print("  MISSING file")
else:
    print("  keys at top:", list(d.keys()))
    # expected keys (adjust to match actual JSON shape once you see it):
    # 'cosine_stats' / 'euclidean_stats' with per-class means/stds/p5/p95
    for cls in ("own_rewrite", "own_paraphrase", "specificity", "unrelated"):
        c = d.get(cls, {})
        row(f"{cls} cosine mean/std",
            c.get("cos_mean"), c.get("cos_std"))
        row(f"{cls} cosine p5/p95",
            c.get("cos_p5"), c.get("cos_p95"))
        row(f"{cls} euclidean mean/std",
            c.get("euc_mean"), c.get("euc_std"))

# --------------------------------------------------------------------------- #
# section: FP / FN analysis (R3.8, prose in tau-Pareto section)
# --------------------------------------------------------------------------- #

sec("fp_fn_qwen0_5b.json  (R3.8 -- FP/FN at tau=0.7)")
d = load("fp_fn_qwen0_5b.json")
if d is None:
    print("  MISSING file")
else:
    print("  keys at top:", list(d.keys()))
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

# --------------------------------------------------------------------------- #
# section: memory-latency benchmark (R3.3, Table: scalability)
# --------------------------------------------------------------------------- #

sec("memlat_qwen0_5b.json  (R3.3 -- Table: scalability)")
d = load("memlat_qwen0_5b.json")
if d is None:
    print("  MISSING file")
else:
    print("  keys at top:", list(d.keys()))
    # expected: per-N row with latency_ms mean+/-std, peak_mem_mb, populate_s
    rows = d.get("rows") or d.get("per_N") or []
    for r in rows:
        row(f"N={r.get('N')}",
            f"lat={r.get('latency_ms')}",
            f"peak_mb={r.get('peak_mem_mb')}",
            f"pop_s/edit={r.get('populate_s_per_edit')}")

# --------------------------------------------------------------------------- #
# section: tau-sweep Pareto (R3.2, Tables: tau-sweep + tau-sweep-1p5b + grace-eps)
# --------------------------------------------------------------------------- #

for label, fname in [
    ("SHARD tau-sweep, Qwen2.5-0.5B + CounterFact",  "tau_pareto_qwen0_5b_cf.json"),
    ("SHARD tau-sweep, Qwen2.5-1.5B + CounterFact",  "tau_pareto_qwen1_5b_cf.json"),
]:
    sec(f"{fname}  ({label})")
    d = load(fname)
    if d is None:
        print("  MISSING file")
        continue
    print("  keys at top:", list(d.keys()))
    # SHARD sub-sweep
    shard = d.get("shard") or d.get("SHARD") or {}
    if shard:
        print("  --- SHARD ---")
        rows = shard.get("rows") or []
        print(f"  {'tau':<8}{'Eff':<10}{'Gen':<10}{'Spec':<10}{'n_fired':<12}")
        for r in rows:
            print(f"  {r.get('tau'):<8}{r.get('Eff'):<10}{r.get('Gen'):<10}"
                  f"{r.get('Spec'):<10}{str(r.get('n_fired')):<12}")
    # GRACE sub-sweep
    grace = d.get("grace") or d.get("GRACE") or {}
    if grace:
        print("  --- GRACE ---")
        rows = grace.get("rows") or []
        print(f"  {'eps':<8}{'Eff':<10}{'Gen':<10}{'Spec':<10}")
        for r in rows:
            print(f"  {r.get('eps'):<8}{r.get('Eff'):<10}{r.get('Gen'):<10}"
                  f"{r.get('Spec'):<10}")

# --------------------------------------------------------------------------- #
# section: multi-seed scaling (R3.4, Table: multiseed-scaling)
# --------------------------------------------------------------------------- #

sec("multi-seed scaling (R3.4 -- Table: multiseed-scaling)")
# The driver mode writes one file per (model, seed); scan all matching names.
for f in sorted(RESULTS_DIR.glob("scaling*seed*.json")):
    d = json.loads(f.read_text())
    print(f"  {f.name}")
    print(f"    model={d.get('model')}  seed={d.get('seed')}  N={d.get('n_edits')}")
    row("Eff/Gen/Spec/Com",
        d.get("Eff@500") or d.get("Eff"),
        d.get("Gen@500") or d.get("Gen"),
        d.get("Spec@500") or d.get("Spec"),
        d.get("Com@500") or d.get("Com"))

# --------------------------------------------------------------------------- #
# section: long stream (R3.7, Table: long-stream)
# --------------------------------------------------------------------------- #

sec("long_stream_qwen0_5b.json  (R3.7 -- Table: long-stream)")
d = load("long_stream_qwen0_5b.json")
if d is None:
    print("  MISSING file")
else:
    print("  keys at top:", list(d.keys()))
    anchors = d.get("anchors") or d.get("per_N") or []
    print(f"  {'N':<8}{'Eff':<10}{'Gen':<10}{'Spec':<10}")
    for r in anchors:
        print(f"  {r.get('N'):<8}{r.get('Eff'):<10}{r.get('Gen'):<10}{r.get('Spec'):<10}")

# --------------------------------------------------------------------------- #
# section: composition oracle (R2.3, Table: composition-oracle)
# --------------------------------------------------------------------------- #

sec("composition_oracle.json  (R2.3 -- Table: composition-oracle)")
d = load("composition_oracle.json")
if d is None:
    print("  MISSING file")
else:
    print("  keys at top:", list(d.keys()))
    per_model = d.get("per_model") or d.get("results") or {}
    for m, r in per_model.items():
        row(f"{m}",
            f"oracle={r.get('oracle_acc')}",
            f"n_probes={r.get('n_probes')}")

# --------------------------------------------------------------------------- #
# section: SHARD-on-7B composition (R2.3, prose in composition section)
# --------------------------------------------------------------------------- #

sec("composition_shard_7b.json  (R2.3 -- SHARD on Qwen2.5-7B)")
d = load("composition_shard_7b.json")
if d is None:
    print("  MISSING file")
else:
    row("model", d.get("model"))
    row("layer", d.get("layer"))
    row("tau", d.get("tau"))
    row("n_edits", d.get("n_edits"))
    row("n_probes", d.get("n_probes"))
    row("Com@500", d.get("Com@500"))

# --------------------------------------------------------------------------- #

sec("done")
print("  If any number in paper.tex or response_to_reviewers.tex does not")
print("  match the corresponding row above, the manuscript is the one that")
print("  is wrong.")
