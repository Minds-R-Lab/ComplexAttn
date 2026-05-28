"""summarize_multiseed.py -- aggregate Mamba multi-seed results.

Scans results/ for SHARD-for-Mamba seed JSONs (v*, one-shot LQR, LQR-GN5,
LQR-GN10, plus GRACE if present), and prints a per-cell table of
mean +/- standard error across seeds for Eff@500, Gen@500, Spec@500.

Handles missing seeds gracefully: cells without all seeds are reported
with whatever's available, plus an n=k indicator.

Usage:
    python summarize_multiseed.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from collections import defaultdict


# Recognised filename patterns. Format:
#   {bench}_{method_tag}_mamba790m_seed{N}.json
# where method_tag identifies the variant.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
if not os.path.isdir(RESULTS_DIR):
    RESULTS_DIR = "results"

VARIANT_TAGS = [
    # (display label, file infix, sort order)
    ("SHARD-v*",      "shard_mamba",              0),
    ("SHARD-LQR-1",   "shard_mamba_lqr",          1),
    ("SHARD-LQR-GN5", "shard_mamba_lqr_gn",       2),
    ("SHARD-LQR-GN10","shard_mamba_lqr_gn10",     3),
    ("GRACE",         "grace_mamba",              4),
]

BENCHMARKS = ["counterfact", "zsre"]
BENCH_LABEL = {"counterfact": "CF", "zsre": "zsRE"}


def parse_filename(path: str):
    """Return (variant_label, bench, seed) or None if not recognised.

    We must disambiguate methods whose tags are prefixes of one another
    (e.g. 'shard_mamba' vs 'shard_mamba_lqr'). Match the most specific tag.
    """
    base = os.path.basename(path)
    m = re.match(r"^(counterfact|zsre)_(.+)_mamba790m_seed(\d+)\.json$", base)
    if not m:
        return None
    bench, mid, seed = m.group(1), m.group(2), int(m.group(3))
    # Find longest matching infix.
    matches = [(label, tag, order) for (label, tag, order) in VARIANT_TAGS
               if mid == tag]
    if not matches:
        return None
    label, _, order = matches[0]
    return (label, bench, seed, order)


def mean_stderr(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    mu = sum(xs) / n
    if n < 2:
        return (mu, float("nan"))
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return (mu, se)


def fmt_mu_se(mu: float, se: float) -> str:
    if math.isnan(mu):
        return "    n/a"
    if math.isnan(se):
        return f"{mu:6.4f}        "
    return f"{mu:6.4f} +/- {se:5.4f}"


def main() -> int:
    pattern = os.path.join(RESULTS_DIR, "*_mamba790m_seed*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"(no result files matching {pattern})")
        return 0

    # cells[(variant, bench)] -> list[(seed, eff, gen, spec, wall)]
    cells: dict[tuple, list] = defaultdict(list)
    for path in files:
        parsed = parse_filename(path)
        if parsed is None:
            continue
        variant, bench, seed, order = parsed
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:
            continue
        r500 = next((r for r in d.get("results", []) if r["N"] == 500), None)
        if r500 is None:
            continue
        eff = r500["efficacy"]["accuracy"]
        gen = r500["generalization"]["accuracy"]
        spec = r500["specificity"]["accuracy"]
        wall = d.get("wall_time_stream_s", float("nan"))
        cells[(order, variant, bench)].append((seed, eff, gen, spec, wall))

    if not cells:
        print(f"(found {len(files)} files but none parsed as Mamba-790M seed JSONs)")
        return 0

    print(f"Multi-seed summary at Mamba-790M, N=500 (scanned {len(files)} files)")
    print("=" * 92)
    print(f"{'Bench':<6} {'Variant':<16} {'seeds':<8} "
          f"{'Eff':<19} {'Gen':<19} {'Spec':<19}")
    print("-" * 92)

    keys = sorted(cells.keys(), key=lambda k: (BENCHMARKS.index(k[2]), k[0]))
    prev_bench = None
    for (order, variant, bench) in keys:
        rows = sorted(cells[(order, variant, bench)], key=lambda r: r[0])
        seeds_seen = [r[0] for r in rows]
        effs  = [r[1] for r in rows]
        gens  = [r[2] for r in rows]
        specs = [r[3] for r in rows]
        mu_e, se_e = mean_stderr(effs)
        mu_g, se_g = mean_stderr(gens)
        mu_s, se_s = mean_stderr(specs)
        seeds_str = "{" + ",".join(str(s) for s in seeds_seen) + "}"
        bench_label = BENCH_LABEL[bench]
        if bench != prev_bench and prev_bench is not None:
            print("-" * 92)
        prev_bench = bench
        print(f"{bench_label:<6} {variant:<16} {seeds_str:<8} "
              f"{fmt_mu_se(mu_e, se_e):<19} "
              f"{fmt_mu_se(mu_g, se_g):<19} "
              f"{fmt_mu_se(mu_s, se_s):<19}")

    print("=" * 92)
    print()
    print("Per-seed detail:")
    for (order, variant, bench) in keys:
        rows = sorted(cells[(order, variant, bench)], key=lambda r: r[0])
        bench_label = BENCH_LABEL[bench]
        for (seed, eff, gen, spec, wall) in rows:
            print(f"  {bench_label:<5} {variant:<16} seed={seed}  "
                  f"Eff={eff:.4f}  Gen={gen:.4f}  Spec={spec:.4f}  wall={wall:7.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
