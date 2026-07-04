"""probe_schemas.py -- One-shot schema probe for the files whose top-level
keys didn't match extract_v2_numbers.py's expectations. Prints ~50 lines.

Run once:
    cd sfib/
    python probe_schemas.py
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).parent / "results"

TARGETS = [
    "tau_pareto_qwen0_5b_cf.json",
    "tau_pareto_qwen1_5b_cf.json",
    "long_stream_qwen0_5b.json",
    "composition_oracle.json",
    "memlat_qwen0_5b.json",
    "scaling_qwen0.5b_counterfact_seed1.json",
]


def show(v, depth=0, max_depth=2):
    ind = "  " * depth
    if isinstance(v, dict):
        for k in list(v.keys())[:8]:
            child = v[k]
            if isinstance(child, dict):
                print(f"{ind}{k!r}: dict(keys={list(child.keys())[:6]})")
                if depth < max_depth:
                    show(child, depth + 1, max_depth)
            elif isinstance(child, list):
                print(f"{ind}{k!r}: list(len={len(child)})")
                if child and isinstance(child[0], dict):
                    print(f"{ind}  first-item keys: {list(child[0].keys())[:8]}")
                    print(f"{ind}  first-item sample: {dict(list(child[0].items())[:5])}")
                elif child:
                    print(f"{ind}  first-item: {child[0]!r}"[:100])
            else:
                s = repr(child)
                if len(s) > 70:
                    s = s[:67] + "..."
                print(f"{ind}{k!r}: {type(child).__name__} = {s}")


for name in TARGETS:
    print()
    print("=" * 60)
    print(f"  {name}")
    print("=" * 60)
    p = R / name
    if not p.exists():
        print("  MISS")
        continue
    d = json.loads(p.read_text())
    print(f"  top-level type: {type(d).__name__}")
    if isinstance(d, dict):
        print(f"  top-level keys: {list(d.keys())}")
        show(d)
    elif isinstance(d, list):
        print(f"  top-level list len: {len(d)}")
        if d:
            print(f"  first-item: {d[0]}"[:200])
