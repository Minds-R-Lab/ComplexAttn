"""ablations_realdata_patches.py -- CounterFact/zsRE rewrite-builder
monkey-patches for AblatedSHARDMethod. Same pattern as grace_realdata_patches.
"""
from __future__ import annotations

import ablations  # noqa: F401  (registers into METHOD_REGISTRY)
from ablations import AblatedSHARDMethod

try:
    from counterfact_data import CounterFactTriple
except Exception:
    CounterFactTriple = None  # type: ignore

try:
    from zsre_data import ZsreTriple
except Exception:
    ZsreTriple = None  # type: ignore


_ORIG = AblatedSHARDMethod._build_rewrite


def _ablated_build_rewrite(self, triple):
    if (CounterFactTriple is not None and isinstance(triple, CounterFactTriple)) \
       or (ZsreTriple is not None and isinstance(triple, ZsreTriple)):
        prompt = triple.prompt_template.format(triple.subject)
        target = triple.target_new
        if not target.startswith(" "):
            target = " " + target
        return prompt, target
    return _ORIG(self, triple)


AblatedSHARDMethod._build_rewrite = _ablated_build_rewrite  # type: ignore[assignment]
