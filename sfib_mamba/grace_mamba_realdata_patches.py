"""grace_mamba_realdata_patches.py -- CF/zsRE rewrite-builder monkeypatches
for GRACEMambaMethod.
"""

from __future__ import annotations

import grace_mamba  # noqa: F401
from grace_mamba import GRACEMambaMethod

try:
    from counterfact_data import CounterFactTriple
except Exception:
    CounterFactTriple = None  # type: ignore

try:
    from zsre_data import ZsreTriple
except Exception:
    ZsreTriple = None  # type: ignore


_ORIG_BUILD = GRACEMambaMethod._build_rewrite
_ORIG_LABEL = GRACEMambaMethod._triple_label


def _build_rewrite_realdata(self, triple):
    if (CounterFactTriple is not None and isinstance(triple, CounterFactTriple)) \
       or (ZsreTriple is not None and isinstance(triple, ZsreTriple)):
        prompt = triple.prompt_template.format(triple.subject)
        target = triple.target_new
        if not target.startswith(" "):
            target = " " + target
        return prompt, target, triple.target_new.strip()
    return _ORIG_BUILD(self, triple)


def _triple_label_realdata(self, triple):
    if (CounterFactTriple is not None and isinstance(triple, CounterFactTriple)) \
       or (ZsreTriple is not None and isinstance(triple, ZsreTriple)):
        return triple.target_new.strip()
    return _ORIG_LABEL(self, triple)


GRACEMambaMethod._build_rewrite = _build_rewrite_realdata  # type: ignore[assignment]
GRACEMambaMethod._triple_label = _triple_label_realdata    # type: ignore[assignment]
