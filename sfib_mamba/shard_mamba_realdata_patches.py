"""shard_mamba_realdata_patches.py -- CF/zsRE rewrite-builder monkeypatches
for SHARDMambaMethod, mirroring grace_realdata_patches and ablations_realdata_patches.
"""

from __future__ import annotations

import shard_mamba  # noqa: F401 -- registers SHARDMambaMethod
from shard_mamba import SHARDMambaMethod

try:
    from counterfact_data import CounterFactTriple
except Exception:
    CounterFactTriple = None  # type: ignore

try:
    from zsre_data import ZsreTriple
except Exception:
    ZsreTriple = None  # type: ignore


_ORIG = SHARDMambaMethod._build_rewrite


def _build_rewrite_realdata(self, triple):
    if (CounterFactTriple is not None and isinstance(triple, CounterFactTriple)) \
       or (ZsreTriple is not None and isinstance(triple, ZsreTriple)):
        prompt = triple.prompt_template.format(triple.subject)
        target = triple.target_new
        if not target.startswith(" "):
            target = " " + target
        return prompt, target, triple.subject
    return _ORIG(self, triple)


SHARDMambaMethod._build_rewrite = _build_rewrite_realdata  # type: ignore[assignment]
