"""grace_realdata_patches.py -- monkeypatches that let GRACEMethod handle
CounterFact and zsRE triples (which have different field names than SFIB's
Triple). Mirrors the pattern run_counterfact.py and run_zsre.py already use
for MEMITMethod, SequentialFTMethod, and AddressableMemoryMethod.

Import this module from inside run_counterfact.py / run_zsre.py *after*
importing grace_method (so the GRACEMethod class is already in the registry).
The patches are applied at module-import time; the runner only needs to
`import grace_realdata_patches` for them to take effect.
"""

from __future__ import annotations

import grace_method  # noqa: F401  -- registers GRACEMethod into METHOD_REGISTRY
from grace_method import GRACEMethod

# ---------------------------------------------------------------------------
# CounterFact triples
# ---------------------------------------------------------------------------
# CounterFactTriple shape:
#   .subject               str
#   .prompt_template       str like "The mother tongue of {} is"
#   .target_new            str like " English"
#   .target_true           str like " French"
#   .paraphrase_prompts    list[str]
#   .neighborhood_prompts  list[str]
# We pick the rewrite prompt (with subject inlined) and aim for target_new.

try:
    from counterfact_data import CounterFactTriple
except Exception:
    CounterFactTriple = None  # type: ignore


# ---------------------------------------------------------------------------
# zsRE triples (API-compatible with CounterFactTriple by design)
# ---------------------------------------------------------------------------

try:
    from zsre_data import ZsreTriple
except Exception:
    ZsreTriple = None  # type: ignore


_ORIG_BUILD_REWRITE = GRACEMethod._build_rewrite
_ORIG_TRIPLE_LABEL = GRACEMethod._triple_label


def _grace_build_rewrite(self, triple):
    """CF/zsRE-aware rewrite: render the prompt with the subject inlined and
    use target_new (the counterfactual) as the answer to insert."""
    if (CounterFactTriple is not None and isinstance(triple, CounterFactTriple)) \
       or (ZsreTriple is not None and isinstance(triple, ZsreTriple)):
        prompt = triple.prompt_template.format(triple.subject)
        target = triple.target_new
        if not target.startswith(" "):
            target = " " + target
        return prompt, target
    return _ORIG_BUILD_REWRITE(self, triple)


def _grace_triple_label(self, triple):
    if (CounterFactTriple is not None and isinstance(triple, CounterFactTriple)) \
       or (ZsreTriple is not None and isinstance(triple, ZsreTriple)):
        return triple.target_new.strip()
    return _ORIG_TRIPLE_LABEL(self, triple)


GRACEMethod._build_rewrite = _grace_build_rewrite      # type: ignore[assignment]
GRACEMethod._triple_label = _grace_triple_label        # type: ignore[assignment]
