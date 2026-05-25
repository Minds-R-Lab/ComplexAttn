"""zsre_data.py — load the zsRE (Levy et al. 2017) knowledge-editing benchmark.

zsRE was originally a zero-shot relation extraction dataset; it was repurposed
for knowledge editing by the ROME / MEMIT line of work and is the second-most-
standard real-data benchmark for sequential editing (after CounterFact). The
MEMIT-formatted JSON we use has the same record shape as CounterFact, with
the addition of a "locality" prompt (about an unrelated entity) used in
place of the "neighborhood" prompts CounterFact uses for specificity.

We download the public copy from the ROME maintainers on first use.

ZsreTriple is API-identical to CounterFactTriple so the run_zsre.py harness
can dispatch to the same methods without changes.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SFIB_DIR = Path(__file__).parent
DATA_DIR = SFIB_DIR / "data" / "zsre"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ZSRE_URL = "https://rome.baulab.info/data/dsets/zsre_mend_eval.json"
ZSRE_PATH = DATA_DIR / "zsre_mend_eval.json"


@dataclass
class ZsreTriple:
    """A zsRE edit. API-identical to CounterFactTriple so the same methods work."""
    case_id: int
    subject: str
    prompt_template: str        # e.g., "Where was {} born?"
    target_new: str             # the edit target
    target_true: str            # the pre-edit answer
    paraphrase_prompts: list[str] = field(default_factory=list)
    # zsRE's "locality" is one paired prompt+answer for specificity testing
    locality_prompt: str = ""
    locality_answer: str = ""

    @property
    def relation(self) -> str:
        return self.prompt_template

    @property
    def obj(self) -> str:
        return self.target_new

    def as_tuple(self):
        return (self.subject, self.relation, self.target_new)

    @property
    def rewrite_prompt(self) -> str:
        return self.prompt_template.format(self.subject)

    # The harness expects neighborhood_prompts; zsRE provides one locality
    # prompt per record. Expose it as a single-element list for compat.
    @property
    def neighborhood_prompts(self) -> list[str]:
        return [self.locality_prompt] if self.locality_prompt else []


def download_zsre(force: bool = False) -> Path:
    if ZSRE_PATH.exists() and not force:
        return ZSRE_PATH
    print(f"[zsre] downloading from {ZSRE_URL}...")
    print(f"[zsre] (this is a ~3 MB file from the ROME maintainers)")
    urllib.request.urlretrieve(ZSRE_URL, ZSRE_PATH)
    print(f"[zsre] saved -> {ZSRE_PATH}")
    return ZSRE_PATH


def _extract_target(field_value):
    """zsRE/MEMIT records sometimes have target as dict {"str": "..."} and
    sometimes as a bare string. Handle both."""
    if isinstance(field_value, dict):
        return field_value.get("str", "")
    return str(field_value) if field_value else ""


def load_zsre(n: int | None = None, seed: int = 0,
                shuffle: bool = True) -> list[ZsreTriple]:
    path = download_zsre()
    with open(path) as f:
        raw = json.load(f)

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(raw)

    if n is not None:
        raw = raw[:n]

    out: list[ZsreTriple] = []
    for i, rec in enumerate(raw):
        # MEMIT's zsRE format has two slightly different layouts in practice;
        # we handle both.
        if "requested_rewrite" in rec:
            rr = rec["requested_rewrite"]
            subject = rr.get("subject", "")
            prompt = rr.get("prompt", "")
            target_new = _extract_target(rr.get("target_new", ""))
            target_true = _extract_target(rr.get("target_true", ""))
            paraphrases = rec.get("paraphrase_prompts", []) or []
            # locality from neighborhood_prompts if present
            locality = ""
            locality_ans = ""
            if rec.get("neighborhood_prompts"):
                # use the first as the single locality prompt
                neigh = rec["neighborhood_prompts"]
                if neigh and isinstance(neigh[0], str):
                    locality = neigh[0]
        else:
            # Direct-shape zsRE: subject, src, alt, rephrase, loc, loc_ans
            subject = rec.get("subject", "")
            # The "src" field is a fully-formed question; turn it into a
            # prompt template by substituting the subject with {} if present.
            src = rec.get("src", "")
            if subject and subject in src:
                prompt = src.replace(subject, "{}", 1)
            else:
                # Couldn't find subject in src; use src as-is. This means
                # rewrite_prompt will just be src, and the .format(subject)
                # call in our base class will be a no-op (no placeholder).
                prompt = src
            target_new = _extract_target(rec.get("alt", ""))
            # zsRE's "answers" is a list; pre-edit answer is the first
            answers = rec.get("answers", [])
            target_true = answers[0] if answers else ""
            paraphrases = []
            if rec.get("rephrase"):
                paraphrases = [rec["rephrase"]]
            locality = rec.get("loc", "")
            locality_ans = rec.get("loc_ans", "")

        out.append(ZsreTriple(
            case_id=int(rec.get("case_id", i)),
            subject=subject,
            prompt_template=prompt,
            target_new=target_new,
            target_true=target_true,
            paraphrase_prompts=[p for p in paraphrases if isinstance(p, str)],
            locality_prompt=locality,
            locality_answer=locality_ans,
        ))
    return out


def zsre_splits(n_edits: int = 500, seed: int = 0
                ) -> tuple[list[ZsreTriple], list[ZsreTriple]]:
    """Returns (edits, held-out specificity set) — mirror of cf_splits."""
    all_ex = load_zsre(n=n_edits + 500, seed=seed, shuffle=True)
    return all_ex[:n_edits], all_ex[n_edits:n_edits + 500]


if __name__ == "__main__":
    examples = load_zsre(n=3)
    print(f"--- 3 sample zsRE triples ---")
    for ex in examples:
        print(f"  case_id={ex.case_id}  subject={ex.subject!r}")
        print(f"    rewrite:     {ex.rewrite_prompt!r}")
        print(f"    target_new:  {ex.target_new!r}  (edit to this)")
        print(f"    target_true: {ex.target_true!r}")
        if ex.paraphrase_prompts:
            print(f"    paraphrase:  {ex.paraphrase_prompts[0]!r}")
        if ex.locality_prompt:
            print(f"    locality:    {ex.locality_prompt!r}  ->  {ex.locality_answer!r}")
        print()
