"""counterfact_data.py — load Meng et al.'s CounterFact benchmark.

CounterFact (Meng et al. 2022, "Locating and Editing Factual Associations
in GPT") is the canonical real-data benchmark for knowledge editing. 21,919
counterfactual edits over real-world entities, each with:

  - A rewrite prompt and a *new* target object (the edit)
  - The original target object (what the model should produce pre-edit)
  - Paraphrase prompts (for generalization testing)
  - Neighborhood prompts about related entities (for specificity testing)

We download the JSON dataset from the ROME maintainers' public server on
first use and cache it locally.

CounterFactTriple is the dataclass we use throughout the SFIB harness; it
exposes .subject, .relation, .obj fields compatible with our existing
Triple interface so the method classes in run_baselines.py don't need
to change. The relation field is a free-form prompt template instead of
a string ID.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SFIB_DIR = Path(__file__).parent
DATA_DIR = SFIB_DIR / "data" / "counterfact"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Public URL hosted by the ROME / MEMIT maintainers (rome.baulab.info)
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
COUNTERFACT_PATH = DATA_DIR / "counterfact.json"


@dataclass
class CounterFactTriple:
    """A CounterFact edit. Compatible with our existing method interfaces
    via the .subject, .relation, .obj attributes plus .as_tuple()."""
    case_id: int
    subject: str
    prompt_template: str        # e.g. "The mother tongue of {} is"
    target_new: str             # the edit target (counterfactual)
    target_true: str            # the pre-edit answer
    paraphrase_prompts: list[str] = field(default_factory=list)
    neighborhood_prompts: list[str] = field(default_factory=list)
    relation_id: str = ""       # optional: dataset's internal relation ID

    # Compatibility with Triple interface
    @property
    def relation(self) -> str:
        return self.relation_id or self.prompt_template

    @property
    def obj(self) -> str:
        return self.target_new

    def as_tuple(self):
        return (self.subject, self.relation, self.target_new)

    @property
    def rewrite_prompt(self) -> str:
        """The fully-formed rewrite prompt (subject substituted in)."""
        return self.prompt_template.format(self.subject)


def download_counterfact(force: bool = False) -> Path:
    """Download CounterFact JSON to local cache. Returns the cache path."""
    if COUNTERFACT_PATH.exists() and not force:
        return COUNTERFACT_PATH
    print(f"[counterfact] downloading from {COUNTERFACT_URL}...")
    print(f"[counterfact] (this is a ~70 MB file from the ROME maintainers)")
    urllib.request.urlretrieve(COUNTERFACT_URL, COUNTERFACT_PATH)
    print(f"[counterfact] saved -> {COUNTERFACT_PATH}")
    return COUNTERFACT_PATH


def load_counterfact(n: int | None = None, seed: int = 0,
                       shuffle: bool = True) -> list[CounterFactTriple]:
    """Load CounterFact records. If n is given, returns the first n (after
    optional shuffling for a stable seed-based ordering)."""
    path = download_counterfact()
    with open(path) as f:
        raw = json.load(f)

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(raw)

    if n is not None:
        raw = raw[:n]

    out: list[CounterFactTriple] = []
    for rec in raw:
        rr = rec["requested_rewrite"]
        para = rec.get("paraphrase_prompts", []) or []
        neigh = rec.get("neighborhood_prompts", []) or []
        out.append(CounterFactTriple(
            case_id=int(rec.get("case_id", -1)),
            subject=rr["subject"],
            prompt_template=rr["prompt"],
            target_new=rr["target_new"]["str"]
                       if isinstance(rr["target_new"], dict)
                       else rr["target_new"],
            target_true=rr["target_true"]["str"]
                       if isinstance(rr["target_true"], dict)
                       else rr["target_true"],
            paraphrase_prompts=[p for p in para if isinstance(p, str)],
            neighborhood_prompts=[p for p in neigh if isinstance(p, str)],
            relation_id=str(rec.get("relation_id", "")),
        ))
    return out


def cf_splits(n_edits: int = 500, seed: int = 0
              ) -> tuple[list[CounterFactTriple], list[CounterFactTriple]]:
    """Split CounterFact into edits and a held-out specificity evaluation set.

    Edits: first n_edits CounterFact triples (after shuffle by seed).
    Held-out specificity: next 500 triples used to measure "did unrelated
        facts survive the edit stream?" — analogous to SFIB's retention.
    The held-out set's *target_true* is the expected answer (pre-edit).
    """
    all_ex = load_counterfact(n=n_edits + 500, seed=seed, shuffle=True)
    return all_ex[:n_edits], all_ex[n_edits:n_edits + 500]


if __name__ == "__main__":
    # Smoke: download + print a few records
    examples = load_counterfact(n=3)
    print(f"--- 3 sample CounterFact triples ---")
    for ex in examples:
        print(f"  case_id={ex.case_id}  subject={ex.subject!r}")
        print(f"    prompt:      {ex.rewrite_prompt!r}")
        print(f"    target_new:  {ex.target_new!r}  (edit to this)")
        print(f"    target_true: {ex.target_true!r}  (original answer)")
        print(f"    paraphrases: {len(ex.paraphrase_prompts)}, neighborhoods: {len(ex.neighborhood_prompts)}")
        if ex.paraphrase_prompts:
            print(f"    para[0]:     {ex.paraphrase_prompts[0]!r}")
        if ex.neighborhood_prompts:
            print(f"    neigh[0]:    {ex.neighborhood_prompts[0]!r}")
        print()
