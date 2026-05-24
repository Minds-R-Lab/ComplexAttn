"""kb_data.py — Sequential Fact Insertion Benchmark (SFIB) data generator.

Generates a synthetic knowledge base over fictional entities with three corpora:

  pretrain  : 5,000 facts about Group A entities (used to train the base model)
  insert    : 1,000 facts about Group B entities (used for sequential insertion)
  compose   : 500 multi-hop queries combining Group A + Group B facts

All entities are deliberately fictional so the base model cannot fall back on
pretraining knowledge. Names are constructed from syllable templates.

Design constraints (pre-registered):

  - Each fact is a triple (subject, relation, object).
  - Relations: lives_in, mayor_of, occupation_of, spouse_of, born_in,
               studied_at, owns_a, allied_with
  - Each fact is rendered into 1 of 3 templated surface forms (to break
    surface-form memorization).
  - Group A and Group B entity sets are DISJOINT (no entity in both).
  - Composition queries pair one Group-A fact with one Group-B fact and
    require traversing the edge.
  - Deterministic given a seed.

Public API:
    generate_kb(seed=0, n_pretrain=5000, n_insert=1000, n_compose=500) -> dict
    render_fact(triple, template_idx=None) -> (text, target)
    render_query(triple, template_idx=None) -> (question, answer)
    render_composition(triple_a, triple_b) -> (question, answer)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Entity / name generation
# ---------------------------------------------------------------------------

# Fictional-sounding syllable pools. Anything composed from these is extremely
# unlikely to appear verbatim in GPT-2's pretraining corpus.
_SYLL_FIRST = ["El", "Bren", "Quor", "Thrann", "Vael", "Mor", "Cyr", "Ith",
                "Sair", "Zar", "Lir", "Korv", "Mae", "Ren", "Tald", "Kael",
                "Yor", "Drel", "Esh", "Pirn", "Vorn", "Glo", "Threnn", "Ash"]
_SYLL_MID  = ["a", "ai", "e", "ie", "o", "ou", "u", "ae", "i", "y"]
_SYLL_LAST = ["dric", "vinn", "thar", "wyn", "rell", "kor", "sten", "vald",
               "fyrn", "lorn", "vex", "thorn", "ven", "qel", "drin", "myr",
               "ral", "soth", "kell", "verth", "brand", "lirn", "kos", "vyr"]

_CITY_PREFIX = ["Vorn", "El", "Threnn", "Kael", "Cyr", "Mor", "Lir", "Drel",
                 "Aesh", "Quor", "Tald", "Ith", "Yor", "Pirn", "Sair", "Glo"]
_CITY_SUFFIX = ["hall", "marsh", "spire", "ford", "reach", "haven", "moor",
                 "vale", "wick", "stead", "mire", "thorpe", "crag", "fast",
                 "stone", "watch", "barrow", "hold", "deep", "garde",
                 "court", "spire2", "keep", "bridge", "harbor", "ridge"]

_OCCUPATIONS = ["baker", "blacksmith", "scholar", "merchant", "fisher",
                 "weaver", "alchemist", "scribe", "cartographer", "miner",
                 "vintner", "carpenter", "herbalist", "navigator", "cleric"]

_UNIVERSITIES_PREFIX = ["Voss", "Threnn", "Kael", "Aelyn", "Mor", "Quor", "Ith",
                         "Cyr", "Drel", "Yor"]
_UNIVERSITIES_SUFFIX = ["Academy", "Conservatory", "Institute", "College",
                         "Lyceum"]

_GOODS = ["bakery", "smithy", "library", "tannery", "vineyard", "tavern",
          "apothecary", "trading-post", "lighthouse", "observatory"]

_FACTIONS = ["Stoneward Order", "Silver Wreath", "Iron Crescent",
              "Lantern Society", "Mossback Guild", "Whitespear Compact",
              "Glassrose League", "Hollow Veil"]


def _make_name(rng: random.Random) -> str:
    return rng.choice(_SYLL_FIRST) + rng.choice(_SYLL_MID) + rng.choice(_SYLL_LAST)


def _make_city(rng: random.Random) -> str:
    return rng.choice(_CITY_PREFIX) + rng.choice(_CITY_SUFFIX)


def _make_university(rng: random.Random) -> str:
    return rng.choice(_UNIVERSITIES_PREFIX) + " " + rng.choice(_UNIVERSITIES_SUFFIX)


# ---------------------------------------------------------------------------
# Relation schema
# ---------------------------------------------------------------------------

@dataclass
class Relation:
    name: str
    subject_type: str   # 'person' | 'city'
    object_type: str    # 'person' | 'city' | 'occupation' | 'university' | 'good' | 'faction'
    # Fact templates: 3 variants per relation, all of which include the same triple.
    # {s} = subject, {o} = object.
    fact_templates: tuple[str, ...]
    # Query templates: question, where the answer is the object.
    query_templates: tuple[tuple[str, str], ...]  # (question, answer-prefix-stripper)


RELATIONS: dict[str, Relation] = {
    "lives_in": Relation(
        name="lives_in", subject_type="person", object_type="city",
        fact_templates=(
            "{s} lives in {o}.",
            "The home of {s} is {o}.",
            "{s} resides in the city of {o}.",
        ),
        query_templates=(
            ("Where does {s} live? Answer:", ""),
            ("What city is {s}'s home? Answer:", ""),
        ),
    ),
    "mayor_of": Relation(
        name="mayor_of", subject_type="city", object_type="person",
        fact_templates=(
            "The mayor of {s} is {o}.",
            "{o} is the mayor of {s}.",
            "{s}'s mayor is {o}.",
        ),
        query_templates=(
            ("Who is the mayor of {s}? Answer:", ""),
            ("{s} is governed by which mayor? Answer:", ""),
        ),
    ),
    "occupation_of": Relation(
        name="occupation_of", subject_type="person", object_type="occupation",
        fact_templates=(
            "{s} works as a {o}.",
            "{s} is a {o} by trade.",
            "The occupation of {s} is {o}.",
        ),
        query_templates=(
            ("What does {s} do for a living? Answer:", ""),
            ("{s}'s occupation is what? Answer:", ""),
        ),
    ),
    "spouse_of": Relation(
        name="spouse_of", subject_type="person", object_type="person",
        fact_templates=(
            "{s} is married to {o}.",
            "{s}'s spouse is {o}.",
            "{o} is the spouse of {s}.",
        ),
        query_templates=(
            ("Who is {s} married to? Answer:", ""),
            ("Who is {s}'s spouse? Answer:", ""),
        ),
    ),
    "born_in": Relation(
        name="born_in", subject_type="person", object_type="city",
        fact_templates=(
            "{s} was born in {o}.",
            "{s}'s birthplace is {o}.",
            "{o} is the birthplace of {s}.",
        ),
        query_templates=(
            ("Where was {s} born? Answer:", ""),
            ("What is {s}'s birthplace? Answer:", ""),
        ),
    ),
    "studied_at": Relation(
        name="studied_at", subject_type="person", object_type="university",
        fact_templates=(
            "{s} studied at {o}.",
            "{s} is an alumnus of {o}.",
            "{o} is where {s} received an education.",
        ),
        query_templates=(
            ("Where did {s} study? Answer:", ""),
            ("Which institution did {s} attend? Answer:", ""),
        ),
    ),
    "owns_a": Relation(
        name="owns_a", subject_type="person", object_type="good",
        fact_templates=(
            "{s} owns a {o}.",
            "{s} is the proprietor of a {o}.",
            "The {o} belongs to {s}.",
        ),
        query_templates=(
            ("What does {s} own? Answer:", ""),
            ("What establishment belongs to {s}? Answer:", ""),
        ),
    ),
    "allied_with": Relation(
        name="allied_with", subject_type="person", object_type="faction",
        fact_templates=(
            "{s} is allied with the {o}.",
            "{s} swears allegiance to the {o}.",
            "The {o} counts {s} among its members.",
        ),
        query_templates=(
            ("Which faction is {s} allied with? Answer:", ""),
            ("What is {s}'s allegiance? Answer:", ""),
        ),
    ),
}


# ---------------------------------------------------------------------------
# Triple type
# ---------------------------------------------------------------------------

@dataclass
class Triple:
    subject: str
    relation: str
    obj: str

    def as_tuple(self): return (self.subject, self.relation, self.obj)


# ---------------------------------------------------------------------------
# KB generator
# ---------------------------------------------------------------------------

@dataclass
class KB:
    pretrain_triples: list[Triple]   = field(default_factory=list)
    insert_triples:   list[Triple]   = field(default_factory=list)
    compose_pairs:    list[tuple[Triple, Triple, str]] = field(default_factory=list)
    # compose_pairs: (pretrain_triple, insert_triple, multi_hop_question_text)
    group_a_entities: set[str]       = field(default_factory=set)
    group_b_entities: set[str]       = field(default_factory=set)

    def to_dict(self):
        return {
            "pretrain": [t.as_tuple() for t in self.pretrain_triples],
            "insert":   [t.as_tuple() for t in self.insert_triples],
            "compose":  [
                {"prereq_pretrain": pa.as_tuple(),
                 "inserted":        ib.as_tuple(),
                 "question":        q}
                for (pa, ib, q) in self.compose_pairs
            ],
            "group_a_entities": sorted(self.group_a_entities),
            "group_b_entities": sorted(self.group_b_entities),
        }


def _fact_to_text(triple: Triple, template_idx: int = 0) -> str:
    rel = RELATIONS[triple.relation]
    tmpl = rel.fact_templates[template_idx % len(rel.fact_templates)]
    return tmpl.format(s=triple.subject, o=triple.obj)


def _triple_to_query(triple: Triple, template_idx: int = 0) -> tuple[str, str]:
    rel = RELATIONS[triple.relation]
    q_tmpl, _ = rel.query_templates[template_idx % len(rel.query_templates)]
    question = q_tmpl.format(s=triple.subject)
    return question, triple.obj


def generate_kb(seed: int = 0,
                n_pretrain: int = 5000,
                n_insert: int = 1000,
                n_compose: int = 500,
                n_entities_a: int = 800,
                n_entities_b: int = 300) -> KB:
    """Generate a deterministic synthetic KB.

    n_entities_a/b: how many *people* in each group (cities are generated
    separately as needed). Avg facts-per-person = n_pretrain / n_entities_a.
    """
    rng = random.Random(seed)
    kb = KB()

    # ---- pools (shared cities, universities, etc.) ----
    cities = []
    seen_cities = set()
    while len(cities) < 250:
        c = _make_city(rng)
        if c not in seen_cities:
            cities.append(c); seen_cities.add(c)

    universities = []
    seen_uni = set()
    while len(universities) < 40:
        u = _make_university(rng)
        if u not in seen_uni:
            universities.append(u); seen_uni.add(u)

    # ---- Group A people ----
    group_a_people = []
    seen_names = set()
    tries = 0
    while len(group_a_people) < n_entities_a:
        tries += 1
        if tries > n_entities_a * 100:
            raise RuntimeError(f'cannot generate {n_entities_a} unique Group A names')
        name = _make_name(rng)
        if name not in seen_names:
            group_a_people.append(name); seen_names.add(name)
    kb.group_a_entities.update(group_a_people)

    # ---- Group B people (disjoint from A) ----
    group_b_people = []
    tries = 0
    while len(group_b_people) < n_entities_b:
        tries += 1
        if tries > n_entities_b * 100:
            raise RuntimeError(f'cannot generate {n_entities_b} unique Group B names')
        name = _make_name(rng)
        if name not in seen_names:
            group_b_people.append(name); seen_names.add(name)
    kb.group_b_entities.update(group_b_people)

    # ---- Reserve mayors (one per city, drawn from Group A) ----
    # Mayors are needed for both pretrain (city -> mayor) and composition queries.
    # Make sure every city has exactly one mayor in Group A so composition works.
    mayors_by_city: dict[str, str] = {}
    for city in cities:
        mayors_by_city[city] = rng.choice(group_a_people)

    # ---- Generate pretrain triples ----
    # For each Group A person, generate ~n_pretrain/n_entities_a relations.
    # Strategy: ensure each person has a 'lives_in' and other relations balanced.
    pretrain = []
    # Dedup key: (subject, relation). The benchmark assumes a fact is uniquely
    # determined by (subject, relation), so we reject any later triple whose
    # (S,R) already exists in the KB. Without this check, the model receives
    # contradictory training signal (e.g., two distinct 'lives_in' cities for
    # the same person) and retention is capped well below 1.0.
    seen_sr: set[tuple[str, str]] = set()

    # Add city->mayor pretrain facts first (these are needed for composition)
    for city, mayor in mayors_by_city.items():
        pretrain.append(Triple(subject=city, relation="mayor_of", obj=mayor))
        seen_sr.add((city, "mayor_of"))

    # Now fill in person-centered facts
    relation_distribution = ["lives_in", "lives_in",   # bias toward this (needed for composition)
                              "occupation_of", "spouse_of", "born_in",
                              "studied_at", "owns_a", "allied_with"]
    # Track each person's 'lives_in' city for composition query construction
    person_to_city: dict[str, str] = {}

    max_attempts = n_pretrain * 50
    attempts = 0
    while len(pretrain) < n_pretrain and attempts < max_attempts:
        attempts += 1
        person = rng.choice(group_a_people)
        rel_name = rng.choice(relation_distribution)
        rel = RELATIONS[rel_name]
        if rel.subject_type != "person":
            continue
        # Reject (S,R) collisions BEFORE sampling the object (cheap)
        if (person, rel_name) in seen_sr:
            continue
        if rel.object_type == "city":
            obj = rng.choice(cities)
            if rel_name == "lives_in":
                person_to_city[person] = obj
        elif rel.object_type == "person":
            obj = rng.choice([p for p in group_a_people if p != person])
        elif rel.object_type == "occupation":
            obj = rng.choice(_OCCUPATIONS)
        elif rel.object_type == "university":
            obj = rng.choice(universities)
        elif rel.object_type == "good":
            obj = rng.choice(_GOODS)
        elif rel.object_type == "faction":
            obj = rng.choice(_FACTIONS)
        else:
            continue
        triple = Triple(subject=person, relation=rel_name, obj=obj)
        pretrain.append(triple)
        seen_sr.add((person, rel_name))
    if len(pretrain) < n_pretrain:
        raise RuntimeError(
            f"only generated {len(pretrain)} unique (S,R) pretrain triples; "
            f"requested {n_pretrain}. Increase n_entities_a or reduce n_pretrain."
        )
    kb.pretrain_triples = pretrain

    # ---- Generate insertion triples (all about Group B people) ----
    # Critical for composition: each insertion is "B-person lives_in <existing city>"
    # so that pretrain's (city -> mayor) edge can be combined.
    insert: list[Triple] = []
    seen_sr_insert: set[tuple[str, str]] = set()
    # First: ensure every Group B person has a 'lives_in' insertion, because that's
    # the bridge for composition queries.
    for person in group_b_people:
        city = rng.choice(cities)
        insert.append(Triple(subject=person, relation="lives_in", obj=city))
        seen_sr_insert.add((person, "lives_in"))
        if len(insert) >= n_insert: break

    # Then fill in additional varied insertions
    max_attempts_ins = n_insert * 50
    attempts_ins = 0
    while len(insert) < n_insert and attempts_ins < max_attempts_ins:
        attempts_ins += 1
        person = rng.choice(group_b_people)
        rel_name = rng.choice(["occupation_of", "spouse_of", "born_in",
                                "studied_at", "owns_a", "allied_with"])
        # Reject (S,R) collisions before sampling object
        if (person, rel_name) in seen_sr_insert:
            continue
        rel = RELATIONS[rel_name]
        if rel.object_type == "city": obj = rng.choice(cities)
        elif rel.object_type == "person":
            # Spouse can be Group A or B; pick Group B for now to keep disjoint
            obj = rng.choice([p for p in group_b_people if p != person])
        elif rel.object_type == "occupation": obj = rng.choice(_OCCUPATIONS)
        elif rel.object_type == "university": obj = rng.choice(universities)
        elif rel.object_type == "good": obj = rng.choice(_GOODS)
        elif rel.object_type == "faction": obj = rng.choice(_FACTIONS)
        triple = Triple(subject=person, relation=rel_name, obj=obj)
        insert.append(triple)
        seen_sr_insert.add((person, rel_name))
    if len(insert) < n_insert:
        raise RuntimeError(
            f"only generated {len(insert)} unique (S,R) insert triples; "
            f"requested {n_insert}. Increase n_entities_b or reduce n_insert."
        )
    kb.insert_triples = insert

    # ---- Generate composition queries ----
    # Standard 2-hop pattern: "Who is the mayor of the city where <B-person> lives?"
    # This requires: insertion fact (B-person, lives_in, city) + pretrain fact (city, mayor_of, mayor)
    compose_pairs: list[tuple[Triple, Triple, str]] = []
    b_lives = [t for t in insert if t.relation == "lives_in"
                                  and t.subject in group_b_people]
    rng.shuffle(b_lives)
    for t_b in b_lives:
        if len(compose_pairs) >= n_compose: break
        city = t_b.obj
        mayor = mayors_by_city.get(city)
        if mayor is None: continue
        # Build the pretrain triple that this depends on
        t_a = Triple(subject=city, relation="mayor_of", obj=mayor)
        # Multi-hop question
        question = (f"Who is the mayor of the city where {t_b.subject} lives? "
                     "Answer:")
        compose_pairs.append((t_a, t_b, question))

    kb.compose_pairs = compose_pairs
    return kb


# ---------------------------------------------------------------------------
# Rendering helpers used by training / evaluation
# ---------------------------------------------------------------------------

def render_train_example(triple: Triple, template_idx: int | None = None,
                          rng: random.Random | None = None) -> str:
    """A single training example (just the fact sentence)."""
    if template_idx is None:
        rng = rng or random.Random()
        template_idx = rng.randint(0, 2)
    return _fact_to_text(triple, template_idx)


def render_eval_query(triple: Triple, template_idx: int = 0) -> tuple[str, str]:
    """Returns (prompt, expected_answer). For exact-match evaluation."""
    return _triple_to_query(triple, template_idx)


def render_composition(pretrain_t: Triple, insert_t: Triple,
                        question: str) -> tuple[str, str]:
    """The composition's expected answer is the OBJECT of the pretrain triple,
    which is the chained answer to the multi-hop query."""
    return question, pretrain_t.obj


# ---------------------------------------------------------------------------
# CLI / smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kb = generate_kb(seed=0, n_pretrain=200, n_insert=50, n_compose=20,
                      n_entities_a=40, n_entities_b=15)
    print(f"Generated KB:")
    print(f"  pretrain triples : {len(kb.pretrain_triples)}")
    print(f"  insert triples   : {len(kb.insert_triples)}")
    print(f"  compose pairs    : {len(kb.compose_pairs)}")
    print(f"  Group A entities : {len(kb.group_a_entities)}")
    print(f"  Group B entities : {len(kb.group_b_entities)}")

    # Sample some examples for hand-check
    print("\n--- 5 pretrain triples (rendered) ---")
    rng = random.Random(0)
    for t in kb.pretrain_triples[:5]:
  