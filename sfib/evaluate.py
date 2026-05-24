"""evaluate.py — Evaluation harness for the SFIB benchmark.

Given a model (any HuggingFace-style causal LM) and the KB, computes the three
pre-registered metrics:

    Insertion@N : did the model learn the N inserted facts?
    Retention@N : do the pretrain facts still answer correctly?
    Composition@N : do multi-hop queries combining old+new answer correctly?

Metric definition (locked):
    Exact-match accuracy after greedy decoding, on a normalized form of the
    target string. Normalization: strip + lowercase + collapse whitespace.
    The target is considered matched if it appears (as a contiguous substring)
    in the first `max_new_tokens` generated tokens after the prompt. We use
    "first occurrence" so trailing chatter doesn't matter.

We use multiple query templates per fact and average over them — this
catches models that memorize surface forms.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from kb_data import (
    KB, Triple, RELATIONS,
    render_eval_query, render_composition,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Strip + lowercase + collapse whitespace + remove trailing punctuation."""
    t = text.strip().lower()
    t = _WS_RE.sub(" ", t)
    # Remove trailing punctuation that GPT-style models often append
    while t and t[-1] in ".,!?;:'\"":
        t = t[:-1]
    return t


def _is_match(generated: str, target: str) -> bool:
    """True iff normalized target appears as a substring in normalized generated."""
    g, t = _normalize(generated), _normalize(target)
    return t in g if t else False


# ---------------------------------------------------------------------------
# Eval data containers
# ---------------------------------------------------------------------------

@dataclass
class EvalExample:
    prompt: str
    target: str
    kind: str   # "insertion" | "retention" | "composition"
    meta: dict = field(default_factory=dict)


def build_eval_set(kb: KB,
                    n_pretrain_eval: int = 500,
                    n_insert_eval: int | None = None,
                    n_compose_eval: int | None = None,
                    queries_per_triple: int = 2,
                    seed: int = 0) -> list[EvalExample]:
    """Build a held-out evaluation set covering all three metrics.

    n_pretrain_eval : number of pretrain triples to randomly hold out for
                       retention testing (drawn from kb.pretrain_triples).
    n_insert_eval   : if None, evaluate all inserted facts.
    n_compose_eval  : if None, evaluate all composition pairs.
    queries_per_triple : how many query templates per triple (averages over them).
    """
    import random
    rng = random.Random(seed)
    examples: list[EvalExample] = []

    # Retention: a random sample of pretrain triples
    pretrain_eval = list(kb.pretrain_triples)
    rng.shuffle(pretrain_eval)
    pretrain_eval = pretrain_eval[:n_pretrain_eval]
    for t in pretrain_eval:
        for q_idx in range(queries_per_triple):
            prompt, target = render_eval_query(t, template_idx=q_idx)
            examples.append(EvalExample(
                prompt=prompt, target=target, kind="retention",
                meta={"triple": t.as_tuple(), "q_idx": q_idx},
            ))

    # Insertion: all (or n_insert_eval) inserted facts
    insert_eval = kb.insert_triples if n_insert_eval is None \
                   else kb.insert_triples[:n_insert_eval]
    for t in insert_eval:
        for q_idx in range(queries_per_triple):
            prompt, target = render_eval_query(t, template_idx=q_idx)
            examples.append(EvalExample(
                prompt=prompt, target=target, kind="insertion",
                meta={"triple": t.as_tuple(), "q_idx": q_idx},
            ))

    # Composition: all (or n_compose_eval) multi-hop queries
    compose_eval = kb.compose_pairs if n_compose_eval is None \
                    else kb.compose_pairs[:n_compose_eval]
    for (ta, tb, qtext) in compose_eval:
        prompt, target = render_composition(ta, tb, qtext)
        examples.append(EvalExample(
            prompt=prompt, target=target, kind="composition",
            meta={"pretrain_triple": ta.as_tuple(),
                  "inserted_triple": tb.as_tuple()},
        ))

    return examples


# ---------------------------------------------------------------------------
# Generation + scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_greedy(model, tokenizer, prompts: list[str],
                     max_new_tokens: int = 12, batch_size: int = 16) -> list[str]:
    """Greedy decode `max_new_tokens` after each prompt. Returns a list of
    GENERATED-ONLY strings (the prompt is stripped from the result).

    CRITICAL: decoder-only models REQUIRE left-padding for batched generation.
    With right-padding, the prompt's last token is at a different position
    for each batch element, and generation produces garbage. We temporarily
    set padding_side='left' inside this function and restore it on exit.
    """
    device = next(model.parameters()).device
    model.eval()
    # Save & set padding_side for generation
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out: list[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                             truncation=True, max_length=256).to(device)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            gen = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            # With left-padding, input length is uniform; new tokens begin at
            # position input_ids.shape[1] for every batch element.
            for j, full_ids in enumerate(gen):
                new_ids = full_ids[input_ids.shape[1]:]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                out.append(text)
    finally:
        tokenizer.padding_side = orig_padding_side
    return out


def score_examples(examples: list[EvalExample], generations: list[str]
                    ) -> dict:
    """Score the generations against examples. Returns aggregate + per-kind."""
    assert len(examples) == len(generations)
    by_kind = {"insertion": [], "retention": [], "composition": []}
    per_example_records = []
    for ex, gen in zip(examples, generations):
        ok = _is_match(gen, ex.target)
        by_kind[ex.kind].append(ok)
        per_example_records.append({
            "kind": ex.kind, "prompt": ex.prompt, "target": ex.target,
            "generated": gen, "correct": ok, "meta": ex.meta,
        })
    summary = {}
    for kind, hits in by_kind.items():
        if not hits:
            summary[kind] = {"n": 0, "accuracy": None}
        else:
            summary[kind] = {"n": len(hits),
                              "accuracy": float(sum(hits) / len(hits))}
    return {"summary": summary, "records": per_example_records}


def evaluate_model(model, tokenizer, examples: list[EvalExample],
                    max_new_tokens: int = 12, batch_size: int = 16) -> dict:
    """End-to-end: generate from prompts, score, return summary."""
    prompts = [ex.prompt for ex in examples]
    generations = generate_greedy(model, tokenizer, prompts,
                                    max_new_tokens=max_new_tokens,
                                    batch_size=batch_size)
    return score_examples(examples, generations)


# ---------------------------------------------------------------------------
# Smoke test (no GPT-2 needed — uses a mock model)
# ---------------------------------------------------------------------------

def _mock_test():
    """Verify the normalizer + scorer logic without needing a real model."""
    # _normalize sanity
    assert _normalize("  Hello, World!  ") == "hello, world"
    assert _normalize("yes") == "yes"
    # _is_match sanity
    assert _is_match("the answer is Threnn.", "threnn")
    assert _is_match("Eldric Stenn", "eldric stenn")
    assert not _is_match("nothing here", "threnn")
    # Build small KB and eval set
    from kb_data import generate_kb
    kb = generate_kb(seed=0, n_pretrain=50, n_insert=20, n_compose=10,
                     n_entities_a=10, n_entities_b=8)
    examples = build_eval_set(kb,
                                n_pretrain_eval=10,
                                n_insert_eval=10,
                                queries_per_triple=2)
    print(f"Built {len(examples)} eval examples")
    # Counts by kind
    by_kind = {"insertion": 0, "retention": 0, "composition": 0}
    for ex in examples: by_kind[ex.kind] += 1
    print(f"  by kind: {by_kind}")
    # Mock perfect / mock random
    perfect = [ex.target for ex in examples]
    half_correct = [ex.target if i % 2 == 0 else "nonsense"
                     for i, ex in enumerate(examples)]
    print(f"\nPerfect-generation score: {score_examples(examples, perfect)['summary']}")
    print(f"\n50% random score:        {score_examples(examples, half_correct)['summary']}")
    print(f"\n=== eval harness sanity OK ===")


if __name__ == "__main__":
    _mock_test()
