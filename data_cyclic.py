"""
Generalized cyclic-rotation task: Z/n for arbitrary n.

This is Exp 3's mod-3 setup with n made a parameter, so we can sweep
across n = 2, 3, 5, 7, 11, 13 and watch how each architecture's
generalization scales with group order.

Each input has:
  - exactly one atom from n classes (A_0, ..., A_{n-1})
  - some number k of TWIRL tokens (each rotates class by +1 mod n)
  - some number of filler tokens (semantically irrelevant)
  - randomized order

Label = (atom_class + k) mod n.

Vocab layout (depends on n):
  0:          PAD
  1:          CLS
  2..n+1:     atom tokens A_0..A_{n-1}
  n+2:        TWIRL
  n+3..n+2+M: fillers (M = NUM_FILLERS)

NUM_FILLERS is held fixed across n so filler distractors are comparable.
"""

import random
import torch
from typing import List, Tuple

PAD_ID  = 0
CLS_ID  = 1
NUM_FILLERS = 10


class CyclicTaskSpec:
    """Holds the vocab layout for a given group order n.

    Why a class instead of module-level constants: when n varies across
    runs we need a fresh vocab each time. This bundles the indices and
    a couple of generator helpers, keeping the contract with the rest
    of the code identical to data_triple.py.
    """
    def __init__(self, n: int):
        assert n >= 2
        self.n           = n
        self.num_classes = n
        self.atom_ids    = tuple(range(2, 2 + n))
        self.twirl_id    = 2 + n
        self.filler_start = self.twirl_id + 1
        self.num_fillers  = NUM_FILLERS
        self.vocab_size   = self.filler_start + NUM_FILLERS


def generate_sentence(spec: CyclicTaskSpec,
                      num_twirls: int,
                      num_fillers: int,
                      atom_class: int,
                      rng: random.Random) -> Tuple[List[int], int]:
    atom_id = spec.atom_ids[atom_class]
    body = [atom_id] + [spec.twirl_id] * num_twirls
    for _ in range(num_fillers):
        body.append(rng.randint(spec.filler_start,
                                 spec.filler_start + spec.num_fillers - 1))
    rng.shuffle(body)

    tokens = [CLS_ID] + body
    label  = (atom_class + num_twirls) % spec.n
    return tokens, label


def make_dataset(spec: CyclicTaskSpec,
                 num_samples: int,
                 depth_range: Tuple[int, int],
                 filler_range: Tuple[int, int] = (0, 8),
                 seed: int = 0):
    rng = random.Random(seed)
    examples, depths, max_len = [], [], 0
    for _ in range(num_samples):
        d = rng.randint(depth_range[0], depth_range[1])
        f = rng.randint(filler_range[0], filler_range[1])
        c = rng.randint(0, spec.n - 1)
        toks, lab = generate_sentence(spec, d, f, c, rng)
        examples.append((toks, lab)); depths.append(d)
        max_len = max(max_len, len(toks))

    tokens = torch.full((num_samples, max_len), PAD_ID, dtype=torch.long)
    labels = torch.empty(num_samples, dtype=torch.long)
    for i, (toks, lab) in enumerate(examples):
        tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        labels[i] = lab
    return tokens, labels, torch.tensor(depths, dtype=torch.long)


def make_depth_stratified_eval(spec: CyclicTaskSpec,
                                samples_per_depth: int,
                                depths: List[int],
                                filler_range: Tuple[int, int] = (0, 8),
                                seed: int = 999):
    rng = random.Random(seed)
    examples, all_depths, max_len = [], [], 0
    for d in depths:
        for _ in range(samples_per_depth):
            f = rng.randint(filler_range[0], filler_range[1])
            c = rng.randint(0, spec.n - 1)
            toks, lab = generate_sentence(spec, d, f, c, rng)
            examples.append((toks, lab)); all_depths.append(d)
            max_len = max(max_len, len(toks))

    N = len(examples)
    tokens = torch.full((N, max_len), PAD_ID, dtype=torch.long)
    labels = torch.empty(N, dtype=torch.long)
    for i, (toks, lab) in enumerate(examples):
        tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        labels[i] = lab
    return tokens, labels, torch.tensor(all_depths, dtype=torch.long)


if __name__ == "__main__":
    rng = random.Random(0)
    for n in (2, 3, 5, 7, 11):
        spec = CyclicTaskSpec(n)
        print(f"n={n:>2}  vocab={spec.vocab_size}  twirl_id={spec.twirl_id}  "
              f"atom_ids={spec.atom_ids}  filler=[{spec.filler_start},"
              f"{spec.filler_start + spec.num_fillers - 1}]")
    # Spot-check a few labels.
    spec = CyclicTaskSpec(5)
    print(f"\nZ/5 examples (chance = 1/5 = 0.20):")
    for c in range(5):
        for d in range(7):
            toks, lab = generate_sentence(spec, d, 1, c, rng)
            assert lab == (c + d) % 5
    print("  all labels match (c + k) mod 5 ✓")
