"""
Mod-3 rotation task (TripleParity).

Generalizes parity-of-negation from Z/2 to Z/3. Each input has:
  - exactly one atom from three classes (A0, A1, A2)
  - some number k of TWIRL tokens (each "rotates" the class by +1 mod 3)
  - some number of filler tokens (semantically irrelevant)
  - randomized order (the task is order-invariant)

Label = (atom_class + k) mod 3   ∈ {0, 1, 2}

Why this task:
  The natural symmetry group is Z/3Z, which is *not* a subgroup of the
  multiplicative group of the real numbers. A real-valued additive
  network with a linear readout cannot solve this at OOD depths: linear
  functions of k cannot be periodic. A complex network on the unit
  circle solves it natively: each TWIRL multiplies the state by
  ω = e^{i·2π/3}, so the unit-circle phase encodes the class directly.
  Gated networks (real or complex) escape the linear-in-k constraint
  via non-linear state updates.

  This is the cleanest separation experiment: Z/2 confounds "complex"
  with "multiplicative composition" because ±1 lives natively in both
  algebras. Z/3 doesn't.

Vocabulary indices:
  0:  PAD
  1:  CLS
  2:  A0    (atom of class 0)
  3:  A1    (atom of class 1)
  4:  A2    (atom of class 2)
  5:  TWIRL (rotate class by +1 mod 3)
  6..N: fillers
"""

import random
import torch
from typing import List, Tuple

PAD_ID   = 0
CLS_ID   = 1
A_IDS    = (2, 3, 4)
A0_ID, A1_ID, A2_ID = A_IDS
TWIRL_ID = 5
FILLER_START = 6
NUM_FILLERS  = 10
VOCAB_SIZE   = FILLER_START + NUM_FILLERS   # 16 total
NUM_CLASSES  = 3


def generate_sentence(num_twirls: int,
                      num_fillers: int,
                      atom_class: int,
                      rng: random.Random) -> Tuple[List[int], int]:
    """Build one (token_ids, label) pair for the mod-3 task."""
    atom_id = A_IDS[atom_class]
    body = [atom_id] + [TWIRL_ID] * num_twirls
    for _ in range(num_fillers):
        body.append(rng.randint(FILLER_START, FILLER_START + NUM_FILLERS - 1))
    rng.shuffle(body)

    tokens = [CLS_ID] + body
    label  = (atom_class + num_twirls) % NUM_CLASSES
    return tokens, label


def make_dataset(num_samples: int,
                 depth_range: Tuple[int, int],
                 filler_range: Tuple[int, int] = (0, 8),
                 seed: int = 0):
    """Tensor dataset; depth (num_twirls) ~ Uniform[depth_range]."""
    rng = random.Random(seed)
    examples, depths, max_len = [], [], 0
    for _ in range(num_samples):
        d = rng.randint(depth_range[0], depth_range[1])
        f = rng.randint(filler_range[0], filler_range[1])
        c = rng.randint(0, NUM_CLASSES - 1)
        toks, lab = generate_sentence(d, f, c, rng)
        examples.append((toks, lab)); depths.append(d)
        max_len = max(max_len, len(toks))

    tokens = torch.full((num_samples, max_len), PAD_ID, dtype=torch.long)
    labels = torch.empty(num_samples, dtype=torch.long)
    for i, (toks, lab) in enumerate(examples):
        tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        labels[i] = lab
    return tokens, labels, torch.tensor(depths, dtype=torch.long)


def make_depth_stratified_eval(samples_per_depth: int,
                                depths: List[int],
                                filler_range: Tuple[int, int] = (0, 8),
                                seed: int = 999):
    """Equal samples at every requested depth, for the OOD curve."""
    rng = random.Random(seed)
    examples, all_depths, max_len = [], [], 0
    for d in depths:
        for _ in range(samples_per_depth):
            f = rng.randint(filler_range[0], filler_range[1])
            c = rng.randint(0, NUM_CLASSES - 1)
            toks, lab = generate_sentence(d, f, c, rng)
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
    print(f"Vocab size: {VOCAB_SIZE}, num classes: {NUM_CLASSES}")
    print("Sample sentences  (depth k, atom class c) -> label = (c+k) mod 3:")
    for c in range(3):
        for d in range(5):
            toks, lab = generate_sentence(d, 2, c, rng)
            expected = (c + d) % 3
            ok = "OK" if lab == expected else "WRONG"
            print(f"  c={c} k={d}  tokens={toks}  label={lab}  ({ok})")
    tokens, labels, depths = make_dataset(8, (0, 5), seed=42)
    print(f"\nDataset shape: {tokens.shape}  labels={labels.tolist()}  depths={depths.tolist()}")
