"""
Synthetic data generator for the scoped-negation parity task.

The task in plain English:
  A sentence contains exactly one truth atom (T or F), some number of NOT
  tokens that flip the valence, and some number of filler tokens that don't
  affect the answer. The label is the final valence: atom_value * (-1)^num_nots.

Why this task:
  - It isolates the algebraic claim. A real-valued transformer must learn to
    count NOT-parity from data. A complex-valued model with phase composition
    can in principle solve it with a single phase-flip primitive.
  - It admits a clean depth-generalization test: train on shallow nesting
    (few NOTs), test on deeper nesting than ever seen during training.
  - Position order is irrelevant to the label — this prevents the real model
    from cheating via a memorized positional template. The model must actually
    count.

Vocabulary indices:
  0: PAD
  1: CLS    (sentinel position used for classification readout)
  2: T      (atom, valence +1)
  3: F      (atom, valence -1)
  4: NOT
  5..N: filler tokens (semantically irrelevant)
"""

import random
import torch
from typing import List, Tuple

PAD_ID  = 0
CLS_ID  = 1
T_ID    = 2
F_ID    = 3
NOT_ID  = 4
FILLER_START = 5
NUM_FILLERS  = 10        # 10 distinct filler words
VOCAB_SIZE   = FILLER_START + NUM_FILLERS   # 15 total


def generate_sentence(num_nots: int,
                      num_fillers: int,
                      atom_value: int,
                      rng: random.Random) -> Tuple[List[int], int]:
    """
    Build one (token_ids, label) pair.

    num_nots:    how many NOT tokens to include (depth of negation)
    num_fillers: how many filler tokens to mix in (acts as distractor noise)
    atom_value:  +1 for T, -1 for F
    rng:         python Random instance for reproducibility

    Returns:
      token_ids:  list of ints starting with CLS, then a shuffled mix of
                  one atom + num_nots NOT tokens + num_fillers fillers
      label:      0 if final valence is +1, 1 if -1   (suitable for BCE)
    """
    atom_id = T_ID if atom_value == +1 else F_ID
    body = [atom_id] + [NOT_ID] * num_nots
    for _ in range(num_fillers):
        body.append(rng.randint(FILLER_START, FILLER_START + NUM_FILLERS - 1))
    rng.shuffle(body)

    tokens = [CLS_ID] + body
    valence = atom_value * ((-1) ** num_nots)
    label = 0 if valence == +1 else 1
    return tokens, label


def make_dataset(num_samples: int,
                 depth_range: Tuple[int, int],
                 filler_range: Tuple[int, int] = (0, 8),
                 seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a tensor dataset with depth (num_nots) sampled uniformly from
    [depth_range[0], depth_range[1]] inclusive.

    Returns:
      tokens:  LongTensor [N, L]      padded with PAD_ID
      labels:  LongTensor [N]         0 or 1
      depths:  LongTensor [N]         num_nots per sample (for stratified eval)
    """
    rng = random.Random(seed)
    examples = []
    depths   = []
    max_len  = 0
    for _ in range(num_samples):
        d = rng.randint(depth_range[0], depth_range[1])
        f = rng.randint(filler_range[0], filler_range[1])
        v = rng.choice([+1, -1])
        toks, lab = generate_sentence(d, f, v, rng)
        examples.append((toks, lab))
        depths.append(d)
        if len(toks) > max_len:
            max_len = len(toks)

    tokens = torch.full((num_samples, max_len), PAD_ID, dtype=torch.long)
    labels = torch.empty(num_samples, dtype=torch.long)
    for i, (toks, lab) in enumerate(examples):
        tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        labels[i] = lab
    return tokens, labels, torch.tensor(depths, dtype=torch.long)


def make_depth_stratified_eval(samples_per_depth: int,
                                depths: List[int],
                                filler_range: Tuple[int, int] = (0, 8),
                                seed: int = 999) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an eval set with an equal number of samples at each requested depth.
    This is what we use for the out-of-distribution depth-generalization plot.
    """
    rng = random.Random(seed)
    examples = []
    all_depths = []
    max_len = 0
    for d in depths:
        for _ in range(samples_per_depth):
            f = rng.randint(filler_range[0], filler_range[1])
            v = rng.choice([+1, -1])
            toks, lab = generate_sentence(d, f, v, rng)
            examples.append((toks, lab))
            all_depths.append(d)
            if len(toks) > max_len:
                max_len = len(toks)

    N = len(examples)
    tokens = torch.full((N, max_len), PAD_ID, dtype=torch.long)
    labels = torch.empty(N, dtype=torch.long)
    for i, (toks, lab) in enumerate(examples):
        tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        labels[i] = lab
    return tokens, labels, torch.tensor(all_depths, dtype=torch.long)


if __name__ == "__main__":
    # Sanity check: print a few examples and verify labels.
    rng = random.Random(0)
    print("Vocab size:", VOCAB_SIZE)
    print("Sample sentences (depth -> tokens -> label):")
    for d in range(5):
        toks, lab = generate_sentence(d, 3, +1, rng)
        expected = 0 if (-1) ** d == 1 else 1
        ok = "OK" if lab == expected else "WRONG"
        print(f"  d={d}  atom=T  tokens={toks}  label={lab} ({ok})")

    tokens, labels, depths = make_dataset(num_samples=8,
                                          depth_range=(0, 3),
                                          seed=42)
    print("\nDataset shape:", tokens.shape, labels.shape, depths.shape)
    print("Depths:", depths.tolist())
