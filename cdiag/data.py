"""
The copy task: a controlled long-range dependency benchmark.

Input sequence structure:
    [t1, t2, ..., t_K]  K random tokens from alphabet {1, ..., A}
    [0, 0, ..., 0]      L zeros (blank)
    [GO]                one "start producing" signal (token A+1)
    [0, 0, ..., 0]      K-1 zeros (waiting for output)

Output target:
    [0, 0, ..., 0]                (positions 1 .. K+L+1, output is irrelevant)
    [t1, t2, ..., t_K]            (positions K+L+2 .. K+L+K+1, must reproduce)

So sequence length is K + L + K = 2K + L. The model must carry K
tokens worth of information across L blank steps.

For SSM use we treat the input as a sequence of one-hot vectors of
dim (A+2), and the model outputs logits of dim (A+1) at each
position. We compute cross-entropy only over the K output positions.

For a copy delay of L, the optimal impulse response of the SSM has
a peak at time L+1 -- a near-delta function offset by the appropriate
amount. This is intrinsically oscillatory (a delta is a sum of
cosines), which is the regime where complex SSMs should help per
the Ran-Milo theory.
"""

import torch


# Vocabulary:
#   0: PAD / blank
#   1..A: data tokens
#   A+1: GO (start producing) signal
#
# Output vocab:
#   0: blank (don't produce data)
#   1..A: data tokens

PAD = 0


def make_batch(batch_size: int, K: int, L: int, A: int,
               device: str = "cpu", seed: int | None = None):
    """Generate a batch for the copy task.

    Returns
    -------
    x_onehot : tensor of shape [B, T, A+2]  one-hot input
    target   : tensor of shape [B, T]        target token ids (0 elsewhere,
                                              data tokens at output positions)
    output_mask : bool tensor [B, T] True at the K output positions
    """
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
    else:
        g = None

    T = 2 * K + L + 1              # total sequence length: K data + L blank + 1 GO + K output
    # Data tokens drawn uniformly from {1, ..., A}
    data = torch.randint(1, A + 1, (batch_size, K), generator=g)

    x = torch.zeros(batch_size, T, dtype=torch.long)
    # Place data tokens at positions 0 .. K-1
    x[:, :K] = data
    # GO signal at position K + L
    x[:, K + L] = A + 1
    # Positions K..K+L-1 and K+L+1..T-1 are blank (already 0)

    target = torch.zeros(batch_size, T, dtype=torch.long)
    target[:, K + L + 1 : K + L + 1 + K] = data

    output_mask = torch.zeros(batch_size, T, dtype=torch.bool)
    output_mask[:, K + L + 1 : K + L + 1 + K] = True

    # One-hot encode input
    x_onehot = torch.nn.functional.one_hot(x, num_classes=A + 2).float()

    return (x_onehot.to(device),
            target.to(device),
            output_mask.to(device))


if __name__ == "__main__":
    x, y, m = make_batch(batch_size=4, K=5, L=20, A=4, seed=0)
    print(f"x shape: {x.shape}      (B, T, vocab_in)")
    print(f"y shape: {y.shape}      (B, T)")
    print(f"m shape: {m.shape}      (B, T)  -- output mask")
    print(f"First sample input tokens (argmax):")
    print(x[0].argmax(-1))
    print(f"First sample target tokens:")
    print(y[0])
    print(f"First sample output mask:")
    print(m[0])
