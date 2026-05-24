"""MemoryNet — retrieval over a learnable codebook as the basic primitive.

Each block:
    q = W_q · x                      (learnable query projection)
    a = softmax(q · K^T / sqrt(d))  (soft attention over learnable keys)
    y = a · V                        (weighted sum of learnable values)

This is single-head attention with a *static* learnable memory (no input
keys/values). The block is doing explicit retrieval against a database
of patterns rather than function approximation.

Why this might be different from ReLU+linear:
- Sample efficiency: the codebook can store rare/atypical training patterns
  directly. The block produces sharp outputs when an input closely matches a
  stored key.
- Discrete-like selectivity through soft attention. As temperature lowers
  (or as keys become well-separated), the block approaches argmax retrieval.
- Naturally interpretable: each prediction has identifiable "nearest codes."

Practical risk on the dynamical task: similar to tropical — if the data's
natural computation is *summing* contributions over time (frequency
integration), then a primitive that *selects* one code at a time can struggle.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class MemoryBlock(nn.Module):
    def __init__(self, in_dim: int, num_codes: int, key_dim: int, out_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(in_dim, key_dim)
        self.keys = nn.Parameter(torch.randn(num_codes, key_dim) / math.sqrt(key_dim))
        self.values = nn.Parameter(torch.randn(num_codes, out_dim) * 0.1)
        self.scale = math.sqrt(key_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(x)                          # (B, key_dim)
        logits = q @ self.keys.T / self.scale           # (B, num_codes)
        attn = logits.softmax(dim=-1)
        return attn @ self.values                       # (B, out_dim)


class MemoryNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        num_layers: int = 2,
        num_codes: int = 40,
        key_dim: int = 16,
        hidden_dim: int = 16,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        blocks = [MemoryBlock(input_dim, num_codes, key_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            blocks.append(MemoryBlock(hidden_dim, num_codes, key_dim, hidden_dim))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    from stacked import DeepMLP
    print(f"{'L':>3s}  {'MemoryNet':>10s}  {'DeepMLP-32':>11s}")
    for L in [1, 2, 4]:
        mem = MemoryNet(num_layers=L)
        mlp = DeepMLP(num_layers=L)
        print(f"{L:>3d}  {count_trainable(mem):>10d}  {count_trainable(mlp):>11d}")
    # forward + grad sanity at L=4
    m = MemoryNet(num_layers=4); m.train()
    x = torch.randn(4, 64); y = torch.randint(0, 4, (4,))
    loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward()
    nz = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    tot = sum(1 for p in m.parameters() if p.requires_grad)
    print(f"\nL=4 gradient sanity: {nz}/{tot} params received nonzero gradient")
