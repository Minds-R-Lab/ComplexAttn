# SHARD-for-Mamba

Adapts the SHARD knowledge-editing primitive (from the
`addressable-memory` branch) to Mamba state-space language models.
The closest prior work is ROMBA (Sen Sharma, Atkinson, Bau, COLM 2024,
[arXiv:2404.03646](https://arxiv.org/abs/2404.03646)), which applied
*single-edit* ROME to Mamba's `out_proj` (W_o) matrix. SHARD-for-Mamba
takes that finding as the starting point and runs sequential streams
of 500+ edits via a slot-bank wrapper attached to the same projection.

## Files

| File | What it does |
|---|---|
| `mamba_adapter.py` | Locates the edit site inside a frozen Mamba model (HuggingFace Mamba-1 / Mamba-2 layouts). |
| `shard_mamba.py` | `SHARDMambaMethod` -- captures `k*` from the input to `out_proj`, optimizes a delta to its output via ROME-style v\*-optimization, stores `(k*, delta_v)` as a slot. Cosine routing with fixed threshold at the last sequence position. |
| `shard_mamba_realdata_patches.py` | Rewrite-builder monkeypatches for CounterFact / zsRE triples. |
| `run_shard_mamba.py` | Standalone runner; imports `cf_eval` / `zsre_eval` from `sfib/` so metrics are identical to the transformer SHARD numbers. |
| `run_shard_mamba_smoke.sh` | Fast 130m smoke test (~15 min on H100). |
| `run_shard_mamba_full.sh` | Full Mamba-2.8B sweep on CounterFact + zsRE (~6-8 hours). |

## Quick start

```bash
cd sfib_mamba

# Fast smoke test -- Mamba-130m, 500 edits CounterFact:
./run_shard_mamba_smoke.sh

# Full sweep -- Mamba-2.8B, matching ROMBA's setup:
./run_shard_mamba_full.sh
```

## Design notes

* **Edit site.** Default is `out_proj` (W_o), the strongest single-edit
  site identified by ROMBA. `--kind` exposes `in_proj` and `x_proj`
  for ablation.
* **Capture position.** Default is `subject_last` (matching ROMBA's
  causal-tracing finding that fact-bearing states localize at the last
  subject token in middle layers). The `--capture_position prompt_last`
  flag captures at the last prompt position (matching SHARD-for-transformer's
  default).
* **Fire position.** Default is `last` (the last sequence position of
  every forward pass, mirroring SHARD-for-transformers). `--fire_position all`
  fires at every position, primarily useful as an ablation that should
  hurt specificity.
* **Hyperparameters.** Default `v_steps=200`, `v_lr=1.0`,
  `v_weight_decay=0`, `v_norm_constraint=20`, `tau=0.7` -- the
  SwiGLU-tuned settings from the transformer SHARD paper, which we
  expect to transfer to Mamba's higher-dimensional activations. Pilot
  ablations on `Mamba-130m` should confirm.

## What this codebase does NOT yet include

* No control-systems extensions (LQR closed-form `delta_v`,
  controllability/observability gramians). Those are planned as a
  second phase once the empirical SHARD-for-Mamba result is validated.
* No GRACE-for-Mamba baseline yet. Adding one is straightforward
  (port `grace_method.py` to the Mamba adapter) and is the natural
  next file to write.
