# ComplexAttn — research experiments on alternatives to standard neural net primitives

This repository contains two related-but-distinct research investigations into
*what the unit of learnable computation in a neural network should be*. Both
threads share a methodology — design a candidate primitive, build a
tightly-scoped benchmark for it, test against strong baselines with multi-seed
error bars, and report honestly whether it wins or loses.

The repository is a research log, not a polished library. Negative results are
kept alongside positive ones. The verdict on each investigation is at the
bottom of its section.

## Contents at a glance

| Thread | Question | Status | Verdict |
|---|---|---|---|
| **RMC** | Can a Hamiltonian flow + Fourier-resonant readout replace the affine-then-nonlinearity primitive? | Complete | Does not beat a matched-parameter MLP on tested tasks. See [`README_RMC.md`](README_RMC.md) for the full writeup. |
| **SFIB** | Can an *addressable memory primitive* solve sequential one-shot fact insertion better than weight-modification methods (sequential FT, LoRA, MEMIT)? | In progress | Baselines characterized; primitive implementation ready to run. |

The two threads share the parent directory, but the file layout makes clear
which thread each file belongs to:

```
new_nural_netwarks/
├── README.md                    ← this file
├── README_RMC.md                ← original RMC investigation writeup
├── requirements.txt
│
├── model.py / verify.py / ablations.py / train.py / run.py
├── benchmark_*.py / sample_efficiency.py / dynamical_*.py
├── multi_seed*.py / coupled_rmc.py / modrelu_rmc.py / memory_net.py
├── stacked.py / tropical_net.py / mrb_net.py / shortest_path_data.py
├── novel_blocks*.py / final_round.py / resnet_huber.py / stress_test.py
├── curriculum_*.py / gemini_v1.py
│   ↑ all RMC-thread code; see README_RMC.md for what each file does
│
├── results/                     ← RMC-thread results (PNGs, JSONs, small .pt)
│
└── sfib/                        ← SFIB-thread code (current work)
    ├── kb_data.py               KB generator
    ├── evaluate.py              eval harness + metric definitions
    ├── pretrain.py              base-model pretraining on the KB
    ├── run_baselines.py         insertion-stream baselines + addressable mem
    ├── diagnose_pretrain.py     diagnostic for retention plateau
    ├── diagnose_in_context.py   diagnostic for in-context learning
    ├── kb_seed0.json            example pre-generated KB (seed=0)
    └── results/                 baseline run outputs (JSONs)
```

Note: the `sfib/checkpoints/` directory (the pretrained GPT-2 backbones — 500
MB to 1.4 GB) is **not** in version control. Regenerate with the pretrain
script (see below).

---

## SFIB — Sequential Fact Insertion Benchmark *(current work)*

### The problem

The motivating claim, distilled to one sentence: *the standard dense neural net
fails at the most basic kind of learning a database can do — inserting one new
fact at a time without forgetting the others*. Sequential fine-tuning, LoRA,
and even MEMIT all degrade catastrophically as the number of insertions grows.
We're testing whether replacing the implicit "fact storage" mechanism of an LM
with an *explicit addressable memory primitive* solves the problem.

### Benchmark design

SFIB (Sequential Fact Insertion Benchmark) is a synthetic knowledge base of
fictional entities with three corpora:

- **2,000 pretrain facts** — taught to a base model during pretraining
- **500 insertion facts** — disjoint entities, inserted one at a time at eval time
- **200 composition pairs** — 2-hop queries requiring chaining one pretrain fact and one inserted fact

At a sequence of checkpoints `N ∈ {0, 1, 10, 50, 100, 250, 500}` we measure:

- **Insertion@N**: accuracy on the first N inserted facts' queries
- **Retention@N**: accuracy on the pretrain facts (measures catastrophic forgetting)
- **Composition@N**: accuracy on 2-hop queries whose inserted half is among the first N

All entities are constructed from fictional syllable pools so they cannot
appear verbatim in the base model's pretraining corpus. The full benchmark
spec is in `sfib/kb_data.py`.

### What we ran and what we found

A clean GPT-2 small (124M params) was pretrained on the 2,000-fact KB. After
fixing a (subject, relation) ambiguity bug in the KB generator that capped any
model at ~88% retention, the backbone hit **97.6% retention** in 6 epochs —
proving the model has the capacity to memorize this amount of factual content.

Four insertion baselines were then measured on this backbone:

| Method | Ins@1 | Ret@1 | Ins@500 | Ret@500 | Headline |
|---|---:|---:|---:|---:|---|
| Frozen (no updates) | — | 0.976 | 0.029 | 0.976 | Lower bound; no learning |
| seq_ft (lr=1e-4) | 1.000 | 0.976 | 0.146 | 0.085 | Classic CF: 89 pp retention lost |
| LoRA-seq (tuned) | 1.000 | 0.546 | 0.008 | 0.004 | Worse than full FT |
| MEMIT (layer 5) | 0.000 | 0.975 | 0.179 | 0.148 | Targeted but doesn't transfer to Q/A eval |

Composition was 0% for every method — 2-hop reasoning is bottlenecked by model
scale, not by retrieval.

Two findings worth keeping regardless of what the primitive does:

1. **GPT-2 small cannot do in-context fact retrieval at any prompt format.**
   We tested 6 prompt templates with oracle retrieval; all of them *degraded*
   accuracy by 10-20 pp compared to no-prepend. The model has the fact in its
   weights *and* in the prompt and still gets it wrong. See
   `sfib/diagnose_in_context.py`. This is an emergent-capability finding at
   model scale, and it rules out RAG-style methods as competitive baselines at
   this scale.

2. **No weight-modification baseline reaches the pre-registered thresholds.**
   The pre-registered targets were Insertion@500 ≥ 0.90, Retention@500 ≥ 0.95,
   Composition@500 ≥ 0.70. The best baseline on insertion was seq_ft at 0.146;
   the best on retention was frozen at 0.976 (but with zero insertion); the
   best from a method that actually updates is MEMIT at 0.148 retention. The
   bar for the addressable primitive is to beat *all four* on Insertion AND
   Retention simultaneously.

### The proposed primitive: an addressable memory layer

The thesis: catastrophic forgetting in MEMIT / seq_ft / LoRA is not because
facts can't be encoded — it's because they're being written into *shared
parameters that fight each other*. Storing each fact in its own slot in an
external memory bank eliminates the interference.

The design (`AddressableMemoryMethod` in `sfib/run_baselines.py`):

- A wrapper around a chosen mid-layer MLP module stores `(k*, delta_v)` pairs
  as slots in an external memory bank
- At forward, the MLP computes its normal output and adds a memory
  contribution via cosine-similarity lookup with top-1 hard selection
- At insertion, the MEMIT v\*-optimization computes the required `delta_v`,
  but instead of *modifying* the down-projection weight, we *store* the pair
  as a new slot
- The base model is never touched

Three properties this design has by construction:

- **No interference between facts**: each insertion is its own slot
- **No retention damage**: base weights frozen, memory acts purely additively
- **No capacity ceiling**: store as many slots as RAM allows

The implementation is in `sfib/run_baselines.py`; first experiment ready to
run with the command below.

### Reproducing the SFIB results

```bash
cd sfib
pip install -r ../requirements.txt
pip install transformers   # also needed for SFIB; not in the RMC requirements

# 1. Pretrain the backbone (~5 min on a 3090 for 6 epochs):
python pretrain.py --max_epochs 15 --n_pretrain 2000 --n_insert 500 --n_compose 200

# 2. Baselines (~1-10 min each):
python run_baselines.py --method frozen
python run_baselines.py --method seq_ft --lr 1e-4
python run_baselines.py --method lora_seq --lr 3e-3 --n_steps 20 --lora_alpha 32
python run_baselines.py --method memit

# 3. The proposed primitive:
python run_baselines.py --method addressable_mem

# Diagnostics:
python diagnose_pretrain.py        # explains retention < 1.0 with per-(S,R) audit
python diagnose_in_context.py      # tests 6 prompt formats for in-context retrieval
```

All results land in `sfib/results/` as small JSON files.

---

## RMC — Resonant Manifold Cells *(earlier work, complete)*

Brief summary: the Resonant Manifold Cell factors the three jobs of a standard
neuron (rotate, mix, select) into three named objects — a learned positive
definite mass matrix (geometry), a learned Hamiltonian flow (dynamics), and a
bank of learned resonant modes (selectivity). Information is processed by
evolving a particle through a symplectic flow and reading out windowed Fourier
coefficients along the trajectory.

Across MNIST, synthetic dynamical-system classification (where the
architecture's prior should fit best), and depth-scaling experiments with
multiple seeds, the RMC does not beat a matched-parameter MLP. The math holds
up — energy conservation, reversibility, gradient flow all verified
numerically — but the structured prior does not translate into measurable
generalization gains on the tasks we tested.

For the full investigation, including ablations and what we'd do next, see
[`README_RMC.md`](README_RMC.md).

---

## Methodology notes

- Every experiment in this repository is run with at least one explicit seed
  control. Multi-seed sweeps are used for any claim about generalization.
- Pre-registered thresholds are set *before* the experiment and reported
  against, even when missed. SFIB's pre-registered Composition threshold of
  0.70 will almost certainly not be reached at GPT-2-small scale; we report
  honestly against it rather than moving the goalposts.
- Negative results are kept and labeled as such. The RMC investigation's
  verdict ("does not beat MLP") is part of the published log.
- The benchmark / harness code is separated from the methods being tested so
  that adding a new method requires only adding a new entry to a registry,
  not modifying the eval logic.

## Affiliation

Minds-R-Lab. Work in progress; no peer-reviewed paper yet. Issues and PRs
welcome.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               