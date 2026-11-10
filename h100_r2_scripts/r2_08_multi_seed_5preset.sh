#!/usr/bin/env bash
# =============================================================================
# R2 revision -- Reviewer 1 point R1.5b
# Multi-seed re-run of the 5-preset ablation (manuscript Section 6.5, Table 9).
#
# R1 produced seed-0 only for the 5-preset Mamba ablation on
# Mamba-790M + CounterFact. Reviewer 1 asks for the three-seed ordering
# since the pure_steer vs ablate_optim gap (0.125 vs 0.126 Gen) is below
# the ~0.0025 seed SEM observed on the STEER CF cell (Table 6).
#
# We re-run only seeds 1 and 2 across the 5 presets; the seed-0 JSONs
# already live under sfib_mamba/results/counterfact_mamba_ablation_*
# _mamba790m_seed0.json and the runner auto-skips them. Entry point is
# the existing sfib_mamba/run_mamba_ablations.py, which exposes
# --preset {shard, ablate_routing, ablate_write, ablate_optim, all_grace}.
#
# Wall-clock: 5 presets x 2 new seeds = 10 runs at ~3 h each = ~30 h on
# one H100-80GB. Run under tmux or nohup.
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/ComplexAttn/ComplexAttn}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for CAND in "$HOME/env/bin/activate" "$REPO_ROOT/.venv/bin/activate" "$HOME/ComplexAttn/env/bin/activate"; do
    if [[ -f "$CAND" ]]; then source "$CAND"; break; fi
  done
fi

cd "$REPO_ROOT/sfib_mamba"
mkdir -p results

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
N_EDITS=500
BENCH=counterfact

run_cell () {
  local PRESET=$1 SEED=$2
  local OUT="results/${BENCH}_mamba_ablation_${PRESET}_mamba790m_seed${SEED}.json"
  if [[ -f "$OUT" ]]; then
    echo "[r2_08] skip (exists): $OUT"
    return 0
  fi
  echo "============================================================="
  echo "[r2_08] preset=${PRESET}  seed=${SEED}  $(date -Iseconds)"
  echo "============================================================="
  python -u run_mamba_ablations.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" \
    --preset "$PRESET" \
    --out "$OUT"
}

echo "[r2_08] $(date -Iseconds) start: 5-preset ablation x seeds {1,2} on Mamba-790M + CF"
echo "[r2_08] venv: ${VIRTUAL_ENV:-<none>}"
echo "[r2_08] cwd:  $(pwd)"

for SEED in 1 2; do
  for PRESET in shard ablate_routing ablate_write ablate_optim all_grace; do
    run_cell "$PRESET" "$SEED"
  done
done

echo
echo "============================================================="
echo "[r2_08] multi-seed 5-preset summary at N=500 (CF, Mamba-790M):"
echo "============================================================="
python - <<'PY'
import glob, json, math, statistics as st
import os
from collections import defaultdict

by_preset = defaultdict(list)
for p in sorted(glob.glob("results/counterfact_mamba_ablation_*_mamba790m_seed*.json")):
    # filename: counterfact_mamba_ablation_<preset>_mamba790m_seed<S>.json
    name = os.path.basename(p)
    rest = name.replace("counterfact_mamba_ablation_", "").replace("_mamba790m_seed", "|")
    preset, tail = rest.split("|")
    seed = int(tail.split(".")[0])
    with open(p) as fh: d = json.load(fh)
    r = next((r for r in d.get("results", []) if r.get("N") == 500), None)
    if r is None: continue
    by_preset[preset].append((
        seed,
        r["efficacy"]["accuracy"],
        r["generalization"]["accuracy"],
        r["specificity"]["accuracy"],
    ))

print(f"{'preset':<16} {'n':>3}  seeds              "
      f"{'Eff mean+-SEM':<22} {'Gen mean+-SEM':<22} {'Spec mean+-SEM':<22}")
order = ["shard","ablate_routing","ablate_write","ablate_optim","all_grace"]
for preset in order:
    rows = sorted(by_preset.get(preset, []))
    if not rows: continue
    seeds = ",".join(str(r[0]) for r in rows)
    def ms(xs):
        m = st.mean(xs); s = st.stdev(xs) if len(xs) > 1 else 0.0
        return m, s / math.sqrt(len(xs))
    eff  = ms([r[1] for r in rows])
    gen  = ms([r[2] for r in rows])
    spec = ms([r[3] for r in rows])
    print(f"{preset:<16} {len(rows):>3}  [{seeds:<10}]  "
          f"{eff[0]:.4f}+-{eff[1]:.4f}       "
          f"{gen[0]:.4f}+-{gen[1]:.4f}       "
          f"{spec[0]:.4f}+-{spec[1]:.4f}")
PY

echo "[r2_08] $(date -Iseconds) done"
echo
echo "Next: replace the single-seed cells in manuscript tab:ablation (§6.5)"
echo "with the mean+-SEM values above."
