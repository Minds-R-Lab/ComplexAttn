#!/usr/bin/env bash
# =============================================================================
# R2 revision -- Reviewer 1 point R1.2
# MQuAKE-CF-3K frozen-base baseline on Mamba-790M.
#
# Fills the `[pending H100 cell]` in manuscript Table 8 (tab:mquake) under §6.5.
# Runs the UNEDITED Mamba-790M on the same shuffled 500-item MQuAKE slice
# used by r1_2_mquake_eval.py for STEER, so the two cells are directly
# comparable. Uses the sibling python script r2_mquake_frozen_baseline.py.
#
# Wall-clock: ~45-60 minutes on one H100-80GB.
# Output:     $REPO_ROOT/sfib_mamba/results/mquake_frozen_mamba790m_seed0.json
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/ComplexAttn/ComplexAttn}"

# Load venv if not already active. Pick the first that exists.
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

# Resolve the frozen-baseline python module. It lives alongside the shell
# script under h100_r2_scripts/, so we tell python to pick it up from there
# while cwd remains sfib_mamba/ so sibling imports (shard_mamba, etc.) still
# work the same way r1_2_mquake_eval.py expects.
H100_DIR="$REPO_ROOT/h100_r2_scripts"
PY_SCRIPT="$H100_DIR/r2_mquake_frozen_baseline.py"

OUT="results/mquake_frozen_mamba790m_seed0.json"
if [[ -f "$OUT" ]]; then
  echo "[r2_09] skip (exists): $OUT"
  exit 0
fi

echo "[r2_09] $(date -Iseconds) start: MQuAKE frozen-base on Mamba-790M"
echo "[r2_09] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[r2_09] venv: ${VIRTUAL_ENV:-<none>}"
echo "[r2_09] cwd:  $(pwd)"
echo "[r2_09] out:  $OUT"

python -u "$PY_SCRIPT" \
  --model "state-spaces/mamba-790m-hf" \
  --layer 32 \
  --seed 0 \
  --n_items 500 \
  --max_new_tokens 32 \
  --out "$OUT"

echo "[r2_09] $(date -Iseconds) done"
echo
echo "Result:"
python - <<PY
import json
d = json.load(open("$OUT"))
print(f"  method:          {d['method']}")
print(f"  benchmark:       {d['benchmark']}")
print(f"  n_items:         {d['n_items_processed']}")
print(f"  single-hop acc:  {d['singlehop_accuracy']:.4f}")
print(f"  multi-hop  acc:  {d['multihop_accuracy']:.4f}")
print(f"  wall time:       {d['wall_s']/3600:.2f} h")
PY
echo
echo "Next: the single-hop and multi-hop numbers above replace the"
echo "[pending H100 cell] row in manuscript tab:mquake."
