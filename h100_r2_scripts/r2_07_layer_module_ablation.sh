#!/usr/bin/env bash
# =============================================================================
# R2 revision -- Reviewer 1 point R1.3
# Layer and module ablation for STEER on Mamba-790M + CounterFact, N = 500.
#
# Populates new Section 6.6 (Layer and module ablation), filling the 9-cell
# grid flagged in the R2 response letter:
#   - Layers: 16, 24, 32 (edit site = W_o)
#   - Modules at layer 32: W_o, in-projection, gate (x-conv)
# The default STEER paper configuration sits at (layer=32, module=W_o).
#
# Rationale: the R1 paper inherits the W_o site from ROMBA's causal tracing.
# This run confirms the choice under STEER's own routing + write primitive,
# rather than relying on an inherited analysis.
#
# Wall-clock: 9 cells x ~45 min each = ~6 hours on one H100.
#
# Output: $RESULTS_DIR/r2_07/cell_<layer>_<module>.json (9 files)
#         $RESULTS_DIR/r2_07/layer_module_grid.json (consolidated table)
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/ComplexAttn}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results_r2}"
OUT_DIR="$RESULTS_DIR/r2_07"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

echo "[r2_07] $(date -Iseconds) start: STEER layer x module ablation on Mamba-790M + CF"

run_cell () {
    local layer="$1"
    local module="$2"    # one of: w_o, in_proj, gate
    local tag="l${layer}_${module}"
    local out_json="$OUT_DIR/cell_${tag}.json"

    if [[ -f "$out_json" ]]; then
        echo "[r2_07]   skip (exists): $tag"
        return 0
    fi

    echo "[r2_07]   run: $tag  ($(date -Iseconds))"
    python -u -m shard_mamba.run_steer \
        --model_name "state-spaces/mamba-790m-hf" \
        --dataset "counterfact" \
        --data_path "data/counterfact/counterfact-edit.json" \
        --N 500 \
        --seed 0 \
        --edit_layer "$layer" \
        --edit_module "$module" \
        --tau 0.70 \
        --gamma 20 \
        --v_steps 25 \
        --v_lr 5e-1 \
        --metrics "eff,gen,spec" \
        --out_json "$out_json"
}

# Layer sweep at module = w_o
for L in 16 24 32; do
    run_cell "$L" "w_o"
done

# Module sweep at layer = 32
for M in in_proj gate; do
    run_cell 32 "$M"
done

# Consolidate into one table JSON
echo "[r2_07] consolidating grid into $OUT_DIR/layer_module_grid.json"
python - <<'PY'
import glob, json, os, re
out_dir = os.environ["OUT_DIR"]
rows = []
for p in sorted(glob.glob(os.path.join(out_dir, "cell_*.json"))):
    tag = re.match(r"cell_l(\d+)_([a-z_]+)\.json", os.path.basename(p)).groups()
    layer, module = int(tag[0]), tag[1]
    with open(p) as f:
        d = json.load(f)
    rows.append({
        "layer": layer, "module": module,
        "eff_at_500":  d["eff_at_N"][str(500)]  if "eff_at_N"  in d else d["eff"],
        "gen_at_500":  d["gen_at_N"][str(500)]  if "gen_at_N"  in d else d["gen"],
        "spec_at_500": d["spec_at_N"][str(500)] if "spec_at_N" in d else d["spec"],
    })
with open(os.path.join(out_dir, "layer_module_grid.json"), "w") as f:
    json.dump(rows, f, indent=2)
print(json.dumps(rows, indent=2))
PY

echo "[r2_07] $(date -Iseconds) done"
echo
echo "Next: patch manuscript.tex with a new Section 6.6 and Table (tab:layer-module-ablation)"
echo "using the 9 rows from $OUT_DIR/layer_module_grid.json."
