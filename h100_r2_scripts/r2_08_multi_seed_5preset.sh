#!/usr/bin/env bash
# =============================================================================
# R2 revision -- Reviewer 1 point R1.5b
# Multi-seed re-run of the five-preset ablation (Section 6.5, Table 9).
#
# R1 reported the ablation at seed 0 only. Seed noise on the STEER CF cell is
# ~0.0025 SEM (Table 6 after R1.5c correction), so the single-seed difference
# between pure_steer (Gen=0.125) and ablate_optim (Gen=0.126) is within noise
# and does not discriminate value-optimizer choice. The reviewer asked for the
# three-seed run to confirm the ordering.
#
# Grid: 5 presets x 3 seeds = 15 runs on Mamba-790M + CounterFact, N = 500.
#
# The 5 presets (routing | write | value-optimizer):
#   pure_steer     = cosine    | additive    | v_star (default STEER)
#   ablate_routing = euclidean | additive    | v_star
#   ablate_write   = cosine    | substitute  | v_star
#   ablate_optim   = cosine    | additive    | finetune
#   all_grace      = euclidean | substitute  | finetune  (= GRACE-for-Mamba)
#
# Wall-clock: 15 runs x ~2 h each = ~30 hours on one H100.
# Recommend running inside tmux or with nohup.
#
# Output: $RESULTS_DIR/r2_08/preset_<name>_seed_<s>.json (15 files)
#         $RESULTS_DIR/r2_08/multi_seed_table.json (consolidated mean ± SEM)
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/ComplexAttn}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results_r2}"
OUT_DIR="$RESULTS_DIR/r2_08"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

echo "[r2_08] $(date -Iseconds) start: 5-preset ablation x 3 seeds on Mamba-790M + CF"

# Preset -> (routing, write, optim)
declare -A PRESET_ROUTING=(
  [pure_steer]="cosine"     [ablate_routing]="euclidean" [ablate_write]="cosine"
  [ablate_optim]="cosine"   [all_grace]="euclidean"
)
declare -A PRESET_WRITE=(
  [pure_steer]="additive"   [ablate_routing]="additive"  [ablate_write]="substitute"
  [ablate_optim]="additive" [all_grace]="substitute"
)
declare -A PRESET_OPTIM=(
  [pure_steer]="v_star"     [ablate_routing]="v_star"    [ablate_write]="v_star"
  [ablate_optim]="finetune" [all_grace]="finetune"
)

run_cell () {
    local preset="$1"
    local seed="$2"
    local out_json="$OUT_DIR/preset_${preset}_seed_${seed}.json"

    if [[ -f "$out_json" ]]; then
        echo "[r2_08]   skip (exists): $preset / seed=$seed"
        return 0
    fi

    echo "[r2_08]   run: $preset / seed=$seed  ($(date -Iseconds))"
    python -u -m shard_mamba.run_steer \
        --model_name "state-spaces/mamba-790m-hf" \
        --dataset "counterfact" \
        --data_path "data/counterfact/counterfact-edit.json" \
        --N 500 \
        --seed "$seed" \
        --edit_layer 32 \
        --edit_module "w_o" \
        --routing "${PRESET_ROUTING[$preset]}" \
        --write_mode "${PRESET_WRITE[$preset]}" \
        --value_optim "${PRESET_OPTIM[$preset]}" \
        --tau 0.70 \
        --gamma 20 \
        --epsilon_init 1.0 \
        --epsilon_expand 0.1 \
        --v_steps 25 \
        --v_lr 5e-1 \
        --finetune_lr 1e-3 \
        --finetune_steps 25 \
        --metrics "eff,gen,spec" \
        --out_json "$out_json"
}

for preset in pure_steer ablate_routing ablate_write ablate_optim all_grace; do
    for seed in 0 1 2; do
        run_cell "$preset" "$seed"
    done
done

# Consolidate: mean ± SEM per preset
echo "[r2_08] consolidating multi-seed grid into $OUT_DIR/multi_seed_table.json"
python - <<'PY'
import glob, json, math, os, re, statistics
out_dir = os.environ["OUT_DIR"]
by_preset = {}
for p in sorted(glob.glob(os.path.join(out_dir, "preset_*_seed_*.json"))):
    m = re.match(r"preset_(.+?)_seed_(\d+)\.json", os.path.basename(p))
    preset, seed = m.group(1), int(m.group(2))
    with open(p) as f:
        d = json.load(f)
    row = {
        "eff":  d["eff_at_N"][str(500)]  if "eff_at_N"  in d else d["eff"],
        "gen":  d["gen_at_N"][str(500)]  if "gen_at_N"  in d else d["gen"],
        "spec": d["spec_at_N"][str(500)] if "spec_at_N" in d else d["spec"],
    }
    by_preset.setdefault(preset, []).append(row)

def mean_sem(xs):
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    sem = s / math.sqrt(len(xs))
    return m, sem

summary = {}
for preset, rows in by_preset.items():
    out = {}
    for k in ("eff", "gen", "spec"):
        m, sem = mean_sem([r[k] for r in rows])
        out[f"{k}_mean"], out[f"{k}_sem"] = m, sem
    out["n_seeds"] = len(rows)
    summary[preset] = out

with open(os.path.join(out_dir, "multi_seed_table.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"{'preset':<16} {'Eff mean±SEM':<20} {'Gen mean±SEM':<20} {'Spec mean±SEM':<20}")
for preset in ("pure_steer","ablate_routing","ablate_write","ablate_optim","all_grace"):
    if preset not in summary: continue
    s = summary[preset]
    print(f"{preset:<16} {s['eff_mean']:.4f}±{s['eff_sem']:.4f}   "
          f"{s['gen_mean']:.4f}±{s['gen_sem']:.4f}   "
          f"{s['spec_mean']:.4f}±{s['spec_sem']:.4f}")
PY

echo "[r2_08] $(date -Iseconds) done"
echo
echo "Next: replace single-seed cells in manuscript.tex tab:ablation with"
echo "the mean±SEM values above; also update the surrounding discussion to"
echo "cite the SEM-based ordering (the single-seed 0.001 gap between pure_steer"
echo "and ablate_optim should now be clearly within SEM)."
