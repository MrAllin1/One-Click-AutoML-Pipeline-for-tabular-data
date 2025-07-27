#!/bin/bash
#SBATCH --job-name=single-fold-train          # name shown by squeue
#SBATCH --partition=dllabdlc_gpu-rtx2080       # adjust if CPU queue is better
#SBATCH --gres=gpu:1                           # GPU not strictly required but harmless
#SBATCH --mem=40G
#SBATCH --time=04:00:00                        # hh:mm:ss
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ── activate your Python environment ────────────────────────────────────
cd /work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template
source automl-tabular-env/bin/activate

# ── optional sanity check for GPU & libraries ──────────────────────────
python - <<'PY'
import torch, sys
try:
    import lightgbm, catboost
except ImportError as e:
    print("❌ Missing lib:", e, file=sys.stderr, flush=True)
print("CUDA available :", torch.cuda.is_available())
print("GPU count      :", torch.cuda.device_count())
print("GPU name       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

# ── set the paths for this fold ──────────────────────────────────────
FOLD_DIR="/work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template/data/bike_sharing_demand/11"
X_TRAIN="$FOLD_DIR/X_train.parquet"
Y_TRAIN="$FOLD_DIR/y_train.parquet"
X_TEST="$FOLD_DIR/X_test.parquet"

OUTPUT_DIR="/work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template/models/tree_based_methods_output/bike_sharing_demand_fold11"

# ── train the model on this fold ─────────────────────────────────────
echo "Training model on fold": "$FOLD_DIR"
python src/models/tree_based_methods/auto_ml_pipeline_project/main.py \
  --X_train "$X_TRAIN" \
  --y_train "$Y_TRAIN" \
  --X_test "$X_TEST" \
  --output_dir "$OUTPUT_DIR"

echo "Done. Model and metrics in $OUTPUT_DIR"
