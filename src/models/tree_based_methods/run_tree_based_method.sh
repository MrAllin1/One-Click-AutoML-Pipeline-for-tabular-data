#!/bin/bash
#SBATCH --job-name=final-model-train             # name shown by squeue
#SBATCH --partition=dllabdlc_gpu-rtx2080         # adjust if CPU queue is better
#SBATCH --gres=gpu:1                             # GPU not strictly required but harmless
#SBATCH --mem=40G
#SBATCH --time=04:00:00                          # hh:mm:ss
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

# ── set the dataset directory and output directory ─────────────────────
DATASET_DIR="/work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template/data/superconductivity"
OUTPUT_DIR="/work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template/models/tree_based_methods_output/superconductivity"

# ── train the final model on all splits ───────────────────────────────
echo "Training final model on all splits"
echo ">> Listing splits in DATASET_DIR (${DATASET_DIR}):"
ls -d "${DATASET_DIR}"/* || echo "  (no dirs found)"
python src/models/tree_based_methods/auto_ml_pipeline_project/final_train.py \
  --dataset_path "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}"

echo "Done. Final model and metrics can be found in ${OUTPUT_DIR}"
