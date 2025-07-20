#!/bin/bash
#SBATCH --job-name=final-model-train             # ← name shown by squeue
#SBATCH --partition=dllabdlc_gpu-rtx2080         # ← adjust if CPU queue is better
#SBATCH --gres=gpu:1                             # GPU not strictly required but harmless
#SBATCH --mem=40G
#SBATCH --time=08:00:00                          # hh:mm:ss
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ── load CUDA (only for sanity checks / optional CatBoost-GPU) ─────────
source /etc/profile.d/modules.sh  
module load cuda/11.7
module load python/3.10-gpu

cd /work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template
source automl-tabular-env/bin/activate

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

# ── set dataset + tier you want to train on ────────────────────────────
export DATASET_DIR="/work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template/data/bike_sharing_demand"
export TIER="S-quick"   # one of: S-quick | S-medium | S-balanced | S-slow

# ── run the AutoML pipeline ────────────────────────────────────────────
cd src/tree_based_methods
python train_final_model.py "${DATASET_DIR}" --tier "${TIER}"

# ── after completion ───────────────────────────────────────────────────
# You’ll find the trained model in:
#   models/$(basename "${DATASET_DIR}")/final_model.pkl
