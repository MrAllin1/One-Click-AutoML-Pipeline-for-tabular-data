#!/bin/bash -l
#SBATCH --job-name=tabpfn-train-predict
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# load modules
module load cuda/11.7

# activate your env
cd /work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template
source automl-tabular-env/bin/activate

# sanity-check
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

# set paths for training and prediction
export DATASET_DIR="$PWD/data/superconductivity"
export MODEL_DIR="$PWD/modelsFinal-superconductivity"

# --------------------
# STEP 1: TRAIN MODEL
# --------------------
cd src/pipeline
python train_model_sequentionaly.py \
    --dataset "$DATASET_DIR" \
    --out-dir "$MODEL_DIR"

# ------------------------
# STEP 2: RUN PREDICTIONS
# ------------------------
# assume train saved a final_model.pkl in $MODEL_DIR
python predict_with_final_model.py \
    --dataset "$DATASET_DIR" \
    --model-file "$MODEL_DIR/final_model.pkl" \
    --fold 1 \
    --output "$MODEL_DIR/y_pred.csv"

python make_predictions_npy.py \
    "$MODEL_DIR" \
    --output "$MODEL_DIR/predictions.npy"
