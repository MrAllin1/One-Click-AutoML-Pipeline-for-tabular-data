#!/bin/bash -l
#SBATCH --job-name=tabpfn-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# load modules
module load cuda/11.7

# activate your env
cd /work/dlclarge2/alidemaa-dl_lab/automl/automl-exam-ss25-tabular-freiburg-template
source automl-tabular-env/bin/activate

# sanity‑check
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

export DATASET_DIR="$PWD/data/yprop_4_1"
export MODEL_DIR="$PWD/modelsFinal-YProp"

# run training (option A: direct script invocation)
cd src/pipeline
python predict_with_final_model.py \
  --dataset $DATASET_DIR \
  --model-file $MODEL_DIR/final_model.pkl \
  --fold 11 \
  --output $MODEL_DIR/y_pred.csv
