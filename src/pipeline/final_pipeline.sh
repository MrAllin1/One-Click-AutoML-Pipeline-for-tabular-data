#!/bin/bash -l
#SBATCH --job-name=tabpfn-train
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

# sanity‑check
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

# set paths
export DATASET_DIR="$PWD/data/exam_dataset"
export MODEL_DIR="$PWD/modelsFinal-exam"

# run training (option A: direct script invocation)
cd src/pipeline
python train_model_sequentionaly.py \
    --dataset "$DATASET_DIR" \
    --out-dir "$MODEL_DIR"
