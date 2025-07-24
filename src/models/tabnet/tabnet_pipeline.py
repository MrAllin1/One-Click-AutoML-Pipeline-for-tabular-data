import argparse
from pathlib import Path
import subprocess
import re
import sys
import io
from typing import Tuple
from src.models.tabnet.tabnet_train import main as _train_main
from src.models.tabnet.tabnet_ensembling import main as _ensemble_main

#-----------------------------
#  Function to run TabNet training via subprocess (unchanged)
#-----------------------------
def run_training(dataset_path: str, output_dir: str):
    print(f"\n[+] Starting TabNet 10-Fold CV Training...")
    train_cmd = [
        "python", "src/models/tabnet/tabnet_train.py",
        "--d", dataset_path,
        "--o", output_dir
    ]
    subprocess.run(train_cmd, check=True)
    print(f"[+] Training completed. Models saved to {output_dir}")

#-----------------------------
# Function to run TabNet ensembling via subprocess (unchanged)
#-----------------------------
def run_ensembling(dataset_path: str, output_dir: str):
    print(f"\n[+] Starting TabNet Ensembling...")
    ensemble_cmd = [
        "python", "src/models/tabnet/tabnet_ensembling.py",
        "--d", dataset_path,
        "--m", output_dir
    ]
    subprocess.run(ensemble_cmd, check=True)
    print(f"[+] Ensembling completed. Check the console for R² scores.")

#-----------------------------
# Function to run full pipeline and return model path + R2
#-----------------------------
def tabnet_final_pipeline(dataset_path: str, output_dir: str) -> Tuple[str, float]:
    """
    Runs TabNet training and ensembling by directly calling the Python functions
    (no subprocess), captures their console output, and returns
    the model directory and the final mean ensemble R² score.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Train via direct function call (capture but ignore train stdout) ---
    print(f"[+] Training TabNet (10-fold CV)...")
    train_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = train_buf
    _train_main(dataset_path, out_dir)
    sys.stdout = old_stdout

    # --- Ensemble via direct function call (capture stdout for R²) ---
    print(f"[+] Ensembling TabNet folds...")
    ens_buf = io.StringIO()
    sys.stdout = ens_buf
    _ensemble_main(dataset_path, out_dir)
    sys.stdout = old_stdout
    ens_output = ens_buf.getvalue()

    # Parse mean R² from ensembling output
    m = re.search(r"Mean Ensemble R2 across folds: ([0-9]+\.[0-9]+)", ens_output)
    if not m:
        raise RuntimeError(
            "Could not parse mean ensemble R2 from output:\n" + ens_output
        )
    r2_mean = float(m.group(1))

    print(f"[+] Pipeline completed. Models in {out_dir}, Mean Ensemble R2: {r2_mean:.4f}")
    return str(out_dir), r2_mean

#-----------------------------
# Main function to run the entire pipeline via subprocess
#-----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline: Train TabNet + Ensemble")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--output_dir", type=str, default="models/tabnet_ensemble", help="Directory to store models")

    args = parser.parse_args()

    run_training(args.dataset_path, args.output_dir)
    run_ensembling(args.dataset_path, args.output_dir)
