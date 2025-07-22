# src/models/tabnet/tabnet_pipeline.py

import argparse
from pathlib import Path
import subprocess


#-----------------------------
#  Function to run TabNet training
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
# Function to run TabNet ensembling
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
# Main function to run the entire pipeline
#-----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline: Train TabNet + Ensemble")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--output_dir", type=str, default="models/tabnet_ensemble", help="Directory to store models")

    args = parser.parse_args()

    run_training(args.dataset_path, args.output_dir)
    run_ensembling(args.dataset_path, args.output_dir)



# usage: python src/models/tabnet/tabnet_pipeline.py --dataset_path bike_sharing_demand --output_dir models/tabnet_ensemble