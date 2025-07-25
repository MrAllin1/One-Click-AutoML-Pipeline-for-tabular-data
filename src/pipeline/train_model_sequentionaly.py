#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch

from models.tabpfn.tabpfn_train import train_tabpfn_ensemble
from models.tree_based_methods.auto_ml_pipeline_project.final_train import tree_based_methods_model
from models.tabnet.tabnet_pipeline import tabnet_final_pipeline

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_each_model_sequentially(dataset: Path, output_dir: Path):
    """
    dataset: Path to dataset root (contains fold parquet files).
    output_dir: Path where models and artifacts will be saved.
    """
    seed = 1

    # TabPFN Ensemble
    tabpfn_model_path, tabpfn_r2 = train_tabpfn_ensemble(
        str(dataset), output_dir=output_dir, seed=seed
    )
    print(f"TabPFN Ensemble Model Path: {tabpfn_model_path}")
    print(f"TabPFN Ensemble R2: {tabpfn_r2}")

    # Tree-based methods
    tree_model_path, tree_r2 = tree_based_methods_model(
        str(dataset), str(output_dir)
    )
    print(f"Tree Based Model Path: {tree_model_path}")
    print(f"Tree Based Model R2: {tree_r2}")

    # TabNet pipeline
    tabnet_model_path, tabnet_r2 = tabnet_final_pipeline(
        str(dataset), str(output_dir)
    )
    print(f"TabNet Model Path: {tabnet_model_path}")
    print(f"TabNet Model R2: {tabnet_r2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential training of TabPFN, tree-based, and TabNet models"
    )
    parser.add_argument(
        "-d", "--dataset",
        required=True,
        help="Path to the dataset root folder containing fold files"
    )
    parser.add_argument(
        "-o", "--out-dir",
        dest="out_dir",
        required=True,
        help="Directory where trained models and outputs will be saved"
    )
    args = parser.parse_args()

    # Resolve paths
    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run training with correct Path types
    train_each_model_sequentially(dataset, out_dir)
