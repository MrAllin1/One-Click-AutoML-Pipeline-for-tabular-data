
import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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


# ─── Training function ────────────────────────────────────────────────────────
def train_each_model_sequentionaly(dataset: str):
    """
    dataset
        Path to dataset root (contains fold parquet files).
    """
    seed = 1
    tabPfnEnsembleModelPath, tabPfnEnsembleR2 = train_tabpfn_ensemble(args.dataset, output_dir="", seed=seed)
    print(f"TabPFN Ensemble Model Path: {tabPfnEnsembleModelPath}")
    print(f"TabPFN Ensemble R2: {tabPfnEnsembleR2}")
    treeBasedModelPath, treeBasedModelR2=tree_based_methods_model(dataset_path=args.dataset, output_dir="")
    print(f"Tree Based Model Path: {treeBasedModelPath}")
    print(f"Tree Based Model R2: {treeBasedModelR2}")
    tabNetModelPath, tabNetR2 = tabnet_final_pipeline(dataset_path=args.dataset, output_dir="")
    print(f"TabNet Model Path: {tabNetModelPath}")
    print(f"TabNet Model R2: {tabNetR2}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TabPFN fold-wise training")
    p.add_argument("-d", "--dataset", required=True, help="dataset root folder")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

