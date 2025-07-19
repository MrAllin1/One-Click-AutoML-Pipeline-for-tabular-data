#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.metrics import r2_score

from data import get_available_folds, load_fold

class PFNAgent(TabPFNRegressor):
    def __init__(self, **kwargs):
        # Force TabPFN to use Apple MPS backend
        super().__init__(device="cuda", **kwargs)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[+] Model saved to {path}")

def main(dataset: str, output_dir: Path):
    folds = get_available_folds(dataset)
    scores = []
    for fold in folds:
        X_tr, X_te, y_tr, y_te = load_fold(dataset, fold)
        print(f"\n[+] Fold {fold}: training on {len(X_tr)} samples")
        agent = PFNAgent()  # uses MPS, no CPU-size guard for ~7k samples
        agent.fit(X_tr, y_tr.values.ravel())

        y_pred = agent.predict(X_te)
        score = r2_score(y_te, y_pred)
        print(f"[+] Fold {fold} R²: {score:.4f}")
        scores.append(score)

        # optionally save each fold’s model
        model_path = output_dir / f"{dataset}_fold{fold}.pkl"
        agent.save(model_path)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\nMean R² over {len(folds)} folds: {mean_score:.4f} ± {std_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",  required=True,
                        help="Name of the dataset folder under data/")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("models"),
                        help="Directory to save per-fold models")
    args = parser.parse_args()
    main(args.dataset, args.out_dir)
