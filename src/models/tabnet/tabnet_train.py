import argparse
from pathlib import Path
import numpy as np
import torch
import pickle
from sklearn.metrics import r2_score
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor

from data import get_available_folds, load_fold
from data.preprocess import preprocess_features 

def train_tabnet(X_tr, y_tr, X_te, y_te):

    def to_numpy_float32(x):
        if isinstance(x, pd.DataFrame):
            # convert all columns to float32
            return x.astype(np.float32).values
        elif isinstance(x, pd.Series):
            return x.astype(np.float32).values
        else:
            x = np.array(x)
            if np.issubdtype(x.dtype, np.number) == False:
                raise ValueError(f"X is not fully numeric. dtype: {x.dtype}")
            return x.astype(np.float32)

    X_tr_np = to_numpy_float32(X_tr)
    X_te_np = to_numpy_float32(X_te)

    y_tr_np = np.asarray(to_numpy_float32(y_tr)).reshape(-1, 1)
    y_te_np = np.asarray(to_numpy_float32(y_te)).reshape(-1, 1)

    clf = TabNetRegressor(
        verbose=0,
        seed=42,
    )

    clf.fit(
        X_tr_np, y_tr_np,
        eval_set=[(X_te_np, y_te_np)],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128
    )
    return clf


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[+] Saved model to {path}")


def main(dataset: str, out_dir: Path):
    folds = get_available_folds(dataset)
    scores = []

    for fold in folds:
        print(f"\n[+] Fold {fold}:")
        X_tr, X_te, y_tr, y_te = load_fold(dataset, fold)

        # preprocess features before training
        X_tr, X_te = preprocess_features(X_tr, X_te)

        model = train_tabnet(X_tr, y_tr, X_te, y_te)

        X_te_np = X_te.astype(np.float32).values  
        y_pred = model.predict(X_te_np).ravel()
        score = r2_score(y_te.values, y_pred)
        print(f"[+] Fold {fold} R²: {score:.4f}")
        scores.append(score)

        save_model(model, out_dir / f"{dataset}_fold{fold}.pkl")

    print(f"\nMean R² over {len(folds)} folds: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("models/tabnet"))
    args = parser.parse_args()
    main(args.dataset, args.out_dir)
