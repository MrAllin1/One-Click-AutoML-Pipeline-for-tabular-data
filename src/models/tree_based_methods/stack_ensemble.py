# stack_ensemble.py  -------------------------------------------------------
import gc, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

BASE_DIR  = Path(__file__).resolve().parent          # …/tree_based_methods
MODEL_DIR = BASE_DIR / "models" / "wine_quality"     # <— adjust if needed
N_SPLITS  = 10                                       # 1 … 10 artefacts

def _arts():
    return [joblib.load(MODEL_DIR / f"lightgbm_split{i}.pkl")
            for i in range(1, N_SPLITS + 1)]

def _prep(df, art):
    df = df[art["columns"]].copy()
    for c in art["categorical_cols"]:
        df[c] = df[c].astype("category")
    return df

# ------------------------------------------------------------------ train
def train_meta():
    arts = _arts()
    Xoof, yoof = [], []
    for split in range(1, N_SPLITS + 1):
        # right: use the actual data folder
        DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "wine_quality"
        base = DATA_DIR / str(split)
        # …/wine_quality/1/
        X_hold = pd.read_parquet(base / "X_test.parquet")
        y_hold = pd.read_parquet(base / "y_test.parquet").squeeze("columns")
        preds = [art["model"].predict(
                    _prep(X_hold, art),
                    num_iteration=art["model"].best_iteration)
                 for art in arts]
        Xoof.append(np.column_stack(preds))
        yoof.append(y_hold.values)
        gc.collect()
    Xoof, yoof = np.vstack(Xoof), np.hstack(yoof)
    meta = RidgeCV(alphas=(1e-3,1e-2,1e-1,1)).fit(Xoof, yoof)
    joblib.dump(meta, MODEL_DIR / "meta_ridge.pkl")
    print(
        "Saved meta-model →", MODEL_DIR / "meta_ridge.pkl",
        "| OOF R² =", round(r2_score(yoof, meta.predict(Xoof)), 4)   # ← use round()
    )


# ---------------------------------------------------------------- predict
def predict(path_in: Path, path_out: Path):
    arts = _arts()
    Xnew = pd.read_parquet(path_in)
    base_preds = [art["model"].predict(
                      _prep(Xnew, art),
                      num_iteration=art["model"].best_iteration)
                  for art in arts]
    meta = joblib.load(MODEL_DIR / "meta_ridge.pkl")
    yhat = meta.predict(np.column_stack(base_preds))
    pd.Series(yhat, name="prediction").to_parquet(path_out)
    print("Wrote stacked predictions →", path_out)

# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    pred = sub.add_parser("predict")
    pred.add_argument("x_test"), pred.add_argument("out")
    args = p.parse_args()

    if args.cmd == "train":
        train_meta()
    else:
        predict(Path(args.x_test).expanduser(),
                Path(args.out).expanduser())
