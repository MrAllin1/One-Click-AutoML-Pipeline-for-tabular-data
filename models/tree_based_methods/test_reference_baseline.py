from pathlib import Path
import joblib, pandas as pd
from sklearn.metrics import r2_score

root = Path("data/wine_quality")
art  = joblib.load(root / "models" / "lightgbm_split1.pkl")   # pick a split

X_test = pd.read_parquet(root / "1" / "X_test.parquet")[art["columns"]]
y_test = pd.read_parquet(root / "1" / "y_test.parquet").squeeze("columns")

for c in art["categorical_cols"]:
    X_test[c] = X_test[c].astype("category")

y_pred = art["model"].predict(X_test, num_iteration=art["model"].best_iteration)
print("R² =", r2_score(y_test, y_pred))
