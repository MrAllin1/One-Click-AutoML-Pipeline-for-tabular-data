# save as: make_predictions_npy.py
import numpy as np
import pandas as pd
from pathlib import Path

csv = Path("y_pred.csv")  # or e.g. Path("outputs/y_pred.csv")
out = Path("data/exam_dataset/predictions.npy")
out.parent.mkdir(parents=True, exist_ok=True)

# Read CSV (handles both header/no-header)
df = pd.read_csv(csv, header='infer')
# pick the prediction column
if "pred" in df.columns:
    arr = df["pred"].to_numpy()
else:
    arr = df.select_dtypes(include=["number"]).iloc[:, 0].to_numpy()

# basic hygiene
if np.isnan(arr).any():
    raise ValueError("Your predictions contain NaNs.")
arr = np.asarray(arr, dtype=np.float32)

np.save(out, arr)
print(f"Saved {arr.shape} to {out}")
