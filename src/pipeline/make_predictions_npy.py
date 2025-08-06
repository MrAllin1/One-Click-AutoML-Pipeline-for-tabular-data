#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Convert a y_pred.csv into a .npy array of predictions."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the directory containing y_pred.csv"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/exam_dataset/predictions.npy"),
        help="Where to save the .npy file (default: %(default)s)"
    )
    args = parser.parse_args()

    csv_path = args.input_dir / "y_pred.csv"
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV (handles both header/no-header)
    df = pd.read_csv(csv_path, header='infer')

    # Pick the prediction column
    if "pred" in df.columns:
        arr = df["pred"].to_numpy()
    else:
        # fallback to first numeric column
        arr = df.select_dtypes(include=["number"]).iloc[:, 0].to_numpy()

    # Basic hygiene
    if np.isnan(arr).any():
        raise ValueError("Your predictions contain NaNs.")
    arr = np.asarray(arr, dtype=np.float32)

    np.save(out_path, arr)
    print(f"Saved array of shape {arr.shape} to {out_path}")

if __name__ == "__main__":
    main()
