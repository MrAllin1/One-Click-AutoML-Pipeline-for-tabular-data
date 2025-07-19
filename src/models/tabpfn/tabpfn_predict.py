#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle
import pandas as pd

from src.data.load import load_full_test_for_specific_dataset

def main(dataset: str, model_path: Path, output_path: Path):
    X_test, y_test = load_full_test_for_specific_dataset(dataset)

    # load via pickle
    with open(model_path, "rb") as f:
        agent = pickle.load(f)

    y_pred = agent.predict(X_test)

    df = pd.DataFrame({
        "prediction": y_pred,
        "target":     y_test.reset_index(drop=True),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Predictions saved to {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dataset",    required=True)
    p.add_argument("-m", "--model-path", required=True, type=Path)
    p.add_argument("-o", "--output-path",required=True, type=Path)
    args = p.parse_args()
    main(args.dataset, args.model_path, args.output_path)
