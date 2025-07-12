import argparse
from pathlib import Path
import pickle
import pandas as pd

from data import load_full_test_for_specific_dataset


def main(dataset: str, model_path: Path, output_path: Path):
    X_test, y_test = load_full_test_for_specific_dataset(dataset)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test_np = X_test.values.astype("float32")
    y_pred = model.predict(X_test_np).ravel()

    df = pd.DataFrame({
        "prediction": y_pred,
        "target": y_test.reset_index(drop=True).values
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-m", "--model-path", required=True, type=Path)
    parser.add_argument("-o", "--output-path", required=True, type=Path)
    args = parser.parse_args()
    main(args.dataset, args.model_path, args.output_path)
