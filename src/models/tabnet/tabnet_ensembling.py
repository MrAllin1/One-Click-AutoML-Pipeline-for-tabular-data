import numpy as np
import pandas as pd
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetRegressor
import joblib
from . import config

def to_numpy_float32(x):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.astype(np.float32).values
    return np.array(x).astype(np.float32)

def load_model(path: Path):
    model = TabNetRegressor()
    model.load_model(str(path))
    return model

def main(dataset_dir: Path, model_dir: Path, output_file: Path = Path(config.ensemble_predictions)):
    X_test = pd.read_parquet(dataset_dir / "X_test.parquet")
    X_test_np = to_numpy_float32(X_test)

    model_paths = sorted(model_dir.glob("tabnet_fold*.zip"))
    if len(model_paths) == 0:
        raise FileNotFoundError(f"No fold models found in {model_dir}")

    scaler_paths = sorted(model_dir.glob("scaler_fold*.pkl"))
    if len(scaler_paths) != len(model_paths):
        raise ValueError("Mismatch between number of models and scalers")

    preds_all = []
    for model_path, scaler_path in zip(model_paths, scaler_paths):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        preds_scaled = model.predict(X_test_np).ravel()
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        preds_all.append(preds)

    ensemble_preds = np.mean(preds_all, axis=0)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_preds = pd.DataFrame({"prediction": ensemble_preds})
    df_preds.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

if __name__ == "__main__":
    import argparse
    from pathlib import Path as _P
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=_P, required=True, help="Path to folder containing X_test.parquet")
    parser.add_argument("--model_dir",   type=_P, required=True, help="Folder where fold models are saved")
    parser.add_argument("--output_file", type=_P, default=_P(config.ensemble_predictions), help="CSV file to save predictions")
    args = parser.parse_args()
    main(args.dataset_dir, args.model_dir, args.output_file)
