# src/models/tabnet/tabnet_ensembling.py

import numpy as np
import pandas as pd
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetRegressor
import joblib

#-----------------------------
# Utility: Function to convert data to float32 numpy arrays
#-----------------------------
def to_numpy_float32(x):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.astype(np.float32).values
    else:
        return np.array(x).astype(np.float32)
    
#-----------------------------
# Function to load a trained TabNet model
#-----------------------------
def load_model(path: Path):
    model = TabNetRegressor()
    model.load_model(str(path))
    return model

#-----------------------------
# Main function to run TabNet ensembling and prediction
#-----------------------------
def main(dataset_dir: Path, model_dir: Path, output_file: Path):
    X_test = pd.read_parquet(dataset_dir / "X_test.parquet")
    X_test_np = to_numpy_float32(X_test)

    # Load all fold models from the model directory
    model_paths = sorted(model_dir.glob("tabnet_fold*.zip"))
    if len(model_paths) == 0:
        raise FileNotFoundError(f"No fold models found in {model_dir}")

    print(f"Found {len(model_paths)} fold models. Loading and predicting")
    # Load each model and its corresponding scaler, then make predictions for the test set
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

    # Ensemble by averaging predictions
    ensemble_preds = np.mean(preds_all, axis=0)

    # save predictions to the output directory as a CSV file y_pred.csv
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_preds = pd.DataFrame({"prediction": ensemble_preds})
    df_preds.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

#-----------------------------
# CLI
#-----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to folder containing X_test.parquet")
    parser.add_argument("--model_dir", type=Path, required=True, help="Folder where fold models are saved")
    parser.add_argument("--output_file", type=Path, default=Path("y_pred.csv"), help="CSV file to save predictions")
    args = parser.parse_args()

    main(args.dataset_dir, args.model_dir, args.output_file)
