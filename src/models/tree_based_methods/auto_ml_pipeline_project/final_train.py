#-------------------------------------
# final_train.py
#-------------------------------------
"""
Entry point to train final model on all splits under a dataset folder.
"""
import argparse
import os

from auto_ml_pipeline.data_processing import load_all_splits
from auto_ml_pipeline.feature_engineering import engineer_features
from auto_ml_pipeline.model_selection import select_and_train
from auto_ml_pipeline.utils import setup_logging, save_model, report_results


def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Pipeline: final model on all splits")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset root with subfolders 1..N containing splits.")
    parser.add_argument("--output_dir", type=str, default="output_final",
                        help="Directory to save the final model and report.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Debug: verify splits directory contents
    try:
        contents = sorted(os.listdir(args.dataset_path))
    except FileNotFoundError:
        raise RuntimeError(f"Dataset path not found: {args.dataset_path}")
    print(f"DEBUG: contents of dataset_path = {contents}")
    if not contents:
        raise RuntimeError(f"No split folders found in {args.dataset_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    # 1. Load and merge all splits into a single train set
    X_all, y_all = load_all_splits(args.dataset_path)
    print(f"DEBUG: Loaded {X_all.shape[0]} rows and {X_all.shape[1]} columns from {args.dataset_path}")
    if X_all.shape[1] == 0:
        raise RuntimeError("No features loaded: X_all has zero columns.")

    # 2. Feature engineering on full data
    X_all_fe, _ = engineer_features(X_all, X_all.copy())
    if X_all_fe.shape[1] == 0:
        raise RuntimeError("No features after engineering: X_all_fe has zero columns.")

    # 3. Model selection and training on full data
    models, metrics = select_and_train(
        X_all_fe, y_all, X_all_fe, y_all,
        output_dir=args.output_dir
    )

    # 4. Save only the primary model
    primary = 'ensemble' if 'ensemble' in models else list(models.keys())[0]
    model_path = os.path.join(args.output_dir, f"final_{primary}.pkl")
    save_model(models[primary], model_path)
    print(f"Saved final model to {model_path}")

    # 5. Report CV metrics used for selection
    report_results(metrics, args.output_dir)
    print(f"Metrics written to {os.path.join(args.output_dir, 'metrics.json')}")

if __name__ == "__main__":
    main()
