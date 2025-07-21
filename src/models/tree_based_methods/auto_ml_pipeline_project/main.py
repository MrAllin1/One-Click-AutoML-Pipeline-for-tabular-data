#-------------------------------------
# main.py
#-------------------------------------
"""
Entry point for split-by-split run (one folder containing X_train.parquet, etc.).
"""
import argparse
import os

from auto_ml_pipeline.data_processing import load_data
from auto_ml_pipeline.feature_engineering import engineer_features
from auto_ml_pipeline.model_selection import select_and_train
from auto_ml_pipeline.utils import setup_logging, save_model, report_results


def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Pipeline: single split run")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to split folder with X_train.parquet, y_train.parquet, etc.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save models and reports.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    # 1. Load and preprocess one split
    X_train, y_train, X_test, y_test = load_data(args.data_path)

    # 2. Feature engineering
    X_train_fe, X_test_fe = engineer_features(X_train, X_test)

    # 3. Model selection, training, HPO, ensembling
    models, metrics = select_and_train(
        X_train_fe, y_train, X_test_fe, y_test,
        output_dir=args.output_dir
    )

    # 4. Save models and report metrics
    for name, model in models.items():
        save_model(model, os.path.join(args.output_dir, f"{name}.pkl"))
    report_results(metrics, args.output_dir)

if __name__ == "__main__":
    main()
