# final_train.py

"""
Entry point: merges **all** training splits, holds out 10% for validation,
selects & retrains the best model on 100% of the training data.
"""
import argparse
import os

from sklearn.model_selection import train_test_split

from config import RANDOM_SEED
# from auto_ml_pipeline.data_processing import load_all_splits
from auto_ml_pipeline.feature_engineering import engineer_features
from auto_ml_pipeline.model_selection import select_and_train, EnsembleModel
from auto_ml_pipeline.hyperparameter_tuning import tune_lgbm, tune_catboost
from auto_ml_pipeline.utils import setup_logging, save_model, report_results
from data.load import load_only_train_for_dataset, DATA_ROOT
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoML Pipeline: final model on train-only dataset"
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to root folder with subfolders 1..N containing X_train.parquet & y_train.parquet.")
    parser.add_argument("--output_dir", type=str, default="output_final",
                        help="Directory to save the final model and report.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        splits = sorted(os.listdir(args.dataset_path))
    except FileNotFoundError:
        raise RuntimeError(f"Dataset path not found: {args.dataset_path}")
    if not splits:
        raise RuntimeError(f"No splits found in {args.dataset_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    # 1) merge all train splits via load_only_train_for_dataset
    ds = Path(args.dataset_path).name
    X_all, y_all = load_only_train_for_dataset(ds, DATA_ROOT)
    print(f"DEBUG: merged train pool = {X_all.shape[0]} rows")

    # 2) hold out 10% for validation
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_all, y_all,
        test_size=0.10,
        random_state=RANDOM_SEED
    )
    print(f"DEBUG: pool={X_pool.shape[0]} rows, val={X_val.shape[0]} rows")

    # 3) feature engineering
    X_pool_fe, X_val_fe = engineer_features(X_pool, X_val)
    if X_pool_fe.shape[1] == 0:
        raise RuntimeError("No features after engineering on pool")

    # 4) model selection on pool→val
    models, metrics = select_and_train(
        X_pool_fe, y_pool,
        X_val_fe, y_val,
        output_dir=args.output_dir
    )
    # after select_and_train in final_train.py
    val_metrics = {f"val_{k}": v for k, v in metrics.items()}
    report_results(val_metrics, args.output_dir)


    # 5) choose best on validation
    best = max(metrics, key=metrics.get)
    print(f"Selected model: {best} (val R² = {metrics[best]:.4f})")

    # 6) retrain best on 100% of all training data
    X_full_fe, _ = engineer_features(X_all, X_all.copy())
    if isinstance(models[best], EnsembleModel):
        m1, m2 = models[best].m1, models[best].m2
        m1.fit(X_full_fe, y_all)
        m2.fit(X_full_fe, y_all)
        final_model = EnsembleModel(m1, m2)
    else:
        base = models[best]
        final_model = base.__class__(**base.get_params())
        final_model.fit(X_full_fe, y_all)

    # 7) save & report only validation metrics
    save_model(final_model, os.path.join(args.output_dir, f"final_{best}.pkl"))
    report_results(metrics, args.output_dir)
    print(f"Validation metrics written to {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
