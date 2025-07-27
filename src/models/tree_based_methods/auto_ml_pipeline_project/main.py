# main.py
"""
Entry point for single-file run via separate train/test parquet files: holds out 10% of train for validation, trains, selects, and saves model.
"""
import argparse
import os

from sklearn.model_selection import train_test_split

from config import RANDOM_SEED
from auto_ml_pipeline.feature_engineering import engineer_features
from auto_ml_pipeline.model_selection import select_and_train, EnsembleModel
from auto_ml_pipeline.utils import setup_logging, save_model, report_results
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoML Pipeline: train from full path parquet files"
    )
    parser.add_argument("--X_train", type=str, required=True,
                        help="Full path to X_train parquet.")
    parser.add_argument("--y_train", type=str, required=True,
                        help="Full path to y_train parquet.")
    parser.add_argument("--X_test", type=str, required=False,
                        help="Full path to X_test parquet (optional, for later inference).", default=None)
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save models and reports.")
    return parser.parse_args()


def load_parquet(path):
    print(f"DEBUG: Loading parquet from {path}")
    df = pd.read_parquet(path)
    print(f"DEBUG: Loaded DataFrame with shape {df.shape} and columns: {list(df.columns)}")
    return df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    # 1) load train and optional test
    X_train = load_parquet(args.X_train)
    y_train = load_parquet(args.y_train).squeeze()
    print(f"DEBUG: y_train series length={y_train.shape} | dtype={y_train.dtype} | head=\n{y_train.head()}")

    if args.X_test:
        X_test = load_parquet(args.X_test)
        print(f"DEBUG: Train/Test split sizes - train: {X_train.shape}, test: {X_test.shape}")
    else:
        print(f"DEBUG: Only training data loaded, size: {X_train.shape}")

    # 2) hold out 10% for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.10,
        random_state=RANDOM_SEED
    )
    print(f"DEBUG: After split → train: {X_tr.shape}, val: {X_val.shape}")
    print(f"DEBUG: y_tr distribution head=\n{y_tr.value_counts().head()}" if y_tr.dtype.name=='category' else "")

    # 3) feature engineering
    X_tr_fe, X_val_fe = engineer_features(X_tr, X_val)
    print(f"DEBUG: Features engineered → X_tr_fe.shape={X_tr_fe.shape}, X_val_fe.shape={X_val_fe.shape}")
    print(f"DEBUG: Feature columns: {list(X_tr_fe.columns)}")
    if X_tr_fe.shape[1] == 0:
        raise RuntimeError("No features after engineering.")

    # 4) model selection on val hold-out
    print("DEBUG: Starting model selection and training...")
    models, metrics = select_and_train(
        X_tr_fe, y_tr,
        X_val_fe, y_val,
        output_dir=args.output_dir
    )
    print(f"DEBUG: Model training complete. Validation metrics: {metrics}")

    # 5) pick best on validation
    best = max(metrics, key=metrics.get)
    print(f"DEBUG: Selected model: {best} with val R² = {metrics[best]:.4f}")

    # 6) retrain best on 100% of original train
    X_full_fe, _ = engineer_features(X_train, X_train.copy())
    print(f"DEBUG: Retraining selected model on full data → features shape: {X_full_fe.shape}")
    if isinstance(models[best], EnsembleModel):
        m1, m2 = models[best].m1, models[best].m2
        m1.fit(X_full_fe, y_train)
        m2.fit(X_full_fe, y_train)
        final_model = EnsembleModel(m1, m2)
        print("DEBUG: Ensemble model retrained.")
    else:
        base = models[best]
        final_model = base.__class__(**base.get_params())
        final_model.fit(X_full_fe, y_train)
        print("DEBUG: Base model retrained.")

    # 7) save and report
    model_path = os.path.join(args.output_dir, "tree_model.pkl")
    save_model(final_model, model_path)
    print(f"DEBUG: Saved final model to {model_path}")
    report_results(metrics, args.output_dir)
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    print(f"DEBUG: Validation metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
