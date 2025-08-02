# final_train.py

"""
Entry point: merges **all** training splits, holds out 10% for validation,
selects & retrains the best model on 100% of the training data,
and logs metrics to TensorBoard.
"""
import argparse
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from .config import RANDOM_SEED, tb_log_subdir
from .auto_ml_pipeline.feature_engineering import engineer_features
from .auto_ml_pipeline.model_selection import select_and_train, EnsembleModel
from .auto_ml_pipeline.utils import setup_logging, save_model, report_results
from data.load import load_only_train_for_dataset, DATA_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoML Pipeline: final model on train-only dataset"
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to root folder with subfolders 1..N containing X_train.parquet & y_train.parquet.")
    parser.add_argument("--output_dir", type=str, default="output_final",
                        help="Directory to save the final model, report, and TensorBoard logs.")
    return parser.parse_args()


def tree_based_methods_model(dataset_path: str, output_dir: str):
    """
    Train and save the best tree-based model on the full training data,
    logging validation R² for each candidate and retrain time.
    """
    # 1) prepare output, logging, and TensorBoard
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    logging.info("Output directory '%s' is ready and logging is configured.", output_dir)

    ds_name = Path(dataset_path).name
    tb_log_dir = output_dir / tb_log_subdir / ds_name / "tree_based_methods" / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    # 2) load and merge all train splits
    ds = Path(dataset_path).name
    X_all, y_all = load_only_train_for_dataset(ds, DATA_ROOT)
    logging.info("Merged training data: %d rows", X_all.shape[0])

    # 3) hold out 10% for validation
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_all, y_all,
        test_size=0.10,
        random_state=RANDOM_SEED
    )
    logging.info("Split data into pool (%d rows) and validation (%d rows)",
                 X_pool.shape[0], X_val.shape[0])

    # 4) feature engineering
    X_pool_fe, X_val_fe = engineer_features(X_pool, X_val)
    if X_pool_fe.shape[1] == 0:
        logging.error("No features generated after engineering on pool")
        raise RuntimeError("No features after engineering on pool")
    logging.info("Feature engineering completed: %d features", X_pool_fe.shape[1])

    # 5) model selection on pool → val, with TensorBoard logging
    models, metrics = select_and_train(
        X_pool_fe, y_pool,
        X_val_fe, y_val,
        output_dir=output_dir,
        tb_writer=writer
    )
    val_metrics = {f"val_{k}": v for k, v in metrics.items()}
    report_results(val_metrics, output_dir)
    logging.info("Model selection complete. Validation metrics: %s", val_metrics)

    # log each candidate's validation R²
    for name, val_r2 in metrics.items():
        writer.add_scalar(f"Tree/val_r2_{name}", val_r2, 0)

    # 6) choose best model
    best_name = max(metrics, key=metrics.get)
    best_r2 = metrics[best_name]
    logging.info("Selected best model '%s' with validation R²=%.4f", best_name, best_r2)
    writer.add_text("Tree/selected_model", best_name, 0)
    writer.add_scalar("Tree/selected_model_r2", best_r2, 0)

    # 7) retrain best on 100% data, timing it
    logging.info("Retraining best model '%s' on full dataset", best_name)
    retrain_start = time.time()
    X_full_fe, _ = engineer_features(X_all, X_all.copy())
    if isinstance(models[best_name], EnsembleModel):
        m1, m2 = models[best_name].m1, models[best_name].m2
        m1.fit(X_full_fe, y_all)
        m2.fit(X_full_fe, y_all)
        final_model = EnsembleModel(m1, m2)
    else:
        base = models[best_name]
        final_model = base.__class__(**base.get_params())
        final_model.fit(X_full_fe, y_all)
    retrain_time = time.time() - retrain_start
    logging.info("Retraining completed in %.2fs", retrain_time)
    writer.add_scalar("Tree/retrain_time_s", retrain_time, 0)

    # 8) save final model
    model_filename = f"final_{best_name}.pkl"
    model_path = os.path.join(output_dir, model_filename)
    save_model(final_model, model_path)
    logging.info("Saved final model to '%s'", model_path)

    writer.close()
    return model_path, best_r2


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

    # Run the same pipeline as above for consistency
    tree_based_methods_model(args.dataset_path, args.output_dir)


if __name__ == "__main__":
    main()
