#!/usr/bin/env python3
"""
Verbose prediction script for TabPFN models **with safe chunked inference**.

This version prevents CUDA out‑of‑memory errors by slicing the test set into
manageable chunks (default = 2 000 rows) and clearing the GPU between slices.
Works for both single TabPFN fold models and EnsemblePFN collections.

If the dataset directory contains `y_test.csv`, the script also computes R² and
stores it next to the prediction CSV as `<output>.r2.txt`.

Examples
--------
Single fold::
    python predict_with_logs.py -d data/superconductivity \
        -m models/fold3.pkl -o preds/fold3.csv

Whole ensemble with 3 000‑row chunks::
    python predict_with_logs.py -d data/superconductivity \
        -m models/ensemble.pkl -o preds/ensemble.csv --rows-per-chunk 3000
"""
from __future__ import annotations

import argparse
import gzip
import logging
import pickle
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Optional: torch just for GPU diagnostics & cache clearing
try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore

from data import load_full_test_for_specific_dataset
from tabpfn_ensemble import EnsemblePFN  # ← adjust import path if needed

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # override any env‑level config to ensure we get our format
)
logger = logging.getLogger("TabPFN-Predict")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _gpu_diagnostics() -> None:
    """Log CUDA availability, GPU name, and basic memory stats (if torch)."""
    if torch is None:
        logger.info("torch not installed → skipping GPU diagnostics")
        return

    cuda_ok = torch.cuda.is_available()
    logger.info("CUDA available: %s", cuda_ok)
    if not cuda_ok:
        return

    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    props = torch.cuda.get_device_properties(dev)
    total = props.total_memory / 2**20  # MiB
    allocated = torch.cuda.memory_allocated(dev) / 2**20
    reserved = torch.cuda.memory_reserved(dev) / 2**20
    logger.info(
        "GPU: %s | memory total: %.0f MiB | allocated: %.0f MiB | reserved: %.0f MiB",
        name,
        total,
        allocated,
        reserved,
    )


def _hydrate_ensemble(ens: "EnsemblePFN", search_dir: Path) -> "EnsemblePFN":
    """Populate an EnsemblePFN with fold models found on disk (if empty)."""
    if ens.models:
        return ens

    logger.info("EnsemblePFN empty → loading fold models from %s", search_dir)
    fold_files = sorted(search_dir.glob("*_fold*.pkl"))
    if not fold_files:
        raise ValueError(
            "EnsemblePFN contains no fold models and none were found on disk.\n"
            "Re‑export the ensemble *after* training, or provide the fold files."
        )
    t0 = time.perf_counter()
    ens.models = [pickle.load(fp.open("rb")) for fp in fold_files]
    logger.info("Loaded %d fold models in %.2f s", len(ens.models), time.perf_counter() - t0)
    return ens


def _load_model(path: Path) -> Union["EnsemblePFN", object]:
    """Load a single‑fold model or EnsemblePFN from *path* (file or directory)."""
    logger.info("Loading model from %s", path)
    t0 = time.perf_counter()

    if path.is_dir():
        ens_pkl = path / "ensemble.pkl"
        if ens_pkl.exists():
            with ens_pkl.open("rb") as f:
                obj = pickle.load(f)
            model = _hydrate_ensemble(obj, path) if isinstance(obj, EnsemblePFN) else obj
        else:
            folds = sorted(path.glob("*_fold*.pkl"))
            if not folds:
                raise FileNotFoundError(f"No ensemble.pkl or *_fold*.pkl files found in {path}")
            model = EnsemblePFN([pickle.load(fp.open("rb")) for fp in folds])
    else:
        if path.suffix not in {".pkl", ".pkl.gz"}:
            raise ValueError("model‑path must be a .pkl/.pkl.gz file or a directory")
        with (path.open("rb") if path.suffix == ".pkl" else gzip.open(path, "rb")) as f:
            obj = pickle.load(f)
        model = _hydrate_ensemble(obj, path.parent) if isinstance(obj, EnsemblePFN) else obj

    logger.info("Model loaded in %.2f s", time.perf_counter() - t0)
    return model

# -----------------------------------------------------------------------------
# Chunked prediction helpers
# -----------------------------------------------------------------------------

def _predict_in_chunks(model, X, rows_per_chunk: int) -> np.ndarray:
    """Run *model.predict* over *X* in row‑chunks to cap peak GPU memory."""
    preds = []
    for start in range(0, len(X), rows_per_chunk):
        end = start + rows_per_chunk
        chunk = X.iloc[start:end]
        chunk_pred = model.predict(chunk).squeeze()
        preds.append(chunk_pred)
        if torch is not None and torch.cuda.is_available():  # clear leftovers
            torch.cuda.empty_cache()
    return np.concatenate(preds)

# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main(dataset_dir: str, model_path: Path, output_path: Path, rows_per_chunk: int) -> None:
    _gpu_diagnostics()

    # -------- data -----------------------------------------------------------
    logger.info("Loading test data from %s", dataset_dir)
    t0 = time.perf_counter()
    X_test, y_test = load_full_test_for_specific_dataset(dataset_dir)
    y_test = y_test.squeeze("columns")
    logger.info("Loaded X_test with shape %s and y_test len %d in %.2f s", X_test.shape, len(y_test), time.perf_counter() - t0)

    # -------- model ----------------------------------------------------------
    model = _load_model(model_path)

    # -------- prediction -----------------------------------------------------
    logger.info("Predicting with rows_per_chunk=%s", rows_per_chunk or "∞ (no chunking)")
    t0 = time.perf_counter()
    if rows_per_chunk and rows_per_chunk > 0:
        y_pred = _predict_in_chunks(model, X_test, rows_per_chunk)
    else:
        y_pred = model.predict(X_test).squeeze()
    pred_time = time.perf_counter() - t0
    logger.info("Prediction done in %.2f s (≈ %.2f ms/sample)", pred_time, 1000 * pred_time / len(X_test))

    # -------- save predictions ----------------------------------------------
    out_df = pd.DataFrame({"prediction": y_pred, "target": y_test.reset_index(drop=True)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("Saved predictions to %s (%.1f kB)", output_path, output_path.stat().st_size / 1024)

    # -------- optional R² ----------------------------------------------------
    try:
        score = r2_score(y_test, y_pred)
    except Exception as exc:  # e.g. y_test missing / wrong len
        logger.warning("Could not compute R²: %s", exc)
    else:
        r2_path = output_path.with_suffix(".r2.txt")
        r2_path.write_text(f"{score:.6f}\n")
        logger.info("R² = %.6f  (saved to %s)", score, r2_path)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Predict with a TabPFN model/ensemble, with OOM‑safe chunking and verbose logs.")
    cli.add_argument("-d", "--dataset", required=True, help="Dataset folder used for training (under data/)")
    cli.add_argument("-m", "--model-path", required=True, type=Path, help="Path to fold/ensemble .pkl or a directory")
    cli.add_argument("-o", "--output-path", required=True, type=Path, help="CSV file to write predictions")
    cli.add_argument("--rows-per-chunk", type=int, default=2000, metavar="N",
                     help="Number of rows per GPU chunk (0 = disable chunking)")
    args = cli.parse_args()

    logger.info("=== TabPFN prediction script started ===")
    start = time.perf_counter()
    try:
        main(args.dataset, args.model_path, args.output_path, args.rows_per_chunk)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Fatal error: %s", exc)
        raise
    finally:
        logger.info("Total runtime: %.2f s", time.perf_counter() - start)
