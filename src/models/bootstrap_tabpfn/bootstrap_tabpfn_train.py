#!/usr/bin/env python3
# bootstrap_tabpfn_train.py

import argparse
import logging
import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor
from torch.utils.tensorboard import SummaryWriter

import optuna
from optuna.exceptions import TrialPruned
from data import load_full_train_for_specific_dataset, get_test_data
from .bootstrap_ensemble import EnsemblePFN
from . import config

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_bootstrap(
    dataset: str,
    output_dir: Path,
    n_bootstrap: int = config.n_bootstrap_default,
    sample_frac: float = config.sample_frac_default,
    seed: int = config.seed_default,
    fold: int = config.fold_default,
    use_optuna: bool = config.use_optuna_default,
    n_trials: int = config.optuna_n_trials_default,
) -> tuple[Path, float]:
    """
    Performs bootstrap ensemble training or Optuna-tuned training.
    Returns final ensemble path and OOB R².
    """
    if use_optuna:
        def objective(trial):
            n_bs = trial.suggest_int("n_bootstrap", *config.tpfn_range_n_bootstrap)
            frac = trial.suggest_float("sample_frac", *config.tpfn_range_sample_frac)
            logger.info(f"🔍 Trial {trial.number}: n_bootstrap={n_bs}, sample_frac={frac:.2f}")
            return _optuna_inner_objective(dataset, seed, fold, n_bs, frac, trial)

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=config.optuna_pruner_n_startup,
            n_warmup_steps=config.optuna_pruner_warmup,
            interval_steps=config.optuna_pruner_interval
        )
        study = optuna.create_study(
            direction=config.optuna_direction,
            sampler=optuna.samplers.TPESampler(seed=config.optuna_sampler_seed),
            pruner=pruner
        )
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        logger.info(f"🏆 Best Params: {best}, R²={study.best_value:.5f}")

        # Retrain final ensemble with best params
        return train_bootstrap(
            dataset=dataset,
            output_dir=output_dir,
            n_bootstrap=best["n_bootstrap"],
            sample_frac=best["sample_frac"],
            seed=seed,
            fold=fold,
            use_optuna=False
        )

    # Manual training below (no Optuna)
    start_time = time.time()
    device = _device()
    ds_name = Path(dataset).name
    outdir = output_dir / ds_name
    outdir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(output_dir / config.tensorboard_runs_dir / ts))
    logger.info(f"Bootstrap training → dataset={ds_name}, device={device}")

    # Load one fold (train only)
    X_full, y_full = load_full_train_for_specific_dataset(dataset)
    n_samples = len(X_full)
    sample_size = int(n_samples * sample_frac)

    # Storage
    oob_preds = defaultdict(list)
    model_paths: list[Path] = []

    for i in range(1, n_bootstrap + 1):
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)

        idx = np.random.choice(n_samples, size=sample_size, replace=True)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[np.unique(idx)] = False

        X_bs = X_full.iloc[idx]
        y_bs = y_full.iloc[idx].values.ravel()

        logger.info(f"[{i}/{n_bootstrap}] bootstrap sample size={sample_size}")
        agent = TabPFNRegressor(device=device,ignore_pretraining_limits=True)
        agent.fit(X_bs, y_bs)

        if oob_mask.any():
            X_oob = X_full[oob_mask]
            pred_oob = agent.predict(X_oob)
            for row_idx, p in zip(np.where(oob_mask)[0], pred_oob):
                oob_preds[row_idx].append(p)

        out_path = outdir / f"bootstrap_{i}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(agent, f)
        model_paths.append(out_path)
        logger.info(f"Saved model → {out_path}")

        del agent
        if device == "cuda":
            torch.cuda.empty_cache()

    # Compute final OOB R²
    final_preds = np.zeros(n_samples)
    used = np.zeros(n_samples, dtype=bool)
    for idx0, preds in oob_preds.items():
        final_preds[idx0] = np.mean(preds)
        used[idx0] = True

    y_true = y_full.values.ravel()[used]
    y_pred = final_preds[used]
    oob_r2 = r2_score(y_true, y_pred)

    logger.info(f"📊 Final OOB R² = {oob_r2:.5f}")
    writer.add_scalar("R2_score/OOB", oob_r2, 0)

    ensemble = EnsemblePFN(model_paths)
    ens_path = outdir / "ensemble.pkl"
    with open(ens_path, "wb") as f:
        pickle.dump(ensemble, f)
    logger.info(f"Saved ensemble → {ens_path}")

    writer.close()
    logger.info(f"Total time: {time.time() - start_time:.1f}s")
    return ens_path, oob_r2


def _optuna_inner_objective(dataset, seed, fold, n_bootstrap, sample_frac, trial):
    X_full, y_full = load_full_train_for_specific_dataset(dataset)
    n_samples = len(X_full)
    sample_size = int(n_samples * sample_frac)
    oob_preds = defaultdict(list)
    device = _device()

    for i in range(1, n_bootstrap + 1):
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)

        idx = np.random.choice(n_samples, size=sample_size, replace=True)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[np.unique(idx)] = False

        agent = TabPFNRegressor(device=device,ignore_pretraining_limits=True)
        agent.fit(X_full.iloc[idx], y_full.iloc[idx].values.ravel())

        if oob_mask.any():
            pred_oob = agent.predict(X_full[oob_mask])
            for row_idx, p in zip(np.where(oob_mask)[0], pred_oob):
                oob_preds[row_idx].append(p)

        preds_arr = np.zeros(n_samples)
        used = np.zeros(n_samples, dtype=bool)
        for idx0, preds in oob_preds.items():
            preds_arr[idx0] = np.mean(preds)
            used[idx0] = True
        y_t = y_full.values.ravel()[used]
        y_p = preds_arr[used]
        intermediate_r2 = r2_score(y_t, y_p)

        trial.report(intermediate_r2, step=i)
        if trial.should_prune():
            logger.info(f"⏸️ Pruned trial {trial.number} at step {i} (R²={intermediate_r2:.4f})")
            raise TrialPruned()

        del agent
        if device == "cuda":
            torch.cuda.empty_cache()

    return intermediate_r2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bootstrap-bagging TabPFN with Optuna pruning")
    p.add_argument("-d", "--dataset",      required=True, help="Dataset root folder")
    p.add_argument("-o", "--out-dir",      default=config.out_dir_default, help="Output directory")
    p.add_argument("--n-bootstrap", type=int, default=config.n_bootstrap_default,   help="Number of bootstraps (manual mode)")
    p.add_argument("--sample-frac", type=float, default=config.sample_frac_default, help="Sample fraction (manual mode)")
    p.add_argument("--seed",        type=int, default=config.seed_default,     help="Random seed")
    p.add_argument("--fold",        type=int, default=config.fold_default,  help="Which fold to use (overrides random)")
    p.add_argument("--optuna",      action="store_true", default=config.use_optuna_default,     help="Enable Optuna Bayesian search")
    p.add_argument("--n-trials",    type=int, default=config.optuna_n_trials_default,    help="Number of Optuna trials (when --optuna)")

    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _, r2 = train_bootstrap(
        dataset=args.dataset,
        output_dir=out,
        n_bootstrap=args.n_bootstrap,
        sample_frac=args.sample_frac,
        seed=args.seed,
        fold=args.fold,
        use_optuna=args.optuna,
        n_trials=args.n_trials
    )
    logger.info(f"📊 Final OOB R²: {r2:.5f}")
