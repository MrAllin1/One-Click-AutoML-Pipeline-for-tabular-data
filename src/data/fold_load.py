
import pandas as pd
from pathlib import Path
from typing import List, Optional

def get_available_folds(dataset: str, data_root: Path) -> List[int]:
    """
    Lists available folds based on folder structure like: data/exam_dataset/1, 2, ...
    """
    dataset_path = data_root / dataset
    fold_dirs = [f for f in dataset_path.iterdir() if f.is_dir() and f.name.isdigit()]
    fold_indices = sorted([int(f.name) for f in fold_dirs])
    return fold_indices


def load_fold(dataset: str, fold: int, data_root: Path = Path(".")):
    fold_path = data_root / dataset / str(fold)

    X_train = pd.read_parquet(fold_path / "X_train.parquet")
    X_test = pd.read_parquet(fold_path / "X_test.parquet")
    y_train = pd.read_parquet(fold_path / "y_train.parquet")
    
    # If y_test exists in some datasets, load conditionally
    y_test_path = fold_path / "y_test.parquet"
    if y_test_path.exists():
        y_test = pd.read_parquet(y_test_path)
    else:
        y_test = None

    return X_train, X_test, y_train, y_test
