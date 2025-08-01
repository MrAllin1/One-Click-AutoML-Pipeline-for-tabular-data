# config.py

# Paths
train_dir = "path/to/train_dataset"
test_dir = "path/to/test_dataset"

# Output locations\ nmodel_dir = "models/tabnet_ensemble"
ensemble_predictions = "models/exam_output/y_pred.csv"

# Cross-Validation Settings
n_splits = 5
shuffle = True
random_state = 42

# Optuna Hyperparameter Optimization
direction = "maximize"
n_trials = 20

# Hyperparameter Search Ranges
n_d = (8, 64)
n_a = (8, 64)
n_steps = (3, 10)
gamma = (1.0, 2.0)
lambda_sparse = {
    "range": (1e-5, 1e-2),
    "log_scale": True,
}
lr = {
    "range": (1e-4, 1e-1),
    "log_scale": True,
}
weight_decay = {
    "range": (1e-6, 1e-3),
    "log_scale": True,
}

# Training Parameters
mask_type = "entmax"
initial_max_epochs = 100
initial_patience = 20
final_max_epochs = 150
final_patience = 30
batch_size = 1024
virtual_batch_size = 128
num_workers = 0
drop_last = False

# Pipeline Defaults (for tabnet_pipeline.py)
cv_folds = 10
cv_trials = 20
