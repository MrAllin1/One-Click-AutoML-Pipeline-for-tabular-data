# config.py for TabPFN bootstrap training (N≈10 000)

# Output paths
out_dir_default         = "models/tabpfn"
tb_log_subdir = "tensorboard_logs"

# Bootstrap sampling
n_bootstrap_default     = 30
sample_frac_default     = 0.8

# Randomization
seed_default            = 42
fold_default            = None  # let simulate_exam_dataset pick

# Optuna settings
use_optuna_default      = True
optuna_direction        = "maximize"
optuna_n_trials_default = 30
optuna_sampler_seed     = seed_default

# Pruner settings
optuna_pruner_n_startup = 5
optuna_pruner_warmup    = 0
optuna_pruner_interval  = 1

# Search ranges
tpfn_range_n_bootstrap  = (10, 20)
tpfn_range_sample_frac  = (0.6, 0.9)
