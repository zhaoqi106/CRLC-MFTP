from copy import deepcopy


DEFAULT_CONFIG = {
    "seed": 20230226,
    "batch_size": 256,
    "fixed_threshold": 0.6,
    "model": {
        "token_dim": 128,
        "n_filters": 128,
        "filter_sizes": [3, 4, 5],
        "pool_size": 10,
        "hidden_dim": 256,
        "dropout": 0.6,
        "expert_dropout": 0.3,
        "candidate_threshold": 0.5,
    },
    "base_training": {
        "epochs": 200,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "lambda_probe": 0.2,
    },
    "residual_training": {
        "max_steps": 500,
        "validate_every": 25,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "lambda_cardinality": 0.2,
        "lambda_mfc": 0.3,
        "lambda_res": 0.3,
        "negative_weight": 0.05,
    },
}


def default_config():
    """Return an independently editable copy of the default settings."""
    return deepcopy(DEFAULT_CONFIG)
