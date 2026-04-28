"""Hyperparameter search space for the FlowPolicy + Franka Kitchen experiment.

The 8 user-requested ranges plus 5 ConditionalUnet1D ranges and 4 boolean flags.
Used by ``scripts/run_kitchen_experiment.py`` together with
``sklearn.model_selection.ParameterSampler`` (n_iter=100).
"""
from __future__ import annotations

from typing import Any, Dict, List

# 1) Hyperparam yang diminta user
USER_GRID: Dict[str, List[Any]] = {
    "epoch": [500, 1000, 3000, 5000],
    "learning_rate": [1e-3, 1e-4, 1e-5, 5e-4],
    "batch_size": [64, 128, 256, 512],
    "num_segments": [1, 2, 3, 4],
    "eps": [1e-2, 1e-3, 1e-4, 1.0],
    "delta": [1e-2, 1e-3, 1e-4, 1.0],
    "n_action_steps": [2, 4, 6, 8],
    "n_obs_steps": [4, 6, 8, 16],
}

# 2) Hyperparam tambahan untuk ConditionalUnet1D (4 nilai per param numerik)
UNET_GRID: Dict[str, List[Any]] = {
    "diffusion_step_embed_dim": [64, 128, 256, 512],
    "down_dims_preset": [
        [128, 256, 512],
        [256, 512, 1024],
        [512, 1024, 2048],
        [256, 512, 1024, 2048],
    ],
    "kernel_size": [3, 5, 7, 9],
    "n_groups": [2, 4, 8, 16],
    "var_fc_hidden_dim": [128, 256, 512, 1024],
    "condition_type": ["film", "add", "cross_attention_film", "mlp_film"],
}

# 3) Boolean flags (2 nilai)
BOOL_GRID: Dict[str, List[bool]] = {
    "use_down_condition": [True, False],
    "use_mid_condition": [True, False],
    "use_up_condition": [True, False],
    "obs_as_global_cond": [True, False],
}


def full_param_grid() -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    grid.update(USER_GRID)
    grid.update(UNET_GRID)
    grid.update(BOOL_GRID)
    return grid


def horizon_from(params: Dict[str, Any]) -> int:
    return int(params["n_obs_steps"]) + int(params["n_action_steps"])


def is_param_compatible(params: Dict[str, Any]) -> bool:
    """Filter trivially-bad combinations BEFORE building the model.

    UNet downsampling halves the time axis once per level, so the horizon
    must be at least ``2 ** (len(down_dims) - 1)``. Also, GroupNorm requires
    each channel size to be divisible by ``n_groups``.
    """
    horizon = horizon_from(params)
    down = params["down_dims_preset"]
    if horizon < 2 ** max(0, len(down) - 1):
        return False
    if any((c % int(params["n_groups"])) != 0 for c in down):
        return False
    if int(params["kernel_size"]) > horizon:
        return False
    return True
