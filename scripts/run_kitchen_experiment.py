"""Orchestrator for the FlowPolicy + Franka Kitchen experiment.

Pipeline (per seed in {0, 42, 101} x type in {no_preprocess, with_preprocess}):
  1. Random search proxy with `sklearn.model_selection.ParameterSampler`
     (n_iter=100, KFold=5 over training episodes; each candidate trained for
     a short proxy budget and scored by mean validation loss).
  2. Best candidate -> full training; success rate measured on the env runner.
  3. Save per-model summary to JSON.
  4. After all 6 models: plot success-rate bar chart, select the winner.
  5. Run inference rollouts on the winner with video + latency saved.

Logic-only file -- the heavy lifting (FlowPolicyLowdim, KitchenLowdimDataset,
KitchenRunner) is reused unchanged from ``flow_policy_3d``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from sklearn.model_selection import KFold, ParameterSampler
from termcolor import cprint
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PKG_ROOT = REPO_ROOT / "FlowPolicy"
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(REPO_ROOT))

# region agent log helpers
def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": "a2c3d3",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open("debug-a2c3d3.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


# endregion

# region agent log import-precheck
try:
    import zarr as _dbg_zarr  # noqa: E402
    import numcodecs as _dbg_numcodecs  # noqa: E402
    _debug_log(
        "H1",
        "scripts/run_kitchen_experiment.py:52",
        "zarr/numcodecs precheck success",
        {
            "zarr_version": getattr(_dbg_zarr, "__version__", "unknown"),
            "numcodecs_version": getattr(_dbg_numcodecs, "__version__", "unknown"),
            "python_version": sys.version.split()[0],
        },
    )
except Exception as _dbg_exc:
    _debug_log(
        "H1",
        "scripts/run_kitchen_experiment.py:64",
        "zarr/numcodecs precheck failure",
        {
            "error_type": type(_dbg_exc).__name__,
            "error": str(_dbg_exc),
            "python_version": sys.version.split()[0],
        },
    )
# endregion

try:
    from flow_policy_3d.dataset.kitchen_lowdim_dataset import KitchenLowdimDataset  # noqa: E402
    from flow_policy_3d.env_runner.kitchen_runner import KitchenRunner  # noqa: E402
    from flow_policy_3d.policy.flowpolicy_lowdim import FlowPolicyLowdim  # noqa: E402
except Exception as _import_exc:
    # region agent log import-failure
    _debug_log(
        "H2",
        "scripts/run_kitchen_experiment.py:82",
        "flow_policy import failure",
        {"error_type": type(_import_exc).__name__, "error": str(_import_exc)},
    )
    # endregion
    raise
from scripts.kitchen_search_space import (  # noqa: E402
    full_param_grid,
    horizon_from,
    is_param_compatible,
)


def _torch_load_checkpoint(path: Path):
    """Kompatibel PyTorch lama/baru (arg ``weights_only`` baru di beberapa versi)."""
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# --------------------------------------------------------------------------- utils
def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_policy(params: Dict[str, Any], state_dim: int, action_dim: int) -> FlowPolicyLowdim:
    horizon = horizon_from(params)
    shape_meta = {
        "obs": {"state": {"shape": [state_dim], "type": "low_dim"}},
        "action": {"shape": [action_dim]},
    }
    cfm = {
        "eps": float(params["eps"]),
        "num_segments": int(params["num_segments"]),
        "boundary": 1,
        "delta": float(params["delta"]),
        "alpha": 1e-5,
        "num_inference_step": 1,
    }
    policy = FlowPolicyLowdim(
        shape_meta=shape_meta,
        horizon=horizon,
        n_obs_steps=int(params["n_obs_steps"]),
        n_action_steps=int(params["n_action_steps"]),
        obs_as_global_cond=bool(params["obs_as_global_cond"]),
        diffusion_step_embed_dim=int(params["diffusion_step_embed_dim"]),
        down_dims=list(params["down_dims_preset"]),
        kernel_size=int(params["kernel_size"]),
        n_groups=int(params["n_groups"]),
        var_fc_hidden_dim=int(params["var_fc_hidden_dim"]),
        condition_type=str(params["condition_type"]),
        use_down_condition=bool(params["use_down_condition"]),
        use_mid_condition=bool(params["use_mid_condition"]),
        use_up_condition=bool(params["use_up_condition"]),
        Conditional_ConsistencyFM=cfm,
    )
    return policy


def _build_dataset(
    params: Dict[str, Any],
    preprocess: bool,
    seed: int,
    cache_path: str,
    split: str,
    episodes_override: Optional[List[Dict[str, np.ndarray]]] = None,
    custom_episode_indices: Optional[np.ndarray] = None,
) -> KitchenLowdimDataset:
    horizon = horizon_from(params)
    return KitchenLowdimDataset(
        horizon=horizon,
        n_obs_steps=int(params["n_obs_steps"]),
        n_action_steps=int(params["n_action_steps"]),
        cache_path=cache_path,
        seed=seed,
        preprocess=preprocess,
        window_ratio=0.25,
        noise_std=0.01,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        split=split,
        episodes=episodes_override,
        custom_episode_indices=custom_episode_indices,
    )


# --------------------------------------------------------------------------- training helpers
def _train_one_pass(
    policy: FlowPolicyLowdim,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    lr: float,
    num_epochs: int,
    max_train_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    progress_desc: str = "train",
) -> Dict[str, float]:
    device = device or _device()
    policy.to(device)
    policy.train()
    optim = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6)

    best_val = float("inf")
    last_val: Optional[float] = None
    last_train: float = float("nan")

    pbar = range(num_epochs)
    if num_epochs > 1:
        pbar = tqdm.tqdm(pbar, desc=progress_desc, leave=False, mininterval=2.0)
    for epoch in pbar:
        epoch_losses: List[float] = []
        for step_idx, batch in enumerate(train_loader):
            batch = {
                "obs": {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()},
                "action": batch["action"].to(device, non_blocking=True),
            }
            loss, _ = policy.compute_loss(batch)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            epoch_losses.append(float(loss.item()))
            if max_train_steps is not None and step_idx + 1 >= max_train_steps:
                break
        last_train = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        if val_loader is not None:
            policy.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        "obs": {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()},
                        "action": batch["action"].to(device, non_blocking=True),
                    }
                    vloss, _ = policy.compute_loss(batch)
                    val_losses.append(float(vloss.item()))
            policy.train()
            last_val = float(np.mean(val_losses)) if val_losses else float("inf")
            best_val = min(best_val, last_val)

    return {
        "train_loss": last_train,
        "val_loss": last_val if last_val is not None else float("nan"),
        "best_val_loss": best_val,
    }


# --------------------------------------------------------------------------- random search
def _kfold_score_candidate(
    params: Dict[str, Any],
    episodes: List[Dict[str, np.ndarray]],
    preprocess: bool,
    seed: int,
    cache_path: str,
    proxy_epochs: int,
    proxy_steps_per_epoch: int,
    cv_folds: int,
    state_dim: int,
    action_dim: int,
) -> float:
    """Return mean validation loss across `cv_folds` of the training episodes."""
    if not is_param_compatible(params):
        return float("inf")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    n_episodes = len(episodes)
    fold_scores: List[float] = []

    horizon = horizon_from(params)
    if horizon > min(ep["state"].shape[0] for ep in episodes):
        return float("inf")

    for tr_idx, va_idx in kf.split(np.arange(n_episodes)):
        try:
            train_ds = _build_dataset(
                params, preprocess, seed, cache_path, split="train",
                episodes_override=episodes, custom_episode_indices=tr_idx,
            )
            val_ds = _build_dataset(
                params, preprocess, seed, cache_path, split="val",
                episodes_override=episodes, custom_episode_indices=va_idx,
            )
        except Exception as exc:
            cprint(f"[search] dataset build failed: {exc}", "red")
            return float("inf")
        if len(train_ds) == 0 or len(val_ds) == 0:
            return float("inf")
        bs = min(int(params["batch_size"]), len(train_ds))
        if bs < 2:
            return float("inf")
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=0, drop_last=False)
        try:
            policy = _build_policy(params, state_dim=state_dim, action_dim=action_dim)
            normalizer = train_ds.get_normalizer()
            policy.set_normalizer(normalizer)
            metrics = _train_one_pass(
                policy, train_loader, val_loader,
                lr=float(params["learning_rate"]),
                num_epochs=proxy_epochs,
                max_train_steps=proxy_steps_per_epoch,
                progress_desc="proxy",
            )
        except Exception as exc:
            cprint(f"[search] proxy training failed: {type(exc).__name__}: {exc}", "red")
            del train_loader, val_loader
            torch.cuda.empty_cache()
            return float("inf")
        fold_scores.append(metrics["best_val_loss"])
        del policy, train_loader, val_loader
        torch.cuda.empty_cache()

    return float(np.mean(fold_scores))


def random_search(
    seed: int,
    preprocess: bool,
    grid: Dict[str, List[Any]],
    n_iter: int,
    cv_folds: int,
    proxy_epochs: int,
    proxy_steps_per_epoch: int,
    episodes: List[Dict[str, np.ndarray]],
    cache_path: str,
    state_dim: int,
    action_dim: int,
    output_dir: Path,
    log_prefix: str,
) -> Dict[str, Any]:
    sampler = ParameterSampler(grid, n_iter=n_iter, random_state=seed)
    best_score = float("inf")
    best_params: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []

    for i, params in enumerate(sampler):
        score = _kfold_score_candidate(
            dict(params), episodes, preprocess, seed, cache_path,
            proxy_epochs, proxy_steps_per_epoch, cv_folds, state_dim, action_dim,
        )
        history.append({"iter": i, "params": dict(params), "score": score})
        cprint(
            f"[{log_prefix}] iter {i + 1}/{n_iter} score={score:.5f} "
            f"best={best_score:.5f}",
            "magenta" if score < best_score else "white",
        )
        if score < best_score:
            best_score = score
            best_params = dict(params)
        # save running history
        with open(output_dir / f"{log_prefix}_search_history.json", "w") as f:
            json.dump({"history": history, "best_score": best_score, "best_params": best_params}, f, indent=2)

    return {"best_params": best_params, "best_score": best_score, "history": history}


# --------------------------------------------------------------------------- full training
def full_train_and_eval(
    best_params: Dict[str, Any],
    preprocess: bool,
    seed: int,
    episodes: List[Dict[str, np.ndarray]],
    cache_path: str,
    state_dim: int,
    action_dim: int,
    output_dir: Path,
    eval_episodes: int,
    rollout_every: int,
    record_video: bool,
) -> Dict[str, Any]:
    """Train the best candidate on the full 70/20/10 split, then eval."""
    train_ds = _build_dataset(best_params, preprocess, seed, cache_path, split="train",
                              episodes_override=episodes)
    val_ds = _build_dataset(best_params, preprocess, seed, cache_path, split="val",
                            episodes_override=episodes)
    test_ds = _build_dataset(best_params, preprocess, seed, cache_path, split="test",
                             episodes_override=episodes)

    bs = int(best_params["batch_size"])
    bs = min(bs, max(2, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=0, drop_last=False)

    policy = _build_policy(best_params, state_dim=state_dim, action_dim=action_dim)
    normalizer = train_ds.get_normalizer()
    policy.set_normalizer(normalizer)

    device = _device()
    policy.to(device)
    optim = torch.optim.AdamW(policy.parameters(),
                              lr=float(best_params["learning_rate"]),
                              betas=(0.95, 0.999), weight_decay=1e-6)

    runner = KitchenRunner(
        output_dir=str(output_dir),
        eval_episodes=eval_episodes,
        max_steps=280,
        n_obs_steps=int(best_params["n_obs_steps"]),
        n_action_steps=int(best_params["n_action_steps"]),
        record_video=record_video,
        target_state_dim=state_dim,
    )

    epoch_history: List[Dict[str, float]] = []
    best_sr = -1.0
    best_ckpt_path = output_dir / "best.ckpt"
    epochs = int(best_params["epoch"])
    cprint(f"[full] training {epochs} epochs (lr={best_params['learning_rate']})", "yellow")

    for ep in tqdm.tqdm(range(epochs), desc="full-train", mininterval=2.0):
        policy.train()
        train_losses: List[float] = []
        for batch in train_loader:
            batch = {
                "obs": {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()},
                "action": batch["action"].to(device, non_blocking=True),
            }
            loss, _ = policy.compute_loss(batch)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            train_losses.append(float(loss.item()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        policy.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    "obs": {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()},
                    "action": batch["action"].to(device, non_blocking=True),
                }
                vloss, _ = policy.compute_loss(batch)
                val_losses.append(float(vloss.item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        sr_log: Dict[str, float] = {}
        if (ep + 1) % rollout_every == 0 or ep == epochs - 1:
            try:
                runner_log = runner.run(policy, save_video=False)
                sr_log = {k: float(v) for k, v in runner_log.items() if isinstance(v, (int, float))}
                sr = sr_log.get("mean_success_rate", 0.0)
                if sr > best_sr:
                    best_sr = sr
                    torch.save(
                        {
                            "policy_state_dict": policy.state_dict(),
                            "normalizer_state_dict": normalizer.state_dict(),
                            "params": best_params,
                            "preprocess": preprocess,
                            "seed": seed,
                            "epoch": ep,
                            "success_rate": sr,
                            "state_dim": state_dim,
                            "action_dim": action_dim,
                        },
                        best_ckpt_path,
                    )
                    cprint(f"[full] new best SR={sr:.4f} -> {best_ckpt_path}", "green")
            except Exception as exc:
                cprint(f"[full] rollout failed: {exc}", "red")

        epoch_history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, **sr_log})
        with open(output_dir / "epoch_history.json", "w") as f:
            json.dump(epoch_history, f, indent=2)

    return {
        "best_success_rate": best_sr,
        "history": epoch_history,
        "best_ckpt_path": str(best_ckpt_path),
        "test_dataset_size": len(test_ds),
    }


# --------------------------------------------------------------------------- inference
def run_inference_on_winner(
    winner_dir: Path,
    eval_episodes: int = 20,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ckpt_path = winner_dir / "best.ckpt"
    payload = _torch_load_checkpoint(ckpt_path)
    params = payload["params"]
    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])

    policy = _build_policy(params, state_dim=state_dim, action_dim=action_dim)
    policy.normalizer.load_state_dict(payload["normalizer_state_dict"])
    policy.load_state_dict(payload["policy_state_dict"])
    policy.to(_device())
    policy.eval()

    out = output_dir or winner_dir
    out.mkdir(parents=True, exist_ok=True)
    runner = KitchenRunner(
        output_dir=str(out),
        eval_episodes=eval_episodes,
        max_steps=280,
        n_obs_steps=int(params["n_obs_steps"]),
        n_action_steps=int(params["n_action_steps"]),
        record_video=True,
        video_filename="best_model_inference.mp4",
        target_state_dim=state_dim,
    )
    log = runner.run(policy, save_video=True)
    log["mean_latency_ms"] = float(log.get("mean_latency_s", 0.0)) * 1000.0
    log["params"] = params
    with open(out / "inference_summary.json", "w") as f:
        json.dump({k: v for k, v in log.items() if not isinstance(v, np.ndarray)}, f, indent=2, default=str)
    return log


# --------------------------------------------------------------------------- plotting
def plot_success_rates(summaries: List[Dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"seed={s['seed']}\n{s['type']}" for s in summaries]
    values = [s["best_success_rate"] for s in summaries]
    winner = int(np.argmax(values))
    colors = ["#4C72B0"] * len(values)
    colors[winner] = "#C44E52"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Success rate (mean fraction of subtasks completed)")
    ax.set_title("FlowPolicy Lowdim - Franka Kitchen (3 seeds x 2 types)")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# --------------------------------------------------------------------------- entrypoint
@dataclass
class ExperimentConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 42, 101])
    types: List[str] = field(default_factory=lambda: ["no_preprocess", "with_preprocess"])
    n_iter: int = 100
    cv_folds: int = 5
    proxy_epochs: int = 5
    proxy_steps_per_epoch: int = 50
    eval_episodes_during_training: int = 5
    rollout_every: int = 100
    inference_episodes: int = 20
    cache_path: str = "data/kitchen_complete_v2_episodes.npz"
    output_root: str = "results_kitchen"


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 42, 101])
    p.add_argument(
        "--types",
        nargs="*",
        choices=["no_preprocess", "with_preprocess"],
        default=["no_preprocess", "with_preprocess"],
        help="Subset tipe pelatihan (default: keduanya seperti spek eksperimen).",
    )
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--proxy_epochs", type=int, default=5)
    p.add_argument("--proxy_steps_per_epoch", type=int, default=50)
    p.add_argument("--rollout_every", type=int, default=100)
    p.add_argument("--inference_episodes", type=int, default=20)
    p.add_argument("--eval_episodes_during_training", type=int, default=5)
    p.add_argument("--cache_path", type=str, default="data/kitchen_complete_v2_episodes.npz")
    p.add_argument("--output_root", type=str, default="results_kitchen")
    args = p.parse_args()
    return ExperimentConfig(
        seeds=args.seeds,
        types=list(args.types),
        n_iter=args.n_iter,
        cv_folds=args.cv,
        proxy_epochs=args.proxy_epochs,
        proxy_steps_per_epoch=args.proxy_steps_per_epoch,
        rollout_every=args.rollout_every,
        inference_episodes=args.inference_episodes,
        eval_episodes_during_training=args.eval_episodes_during_training,
        cache_path=args.cache_path,
        output_root=args.output_root,
    )


def main() -> None:
    cfg = parse_args()
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load dataset once via the dataset class (it caches to disk for next runs)
    cprint("[main] loading kitchen dataset (this triggers Minari download once)", "cyan")
    base_ds = KitchenLowdimDataset(
        horizon=12, n_obs_steps=4, n_action_steps=8,
        cache_path=cfg.cache_path, seed=0, preprocess=False, split="train",
    )
    episodes = base_ds.episodes
    state_dim = base_ds.state_dim
    action_dim = base_ds.action_dim
    cprint(f"[main] state_dim={state_dim} action_dim={action_dim} n_episodes={len(episodes)}", "yellow")

    grid = full_param_grid()
    summaries: List[Dict[str, Any]] = []

    for seed in cfg.seeds:
        for tipe in cfg.types:
            preprocess = (tipe == "with_preprocess")
            run_dir = output_root / f"seed{seed}_{tipe}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cprint(f"\n=== seed={seed} type={tipe} preprocess={preprocess} ===", "cyan")
            _set_seed(seed)

            search_log = random_search(
                seed=seed,
                preprocess=preprocess,
                grid=grid,
                n_iter=cfg.n_iter,
                cv_folds=cfg.cv_folds,
                proxy_epochs=cfg.proxy_epochs,
                proxy_steps_per_epoch=cfg.proxy_steps_per_epoch,
                episodes=episodes,
                cache_path=cfg.cache_path,
                state_dim=state_dim,
                action_dim=action_dim,
                output_dir=run_dir,
                log_prefix=f"seed{seed}_{tipe}",
            )
            best_params = search_log["best_params"]
            if best_params is None:
                cprint("[main] random search produced no valid candidate; skipping full train", "red")
                continue
            with open(run_dir / "best_params.json", "w") as f:
                json.dump(best_params, f, indent=2)

            cprint(f"[main] best score={search_log['best_score']:.5f}", "magenta")
            cprint(f"[main] best params={best_params}", "magenta")

            full = full_train_and_eval(
                best_params=best_params,
                preprocess=preprocess,
                seed=seed,
                episodes=episodes,
                cache_path=cfg.cache_path,
                state_dim=state_dim,
                action_dim=action_dim,
                output_dir=run_dir,
                eval_episodes=cfg.eval_episodes_during_training,
                rollout_every=cfg.rollout_every,
                record_video=False,
            )
            summary = {
                "seed": seed,
                "type": tipe,
                "best_params": best_params,
                "best_search_score": search_log["best_score"],
                "best_success_rate": float(full["best_success_rate"]),
                "best_ckpt_path": full["best_ckpt_path"],
            }
            summaries.append(summary)
            with open(run_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

    if not summaries:
        cprint("[main] no summaries to plot", "red")
        return

    with open(output_root / "all_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    plot_path = output_root / "success_rate.png"
    plot_success_rates(summaries, plot_path)
    cprint(f"[main] plot saved -> {plot_path}", "cyan")

    if len(summaries) == 1:
        winner = summaries[0]
    else:
        winner = max(summaries, key=lambda s: s["best_success_rate"])
    cprint(
        f"[main] winner: seed={winner['seed']} type={winner['type']} "
        f"SR={winner['best_success_rate']:.4f}",
        "green",
    )

    winner_dir = output_root / f"seed{winner['seed']}_{winner['type']}"
    inference_dir = output_root / "inference_winner"
    inference_dir.mkdir(parents=True, exist_ok=True)
    inf = run_inference_on_winner(
        winner_dir=winner_dir,
        eval_episodes=cfg.inference_episodes,
        output_dir=inference_dir,
    )
    cprint(
        f"[main] inference SR={inf['mean_success_rate']:.4f} "
        f"latency={inf['mean_latency_ms']:.2f} ms",
        "green",
    )


if __name__ == "__main__":
    main()
