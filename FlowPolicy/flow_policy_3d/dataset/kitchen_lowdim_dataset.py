"""Lowdim Franka-Kitchen dataset based on Minari `D4RL/kitchen/complete-v2`.

Two preprocessing modes are supported:

* ``preprocess=False`` -- standard sliding window of length ``horizon`` with
  ``stride=1`` per episode; episode-level split 70/20/10; no noise.
* ``preprocess=True``  -- per episode, panjang jendela
  ``W = max(horizon, int(round(window_ratio * T)))`` (default ``window_ratio=0.25``),
  geser ``stride=1``; dari setiap segmen ``[s:s+W]`` dipakai **langkah terakhir**
  sepanjang ``horizon`` untuk BC; split 70/20/10 di level window; noise Gaussian
  pada state **hanya** split train.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from termcolor import cprint

from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.dataset.base_dataset import BaseDataset
from flow_policy_3d.model.common.normalizer import LinearNormalizer


def _flatten_obs(obs) -> np.ndarray:
    """Convert a Minari observation entry to a 2-D float32 numpy array."""
    if isinstance(obs, np.ndarray):
        return np.asarray(obs, dtype=np.float32)
    if isinstance(obs, dict):
        # Prefer the common 'observation' key, then concatenate desired_goal
        # values (if dict) so that the goal context is still visible.
        parts: List[np.ndarray] = []
        if "observation" in obs:
            parts.append(np.asarray(obs["observation"], dtype=np.float32))
        if "desired_goal" in obs:
            dg = obs["desired_goal"]
            if isinstance(dg, dict):
                for v in dg.values():
                    parts.append(np.asarray(v, dtype=np.float32))
            else:
                parts.append(np.asarray(dg, dtype=np.float32))
        if not parts:
            raise ValueError(f"Unsupported observation dict keys: {list(obs.keys())}")
        # Broadcast each (T,) vector to (T, k) if needed and concatenate on last dim
        T = parts[0].shape[0]
        normalised = []
        for p in parts:
            if p.ndim == 1:
                p = p.reshape(T, -1) if p.shape[0] == T else p[None, :].repeat(T, axis=0)
            normalised.append(p)
        return np.concatenate(normalised, axis=-1).astype(np.float32)
    raise TypeError(f"Unsupported observation type: {type(obs)}")


def _load_minari_episodes(dataset_id: str) -> List[Dict[str, np.ndarray]]:
    """Return list of {'state': (T, Ds), 'action': (T, Da)} numpy episodes."""
    import minari  # lazy import so the package isn't required at import time

    try:
        ds = minari.load_dataset(dataset_id, download=True)
    except TypeError:
        # older minari versions: download separately
        try:
            minari.download_dataset(dataset_id)
        except Exception:
            pass
        ds = minari.load_dataset(dataset_id)

    episodes: List[Dict[str, np.ndarray]] = []
    for ep in ds.iterate_episodes():
        states = _flatten_obs(ep.observations)
        actions = np.asarray(ep.actions, dtype=np.float32)
        # observations have length T+1, actions have length T -> trim states
        T = actions.shape[0]
        if states.shape[0] == T + 1:
            states = states[:T]
        elif states.shape[0] != T:
            T = min(states.shape[0], T)
            states = states[:T]
            actions = actions[:T]
        if T < 2:
            continue
        episodes.append({"state": states, "action": actions})
    if not episodes:
        raise RuntimeError(f"Minari dataset '{dataset_id}' produced 0 usable episodes")
    return episodes


def _save_episode_cache(episodes: List[Dict[str, np.ndarray]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "n_episodes": len(episodes),
    }
    for i, ep in enumerate(episodes):
        payload[f"state_{i}"] = ep["state"]
        payload[f"action_{i}"] = ep["action"]
    np.savez_compressed(path, **payload)


def _load_episode_cache(path: str) -> List[Dict[str, np.ndarray]]:
    arr = np.load(path, allow_pickle=False)
    n = int(arr["n_episodes"])
    return [{"state": arr[f"state_{i}"], "action": arr[f"action_{i}"]} for i in range(n)]


class KitchenLowdimDataset(BaseDataset):
    """Behavior-cloning dataset for Franka Kitchen (lowdim, state-based)."""

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        dataset_id: str = "D4RL/kitchen/complete-v2",
        cache_path: Optional[str] = None,
        seed: int = 42,
        preprocess: bool = False,
        window_ratio: float = 0.25,
        noise_std: float = 0.01,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        split: str = "train",
        episodes: Optional[List[Dict[str, np.ndarray]]] = None,
        custom_episode_indices: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got '{split}'")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train+val+test ratios must sum to 1.0")

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.preprocess = preprocess
        self.window_ratio = window_ratio
        self.noise_std = noise_std
        self.split = split
        self.seed = seed
        self.dataset_id = dataset_id
        self.cache_path = cache_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # ---- load raw episodes (from cache, override, or minari) ----
        if episodes is not None:
            self.episodes = episodes
        elif cache_path is not None and os.path.isfile(cache_path):
            cprint(f"[KitchenLowdimDataset] loading cache {cache_path}", "cyan")
            self.episodes = _load_episode_cache(cache_path)
        else:
            cprint(f"[KitchenLowdimDataset] loading Minari '{dataset_id}'", "cyan")
            self.episodes = _load_minari_episodes(dataset_id)
            if cache_path is not None:
                _save_episode_cache(self.episodes, cache_path)

        self.state_dim = self.episodes[0]["state"].shape[-1]
        self.action_dim = self.episodes[0]["action"].shape[-1]
        cprint(
            f"[KitchenLowdimDataset] n_episodes={len(self.episodes)} "
            f"state_dim={self.state_dim} action_dim={self.action_dim} "
            f"preprocess={preprocess} split={split}",
            "yellow",
        )

        # ---- build window indices per episode ----
        # Each tuple = (episode_idx, start_offset). Output slice length is always ``horizon``.
        all_windows: List[Tuple[int, int]] = []
        rng = np.random.default_rng(seed)

        def _preprocess_window_len(ep_len: int) -> int:
            return max(horizon, int(round(window_ratio * ep_len)))

        if preprocess:
            for ei, ep in enumerate(self.episodes):
                T = ep["state"].shape[0]
                W = _preprocess_window_len(T)
                if W > T:
                    continue
                all_windows.extend([(ei, s) for s in range(0, T - W + 1)])
        else:
            for ei, ep in enumerate(self.episodes):
                T = ep["state"].shape[0]
                if T < horizon:
                    continue
                all_windows.extend([(ei, s) for s in range(0, T - horizon + 1)])

        if not all_windows:
            raise RuntimeError("No windows produced; check horizon vs episode length.")

        all_windows_arr = np.array(all_windows, dtype=np.int64)

        # ---- 70/20/10 split ----
        if custom_episode_indices is not None:
            # external orchestrator pre-decided which episode indices belong to this dataset
            ep_set = set(int(x) for x in custom_episode_indices)
            mask = np.array([w[0] in ep_set for w in all_windows], dtype=bool)
            self.windows = all_windows_arr[mask]
        else:
            if preprocess:
                # window-level split
                perm = rng.permutation(len(all_windows_arr))
                n = len(all_windows_arr)
                n_train = int(round(train_ratio * n))
                n_val = int(round(val_ratio * n))
                if split == "train":
                    chosen = perm[:n_train]
                elif split == "val":
                    chosen = perm[n_train : n_train + n_val]
                else:
                    chosen = perm[n_train + n_val :]
                self.windows = all_windows_arr[chosen]
            else:
                # episode-level split (deterministic by seed)
                ep_perm = rng.permutation(len(self.episodes))
                n_ep = len(self.episodes)
                n_tr = max(1, int(round(train_ratio * n_ep)))
                n_va = max(1, int(round(val_ratio * n_ep)))
                if n_tr + n_va >= n_ep:
                    n_va = max(1, n_ep - n_tr - 1)
                if split == "train":
                    chosen_ep = set(int(x) for x in ep_perm[:n_tr])
                elif split == "val":
                    chosen_ep = set(int(x) for x in ep_perm[n_tr : n_tr + n_va])
                else:
                    chosen_ep = set(int(x) for x in ep_perm[n_tr + n_va :])
                if len(chosen_ep) == 0:
                    chosen_ep = {int(ep_perm[0])}
                mask = np.array([int(w[0]) in chosen_ep for w in all_windows_arr], dtype=bool)
                self.windows = all_windows_arr[mask]

        cprint(
            f"[KitchenLowdimDataset] split='{split}' n_windows={len(self.windows)}",
            "yellow",
        )

    # --------------- BaseDataset API ---------------
    def __len__(self) -> int:
        return int(len(self.windows))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ei, s = int(self.windows[idx][0]), int(self.windows[idx][1])
        ep = self.episodes[ei]
        T = ep["state"].shape[0]
        if self.preprocess:
            W = max(self.horizon, int(round(self.window_ratio * T)))
            start = s + W - self.horizon
            state_seq = ep["state"][start : start + self.horizon].astype(np.float32)
            action_seq = ep["action"][start : start + self.horizon].astype(np.float32)
        else:
            state_seq = ep["state"][s : s + self.horizon].astype(np.float32)
            action_seq = ep["action"][s : s + self.horizon].astype(np.float32)

        if self.preprocess and self.split == "train" and self.noise_std > 0:
            state_seq = state_seq + np.random.randn(*state_seq.shape).astype(np.float32) * self.noise_std

        sample = {
            "obs": {"state": state_seq},
            "action": action_seq,
        }
        return dict_apply(sample, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)

    # --------------- Compatibility helpers ---------------
    def get_validation_dataset(self) -> "KitchenLowdimDataset":
        return self.spawn_split("val")

    def get_test_dataset(self) -> "KitchenLowdimDataset":
        return self.spawn_split("test")

    def spawn_split(self, split: str) -> "KitchenLowdimDataset":
        return KitchenLowdimDataset(
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            dataset_id=self.dataset_id,
            cache_path=None,  # already loaded
            seed=self.seed,
            preprocess=self.preprocess,
            window_ratio=self.window_ratio,
            noise_std=self.noise_std,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            split=split,
            episodes=self.episodes,
        )

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        states = np.concatenate([ep["state"] for ep in self.episodes], axis=0)
        actions = np.concatenate([ep["action"] for ep in self.episodes], axis=0)
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"state": states, "action": actions}, last_n_dims=1, mode=mode, **kwargs
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        acts = np.concatenate([ep["action"] for ep in self.episodes], axis=0)
        return torch.from_numpy(acts.astype(np.float32))

    @property
    def shape_meta(self) -> Dict:
        return {
            "obs": {"state": {"shape": [self.state_dim], "type": "low_dim"}},
            "action": {"shape": [self.action_dim]},
        }
