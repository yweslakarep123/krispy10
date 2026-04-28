"""Franka Kitchen runner using gymnasium-robotics + minari-format observations.

Mirrors the public API of ``flow_policy_3d.env_runner.adroit_runner.AdroitRunner``
but for the lowdim Franka Kitchen environment. It also exposes hooks used by
``scripts/run_kitchen_experiment.py`` to record success-rate, mean inference
latency and an MP4 rollout video.
"""
from __future__ import annotations

import collections
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import tqdm
from termcolor import cprint

from flow_policy_3d.env_runner.base_runner import BaseRunner
from flow_policy_3d.policy.base_policy import BasePolicy

DEFAULT_TASKS = ["microwave", "kettle", "bottom burner", "light switch"]


def _flatten_kitchen_obs(obs) -> np.ndarray:
    """Flatten gymnasium-robotics kitchen observation into a 1-D float32 vector."""
    if isinstance(obs, np.ndarray):
        return np.asarray(obs, dtype=np.float32).reshape(-1)
    if isinstance(obs, dict):
        parts: List[np.ndarray] = []
        if "observation" in obs:
            parts.append(np.asarray(obs["observation"], dtype=np.float32).reshape(-1))
        if "desired_goal" in obs:
            dg = obs["desired_goal"]
            if isinstance(dg, dict):
                for v in dg.values():
                    parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
            else:
                parts.append(np.asarray(dg, dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=0).astype(np.float32)
    raise TypeError(f"Unsupported observation type: {type(obs)}")


def _build_kitchen_env(tasks: List[str], render_mode: Optional[str], max_steps: int):
    import gymnasium as gym
    try:
        import gymnasium_robotics  # noqa: F401
        gym.register_envs(gymnasium_robotics)
    except Exception:
        pass
    env = gym.make(
        "FrankaKitchen-v1",
        tasks_to_complete=list(tasks),
        render_mode=render_mode,
        max_episode_steps=max_steps,
    )
    return env


class KitchenRunner(BaseRunner):
    def __init__(
        self,
        output_dir: str,
        eval_episodes: int = 10,
        max_steps: int = 280,
        n_obs_steps: int = 4,
        n_action_steps: int = 4,
        fps: int = 12,
        tqdm_interval_sec: float = 5.0,
        tasks: Optional[List[str]] = None,
        record_video: bool = True,
        video_filename: str = "rollout.mp4",
        target_state_dim: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.tasks = list(tasks) if tasks is not None else list(DEFAULT_TASKS)
        self.record_video = record_video
        self.video_filename = video_filename
        self.target_state_dim = target_state_dim

    # ------------------------------------------------------------------ helpers
    def _coerce_state_dim(self, state: np.ndarray) -> np.ndarray:
        """Pad or truncate the flattened observation to ``target_state_dim``."""
        if self.target_state_dim is None:
            return state
        cur = state.shape[-1]
        if cur == self.target_state_dim:
            return state
        if cur > self.target_state_dim:
            return state[: self.target_state_dim]
        pad = np.zeros((self.target_state_dim - cur,), dtype=np.float32)
        return np.concatenate([state, pad], axis=0)

    def _stack_history(self, history: collections.deque) -> np.ndarray:
        # Ensure exactly n_obs_steps entries by padding with the oldest frame.
        while len(history) < self.n_obs_steps:
            history.appendleft(history[0])
        arr = np.stack(list(history)[-self.n_obs_steps :], axis=0)
        return arr.astype(np.float32)

    # ------------------------------------------------------------------ main API
    def run(
        self,
        policy: BasePolicy,
        save_video: Optional[bool] = None,
        video_path: Optional[str] = None,
    ) -> Dict:
        device = policy.device
        record_video = self.record_video if save_video is None else bool(save_video)
        render_mode = "rgb_array" if record_video else None

        env = _build_kitchen_env(self.tasks, render_mode=render_mode, max_steps=self.max_steps)

        all_success_fraction: List[float] = []
        all_full_success: List[float] = []
        all_step_latency: List[float] = []
        all_video_frames: List[np.ndarray] = []

        for ep_idx in tqdm.tqdm(
            range(self.eval_episodes),
            desc="Eval Kitchen",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            obs, info = env.reset(seed=int(ep_idx) + 12345)
            if hasattr(policy, "reset"):
                policy.reset()

            obs_flat = _flatten_kitchen_obs(obs)
            obs_flat = self._coerce_state_dim(obs_flat)
            history: collections.deque = collections.deque(maxlen=max(self.n_obs_steps, 1))
            history.append(obs_flat)

            terminated = truncated = False
            step_count = 0
            total_time = 0.0
            tasks_completed = set()

            while not (terminated or truncated) and step_count < self.max_steps:
                state_hist = self._stack_history(history)  # (To, Ds)
                obs_dict_input = {
                    "state": torch.from_numpy(state_hist[None, ...]).to(device=device, dtype=torch.float32)
                }

                t0 = time.time()
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)
                t1 = time.time()
                total_time += t1 - t0

                action_seq = action_dict["action"].detach().to("cpu").numpy()[0]  # (Ta, Da)

                for ai in range(action_seq.shape[0]):
                    action = action_seq[ai].astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)

                    if record_video:
                        try:
                            frame = env.render()
                            if frame is not None:
                                all_video_frames.append(np.asarray(frame))
                        except Exception:
                            pass

                    if isinstance(info, dict):
                        completed = info.get("completed_tasks") or info.get("tasks_to_complete_at_t", [])
                        if isinstance(completed, (list, tuple, set)):
                            for tname in completed:
                                tasks_completed.add(tname)

                    obs_flat = _flatten_kitchen_obs(obs)
                    obs_flat = self._coerce_state_dim(obs_flat)
                    history.append(obs_flat)

                    step_count += 1
                    if terminated or truncated or step_count >= self.max_steps:
                        break

            # Some env versions don't expose completed task list; fall back to reward sum.
            if len(tasks_completed) == 0 and isinstance(info, dict) and "remaining_tasks" in info:
                remaining = info["remaining_tasks"]
                tasks_completed = set(self.tasks) - set(remaining)
            success_fraction = float(len(tasks_completed)) / max(1, len(self.tasks))
            full_success = float(len(tasks_completed) >= len(self.tasks))

            all_success_fraction.append(success_fraction)
            all_full_success.append(full_success)
            if step_count > 0:
                all_step_latency.append(total_time / step_count)

        env.close()

        log: Dict = {}
        log["mean_success_rate"] = float(np.mean(all_success_fraction)) if all_success_fraction else 0.0
        log["mean_full_success"] = float(np.mean(all_full_success)) if all_full_success else 0.0
        log["mean_latency_s"] = float(np.mean(all_step_latency)) if all_step_latency else 0.0
        log["test_mean_score"] = log["mean_success_rate"]
        log["n_episodes"] = self.eval_episodes
        cprint(
            f"[KitchenRunner] success_rate={log['mean_success_rate']:.4f} "
            f"full_success={log['mean_full_success']:.4f} "
            f"latency={log['mean_latency_s'] * 1000:.2f} ms",
            "green",
        )

        if record_video and all_video_frames:
            vp = video_path or os.path.join(self.output_dir, self.video_filename)
            os.makedirs(os.path.dirname(vp) or ".", exist_ok=True)
            self._write_video(all_video_frames, vp, fps=self.fps)
            log["video_path"] = vp
            cprint(f"[KitchenRunner] saved video -> {vp}", "cyan")

        return log

    @staticmethod
    def _write_video(frames: List[np.ndarray], path: str, fps: int) -> None:
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
                for frame in frames:
                    writer.append_data(np.asarray(frame))
        except Exception as exc:
            cprint(f"[KitchenRunner] imageio writer failed: {exc}", "red")
            try:
                import imageio
                imageio.mimsave(path, frames, fps=fps)
            except Exception as exc2:
                cprint(f"[KitchenRunner] fallback writer failed: {exc2}", "red")
