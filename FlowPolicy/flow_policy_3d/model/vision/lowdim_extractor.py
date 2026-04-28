"""Lowdim (state-based) observation encoder for FlowPolicy.

Replaces the PointNet-based ``FlowPolicyEncoder`` so that FlowPolicy can be
applied to environments without point clouds (e.g. Franka Kitchen).
"""
from typing import Dict, List, Tuple, Type

import torch
import torch.nn as nn
from termcolor import cprint


def _create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
) -> List[nn.Module]:
    if len(net_arch) > 0:
        modules: List[nn.Module] = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
    last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
    modules.append(nn.Linear(last_layer_dim, output_dim))
    return modules


class LowdimEncoder(nn.Module):
    """State-only MLP encoder.

    Args:
        observation_space: dict whose key ``state_key`` maps to a shape tuple/list.
        out_channel: target embedding size.
        state_mlp_size: tuple of hidden widths; the last element is the projection size
            and the rest form the hidden architecture.
        state_key: which key in ``observation_space`` carries the state vector.
    """

    def __init__(
        self,
        observation_space: Dict,
        out_channel: int = 128,
        state_mlp_size: Tuple[int, ...] = (128, 128),
        state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
        state_key: str = "state",
    ) -> None:
        super().__init__()
        if state_key not in observation_space:
            raise KeyError(
                f"[LowdimEncoder] expected key '{state_key}' in observation_space, "
                f"got keys={list(observation_space.keys())}"
            )

        self.state_key = state_key
        self.state_shape = observation_space[state_key]
        if len(self.state_shape) != 1:
            raise ValueError(
                f"[LowdimEncoder] state must be 1-D, got shape={self.state_shape}"
            )

        if len(state_mlp_size) == 0:
            raise RuntimeError("state_mlp_size must contain at least one element")
        if len(state_mlp_size) == 1:
            net_arch: List[int] = []
        else:
            net_arch = list(state_mlp_size[:-1])
        proj_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(
            *_create_mlp(self.state_shape[0], proj_dim, net_arch, state_mlp_activation_fn)
        )
        self.projection = nn.Linear(proj_dim, out_channel)
        self.n_output_channels = out_channel

        cprint(
            f"[LowdimEncoder] state_dim={self.state_shape[0]} -> mlp{state_mlp_size} -> {out_channel}",
            "yellow",
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        state = observations[self.state_key]
        feat = self.state_mlp(state)
        feat = self.projection(feat)
        return feat

    def output_shape(self) -> int:
        return self.n_output_channels
