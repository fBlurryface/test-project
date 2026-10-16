from __future__ import annotations

from typing import Protocol
import numpy as np

from .task_frames import CANONICAL_ACTION_NAMES


POLICY_ACTION_DIM = 6


class Policy(Protocol):
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Return one canonical local 6D EE delta."""
        ...


def validate_action(action6) -> np.ndarray:
    action = np.asarray(action6, dtype=float).reshape(POLICY_ACTION_DIM).copy()
    if not np.all(np.isfinite(action)):
        raise ValueError("Policy action contains NaN/Inf")
    return action


class ZeroPolicy:
    """Safe placeholder proving that the simulator-policy interface is wired."""

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        _ = observation
        return np.zeros(POLICY_ACTION_DIM, dtype=np.float32)


__all__ = [
    "CANONICAL_ACTION_NAMES",
    "POLICY_ACTION_DIM",
    "Policy",
    "ZeroPolicy",
    "validate_action",
]
