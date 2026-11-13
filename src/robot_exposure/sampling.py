"""Explicit episode/window exposure controls.

The functions operate on the number of legal training windows in each
demonstration. They return normalized probabilities over demonstrations; the
caller can then sample a legal window uniformly within the selected demo.
"""
from __future__ import annotations

import numpy as np


def _counts(window_counts: object) -> np.ndarray:
    counts = np.asarray(window_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("window_counts must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("window_counts must contain finite positive values")
    return counts


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("exposure weights must have a positive finite sum")
    return values / total


def episode_probabilities(window_counts: object, gamma: float) -> np.ndarray:
    """Return probabilities proportional to ``N_i ** gamma``.

    ``gamma=1`` is window-uniform (each legal window has equal mass),
    ``gamma=0`` is episode-uniform, and ``gamma=-1`` is inverse-duration.
    """
    counts = _counts(window_counts)
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    return _normalize(np.power(counts, float(gamma)))


def window_uniform_probabilities(window_counts: object) -> np.ndarray:
    """Exposure induced by uniform sampling over all legal windows."""
    return episode_probabilities(window_counts, gamma=1.0)


def episode_uniform_probabilities(window_counts: object) -> np.ndarray:
    """Equal exposure for every demonstration."""
    return episode_probabilities(window_counts, gamma=0.0)


def inverse_duration_probabilities(window_counts: object) -> np.ndarray:
    """Inverse-duration exposure (``gamma=-1``)."""
    return episode_probabilities(window_counts, gamma=-1.0)
