"""Reusable exposure controls for fixed-budget imitation-learning studies."""

from .sampling import (
    episode_probabilities,
    episode_uniform_probabilities,
    inverse_duration_probabilities,
    window_uniform_probabilities,
)

__all__ = [
    "episode_probabilities",
    "episode_uniform_probabilities",
    "inverse_duration_probabilities",
    "window_uniform_probabilities",
]
