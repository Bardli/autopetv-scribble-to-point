"""Prompt-channel encoders for DKFZ interactive inference variants."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy import ndimage


PROMPT_ENCODINGS = ("point_edt", "scribble_inverse_edt", "scribble_distance")
SCRIBBLE_DISTANCE_MODES = ("inverse_clipped", "normalized_distance")


def preprocessed_seed_map(
    points: list[dict[str, Any]],
    shape: tuple[int, ...],
    properties: dict[str, Any],
    label_name: str,
) -> np.ndarray:
    from nnunetv2.training.dataloading.utils import preprocess_point

    seeds = np.zeros(shape, dtype=bool)
    for click in points:
        if click.get("name") != label_name:
            continue
        coord = preprocess_point(click["point"], properties, shape)
        if len(coord) != len(shape):
            continue
        clipped = tuple(int(max(0, min(shape[i] - 1, round(float(coord[i]))))) for i in range(len(shape)))
        seeds[clipped] = True
    return seeds


def nearest_distance_map(seeds: np.ndarray) -> np.ndarray:
    if not np.any(seeds):
        return np.zeros(seeds.shape, dtype=np.float32)
    return ndimage.distance_transform_edt(~seeds).astype(np.float32)


def encode_distance(
    distance: np.ndarray,
    *,
    max_distance: float | None,
    mode: str,
) -> np.ndarray:
    if mode not in SCRIBBLE_DISTANCE_MODES:
        raise ValueError(f"Unsupported scribble distance mode: {mode}")
    if not np.any(distance):
        return distance.astype(np.float32)

    if max_distance is None or max_distance <= 0:
        max_distance = float(np.linalg.norm(distance.shape))
    max_distance = max(float(max_distance), 1e-6)

    clipped = np.minimum(distance, max_distance) / max_distance
    if mode == "normalized_distance":
        return clipped.astype(np.float32)
    return (1.0 - clipped).astype(np.float32)


def make_scribble_distance_encoder(
    *,
    max_distance: float | None = None,
    mode: str = "inverse_clipped",
):
    """Return a DKFZ-compatible click encoder using whole-scribble nearest EDT.

    The returned callable has the same signature as DKFZ's
    ``sparse_to_dense_point_nnInteractive``. It maps all foreground points to a
    foreground seed mask and all background points to a background seed mask,
    computes the Euclidean distance to the nearest seed in each class, and then
    emits two dense prompt channels.
    """

    def encoder(
        points: list[dict[str, Any]],
        shape: tuple[int, ...],
        properties: dict[str, Any],
        sigma: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_seed = preprocessed_seed_map(points, shape, properties, "tumor")
        neg_seed = preprocessed_seed_map(points, shape, properties, "background")

        pos = encode_distance(
            nearest_distance_map(pos_seed),
            max_distance=max_distance,
            mode=mode,
        )
        neg = encode_distance(
            nearest_distance_map(neg_seed),
            max_distance=max_distance,
            mode=mode,
        )
        return torch.from_numpy(pos), torch.from_numpy(neg)

    return encoder


def parse_optional_positive_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        return None
    return out
