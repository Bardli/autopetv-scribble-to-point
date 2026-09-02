"""Sparse seed selection for scribble prompts (experiment E-C sparse-seed).

Motivation
----------
The DKFZ interactive base was trained on *sparse* isolated point prompts (<=10),
each rendered as a small radius-2 EDT ball. Our closed-loop harness instead seeds
the *whole* accumulated scribble stroke (every stroke voxel becomes a distance-0
seed) and renders a radius-``max_distance`` tube. This module subsamples an
accumulated per-class stroke into a few sparse seed voxels so that the existing
radius-2 encoder renders a handful of isolated cones -- closer to the base's
native prompt distribution.

Coordinate space
----------------
``points`` are ``[i, j, k]`` voxel coordinates in the label / PET array index
space (they come from ``np.argwhere`` on the label-derived scribble volume). The
optional ``pet`` array must share that exact grid, so ``pet[i, j, k]`` is the SUV
lookup for a seed.

Schemes
-------
- ``whole_stroke``      : identity, return every point (legacy path, default).
- ``skeleton_centroid`` : split each connected stroke into ~``n_seeds`` equal
                          arc-length segments; seed = the mid voxel of each
                          segment. Pure geometry, no PET (experiment E-C1).
- ``skeleton_suv_peak`` : same segmentation, but seed = the voxel with the
                          maximum PET SUV within the segment (optionally within a
                          Chebyshev neighbourhood of radius ``suv_radius``).
                          Needs ``pet`` (experiment E-C2).
"""

from __future__ import annotations

import math

import numpy as np

SEED_SCHEMES = ("whole_stroke", "skeleton_centroid", "skeleton_suv_peak")

Point = list[int]


def _connected_components(points: list[Point]) -> list[list[Point]]:
    """Group voxel coordinates into 26-connected components.

    Strokes are tiny (tens of voxels); a plain BFS over a coordinate set is more
    than fast enough and avoids allocating a full whole-body volume.
    """
    remaining = {tuple(int(v) for v in p) for p in points}
    components: list[list[Point]] = []
    offsets = [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if not (di == 0 and dj == 0 and dk == 0)
    ]
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            ci, cj, ck = stack.pop()
            for di, dj, dk in offsets:
                nb = (ci + di, cj + dj, ck + dk)
                if nb in remaining:
                    remaining.discard(nb)
                    stack.append(nb)
                    comp.append(nb)
        components.append([list(c) for c in comp])
    return components


def _order_component(comp: list[Point]) -> list[Point]:
    """Order a thin stroke component along its arc via a greedy walk.

    Start from one end of the component's diameter (the farthest-apart pair of
    voxels) and repeatedly step to the nearest unvisited voxel. For a ~1-voxel
    wide skeleton this recovers arc-length order.
    """
    if len(comp) <= 2:
        return comp
    arr = np.asarray(comp, dtype=np.float64)
    # Diameter endpoints: pairwise squared distances, take the argmax pair.
    diff = arr[:, None, :] - arr[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    start_idx = int(np.unravel_index(int(np.argmax(d2)), d2.shape)[0])

    n = len(comp)
    visited = np.zeros(n, dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    cur = start_idx
    for _ in range(n - 1):
        dist = d2[cur].copy()
        dist[visited] = np.inf
        nxt = int(np.argmin(dist))
        visited[nxt] = True
        order.append(nxt)
        cur = nxt
    return [comp[i] for i in order]


def _split_segments(ordered: list[Point], k: int) -> list[list[Point]]:
    """Split an ordered voxel list into ``k`` contiguous ~equal arc-length chunks."""
    k = max(1, min(k, len(ordered)))
    return [list(chunk) for chunk in np.array_split(np.asarray(ordered, dtype=object), k)]


def _allocate_seeds(sizes: list[int], n_seeds: int) -> list[int]:
    """Allocate ``~n_seeds`` seeds across components: min 1 each, else by size."""
    total = float(sum(sizes)) or 1.0
    alloc = [max(1, int(round(n_seeds * s / total))) for s in sizes]
    return alloc


def _neighbourhood_suv_argmax(
    seg: list[Point], pet: np.ndarray, radius: int
) -> Point:
    """Return the voxel of maximum PET SUV in/around a segment.

    ``radius == 0`` searches only the segment's own voxels (kept on-stroke, the
    clean C1-vs-C2 contrast). ``radius > 0`` also scans a Chebyshev cube around
    each segment voxel, clamped to the image bounds.
    """
    shape = pet.shape
    if radius <= 0:
        candidates = seg
    else:
        seen: set[tuple[int, int, int]] = set()
        candidates = []
        for i, j, k in seg:
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    for dk in range(-radius, radius + 1):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                            key = (ni, nj, nk)
                            if key not in seen:
                                seen.add(key)
                                candidates.append([ni, nj, nk])
    best = candidates[0]
    best_val = -math.inf
    for i, j, k in candidates:
        val = float(pet[i, j, k])
        if val > best_val:
            best_val = val
            best = [int(i), int(j), int(k)]
    return best


def select_seeds(
    points: list[Point],
    *,
    scheme: str,
    n_seeds: int = 5,
    pet: np.ndarray | None = None,
    suv_radius: int = 0,
) -> list[Point]:
    """Subsample an accumulated per-class stroke into sparse seed voxels."""
    if scheme not in SEED_SCHEMES:
        raise ValueError(f"Unsupported seed scheme: {scheme}")
    if scheme == "whole_stroke":
        return [list(p) for p in points]
    if not points:
        return []
    if scheme == "skeleton_suv_peak" and pet is None:
        raise ValueError("skeleton_suv_peak requires a PET array")

    components = _connected_components(points)
    alloc = _allocate_seeds([len(c) for c in components], n_seeds)

    seeds: list[Point] = []
    for comp, k in zip(components, alloc):
        ordered = _order_component(comp)
        for seg in _split_segments(ordered, k):
            if not seg:
                continue
            if scheme == "skeleton_centroid":
                seeds.append([int(v) for v in seg[len(seg) // 2]])
            else:  # skeleton_suv_peak
                seeds.append(_neighbourhood_suv_argmax(seg, pet, suv_radius))
    # De-duplicate while preserving order (segments can share a mid/peak voxel).
    unique: list[Point] = []
    seen: set[tuple[int, int, int]] = set()
    for s in seeds:
        key = (s[0], s[1], s[2])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique
