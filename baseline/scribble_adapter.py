"""Convert accumulated autoPET V scribbles into DKFZ GC point prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MAX_FG_POINTS = 0
MAX_BG_POINTS = 0


def load_scribbles(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def split_points(data: dict[str, Any]) -> dict[str, list[list[int]]]:
    if "points" in data:
        out = {"tumor": [], "background": []}
        for item in data.get("points", []):
            name = str(item.get("name", "")).lower()
            point = normalize_point(item.get("point"))
            if name in {"tumor", "foreground", "fg", "positive", "pos"}:
                out["tumor"].append(point)
            elif name in {"background", "bg", "negative", "neg"}:
                out["background"].append(point)
        return out

    return {
        "tumor": [normalize_point(p) for p in data.get("tumor", [])],
        "background": [normalize_point(p) for p in data.get("background", [])],
    }


def normalize_point(point: Any) -> list[int]:
    if point is None or len(point) != 3:
        raise ValueError(f"Expected a 3D point, got {point!r}")
    return [int(round(float(v))) for v in point]


def uniform_subsample(points: list[list[int]], max_points: int) -> list[list[int]]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    if max_points == 1:
        return [points[len(points) // 2]]

    last = len(points) - 1
    indices = [round(i * last / (max_points - 1)) for i in range(max_points)]
    return [points[i] for i in indices]


def to_gc_points(
    data: dict[str, Any],
    max_fg_points: int = MAX_FG_POINTS,
    max_bg_points: int = MAX_BG_POINTS,
) -> dict[str, Any]:
    split = split_points(data)
    fg = uniform_subsample(split["tumor"], max_fg_points)
    bg = uniform_subsample(split["background"], max_bg_points)
    return {
        "version": {"major": 1, "minor": 0},
        "type": "Multiple points",
        "points": [
            *[{"point": point, "name": "tumor"} for point in fg],
            *[{"point": point, "name": "background"} for point in bg],
        ],
    }


def write_gc_points(data: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-fg-points", type=int, default=MAX_FG_POINTS)
    parser.add_argument("--max-bg-points", type=int, default=MAX_BG_POINTS)
    args = parser.parse_args()

    adapted = to_gc_points(
        load_scribbles(args.input),
        max_fg_points=args.max_fg_points,
        max_bg_points=args.max_bg_points,
    )
    write_gc_points(adapted, args.output)
    print(
        f"wrote {args.output}: "
        f"{sum(p['name'] == 'tumor' for p in adapted['points'])} tumor, "
        f"{sum(p['name'] == 'background' for p in adapted['points'])} background"
    )


if __name__ == "__main__":
    main()
