#!/usr/bin/env python3
"""Convert AutoPET V lesion scribbles into DKFZ interactive point prompts.

This adapter does not load, train, alter, or redistribute a segmentation model.
It only normalizes a JSON scribble record to the point-prompt JSON consumed by
the DKFZ interactive inference interface.

Accepted input schemas
----------------------
1. {"points": [{"point": [x, y, z], "name": "tumor" | "background"}, ...]}
2. {"tumor": [[x, y, z], ...], "background": [[x, y, z], ...]}

Coordinates are rounded to integer voxel coordinates. A cap of 0 means that
all points of that polarity are retained; caps greater than zero select evenly
spaced points from the input order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

Point = list[int]
PointGroups = dict[str, list[Point]]

_FOREGROUND_NAMES = {"tumor", "foreground", "fg", "positive", "pos"}
_BACKGROUND_NAMES = {"background", "bg", "negative", "neg"}


def normalize_point(value: Any) -> Point:
    """Validate and round one XYZ coordinate triple."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected one 3D point, got {value!r}")
    try:
        return [int(round(float(coordinate))) for coordinate in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Point coordinates must be numeric, got {value!r}") from error


def split_points(data: dict[str, Any]) -> PointGroups:
    """Return normalized foreground and background point lists from either schema."""
    if "points" not in data:
        return {
            "tumor": [normalize_point(point) for point in data.get("tumor", [])],
            "background": [normalize_point(point) for point in data.get("background", [])],
        }

    raw_points = data["points"]
    if not isinstance(raw_points, list):
        raise ValueError("'points' must be a list")
    groups: PointGroups = {"tumor": [], "background": []}
    for item in raw_points:
        if not isinstance(item, dict):
            raise ValueError(f"Each point entry must be an object, got {item!r}")
        name = str(item.get("name", "")).lower()
        point = normalize_point(item.get("point"))
        if name in _FOREGROUND_NAMES:
            groups["tumor"].append(point)
        elif name in _BACKGROUND_NAMES:
            groups["background"].append(point)
    return groups


def uniform_subsample(points: list[Point], maximum: int) -> list[Point]:
    """Keep all points for a non-positive cap, otherwise sample uniformly."""
    if maximum <= 0 or len(points) <= maximum:
        return list(points)
    if maximum == 1:
        return [points[len(points) // 2]]
    last = len(points) - 1
    indices = [round(index * last / (maximum - 1)) for index in range(maximum)]
    return [points[index] for index in indices]


def to_dkfz_points(
    data: dict[str, Any], *, max_fg_points: int = 0, max_bg_points: int = 0
) -> dict[str, Any]:
    """Convert an AutoPET V scribble JSON object to DKFZ Multiple-points JSON."""
    groups = split_points(data)
    foreground = uniform_subsample(groups["tumor"], max_fg_points)
    background = uniform_subsample(groups["background"], max_bg_points)
    return {
        "version": {"major": 1, "minor": 0},
        "type": "Multiple points",
        "points": [
            *({"point": point, "name": "tumor"} for point in foreground),
            *({"point": point, "name": "background"} for point in background),
        ],
    }


def _self_test() -> None:
    """Run small dependency-free contract checks for the public adapter."""
    converted = to_dkfz_points(
        {
            "points": [
                {"point": [1.2, 2.5, 3.7], "name": "positive"},
                {"point": [4, 5, 6], "name": "negative"},
            ]
        }
    )
    assert converted["points"] == [
        {"point": [1, 2, 4], "name": "tumor"},
        {"point": [4, 5, 6], "name": "background"},
    ]
    assert uniform_subsample([[0, 0, 0], [1, 1, 1], [2, 2, 2]], 0) == [
        [0, 0, 0], [1, 1, 1], [2, 2, 2]
    ]
    assert uniform_subsample([[0, 0, 0], [1, 1, 1], [2, 2, 2]], 2) == [
        [0, 0, 0], [2, 2, 2]
    ]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Input scribble JSON")
    parser.add_argument("--output", type=Path, help="Output DKFZ point JSON")
    parser.add_argument("--max-fg-points", type=int, default=0)
    parser.add_argument("--max-bg-points", type=int, default=0)
    parser.add_argument("--self-test", action="store_true", help="Run dependency-free checks")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        print("self-test: PASS")
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")

    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON value must be an object")
        converted = to_dkfz_points(
            data, max_fg_points=args.max_fg_points, max_bg_points=args.max_bg_points
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
