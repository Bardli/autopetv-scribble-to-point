#!/usr/bin/env python3
"""Fail early when the two non-redistributed DKFZ checkpoint folds are absent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_RELATIVE_PATHS = (
    "dataset.json",
    "dataset_fingerprint.json",
    "plans.json",
    "fold_0/checkpoint_final.pth",
    "fold_5/checkpoint_final.pth",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("third_party/autoPET-interactive/_model"),
        help="Directory containing the official DKFZ _model files.",
    )
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    missing = [path for path in REQUIRED_RELATIVE_PATHS if not (model_dir / path).is_file()]
    if missing:
        print(f"Model directory is incomplete: {model_dir}", file=sys.stderr)
        for path in missing:
            print(f"  missing: {path}", file=sys.stderr)
        print(
            "Obtain the official DKFZ checkpoint, then retain only fold_0 and "
            "fold_5 plus dataset.json, dataset_fingerprint.json, and plans.json.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"Model layout verified: {model_dir}")


if __name__ == "__main__":
    main()
