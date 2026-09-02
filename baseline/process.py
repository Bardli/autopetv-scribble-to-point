"""Run DKFZ autoPET IV interactive inference with minimal autoPET V adapters."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from baseline.dkfz_runner import (  # noqa: E402
    DEFAULT_AP3_REPO,
    DEFAULT_DKFZ_REPO,
    DEFAULT_TRACER_CKPT,
    DkfzRunner,
    parse_folds,
    run_step,
)
from baseline.prompt_encoding import PROMPT_ENCODINGS, SCRIBBLE_DISTANCE_MODES  # noqa: E402
from baseline.scribble_adapter import MAX_BG_POINTS, MAX_FG_POINTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/input", type=Path)
    parser.add_argument("--output-root", default="/output", type=Path)
    parser.add_argument("--clicks-json", default=None, type=Path)
    parser.add_argument("--dkfz-repo", default=DEFAULT_DKFZ_REPO, type=Path)
    parser.add_argument("--autopet3-repo", default=DEFAULT_AP3_REPO, type=Path)
    parser.add_argument("--tracer-checkpoint", default=DEFAULT_TRACER_CKPT, type=Path)
    parser.add_argument("--max-fg-points", default=MAX_FG_POINTS, type=int)
    parser.add_argument("--max-bg-points", default=MAX_BG_POINTS, type=int)
    parser.add_argument("--folds", default="0", help="Comma-separated DKFZ checkpoint folds, for example: 0 or 0,5")
    parser.add_argument("--prompt-encoding", default="point_edt", choices=PROMPT_ENCODINGS)
    parser.add_argument("--scribble-distance-mode", default="inverse_clipped", choices=SCRIBBLE_DISTANCE_MODES)
    parser.add_argument("--scribble-distance-max", default=None, type=float)
    parser.add_argument("--enable-tracer-suv-filter", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    runner = DkfzRunner(
        args.dkfz_repo,
        parse_folds(args.folds),
        device=args.device,
        prompt_encoding=args.prompt_encoding,
        scribble_distance_mode=args.scribble_distance_mode,
        scribble_distance_max=args.scribble_distance_max,
    )
    result = run_step(
        runner,
        args.input_root,
        args.output_root,
        clicks_json=args.clicks_json,
        max_fg_points=args.max_fg_points,
        max_bg_points=args.max_bg_points,
        enable_suv_filter=args.enable_tracer_suv_filter,
        autopet3_repo=args.autopet3_repo,
        tracer_checkpoint=args.tracer_checkpoint,
        device=args.device,
    )
    print(f"Runner initialization: {runner.init_timings['runner_init_s']:.3f}s")
    print("Step timings:", {key: round(value, 3) for key, value in sorted(result.timings.items())})
    if result.cuda_memory:
        print("CUDA memory:", result.cuda_memory)


if __name__ == "__main__":
    main()
