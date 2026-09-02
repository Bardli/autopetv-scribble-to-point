#!/usr/bin/env python
"""Grand Challenge container entry for the AutoPET V two-fold DKFZ ensemble.

This is a thin container shim around the real base-path inference in
``baseline/process.py``. It does two container-mechanics jobs and NO inference
logic of its own:

1. Stage ``/input`` into a writable working copy.

   Grand Challenge mounts ``/input`` read-only. The base path
   ``baseline.dkfz_runner.run_step`` converts the GC scribbles/clicks into DKFZ
   point prompts and writes the adapted ``lesion-clicks.json`` back into its
   ``--input-root`` (see ``prepare_adapted_input``). Writing into a read-only
   ``/input`` would fail, so we copy the GC input tree into a writable staging
   directory and point ``--input-root`` there. Outputs still go to ``/output``.

2. Resolve container paths that the base ``process.py`` defaults would otherwise
   point at host-relative locations:
     * ``--dkfz-repo`` -> the packaged vendor DKFZ repo (whose ``_model`` dir
       holds the selected checkpoint folds), instead of the repo-relative default.
     * ``--input-root`` / ``--output-root`` -> the staged input and ``/output``.

All model/prompt flags (``--prompt-encoding``, ``--max-fg-points``,
``--max-bg-points``, ``--folds``) come from the Docker CMD and are forwarded
verbatim to ``baseline/process.py``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SRC_INPUT = Path(os.environ.get("APV_INPUT_ROOT", "/input"))
STAGE_INPUT = Path(os.environ.get("APV_STAGE_INPUT", "/opt/algorithm/work/input"))
OUTPUT_ROOT = Path(os.environ.get("APV_OUTPUT_ROOT", "/output"))
DKFZ_REPO = Path(
    os.environ.get("APV_DKFZ_REPO", "/opt/algorithm/vendor/autoPET-interactive")
)


def stage_input() -> None:
    """Copy the (possibly read-only) GC input tree into a writable location."""
    if not SRC_INPUT.is_dir():
        raise RuntimeError(f"Missing GC input directory: {SRC_INPUT}")
    if STAGE_INPUT.exists():
        shutil.rmtree(STAGE_INPUT)
    STAGE_INPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC_INPUT, STAGE_INPUT)
    print(f"[entry] staged input {SRC_INPUT} -> {STAGE_INPUT}", flush=True)


def main() -> None:
    if not DKFZ_REPO.is_dir():
        raise RuntimeError(f"Missing packaged DKFZ repo: {DKFZ_REPO}")
    (OUTPUT_ROOT / "images" / "tumor-lesion-segmentation").mkdir(
        parents=True, exist_ok=True
    )
    stage_input()

    passthrough = sys.argv[1:]  # fixed model/prompt flags from the Docker CMD
    sys.argv = [
        "process.py",
        "--input-root",
        str(STAGE_INPUT),
        "--output-root",
        str(OUTPUT_ROOT),
        "--dkfz-repo",
        str(DKFZ_REPO),
        *passthrough,
    ]
    print(f"[entry] baseline.process argv: {sys.argv[1:]}", flush=True)

    from baseline.process import main as process_main

    process_main()


if __name__ == "__main__":
    main()
