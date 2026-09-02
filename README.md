# autoPET V scribble-to-point container

Source code for the FightTumor AutoPET V candidate. The container reuses DKFZ's
autoPET IV interactive model and translates AutoPET V scribbles into the point
prompt format expected by that model. The final configuration uses DKFZ folds
0 and 5 and the model's built-in 8-view mirroring test-time augmentation.

## Contents and provenance

- `baseline/` contains the Challenge input adapter and inference wrapper.
- `process_entry.py` stages read-only Grand Challenge input before inference.
- `third_party/autoPET-interactive/` is an unmodified, pinned DKFZ source
  checkout; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- `Dockerfile` is the complete build recipe for the final two-fold image.

No patient data, model checkpoints, Docker export, or evaluation output is
committed to this repository.

## Model prerequisite

Download the official final checkpoint referenced in the vendored
[DKFZ README](third_party/autoPET-interactive/README.md). Its redistribution
terms should be checked with the upstream authors. Put the required files in
the following local-only location:

```text
third_party/autoPET-interactive/_model/
├── dataset.json
├── dataset_fingerprint.json
├── plans.json
├── fold_0/checkpoint_final.pth
└── fold_5/checkpoint_final.pth
```

The original DKFZ release may contain further folds. They are not needed by
this candidate. The model directory is ignored by Git; do not commit it.

## Build

Use Docker on an NVIDIA Linux machine. From the repository root:

```bash
bash scripts/build.sh autopetv-scribble-to-point:2fold-8tta
```

The script checks the local checkpoint layout before invoking Docker. The
build itself does not require a GPU; running the resulting image does.

The final image entrypoint is fixed to:

```text
--prompt-encoding point_edt --max-fg-points 0 --max-bg-points 0 --folds 0,5
```

To export an image for Grand Challenge upload:

```bash
bash scripts/export.sh autopetv-scribble-to-point:2fold-8tta candidate.tar.gz
```

## Running locally

Grand Challenge supplies `/input` and expects the segmentation under
`/output/images/tumor-lesion-segmentation`. For a local smoke run, mount
equivalent directories and provide the NVIDIA runtime:

```bash
docker run --rm --gpus all \
  -v "$PWD/example-input:/input:ro" \
  -v "$PWD/example-output:/output" \
  autopetv-scribble-to-point:2fold-8tta
```

Do not use Challenge test data in a public repository. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact candidate specification.

## License

The adapter code is released under Apache License 2.0. The vendored DKFZ source
retains its original Apache License 2.0 and attribution. Checkpoint weights are
not part of this repository.
