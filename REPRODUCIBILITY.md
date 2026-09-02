# Final-container specification

This repository rebuilds the final FightTumor candidate for AutoPET V.

| Component | Final setting |
| --- | --- |
| Base method | DKFZ autoPET IV interactive inference |
| Adapter | Convert Grand Challenge scribbles to point prompts |
| Prompt representation | `point_edt` |
| Fold ensemble | DKFZ folds `0,5`, averaged as equal logits |
| Test-time augmentation | nnU-Net mirroring over the three spatial axes: identity plus seven flips (8 views) |
| Sliding-window step | `0.5` |
| Gaussian weighting | enabled |
| Tracer SUV post-filter | disabled |

The image uses the pinned PyTorch CUDA 12.6 runtime declared in `Dockerfile`.
The Docker `CMD` is the immutable final invocation:

```text
--prompt-encoding point_edt --max-fg-points 0 --max-bg-points 0 --folds 0,5
```

The original Docker export was built as
`autopetv_gc_dkfz_2fold_tta_v1:2fold_tta_20260830b`. This source repository
does not contain image exports, challenge data, or checkpoint weights.
