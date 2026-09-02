"""Persistent DKFZ autoPET interactive inference runner."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import torch

from baseline.prompt_encoding import (
    PROMPT_ENCODINGS,
    SCRIBBLE_DISTANCE_MODES,
    make_scribble_distance_encoder,
    parse_optional_positive_float,
)
from baseline.scribble_adapter import MAX_BG_POINTS, MAX_FG_POINTS, load_scribbles, to_gc_points, write_gc_points
from baseline.tracer_classifier import classify_pet_mha

DEFAULT_DKFZ_REPO = Path("external/autopet5_baseline/autoPET-interactive")
DEFAULT_AP3_REPO = Path("external/autopet5_baseline/autoPETIII")
DEFAULT_TRACER_CKPT = Path("external/autopet5_baseline/weights/tracer_classifier.pt")

SUV_THRESHOLDS = {
    "fdg": 1.5,
    "psma": 1.0,
}


@dataclass
class StepResult:
    output_mha: Path
    timings: dict[str, float] = field(default_factory=dict)
    cuda_memory: dict[str, int] = field(default_factory=dict)
    adapted_clicks: dict[str, Any] | None = None
    tracer: str | None = None
    prompt_encoding: str = "point_edt"


@dataclass(frozen=True)
class InputCacheKey:
    ct_path: Path
    ct_mtime_ns: int
    ct_size: int
    pet_path: Path
    pet_mtime_ns: int
    pet_size: int


@dataclass
class CachedInput:
    key: InputCacheKey
    input_array: np.ndarray
    spacing: tuple[float, ...]
    direction: tuple[float, ...]
    origin: tuple[float, ...]
    uuid: str
    props_spacing: tuple[float, ...]


def parse_folds(value: str) -> tuple[int, ...]:
    folds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not folds:
        raise ValueError("--folds must contain at least one fold id")
    return folds


def first_mha(path: Path) -> Path:
    files = sorted(path.glob("*.mha"))
    if not files:
        raise FileNotFoundError(f"No .mha files found under {path}")
    return files[0]


def input_cache_key(input_root: Path) -> InputCacheKey:
    ct_path = first_mha(input_root / "images" / "ct").resolve()
    pet_path = first_mha(input_root / "images" / "pet").resolve()
    ct_stat = ct_path.stat()
    pet_stat = pet_path.stat()
    return InputCacheKey(
        ct_path=ct_path,
        ct_mtime_ns=ct_stat.st_mtime_ns,
        ct_size=ct_stat.st_size,
        pet_path=pet_path,
        pet_mtime_ns=pet_stat.st_mtime_ns,
        pet_size=pet_stat.st_size,
    )


def prepare_adapted_input(
    input_root: Path,
    raw_clicks_path: Path,
    max_fg_points: int,
    max_bg_points: int,
) -> tuple[dict[str, Any], float]:
    start = time.time()
    if raw_clicks_path.exists():
        adapted = to_gc_points(
            load_scribbles(raw_clicks_path),
            max_fg_points=max_fg_points,
            max_bg_points=max_bg_points,
        )
    else:
        adapted = {"version": {"major": 1, "minor": 0}, "type": "Multiple points", "points": []}
    write_gc_points(adapted, input_root / "lesion-clicks.json")
    return adapted, time.time() - start


def apply_tracer_suv_filter(
    output_root: Path,
    input_root: Path,
    tracer: str,
    fg_points: list[list[int]],
) -> None:
    threshold = SUV_THRESHOLDS[tracer]
    seg_path = first_mha(output_root / "images" / "tumor-lesion-segmentation")
    pet_path = first_mha(input_root / "images" / "pet")

    seg_img = sitk.ReadImage(str(seg_path))
    pet_img = sitk.ReadImage(str(pet_path))
    seg = sitk.GetArrayFromImage(seg_img) > 0
    pet = sitk.GetArrayFromImage(pet_img)

    labeled, num_components = ndimage.label(seg)
    keep = np.zeros(num_components + 1, dtype=bool)
    keep[0] = False

    protected = set()
    for x, y, z in fg_points:
        if 0 <= z < labeled.shape[0] and 0 <= y < labeled.shape[1] and 0 <= x < labeled.shape[2]:
            component_id = int(labeled[z, y, x])
            if component_id > 0:
                protected.add(component_id)

    for component_id in range(1, num_components + 1):
        component = labeled == component_id
        if component_id in protected:
            keep[component_id] = True
        else:
            keep[component_id] = bool(np.nanmax(pet[component]) >= threshold)

    filtered = keep[labeled].astype(np.uint8)
    out_img = sitk.GetImageFromArray(filtered)
    out_img.CopyInformation(seg_img)
    sitk.WriteImage(out_img, str(seg_path), useCompression=True)
    print(
        f"tracer SUV filter: tracer={tracer}, threshold={threshold}, "
        f"kept={int(keep.sum())}/{num_components} components, output={seg_path}"
    )


class DkfzRunner:
    """Hold DKFZ model state and run per-case/step prediction calls."""

    def __init__(
        self,
        dkfz_repo: Path,
        folds: tuple[int, ...],
        *,
        device: str | None = None,
        verbose: bool = True,
        allow_tqdm: bool = True,
        prompt_encoding: str = "point_edt",
        scribble_distance_mode: str = "inverse_clipped",
        scribble_distance_max: float | None = None,
    ) -> None:
        self.dkfz_repo = dkfz_repo
        self.folds = folds
        self.verbose = verbose
        self.allow_tqdm = allow_tqdm
        self.prompt_encoding = prompt_encoding
        self.scribble_distance_mode = (
            "normalized_distance" if prompt_encoding == "scribble_distance" else scribble_distance_mode
        )
        self.scribble_distance_max = parse_optional_positive_float(scribble_distance_max)
        self.init_timings: dict[str, float] = {}
        self._input_cache: CachedInput | None = None

        init_start = time.time()
        module_start = time.time()
        self.module = self._load_dkfz_module(dkfz_repo)
        self.init_timings["module_load_s"] = time.time() - module_start

        if verbose:
            self.module._show_torch_cuda_info()
        self._configure_prompt_encoding()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        predictor_start = time.time()
        self.predictor = self.module.autoPETPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=verbose,
            verbose_preprocessing=False,
            allow_tqdm=allow_tqdm,
        )
        print(f"DKFZ folds: {folds}")
        self.predictor.initialize_from_trained_model_folder(
            dkfz_repo / "_model",
            use_folds=folds,
            checkpoint_name="checkpoint_final.pth",
        )
        self.init_timings["predictor_init_s"] = time.time() - predictor_start
        self.init_timings["runner_init_s"] = time.time() - init_start

    def _configure_prompt_encoding(self) -> None:
        if self.prompt_encoding not in PROMPT_ENCODINGS:
            raise ValueError(f"Unsupported prompt encoding: {self.prompt_encoding}")
        if self.scribble_distance_mode not in SCRIBBLE_DISTANCE_MODES:
            raise ValueError(f"Unsupported scribble distance mode: {self.scribble_distance_mode}")

        predictor_module = sys.modules.get("nnunetv2.inference.autopet_predictor")
        if predictor_module is None:
            import nnunetv2.inference.autopet_predictor as predictor_module  # type: ignore[no-redef]

        if not hasattr(predictor_module, "_apv_original_sparse_to_dense_point_nnInteractive"):
            predictor_module._apv_original_sparse_to_dense_point_nnInteractive = (  # type: ignore[attr-defined]
                predictor_module.sparse_to_dense_point_nnInteractive
            )

        if self.prompt_encoding == "point_edt":
            predictor_module.sparse_to_dense_point_nnInteractive = (  # type: ignore[attr-defined]
                predictor_module._apv_original_sparse_to_dense_point_nnInteractive  # type: ignore[attr-defined]
            )
            return

        predictor_module.sparse_to_dense_point_nnInteractive = make_scribble_distance_encoder(  # type: ignore[attr-defined]
            max_distance=self.scribble_distance_max,
            mode=self.scribble_distance_mode,
        )

    @staticmethod
    def _load_dkfz_module(dkfz_repo: Path):
        resolved = str(dkfz_repo.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        spec = importlib.util.spec_from_file_location(
            "dkfz_autopet_interactive_inference",
            dkfz_repo / "inference.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {dkfz_repo / 'inference.py'}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_or_get_cached_input(self, input_root: Path) -> tuple[CachedInput, bool, float]:
        key = input_cache_key(input_root)
        if self._input_cache is not None and self._input_cache.key == key:
            return self._input_cache, True, 0.0

        read_start = time.time()
        input_array_ct, spacing, direction, origin, uuid = self.module.load_image_file_as_array(
            location=input_root / "images" / "ct",
        )
        input_array_pet, _, _, _, _ = self.module.load_image_file_as_array(
            location=input_root / "images" / "pet",
        )
        spacing_tuple = tuple(float(x) for x in spacing)
        cached = CachedInput(
            key=key,
            input_array=np.stack([input_array_ct, input_array_pet]).astype(np.half),
            spacing=spacing_tuple,
            direction=tuple(float(x) for x in direction),
            origin=tuple(float(x) for x in origin),
            uuid=uuid,
            props_spacing=tuple(reversed(spacing_tuple)),
        )
        self._input_cache = cached
        return cached, False, time.time() - read_start

    def _cuda_memory_snapshot(self, prefix: str) -> dict[str, int]:
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return {}
        return {
            f"{prefix}_memory_allocated": int(torch.cuda.memory_allocated(self.device)),
            f"{prefix}_memory_reserved": int(torch.cuda.memory_reserved(self.device)),
            f"{prefix}_max_memory_allocated": int(torch.cuda.max_memory_allocated(self.device)),
            f"{prefix}_max_memory_reserved": int(torch.cuda.max_memory_reserved(self.device)),
        }

    def predict(self, input_root: Path, output_root: Path) -> StepResult:
        total_start = time.time()
        timings: dict[str, float] = {}
        cuda_memory: dict[str, int] = {}

        cached_input, cache_hit, input_read_s = self._load_or_get_cached_input(input_root)
        clicks = self.module.load_json(input_root / "lesion-clicks.json")
        timings["input_read_s"] = input_read_s
        timings["input_cache_hit"] = 1.0 if cache_hit else 0.0

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        cuda_memory.update(self._cuda_memory_snapshot("before_predict"))

        predict_start = time.time()
        with torch.inference_mode():
            ret = self.predictor.predict_single_npy_array(
                cached_input.input_array,
                {"spacing": list(cached_input.props_spacing)},
                clicks,
                self.module.POINT_WIDTH,
                None,
                None,
                False,
            )
        timings["model_predict_total_s"] = time.time() - predict_start
        print("Time taken for prediction: ", timings["model_predict_total_s"])

        cuda_memory.update(self._cuda_memory_snapshot("after_predict"))

        write_start = time.time()
        output_dir = output_root / "images" / "tumor-lesion-segmentation"
        self.module.write_array_as_image_file(
            location=output_dir,
            array=ret,
            spacing=cached_input.spacing,
            direction=cached_input.direction,
            origin=cached_input.origin,
            uuid=cached_input.uuid,
        )
        output_mha = output_dir / f"{cached_input.uuid}.mha"
        timings["output_write_s"] = time.time() - write_start
        timings["step_total_s"] = time.time() - total_start
        print("Saved.")
        return StepResult(output_mha=output_mha, timings=timings, cuda_memory=cuda_memory)


def run_step(
    runner: DkfzRunner,
    input_root: Path,
    output_root: Path,
    *,
    clicks_json: Path | None = None,
    max_fg_points: int = MAX_FG_POINTS,
    max_bg_points: int = MAX_BG_POINTS,
    enable_suv_filter: bool = False,
    autopet3_repo: Path = DEFAULT_AP3_REPO,
    tracer_checkpoint: Path = DEFAULT_TRACER_CKPT,
    device: str | None = None,
) -> StepResult:
    raw_clicks = clicks_json or input_root / "lesion-clicks.json"
    adapted, scribble_adapt_s = prepare_adapted_input(input_root, raw_clicks, max_fg_points, max_bg_points)

    tracer = None
    tracer_classify_s = 0.0
    if enable_suv_filter:
        tracer_start = time.time()
        pet_mha = first_mha(input_root / "images" / "pet")
        tracer = classify_pet_mha(pet_mha, tracer_checkpoint, autopet3_repo, device)
        tracer_classify_s = time.time() - tracer_start
        print(f"tracer classifier: {tracer}")

    result = runner.predict(input_root, output_root)
    result.timings["scribble_adapt_s"] = scribble_adapt_s
    result.timings["tracer_classify_s"] = tracer_classify_s
    result.adapted_clicks = adapted
    result.tracer = tracer
    result.prompt_encoding = runner.prompt_encoding

    if enable_suv_filter:
        filter_start = time.time()
        fg_points = [p["point"] for p in adapted["points"] if p["name"] == "tumor"]
        apply_tracer_suv_filter(output_root, input_root, tracer, fg_points)
        result.timings["suv_filter_s"] = time.time() - filter_start
    else:
        result.timings["suv_filter_s"] = 0.0

    return result
