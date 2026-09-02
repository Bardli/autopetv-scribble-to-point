"""Run the autoPET III FDG/PSMA MIP classifier on an MHA PET image."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path

import SimpleITK as sitk
import torch


def mha_to_nifti(pet_mha_path: Path, out_path: Path) -> Path:
    image = sitk.ReadImage(str(pet_mha_path))
    sitk.WriteImage(image, str(out_path), useCompression=True)
    return out_path


def load_autopet3_module(autopet3_repo: Path):
    module_path = autopet3_repo / "classify_pet.py"
    spec = importlib.util.spec_from_file_location("autopet3_classify_pet", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Upstream MIP_transform.__call__ checks a module-level `transform` name.
    module.transform = None
    return module


def classify_pet_mha(
    pet_mha_path: Path,
    checkpoint_path: Path,
    autopet3_repo: Path,
    device: str | None = None,
) -> str:
    module = load_autopet3_module(autopet3_repo)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with tempfile.TemporaryDirectory(prefix="autopet_tracer_") as tmpdir:
        pet_nii = mha_to_nifti(pet_mha_path, Path(tmpdir) / "pet_for_classifier.nii.gz")
        pet = module.reorient_image(str(pet_nii))
        mip_cor, mip_sag = module.get_mips(pet)

    model = module.MipClassifier()
    state = torch.load(str(checkpoint_path), map_location=device_obj)
    model.load_state_dict(state)
    model = model.to(device_obj)
    model.eval()

    mip_cor = module.MIP_transform(resize=(224, 224))(mip_cor).unsqueeze(0).to(device_obj)
    mip_sag = module.MIP_transform(resize=(224, 224))(mip_sag).unsqueeze(0).to(device_obj)

    with torch.no_grad():
        logit = model(mip_cor, mip_sag)
        prob_psma = torch.sigmoid(logit).item()
    return "psma" if prob_psma > 0.5 else "fdg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pet-mha", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--autopet3-repo", required=True, type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(classify_pet_mha(args.pet_mha, args.checkpoint, args.autopet3_repo, args.device))


if __name__ == "__main__":
    main()
