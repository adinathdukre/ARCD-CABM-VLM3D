"""Export CT-CLIP LiPro logits for CT-RATE validation volumes.

This is an image-only loader for the public CT_LiPro checkpoint.  The
upstream inference script constructs a 110M-parameter BERT and executes it on
an empty string even though the LiPro head only consumes image latents.  We
load just CTViT, its visual projection, and the 18-label classifier instead.
The resulting CSV uses the same schema as ``ct_rate_calibrate_ensemble.py``
so the two foundation models can be evaluated and calibrated together.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_PREFIX = "target::"
LOGIT_PREFIX = "logit::"
PROB_PREFIX = "probability::"


def patient_id(case_id: str) -> str:
    return "_".join(case_id.split("_")[:2])


def family_id(case_id: str) -> str:
    return "_".join(case_id.split("_")[:3])


def shard_path(output: Path, rank: int, world_size: int) -> Path:
    return output.with_name(
        f"{output.stem}.part-{rank:05d}-of-{world_size:05d}{output.suffix}"
    )


def volume_path(data_root: Path, volume_name: str) -> Path:
    case_id = volume_name.removesuffix(".nii.gz")
    pieces = case_id.split("_")
    if len(pieces) < 4:
        raise ValueError(f"Unexpected CT-RATE volume name: {volume_name}")
    patient = "_".join(pieces[:2])
    family = "_".join(pieces[:3])
    path = data_root / patient / family / volume_name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_volume(
    path: Path, metadata: pd.Series | None = None, intensity_mode: str = "hu"
) -> torch.Tensor:
    """Prepare a fixed-v2 NIfTI for CT-CLIP.

    CT-RATE ``*_fixed`` files already store Hounsfield units.  The upstream
    loader still contains its v1 DICOM rescale step, which double-applies the
    intercept to fixed-v2 files and also cannot be reproduced for hidden
    challenge inputs (there is no sidecar metadata CSV).  ``dicom`` is kept as
    an explicit legacy ablation, never as an automatic transform.
    """
    nii = nib.load(str(path))
    image = np.asarray(nii.dataobj, dtype=np.float32)
    x_spacing, y_spacing, z_spacing = map(float, nii.header.get_zooms()[:3])

    if intensity_mode == "hu":
        pass
    elif intensity_mode == "dicom":
        if metadata is None:
            raise ValueError("Legacy DICOM rescaling requires metadata")
        slope = float(metadata["RescaleSlope"])
        intercept = float(metadata["RescaleIntercept"])
        image = slope * image + intercept
    else:
        raise ValueError(f"Unknown intensity mode: {intensity_mode}")
    image = np.clip(image, -1000.0, 1000.0)
    # NIfTI is X,Y,Z; CTViT preprocessing resamples in Z,X,Y order.
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).unsqueeze(0)
    original = tensor.shape[2:]
    current_spacing = (z_spacing, x_spacing, y_spacing)
    target_spacing = (1.5, 0.75, 0.75)
    resized = tuple(
        max(1, int(original[i] * current_spacing[i] / target_spacing[i]))
        for i in range(3)
    )
    tensor = F.interpolate(
        tensor, size=resized, mode="trilinear", align_corners=False
    )[0, 0]
    # Convert Z,X,Y to X,Y,Z, scale to [-1, 1], then center crop/pad.
    tensor = tensor.permute(1, 2, 0).div_(1000.0)
    target = (480, 480, 240)
    starts = [max((tensor.shape[i] - target[i]) // 2, 0) for i in range(3)]
    ends = [min(starts[i] + target[i], tensor.shape[i]) for i in range(3)]
    tensor = tensor[
        starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]
    ]
    pads: list[int] = []
    for axis in (2, 1, 0):
        missing = target[axis] - tensor.shape[axis]
        before = missing // 2
        pads.extend((before, missing - before))
    tensor = F.pad(tensor, pads, value=-1.0)
    # CTViT expects C,T,H,W.
    return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()


class ImageOnlyLiPro(nn.Module):
    def __init__(self, ctvit_source: Path):
        super().__init__()
        sys.path.insert(0, str(ctvit_source))
        from transformer_maskgit import CTViT

        self.visual_transformer = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
        )
        self.to_visual_latent = nn.Linear(294912, 512, bias=False)
        self.classifier = nn.Linear(512, 18)

    def load_public_checkpoint(self, checkpoint_path: Path) -> None:
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
        visual_prefix = "trained_model.visual_transformer."
        visual = {
            key[len(visual_prefix) :]: value
            for key, value in state.items()
            if key.startswith(visual_prefix)
        }
        incompatible = self.visual_transformer.load_state_dict(visual, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Incompatible CTViT checkpoint: {incompatible}")
        self.to_visual_latent.load_state_dict(
            {"weight": state["trained_model.to_visual_latent.weight"]}
        )
        self.classifier.load_state_dict(
            {"weight": state["classifier.weight"], "bias": state["classifier.bias"]}
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoded = self.visual_transformer(image, return_encoded_tokens=True)
        image_features = encoded.mean(dim=1).reshape(encoded.shape[0], -1)
        latents = F.normalize(self.to_visual_latent(image_features), dim=-1)
        return self.classifier(F.relu(latents))


@torch.inference_mode()
def predict(args: argparse.Namespace) -> None:
    rank = args.rank
    if rank is None:
        rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = args.world_size
    if world_size is None:
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} must be in [0, {world_size})")

    metadata = pd.read_csv(args.metadata).set_index("VolumeName", drop=False)
    labels = pd.read_csv(args.labels)
    label_names = list(labels.columns[1:])
    if len(label_names) != 18:
        raise ValueError(f"Expected 18 labels, found {len(label_names)}")
    if args.limit is not None:
        labels = labels.iloc[: int(args.limit)]
    labels = labels.iloc[rank::world_size].copy()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = ImageOnlyLiPro(Path(args.ctvit_source).resolve())
    model.load_public_checkpoint(Path(args.checkpoint).resolve())
    model.to(device).eval()
    print(
        f"rank {rank}/{world_size}: {len(labels)} volumes; "
        f"checkpoint={Path(args.checkpoint).name}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    data_root = Path(args.data_root).resolve()
    use_amp = device.type == "cuda" and args.amp != "none"
    amp_dtype = torch.bfloat16 if args.amp == "bfloat16" else torch.float16
    for local_index, (_, label_row) in enumerate(labels.iterrows()):
        volume_name = str(label_row["VolumeName"])
        case_id = volume_name.removesuffix(".nii.gz")
        if volume_name not in metadata.index:
            raise KeyError(f"No metadata for {volume_name}")
        image = load_volume(
            volume_path(data_root, volume_name),
            metadata.loc[volume_name],
            intensity_mode=args.intensity_mode,
        )
        image = image.unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(image)
        logits = logits.float().cpu()[0]
        probabilities = torch.sigmoid(logits)
        row: dict[str, object] = {
            "case_id": case_id,
            "patient_id": patient_id(case_id),
            "family_id": family_id(case_id),
        }
        for class_index, label_name in enumerate(label_names):
            row[f"{TARGET_PREFIX}{label_name}"] = int(label_row[label_name])
            row[f"{LOGIT_PREFIX}{label_name}"] = float(logits[class_index])
            row[f"{PROB_PREFIX}{label_name}"] = float(probabilities[class_index])
        rows.append(row)
        if local_index % 10 == 0:
            print(
                f"rank {rank}: processed {local_index + 1}/{len(labels)} "
                f"({volume_name})",
                flush=True,
            )

    output = Path(args.output).resolve()
    destination = shard_path(output, rank, world_size) if world_size > 1 else output
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("case_id")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)
    print(f"rank {rank}: wrote {len(frame)} rows to {destination}", flush=True)


def merge(args: argparse.Namespace) -> None:
    parts = [Path(part).resolve() for part in args.parts]
    frames = [pd.read_csv(part) for part in parts]
    merged = pd.concat(frames, ignore_index=True).sort_values("case_id")
    if merged["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs across prediction shards")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Merged {len(parts)} shards / {len(merged)} rows into {output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    pred = commands.add_parser("predict")
    pred.add_argument("--data-root", required=True)
    pred.add_argument("--metadata", required=True)
    pred.add_argument("--labels", required=True)
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--ctvit-source", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--device", default="cuda")
    pred.add_argument("--amp", choices=("none", "float16", "bfloat16"), default="none")
    pred.add_argument(
        "--intensity-mode",
        choices=("hu", "dicom"),
        default="hu",
        help="Use HU for fixed-v2 NIfTI; dicom reproduces the legacy rescale ablation",
    )
    pred.add_argument("--rank", type=int)
    pred.add_argument("--world-size", type=int)
    pred.add_argument("--limit", type=int)
    pred.set_defaults(function=predict)
    merger = commands.add_parser("merge")
    merger.add_argument("--parts", nargs="+", required=True)
    merger.add_argument("--output", required=True)
    merger.set_defaults(function=merge)
    return root


if __name__ == "__main__":
    parsed = parser().parse_args()
    parsed.function(parsed)
