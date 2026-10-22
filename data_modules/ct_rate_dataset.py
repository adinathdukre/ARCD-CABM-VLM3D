"""Direct NIfTI loader for CT-RATE multi-label classification.

The generic dataset bundled with this repository expects a large precomputed
``.npy`` cache.  CT-RATE is already available as corrected NIfTI volumes, so
this loader samples axial slices directly, applies the three CT windows used
by the challenge baseline, and resizes in-plane without materializing another
multi-terabyte copy of the dataset.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import lightning as L
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_LABEL_COLS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

CT_WINDOWS = (
    (-1000.0, 1000.0),
    (-150.0, 250.0),
    (-1000.0, 400.0),
)

# Compact, bounded acquisition representation used only by the training-time
# reconstruction gate.  The challenge model remains a single-volume image
# model and does not require a metadata sidecar at inference time.
ACQUISITION_FEATURE_DIM = 13


def _numbers(value: object) -> list[float]:
    if value is None:
        return []
    return [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(value))]


def _first_number(value: object, default: float = 0.0) -> float:
    values = _numbers(value)
    return values[0] if values else default


def _acquisition_features(row: dict[str, str]) -> np.ndarray:
    """Return bounded physics/acquisition features without identifiers."""
    kernel_numbers = [x for x in _numbers(row.get("ConvolutionKernel")) if x <= 100]
    kernel_strength = max(kernel_numbers, default=0.0)
    xy_values = _numbers(row.get("XYSpacing"))
    xy_spacing = float(np.mean(xy_values)) if xy_values else 0.0
    manufacturer = (row.get("Manufacturer") or "").lower()
    manufacturer_one_hot = [
        float("siemens" in manufacturer),
        float("philips" in manufacturer),
        float("ge " in f"{manufacturer} " or "general electric" in manufacturer),
        float("canon" in manufacturer or "toshiba" in manufacturer),
    ]
    if not any(manufacturer_one_hot):
        manufacturer_one_hot.append(1.0)
    else:
        manufacturer_one_hot.append(0.0)

    # Fixed transforms avoid fitting normalization statistics on validation.
    # Every feature is approximately in [0, 1], and missing values map to 0.
    continuous = [
        min(kernel_strength / 80.0, 1.5),
        float(bool(kernel_numbers)),
        min(xy_spacing / 2.0, 1.5),
        min(_first_number(row.get("ZSpacing")) / 5.0, 1.5),
        min(math.log1p(max(_first_number(row.get("NumberofSlices")), 0.0)) / 8.0, 1.5),
        min(math.log1p(max(_first_number(row.get("Rows")), 0.0)) / 7.0, 1.5),
        min(max(_first_number(row.get("ReconstructionDiameter")), 0.0) / 500.0, 1.5),
        min(math.log1p(max(_first_number(row.get("CTDIvol")), 0.0)) / 5.0, 1.5),
    ]
    return np.asarray([*continuous, *manufacturer_one_hot], dtype=np.float32)


def _read_metadata(path: Optional[str]) -> dict[str, np.ndarray]:
    if not path:
        return {}
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"CT-RATE metadata not found: {metadata_path}")
    features = {}
    with metadata_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "VolumeName" not in (reader.fieldnames or []):
            raise ValueError(f"{metadata_path} is missing VolumeName")
        for row in reader:
            features[_strip_nii_suffix(row["VolumeName"])] = _acquisition_features(row)
    return features


def _strip_nii_suffix(name: str) -> str:
    name = name.strip()
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _volume_path(root: Path, volume_name: str) -> Path:
    """Map ``train_1_a_2`` to ``train_1/train_1_a/train_1_a_2.nii.gz``."""
    parts = volume_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CT-RATE volume name: {volume_name!r}")
    patient = "_".join(parts[:2])
    scan = "_".join(parts[:3])
    return root / patient / scan / f"{volume_name}.nii.gz"


def _scan_family(volume_name: str) -> str:
    """Return the acquisition family shared by repeat reconstructions."""
    parts = volume_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CT-RATE volume name: {volume_name!r}")
    return "_".join(parts[:3])


def _read_exclusions(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    exclusion_path = Path(path)
    if not exclusion_path.exists():
        raise FileNotFoundError(f"Exclusion list not found: {exclusion_path}")
    excluded = set()
    for line in exclusion_path.read_text().splitlines():
        line = line.strip()
        if line:
            excluded.add(_strip_nii_suffix(Path(line).name))
    return excluded


class CTRateDataset(Dataset):
    def __init__(
        self,
        root: str,
        labels_path: str,
        label_cols: Sequence[str] = DEFAULT_LABEL_COLS,
        num_slices: int = 96,
        image_size: int = 224,
        exclusion_path: Optional[str] = None,
        limit: Optional[int] = None,
        transform: Optional[Callable] = None,
        group_by_family: bool = False,
        pair_reconstructions: bool = False,
        metadata_path: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.label_cols = list(label_cols)
        self.num_slices = int(num_slices)
        self.image_size = int(image_size)
        self.transform = transform
        self.group_by_family = bool(group_by_family)
        self.pair_reconstructions = bool(pair_reconstructions)
        if self.pair_reconstructions and not self.group_by_family:
            raise ValueError("pair_reconstructions requires group_by_family=True")
        excluded = _read_exclusions(exclusion_path)
        self.metadata = _read_metadata(metadata_path)

        if not self.root.exists():
            raise FileNotFoundError(f"CT-RATE volume root not found: {self.root}")

        records = []
        with Path(labels_path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in ["VolumeName", *self.label_cols] if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{labels_path} is missing columns: {missing}")
            for row in reader:
                volume_name = _strip_nii_suffix(row["VolumeName"])
                if volume_name in excluded:
                    continue
                path = _volume_path(self.root, volume_name)
                if not path.exists():
                    continue
                labels = np.asarray([float(row[c]) for c in self.label_cols], dtype=np.float32)
                records.append((volume_name, path, labels))

        if limit is not None:
            records = records[: int(limit)]
        if not records:
            raise RuntimeError(f"No usable CT-RATE records found under {self.root}")
        if self.group_by_family:
            families = {}
            for record in records:
                families.setdefault(_scan_family(record[0]), []).append(record)
            for family_name, family_records in families.items():
                reference = family_records[0][2]
                if any(not np.array_equal(reference, r[2]) for r in family_records[1:]):
                    raise ValueError(f"Label mismatch within scan family {family_name}")
            self.items = list(families.values())
            label_records = [family[0] for family in self.items]
        else:
            self.items = records
            label_records = records

        positives = np.stack([r[2] for r in label_records]).sum(axis=0).astype(int)
        unit = "scan families" if self.group_by_family else "volumes"
        print(f"CTRateDataset: {len(self.items)} {unit} ({len(records)} volumes) from {self.root}")
        for name, count in zip(self.label_cols, positives):
            print(f"  {name:40s} {count:6d} ({100.0 * count / len(self.items):5.2f}%)")

    def __len__(self) -> int:
        return len(self.items)

    def _load_volume(self, path: Path) -> torch.Tensor:
        image = nib.load(path)
        depth = int(image.shape[2])
        indices = np.rint(np.linspace(0, depth - 1, self.num_slices)).astype(np.int64)

        # Nibabel's ArrayProxy cannot apply a NumPy fancy-index array directly.
        # CT-RATE volumes are gzip-compressed, so decompress the scan once and
        # select the uniformly spaced planes from the in-memory array.
        hu = np.asarray(image.dataobj, dtype=np.float32)[..., indices]
        slices = torch.from_numpy(np.ascontiguousarray(hu.transpose(2, 0, 1))).unsqueeze(1)
        slices = F.interpolate(
            slices,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        channels = []
        for lower, upper in CT_WINDOWS:
            channels.append(((slices.clamp(lower, upper) - lower) / (upper - lower)))
        # [C, H, W, S], matching AnyMC3D's volume convention.
        return torch.stack(channels, dim=0).permute(0, 2, 3, 1).contiguous()

    def _load_record(self, record):
        volume_name, path, labels = record
        volume = self._load_volume(path)
        if self.transform is not None:
            volume = self.transform({"image": volume})["image"]
        return volume, torch.from_numpy(labels.copy()), volume_name

    def _metadata_tensor(self, volume_name: str) -> torch.Tensor:
        features = self.metadata.get(volume_name)
        if features is None:
            features = np.zeros(ACQUISITION_FEATURE_DIM, dtype=np.float32)
        return torch.from_numpy(features.copy())

    def __getitem__(self, index: int):
        if not self.group_by_family:
            return self._load_record(self.items[index])

        family = self.items[index]
        first_idx = int(torch.randint(len(family), (1,)).item())
        first = self._load_record(family[first_idx])
        if not self.pair_reconstructions:
            return first

        if len(family) > 1:
            offset = int(torch.randint(1, len(family), (1,)).item())
            second_idx = (first_idx + offset) % len(family)
            pair_valid = True
        else:
            second_idx = first_idx
            pair_valid = False
        second_volume, _, _ = self._load_record(family[second_idx])
        volume, labels, volume_name = first
        if not self.metadata:
            return volume, labels, volume_name, second_volume, torch.tensor(pair_valid)
        second_name = family[second_idx][0]
        return (
            volume,
            labels,
            volume_name,
            second_volume,
            torch.tensor(pair_valid),
            self._metadata_tensor(volume_name),
            self._metadata_tensor(second_name),
        )


class CTRateDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_root: str,
        val_root: str,
        train_labels_path: str,
        val_labels_path: str,
        train_exclusion_path: Optional[str] = None,
        val_exclusion_path: Optional[str] = None,
        label_cols: Sequence[str] = DEFAULT_LABEL_COLS,
        num_slices: int = 96,
        image_size: int = 224,
        batch_size: int = 1,
        num_workers: int = 2,
        train_limit: Optional[int] = None,
        val_limit: Optional[int] = None,
        task: str = "multilabel",
        fold: int = 0,
        family_balanced: bool = False,
        pair_reconstructions: bool = False,
        train_metadata_path: Optional[str] = None,
        val_metadata_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if task != "multilabel":
            raise ValueError("CT-RATE is configured as an 18-label multilabel task")
        self.train_root = train_root
        self.val_root = val_root
        self.train_labels_path = train_labels_path
        self.val_labels_path = val_labels_path
        self.train_exclusion_path = train_exclusion_path
        self.val_exclusion_path = val_exclusion_path
        self.label_cols = list(label_cols)
        self.num_slices = num_slices
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.task = task
        # CT-RATE has a predefined split; this value is accepted only so the
        # generic multi-fold training entry point can instantiate this module.
        self.fold = int(fold)
        self.family_balanced = bool(family_balanced)
        self.pair_reconstructions = bool(pair_reconstructions)
        self.train_metadata_path = train_metadata_path
        self.val_metadata_path = val_metadata_path
        self.train_transform = None
        self.eval_transform = None
        self._available_splits = {"train", "val"}

    def _dataset(self, split: str) -> CTRateDataset:
        is_train = split == "train"
        return CTRateDataset(
            root=self.train_root if is_train else self.val_root,
            labels_path=self.train_labels_path if is_train else self.val_labels_path,
            label_cols=self.label_cols,
            num_slices=self.num_slices,
            image_size=self.image_size,
            exclusion_path=self.train_exclusion_path if is_train else self.val_exclusion_path,
            limit=self.train_limit if is_train else self.val_limit,
            transform=self.train_transform if is_train else self.eval_transform,
            group_by_family=is_train and (
                self.family_balanced or self.pair_reconstructions
            ),
            pair_reconstructions=is_train and self.pair_reconstructions,
            metadata_path=(
                self.train_metadata_path if is_train else self.val_metadata_path
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset("train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset("val"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
