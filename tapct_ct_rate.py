"""Frozen TAP-CT feature extraction and CT-RATE multi-label head training.

The extractor represents each 96-slice study as eight TAP-CT 12-slice CLS
tokens.  The classifier uses one learned query per abnormality to pool those
tokens.  Feature extraction is sharded so independent one-GPU Slurm array
tasks can process the full dataset without distributed coordination.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel

from data_modules.ct_rate_dataset import (
    DEFAULT_LABEL_COLS,
    _read_exclusions,
    _scan_family,
    _strip_nii_suffix,
    _volume_path,
)


DEFAULT_MODEL = "<MODEL_ROOT>/tap-ct-b-3d"
DEFAULT_DATA = "<CT_RATE_ROOT>"
DEFAULT_OUTPUT = (
    "<SCRATCH_ROOT>/Med-SenseNova-Vision/"
    "artifacts/tapct-classification"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_paths(data_root: Path, split: str) -> tuple[Path, Path, Path]:
    if split not in {"train", "valid"}:
        raise ValueError(f"Unsupported split: {split}")
    volume_dir = data_root / ("train_fixed" if split == "train" else "valid_fixed")
    labels = data_root / "multi_abnormality_labels" / f"{split}_predicted_labels.csv"
    exclusions = data_root / "metadata" / f"no_chest_{split}.txt"
    return volume_dir, labels, exclusions


def load_records(data_root: Path, split: str, family_balanced: bool) -> list[dict]:
    volume_dir, labels_path, exclusions_path = split_paths(data_root, split)
    excluded = _read_exclusions(str(exclusions_path))
    records: list[dict] = []
    with labels_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["VolumeName", *DEFAULT_LABEL_COLS]
        missing = [key for key in required if key not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{labels_path} is missing columns: {missing}")
        for row in reader:
            name = _strip_nii_suffix(row["VolumeName"])
            if name in excluded:
                continue
            path = _volume_path(volume_dir, name)
            if not path.is_file():
                continue
            records.append(
                {
                    "name": name,
                    "family": _scan_family(name),
                    "path": path,
                    "labels": np.asarray(
                        [float(row[column]) for column in DEFAULT_LABEL_COLS],
                        dtype=np.float32,
                    ),
                }
            )

    if family_balanced:
        representatives: dict[str, dict] = {}
        for record in records:
            representatives.setdefault(record["family"], record)
        records = list(representatives.values())
    if not records:
        raise RuntimeError(f"No usable records found for {split}")
    return records


def load_tapct_volume(path: Path, num_slices: int = 96) -> torch.Tensor:
    """Return a normalized tensor shaped [1, D, 224, 224]."""
    image = nib.as_closest_canonical(nib.load(path))
    depth = int(image.shape[2])
    indices = np.rint(np.linspace(0, depth - 1, num_slices)).astype(np.int64)
    hu = np.asarray(image.dataobj, dtype=np.float32)[..., indices]
    # nib.as_closest_canonical gives RAS; TAP-CT was evaluated in LPS.
    hu = np.ascontiguousarray(hu[::-1, ::-1, :].transpose(2, 0, 1))
    volume = torch.from_numpy(hu).unsqueeze(1)
    volume = F.interpolate(volume, size=(224, 224), mode="bilinear", align_corners=False)
    volume = volume.squeeze(1).clamp_(-1008.0, 822.0)
    volume = (volume - (-86.80862426757812)) / 322.63470458984375
    return volume.unsqueeze(0)


@torch.inference_mode()
def encode_volume(
    model: nn.Module,
    volume: torch.Tensor,
    device: torch.device,
    chunk_depth: int = 12,
    chunk_batch_size: int = 1,
) -> torch.Tensor:
    if volume.shape[1] % chunk_depth:
        raise ValueError("num_slices must be divisible by chunk_depth")
    chunks = volume.unfold(1, chunk_depth, chunk_depth).permute(1, 0, 4, 2, 3)
    features = []
    for start in range(0, len(chunks), chunk_batch_size):
        batch = chunks[start : start + chunk_batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixel_values=batch)
        features.append(output.pooler_output.float().cpu())
    return torch.cat(features, dim=0)


def load_tapct(model_path: str, device: torch.device) -> nn.Module:
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.requires_grad_(False).eval().to(device)
    return model


def run_smoke(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The TAP-CT smoke test requires a CUDA GPU")
    records = load_records(Path(args.data_root), "valid", family_balanced=False)
    device = torch.device("cuda")
    model = load_tapct(args.model_path, device)
    volume = load_tapct_volume(records[0]["path"], args.num_slices)
    features = encode_volume(
        model,
        volume,
        device,
        chunk_depth=args.chunk_depth,
        chunk_batch_size=args.chunk_batch_size,
    )
    expected_chunks = args.num_slices // args.chunk_depth
    if features.shape != (expected_chunks, 768):
        raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")
    print(
        json.dumps(
            {
                "status": "ok",
                "volume": records[0]["name"],
                "input_shape": list(volume.shape),
                "feature_shape": list(features.shape),
                "feature_finite": bool(torch.isfinite(features).all()),
                "gpu": torch.cuda.get_device_name(0),
            },
            indent=2,
        )
    )


def run_extract(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires a CUDA GPU")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    family_balanced = args.split == "train"
    all_records = load_records(Path(args.data_root), args.split, family_balanced)
    records = all_records[args.shard_index :: args.num_shards]
    if args.limit:
        records = records[: args.limit]
    print(
        f"Extracting {len(records)}/{len(all_records)} {args.split} records "
        f"for shard {args.shard_index}/{args.num_shards}"
    )

    device = torch.device("cuda")
    model = load_tapct(args.model_path, device)
    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    for index, record in enumerate(records, start=1):
        volume = load_tapct_volume(record["path"], args.num_slices)
        feature = encode_volume(
            model,
            volume,
            device,
            chunk_depth=args.chunk_depth,
            chunk_batch_size=args.chunk_batch_size,
        )
        feature_rows.append(feature.numpy().astype(np.float16))
        labels.append(record["labels"])
        names.append(record["name"])
        if index == 1 or index % 25 == 0:
            print(f"[{index:5d}/{len(records):5d}] {record['name']}", flush=True)

    output_dir = Path(args.output_dir) / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{args.split}-shard-{args.shard_index:03d}-of-{args.num_shards:03d}.npz"
    )
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        features=np.stack(feature_rows),
        labels=np.stack(labels),
        names=np.asarray(names),
    )
    os.replace(temporary, output_path)
    print(f"Saved {output_path} ({output_path.stat().st_size / 2**20:.1f} MiB)")


class DiseaseQueryHead(nn.Module):
    def __init__(self, embed_dim: int = 768, num_classes: int = 18, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.queries = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.class_weight = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.norm(features)
        queries = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        pooled, _ = self.attention(queries, features, features, need_weights=False)
        pooled = self.dropout(pooled)
        return (pooled * self.class_weight.unsqueeze(0)).sum(-1) / math.sqrt(
            pooled.shape[-1]
        ) + self.bias


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    positive = probabilities
    negative = 1.0 - probabilities
    if clip:
        negative = (negative + clip).clamp(max=1.0)
    loss = targets * torch.log(positive.clamp(min=1e-8))
    loss += (1.0 - targets) * torch.log(negative.clamp(min=1e-8))
    if gamma_pos > 0 or gamma_neg > 0:
        probability_target = positive * targets + negative * (1.0 - targets)
        gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
        loss *= torch.pow(1.0 - probability_target, gamma)
    return -loss.mean()


def read_feature_shards(output_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted((output_dir / "features").glob(f"{split}-shard-*.npz"))
    if not paths:
        raise FileNotFoundError(f"No {split} feature shards found under {output_dir / 'features'}")
    features, labels, names = [], [], []
    for path in paths:
        with np.load(path) as shard:
            features.append(shard["features"])
            labels.append(shard["labels"])
            names.append(shard["names"])
    features_array = np.concatenate(features)
    labels_array = np.concatenate(labels)
    names_array = np.concatenate(names)
    order = np.argsort(names_array)
    return features_array[order], labels_array[order], names_array[order]


@torch.inference_mode()
def predict_head(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for features, targets in loader:
        logits = model(features.to(device, dtype=torch.float32))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def macro_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    scores = []
    for class_index in range(labels.shape[1]):
        if np.unique(labels[:, class_index]).size == 2:
            scores.append(roc_auc_score(labels[:, class_index], probabilities[:, class_index]))
    return float(np.mean(scores))


def optimal_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 181)
    for class_index in range(labels.shape[1]):
        values = [
            f1_score(labels[:, class_index], probabilities[:, class_index] >= threshold, zero_division=0)
            for threshold in grid
        ]
        thresholds[class_index] = grid[int(np.argmax(values))]
    return thresholds


def run_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    train_features, train_labels, _ = read_feature_shards(output_dir, "train")
    val_features, val_labels, val_names = read_feature_shards(output_dir, "valid")
    print(f"Train features: {train_features.shape}; validation: {val_features.shape}")

    train_dataset = TensorDataset(
        torch.from_numpy(train_features.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_features.astype(np.float32)),
        torch.from_numpy(val_labels.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiseaseQueryHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run_dir = output_dir / f"head-seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_auroc = -math.inf
    best_epoch = -1
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for features, targets in train_loader:
            features = features.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = asymmetric_loss(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        probabilities, labels = predict_head(model, val_loader, device)
        auroc = macro_auroc(labels, probabilities)
        thresholds = optimal_thresholds(labels, probabilities)
        macro_f1 = float(
            f1_score(labels, probabilities >= thresholds, average="macro", zero_division=0)
        )
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.6f} "
            f"val_auroc={auroc:.6f} val_f1={macro_f1:.6f}",
            flush=True,
        )
        if auroc > best_auroc + 1e-5:
            best_auroc = auroc
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "thresholds": thresholds,
                    "val_auroc": auroc,
                    "val_macro_f1": macro_f1,
                    "epoch": epoch,
                    "label_cols": DEFAULT_LABEL_COLS,
                    "model_path": args.model_path,
                    "num_slices": args.num_slices,
                    "chunk_depth": args.chunk_depth,
                },
                run_dir / "best.pt",
            )
            np.savez(
                run_dir / "validation_predictions.npz",
                names=val_names,
                labels=labels,
                probabilities=probabilities,
                thresholds=thresholds,
            )
        else:
            patience += 1
        if patience >= args.patience:
            break

    metrics = {"best_epoch": best_epoch, "val_auroc": best_auroc}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "extract", "train"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model-path", default=DEFAULT_MODEL)
        subparser.add_argument("--data-root", default=DEFAULT_DATA)
        subparser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
        subparser.add_argument("--num-slices", type=int, default=96)
        subparser.add_argument("--chunk-depth", type=int, default=12)

    smoke = subparsers.choices["smoke"]
    smoke.add_argument("--chunk-batch-size", type=int, default=1)

    extract = subparsers.choices["extract"]
    extract.add_argument("--split", choices=["train", "valid"], required=True)
    extract.add_argument("--shard-index", type=int, required=True)
    extract.add_argument("--num-shards", type=int, required=True)
    extract.add_argument("--chunk-batch-size", type=int, default=1)
    extract.add_argument("--limit", type=int)

    train = subparsers.choices["train"]
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        run_smoke(args)
    elif args.command == "extract":
        run_extract(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
