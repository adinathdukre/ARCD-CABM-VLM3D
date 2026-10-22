"""CT-RATE validation prediction, calibration, and logit ensembling.

The challenge evaluates multi-label predictions at a fixed probability
threshold.  This utility keeps evaluation honest by assigning every volume
from the same patient to either a calibration subset or an evaluation subset,
never both.

Examples
--------
Distributed prediction under a four-task Slurm allocation::

    srun -n 4 python ct_rate_calibrate_ensemble.py predict \
        --config checkpoints/run/config.yaml \
        --checkpoint checkpoints/run/best.ckpt \
        --output calibration/adq.csv

Merge the four independently written shards::

    python ct_rate_calibrate_ensemble.py merge \
        --parts calibration/adq.part-*-of-4.csv \
        --output calibration/adq.csv

Fit thresholds and ensemble weights on patient-disjoint calibration data and
report performance on the held-out patients::

    python ct_rate_calibrate_ensemble.py calibrate \
        --predictions calibration/adq.csv calibration/cdra.csv \
        --names adq cdra --output-dir calibration/adq_cdra
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset


TARGET_PREFIX = "target::"
LOGIT_PREFIX = "logit::"
PROB_PREFIX = "probability::"


def patient_id_from_case_id(case_id: str) -> str:
    """Map ``valid_123_a_2`` to patient ``valid_123``."""
    parts = str(case_id).split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CT-RATE case id: {case_id!r}")
    return "_".join(parts[:2])


def family_id_from_case_id(case_id: str) -> str:
    """Map ``valid_123_a_2`` to acquisition family ``valid_123_a``."""
    parts = str(case_id).split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CT-RATE case id: {case_id!r}")
    return "_".join(parts[:3])


def shard_path(output: Path, rank: int, world_size: int) -> Path:
    return output.with_name(
        f"{output.stem}.part-{rank:05d}-of-{world_size:05d}{output.suffix}"
    )


def _load_model(config_path: Path, checkpoint_path: Path):
    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint and config are incompatible: "
            f"missing={list(incompatible.missing_keys)}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    return cfg, model


@torch.inference_mode()
def predict(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    rank = args.rank
    if rank is None:
        rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = args.world_size
    if world_size is None:
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} must be in [0, {world_size})")

    cfg, model = _load_model(config_path, checkpoint_path)
    if cfg.model.get("task") != "multilabel":
        raise ValueError("The supplied model is not configured for multilabel")

    # Instantiate exactly the validation loader used in training, overriding
    # only worker and batch counts. Validation augmentation is deterministic.
    data_overrides = {
        "fold": int(cfg.data.module.get("fold", 0)),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
    }
    if args.limit is not None:
        data_overrides["val_limit"] = int(args.limit)
    dm = instantiate(cfg.data.module, **data_overrides)
    from data_modules.data_augmentation import build_val_transforms

    dm.eval_transform = build_val_transforms()
    dataset = dm._dataset("val")
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    model.to(device).eval()
    use_amp = device.type == "cuda"
    label_names = list(cfg.data.module.label_cols)
    modality = str(cfg.model.modalities[0])
    rows: list[dict[str, object]] = []

    print(
        f"rank {rank}/{world_size}: {len(indices)} of {len(dataset)} volumes; "
        f"checkpoint={checkpoint_path.name}"
    )
    for batch_index, (volumes, targets, case_ids) in enumerate(loader):
        volumes = volumes.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits, _ = model.model({modality: volumes})
        logits_np = logits.float().cpu().numpy()
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        targets_np = targets.float().numpy()

        for row_index, case_id in enumerate(case_ids):
            case_id = str(case_id)
            row: dict[str, object] = {
                "case_id": case_id,
                "patient_id": patient_id_from_case_id(case_id),
                "family_id": family_id_from_case_id(case_id),
            }
            for class_index, label_name in enumerate(label_names):
                row[f"{TARGET_PREFIX}{label_name}"] = int(
                    targets_np[row_index, class_index]
                )
                row[f"{LOGIT_PREFIX}{label_name}"] = float(
                    logits_np[row_index, class_index]
                )
                row[f"{PROB_PREFIX}{label_name}"] = float(
                    probabilities[row_index, class_index]
                )
            rows.append(row)
        if batch_index % 50 == 0:
            print(f"rank {rank}: processed {len(rows)}/{len(indices)}")

    destination = shard_path(output, rank, world_size) if world_size > 1 else output
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("case_id")
    frame.to_csv(destination, index=False)
    print(f"rank {rank}: wrote {len(frame)} predictions to {destination}")


def prediction_columns(frame: pd.DataFrame) -> list[str]:
    labels = [c[len(TARGET_PREFIX):] for c in frame if c.startswith(TARGET_PREFIX)]
    if not labels:
        raise ValueError("No target columns found")
    for label in labels:
        for prefix in (LOGIT_PREFIX, PROB_PREFIX):
            column = f"{prefix}{label}"
            if column not in frame:
                raise ValueError(f"Missing prediction column: {column}")
    return labels


def merge_predictions(args: argparse.Namespace) -> None:
    part_paths = [Path(path).resolve() for path in args.parts]
    if not part_paths:
        raise ValueError("At least one prediction shard is required")
    frames = [pd.read_csv(path) for path in part_paths]
    labels = prediction_columns(frames[0])
    expected = set(frames[0].columns)
    for path, frame in zip(part_paths[1:], frames[1:]):
        if set(frame.columns) != expected:
            raise ValueError(f"Prediction schema mismatch in {path}")
    merged = pd.concat(frames, ignore_index=True).sort_values("case_id")
    duplicates = merged["case_id"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = merged.loc[duplicates, "case_id"].head().tolist()
        raise ValueError(f"Duplicate case ids across shards: {duplicate_ids}")
    # Exercise full schema validation after concatenation.
    prediction_columns(merged)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(
        f"Merged {len(part_paths)} shards and {len(merged)} cases "
        f"({len(labels)} labels) into {output}"
    )


def _safe_logit(probability: np.ndarray | float) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    probability = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(probability / (1.0 - probability))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    positive = logits >= 0
    result = np.empty_like(logits, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result


def best_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the exact observed-score threshold maximizing binary F1."""
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-probabilities, kind="stable")
    sorted_targets = targets[order]
    sorted_probabilities = probabilities[order]
    true_positives = np.cumsum(sorted_targets)
    predicted_positives = np.arange(1, len(targets) + 1)
    total_positives = int(targets.sum())
    denominator = predicted_positives + total_positives
    f1 = np.divide(
        2.0 * true_positives,
        denominator,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=denominator > 0,
    )
    # Evaluate only after the last member of a tied-score group.
    group_end = np.r_[
        sorted_probabilities[:-1] != sorted_probabilities[1:], True
    ]
    candidate_indices = np.flatnonzero(group_end)
    if len(candidate_indices) == 0 or total_positives == 0:
        return 0.5
    candidate_f1 = f1[candidate_indices]
    best_score = candidate_f1.max()
    tied = candidate_indices[np.isclose(candidate_f1, best_score)]
    # Prefer the threshold closest to 0.5 when multiple settings tie.
    best_index = tied[
        np.argmin(np.abs(sorted_probabilities[tied] - 0.5))
    ]
    return float(np.clip(sorted_probabilities[best_index], 1.0e-5, 1 - 1.0e-5))


def official_crg_score(
    targets: np.ndarray, predictions: np.ndarray
) -> float:
    """Exact binary-classification CRG used by the official evaluator."""
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape:
        raise ValueError("CRG targets and predictions must have identical shapes")
    positives = int(targets.sum())
    if positives == 0:
        raise ValueError("CRG is undefined when ground truth has no positives")
    total = int(targets.size)
    true_positives = int(((predictions == 1) & (targets == 1)).sum())
    false_negatives = int(((predictions == 0) & (targets == 1)).sum())
    false_positives = int(((predictions == 1) & (targets == 0)).sum())
    reward = (total - positives) / (2.0 * positives)
    maximum = (total - positives) / 2.0
    raw_score = (
        reward * true_positives
        - reward * false_negatives
        - false_positives
    )
    denominator = 2.0 * maximum - raw_score
    return float(maximum / denominator) if denominator != 0 else 0.0


def best_crg_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    positive_reward: float,
) -> float:
    """Maximize the additive official-CRG utility for one label."""
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-probabilities, kind="stable")
    sorted_targets = targets[order]
    sorted_probabilities = probabilities[order]
    true_positives = np.cumsum(sorted_targets)
    false_positives = np.cumsum(1 - sorted_targets)
    group_end = np.r_[
        sorted_probabilities[:-1] != sorted_probabilities[1:], True
    ]
    candidate_indices = np.flatnonzero(group_end)
    utility = (
        2.0 * float(positive_reward) * true_positives[candidate_indices]
        - false_positives[candidate_indices]
    )
    if len(utility) == 0 or utility.max() <= 0:
        return float(min(1.0, np.nextafter(probabilities.max(), 1.0)))
    best_score = utility.max()
    tied = candidate_indices[np.isclose(utility, best_score)]
    best_index = tied[
        np.argmin(np.abs(sorted_probabilities[tied] - 0.5))
    ]
    return float(np.clip(sorted_probabilities[best_index], 1.0e-5, 1 - 1.0e-5))


def blended_thresholds(
    f1_thresholds: np.ndarray,
    crg_thresholds: np.ndarray,
    crg_fraction: float,
) -> np.ndarray:
    """Interpolate operating points in logit space."""
    if not 0.0 <= crg_fraction <= 1.0:
        raise ValueError("crg_fraction must be in [0, 1]")
    return _sigmoid(
        (1.0 - crg_fraction) * _safe_logit(f1_thresholds)
        + crg_fraction * _safe_logit(crg_thresholds)
    )


def _binary_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float("nan")
    return float(roc_auc_score(targets, probabilities))


def metric_summary(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | float = 0.5,
) -> dict[str, object]:
    thresholds = np.broadcast_to(
        np.asarray(thresholds, dtype=np.float64), probabilities.shape[1:]
    )
    predictions = probabilities >= thresholds[None, :]
    per_class_auc = np.asarray([
        _binary_auroc(targets[:, index], probabilities[:, index])
        for index in range(targets.shape[1])
    ])
    per_class_f1 = np.asarray([
        f1_score(
            targets[:, index], predictions[:, index], zero_division=0
        )
        for index in range(targets.shape[1])
    ])
    return {
        "macro_auroc": float(np.nanmean(per_class_auc)),
        "macro_f1": float(np.mean(per_class_f1)),
        "macro_accuracy": float((predictions == targets).mean()),
        "crg": official_crg_score(targets, predictions),
        "per_class_auroc": per_class_auc.tolist(),
        "per_class_f1": per_class_f1.tolist(),
    }


def choose_patient_split(
    targets: np.ndarray,
    patient_ids: np.ndarray,
    calibration_fraction: float,
    seed: int,
    candidates: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a patient-disjoint split with closely matched prevalences."""
    if not 0.1 <= calibration_fraction <= 0.9:
        raise ValueError("calibration_fraction must be between 0.1 and 0.9")
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(candidates):
        splitter = GroupShuffleSplit(
            n_splits=1,
            train_size=calibration_fraction,
            random_state=seed + offset,
        )
        calibration, evaluation = next(
            splitter.split(targets, groups=patient_ids)
        )
        # Both halves must contain positives and negatives for every class.
        if any(
            np.unique(targets[index, class_index]).size < 2
            for index in (calibration, evaluation)
            for class_index in range(targets.shape[1])
        ):
            continue
        prevalence_gap = np.abs(
            targets[calibration].mean(axis=0)
            - targets[evaluation].mean(axis=0)
        ).mean()
        size_gap = abs(len(calibration) / len(targets) - calibration_fraction)
        score = float(prevalence_gap + size_gap)
        if best is None or score < best[0]:
            best = (score, calibration, evaluation)
    if best is None:
        raise RuntimeError("Could not construct a valid patient-grouped split")
    calibration, evaluation = best[1], best[2]
    if set(patient_ids[calibration]) & set(patient_ids[evaluation]):
        raise AssertionError("Patient leakage detected")
    return calibration, evaluation


def _integer_compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in _integer_compositions(total - first, parts - 1):
            yield (first, *remainder)


def fit_ensemble(
    targets: np.ndarray,
    logits_by_model: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-class simplex weights and F1 operating points.

    Weight selection balances class AUROC and the best attainable F1 on the
    calibration patients. Thresholds are fitted only after blending.
    """
    if resolution < 1:
        raise ValueError("resolution must be positive")
    model_count, _, class_count = logits_by_model.shape
    candidates = np.asarray(
        list(_integer_compositions(resolution, model_count)), dtype=np.float64
    ) / resolution
    weights = np.zeros((class_count, model_count), dtype=np.float64)
    thresholds = np.zeros(class_count, dtype=np.float64)
    for class_index in range(class_count):
        y = targets[:, class_index]
        best: tuple[float, float, np.ndarray, float] | None = None
        for candidate in candidates:
            blended_logit = np.sum(
                logits_by_model[:, :, class_index] * candidate[:, None], axis=0
            )
            probability = _sigmoid(blended_logit)
            threshold = best_f1_threshold(y, probability)
            f1 = f1_score(y, probability >= threshold, zero_division=0)
            auroc = _binary_auroc(y, probability)
            objective = float(f1 + (0.0 if math.isnan(auroc) else auroc))
            # Prefer simpler, more uniform weights when the score ties.
            uniform_distance = float(
                np.square(candidate - 1.0 / model_count).sum()
            )
            ranking = (objective, -uniform_distance)
            if best is None or ranking > (best[0], best[1]):
                best = (objective, -uniform_distance, candidate.copy(), threshold)
        assert best is not None
        weights[class_index] = best[2]
        thresholds[class_index] = best[3]
    return weights, thresholds


def _aligned_prediction_data(
    paths: Sequence[str], names: Sequence[str] | None
) -> tuple[list[str], pd.DataFrame, np.ndarray, np.ndarray]:
    if names is None:
        names = [Path(path).stem for path in paths]
    if len(paths) != len(names):
        raise ValueError("--names must have the same length as --predictions")
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique")

    frames = []
    for path in paths:
        frame = pd.read_csv(path).sort_values("case_id").reset_index(drop=True)
        prediction_columns(frame)
        frames.append(frame)
    labels = prediction_columns(frames[0])
    reference_cases = frames[0]["case_id"].astype(str).tolist()
    target_columns = [f"{TARGET_PREFIX}{label}" for label in labels]
    reference_targets = frames[0][target_columns].to_numpy(dtype=np.int64)
    for path, frame in zip(paths[1:], frames[1:]):
        if frame["case_id"].astype(str).tolist() != reference_cases:
            raise ValueError(f"Case IDs do not align in {path}")
        if not np.array_equal(
            frame[target_columns].to_numpy(dtype=np.int64), reference_targets
        ):
            raise ValueError(f"Targets do not align in {path}")
    logits = np.stack([
        frame[[f"{LOGIT_PREFIX}{label}" for label in labels]].to_numpy(
            dtype=np.float64
        )
        for frame in frames
    ])
    return list(names), frames[0], reference_targets, logits


def calibrate(args: argparse.Namespace) -> None:
    names, frame, targets, logits = _aligned_prediction_data(
        args.predictions, args.names
    )
    labels = prediction_columns(frame)
    patient_ids = frame["patient_id"].astype(str).to_numpy()
    calibration, evaluation = choose_patient_split(
        targets,
        patient_ids,
        calibration_fraction=float(args.calibration_fraction),
        seed=int(args.seed),
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crg_fraction = float(getattr(args, "crg_fraction", 0.45))
    calibration_targets = targets[calibration]
    calibration_positives = int(calibration_targets.sum())
    positive_reward = (
        calibration_targets.size - calibration_positives
    ) / (2.0 * calibration_positives)

    def threshold_variants(probabilities: np.ndarray):
        f1_thresholds = np.asarray([
            best_f1_threshold(
                targets[calibration, class_index],
                probabilities[calibration, class_index],
            )
            for class_index in range(len(labels))
        ])
        crg_thresholds = np.asarray([
            best_crg_threshold(
                targets[calibration, class_index],
                probabilities[calibration, class_index],
                positive_reward,
            )
            for class_index in range(len(labels))
        ])
        balanced = blended_thresholds(
            f1_thresholds, crg_thresholds, crg_fraction
        )
        return f1_thresholds, crg_thresholds, balanced

    def append_summary(
        model_name: str,
        mode: str,
        probabilities: np.ndarray,
        threshold: np.ndarray | float,
    ) -> None:
        metrics = metric_summary(
            targets[evaluation], probabilities[evaluation], threshold
        )
        summary_rows.append({
            "model": model_name,
            "mode": mode,
            "n_cases": len(evaluation),
            "n_patients": len(set(patient_ids[evaluation])),
            "crg": metrics["crg"],
            "macro_auroc": metrics["macro_auroc"],
            "macro_f1": metrics["macro_f1"],
            "macro_accuracy": metrics["macro_accuracy"],
        })

    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for model_index, name in enumerate(names):
        probabilities = _sigmoid(logits[model_index])
        f1_thresholds, crg_thresholds, balanced_thresholds = (
            threshold_variants(probabilities)
        )
        for mode, threshold in (
            ("raw", 0.5),
            ("f1_calibrated", f1_thresholds),
            ("crg_calibrated", crg_thresholds),
            ("balanced_calibrated", balanced_thresholds),
        ):
            append_summary(name, mode, probabilities, threshold)
        for class_index, label in enumerate(labels):
            threshold_rows.append({
                "model": name,
                "label": label,
                "threshold": balanced_thresholds[class_index],
                "submission_logit_bias": -float(
                    _safe_logit(balanced_thresholds[class_index])
                ),
                "f1_threshold": f1_thresholds[class_index],
                "crg_threshold": crg_thresholds[class_index],
                "crg_fraction": crg_fraction,
            })

    # Always report the uniform ensemble; it is a low-variance baseline.
    uniform_logits = logits.mean(axis=0)
    uniform_probabilities = _sigmoid(uniform_logits)
    uniform_f1, uniform_crg, uniform_balanced = threshold_variants(
        uniform_probabilities
    )
    for mode, threshold in (
        ("raw", 0.5),
        ("f1_calibrated", uniform_f1),
        ("crg_calibrated", uniform_crg),
        ("balanced_calibrated", uniform_balanced),
    ):
        append_summary(
            "uniform_ensemble", mode, uniform_probabilities, threshold
        )
    uniform_parameter_rows = []
    for class_index, label in enumerate(labels):
        row: dict[str, object] = {
            "label": label,
            "threshold": uniform_balanced[class_index],
            "submission_logit_bias": -float(
                _safe_logit(uniform_balanced[class_index])
            ),
            "f1_threshold": uniform_f1[class_index],
            "crg_threshold": uniform_crg[class_index],
            "crg_fraction": crg_fraction,
        }
        for name in names:
            row[f"weight::{name}"] = 1.0 / len(names)
        uniform_parameter_rows.append(row)

    weights, ensemble_thresholds = fit_ensemble(
        targets[calibration], logits[:, calibration], int(args.weight_resolution)
    )
    ensemble_logits = np.einsum("cm,mnc->nc", weights, logits)
    ensemble_probabilities = _sigmoid(ensemble_logits)
    _, fitted_crg, fitted_balanced = threshold_variants(ensemble_probabilities)
    for mode, threshold in (
        ("raw", 0.5),
        ("f1_calibrated", ensemble_thresholds),
        ("crg_calibrated", fitted_crg),
        ("balanced_calibrated", fitted_balanced),
    ):
        append_summary(
            "fitted_ensemble", mode, ensemble_probabilities, threshold
        )

    weight_rows = []
    for class_index, label in enumerate(labels):
        row: dict[str, object] = {
            "label": label,
            "threshold": fitted_balanced[class_index],
            "submission_logit_bias": -float(
                _safe_logit(fitted_balanced[class_index])
            ),
            "f1_threshold": ensemble_thresholds[class_index],
            "crg_threshold": fitted_crg[class_index],
            "crg_fraction": crg_fraction,
        }
        for model_index, name in enumerate(names):
            row[f"weight::{name}"] = weights[class_index, model_index]
        weight_rows.append(row)

    split_frame = frame[["case_id", "patient_id", "family_id"]].copy()
    split_frame["subset"] = ""
    split_frame.loc[calibration, "subset"] = "calibration"
    split_frame.loc[evaluation, "subset"] = "evaluation"
    if (split_frame["subset"] == "").any():
        raise AssertionError("Unassigned cases in calibration split")

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "single_model_thresholds.csv", index=False
    )
    pd.DataFrame(uniform_parameter_rows).to_csv(
        output_dir / "uniform_ensemble_parameters.csv", index=False
    )
    pd.DataFrame(weight_rows).to_csv(
        output_dir / "ensemble_parameters.csv", index=False
    )
    split_frame.to_csv(output_dir / "patient_split.csv", index=False)

    metadata = {
        "prediction_files": [str(Path(path).resolve()) for path in args.predictions],
        "model_names": names,
        "seed": int(args.seed),
        "calibration_fraction": float(args.calibration_fraction),
        "calibration_cases": int(len(calibration)),
        "evaluation_cases": int(len(evaluation)),
        "calibration_patients": int(len(set(patient_ids[calibration]))),
        "evaluation_patients": int(len(set(patient_ids[evaluation]))),
        "patient_overlap": 0,
        "weight_resolution": int(args.weight_resolution),
        "crg_fraction": crg_fraction,
        "crg_implementation": "forithmus/VLM3D-Dockers abnclass_evaluation",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nPatient-safe calibration outputs written to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--config", required=True)
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--batch-size", type=int, default=1)
    prediction.add_argument("--num-workers", type=int, default=4)
    prediction.add_argument(
        "--limit", type=int, default=None,
        help="Optional validation-volume limit for smoke tests",
    )
    prediction.add_argument("--device", default="cuda")
    prediction.add_argument("--rank", type=int, default=None)
    prediction.add_argument("--world-size", type=int, default=None)
    prediction.set_defaults(func=predict)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--parts", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=merge_predictions)

    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--predictions", nargs="+", required=True)
    calibration.add_argument("--names", nargs="+", default=None)
    calibration.add_argument("--output-dir", required=True)
    calibration.add_argument("--calibration-fraction", type=float, default=0.5)
    calibration.add_argument("--seed", type=int, default=2026)
    calibration.add_argument("--weight-resolution", type=int, default=10)
    calibration.add_argument(
        "--crg-fraction", type=float, default=0.45,
        help="Logit-space interpolation from F1 (0) to CRG (1) thresholds",
    )
    calibration.set_defaults(func=calibrate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
