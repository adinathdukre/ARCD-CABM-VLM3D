#!/usr/bin/env python3
"""Extract real embeddings, slice attention, relation attention, and ACER gates.

The resulting NPZ is intentionally compact and contains only public-validation
derived quantities needed for manuscript figures.  It never fabricates spatial
saliency: AnyMC3D exposes disease-specific *slice* attention, so that is what is
stored and visualized.
"""
from __future__ import annotations

import argparse
import gc
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ct_rate_calibrate_ensemble import _load_model, patient_id_from_case_id
from data_modules.data_augmentation import build_val_transforms


ROOT = Path(__file__).resolve().parent
MODEL_SPECS = OrderedDict(
    [
        (
            "Controlled baseline",
            (
                ROOT / "checkpoints/ct_rate_controlled_baseline_dinov2_vitb_96s_stage1_v2_fold0/config.yaml",
                ROOT / "checkpoints/ct_rate_controlled_baseline_dinov2_vitb_96s_stage1_v2_fold0/epoch=04-val_auroc=0.8492.ckpt",
            ),
        ),
        (
            "ADQ-MR",
            (
                ROOT / "checkpoints/ct_rate_adq_mr_dinov2_vitb_96s_stage1_v2_fold0/config.yaml",
                ROOT / "checkpoints/ct_rate_adq_mr_dinov2_vitb_96s_stage1_v2_fold0/epoch=04-val_auroc=0.8569.ckpt",
            ),
        ),
        (
            "CDRA",
            (
                ROOT / "checkpoints/ct_rate_adq_cdra_dinov2_vitb_96s_stage2_v1_fold0/config.yaml",
                ROOT / "checkpoints/ct_rate_adq_cdra_dinov2_vitb_96s_stage2_v1_fold0/epoch=00-val_auroc=0.8575.ckpt",
            ),
        ),
        (
            "CRS-ASL",
            (
                ROOT / "checkpoints/ct_rate_adq_crs_asl_dinov2_vitb_96s_stage2_v1_fold0/config.yaml",
                ROOT / "checkpoints/ct_rate_adq_crs_asl_dinov2_vitb_96s_stage2_v1_fold0/epoch=00-val_auroc=0.8603.ckpt",
            ),
        ),
        (
            "Practical ACER",
            (
                Path(
                    "<SCRATCH_ROOT>/Med-SenseNova-Vision/artifacts/"
                    "anymc3d-classification/ct_rate_acer_crs_dinov2_vitb_96s_v1_fold0/config.yaml"
                ),
                Path(
                    "<SCRATCH_ROOT>/Med-SenseNova-Vision/artifacts/"
                    "anymc3d-classification/ct_rate_acer_crs_dinov2_vitb_96s_v1_fold0/"
                    "epoch=01-val_auroc=0.8616.ckpt"
                ),
            ),
        ),
    ]
)

DISPLAY_LABELS = [
    "Cardiomegaly",
    "Hiatal hernia",
    "Emphysema",
    "Pleural effusion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "<SCRATCH_ROOT>/paper_manuscript_stage/"
            "data/model_visuals.npz"
        ),
    )
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--gate-pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="Checkpoint subset to extract; defaults to all five configurations.",
    )
    return parser.parse_args()


def build_validation_dataset(config_path: Path):
    cfg = OmegaConf.load(config_path)
    dm = instantiate(
        cfg.data.module,
        fold=int(cfg.data.module.get("fold", 0)),
        batch_size=1,
        num_workers=0,
    )
    dm.eval_transform = build_val_transforms()
    dataset = dm._dataset("val")
    index_by_case = {record[0]: i for i, record in enumerate(dataset.items)}
    if len(index_by_case) != len(dataset):
        raise RuntimeError("Validation case identifiers are not unique")
    return cfg, dataset, index_by_case


def select_cases(
    predictions: pd.DataFrame,
    index_by_case: dict[str, int],
    label_names: list[str],
    per_class: int,
    seed: int,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Select a patient-unique, balanced subset without prediction cherry-picking."""
    rng = np.random.default_rng(seed)
    chosen_ids: list[str] = []
    chosen_group: list[int] = []
    used_patients: set[str] = set()
    target_columns = [f"target::{name}" for name in DISPLAY_LABELS]
    if any(column not in predictions for column in target_columns):
        raise ValueError("Archived predictions are missing display-label targets")

    for class_index, label in enumerate(DISPLAY_LABELS):
        mask = predictions[f"target::{label}"].to_numpy(dtype=int) == 1
        # Give each selected point one unambiguous color among the four display
        # labels. Other CT-RATE abnormalities may still co-occur.
        mask &= predictions[target_columns].sum(axis=1).to_numpy(dtype=int) == 1
        candidates = sorted(
            case_id
            for case_id in predictions.loc[mask, "case_id"].astype(str)
            if case_id in index_by_case
        )
        rng.shuffle(candidates)
        accepted = 0
        for case_id in candidates:
            patient_id = patient_id_from_case_id(case_id)
            if patient_id in used_patients:
                continue
            chosen_ids.append(case_id)
            chosen_group.append(class_index)
            used_patients.add(patient_id)
            accepted += 1
            if accepted == per_class:
                break
        if accepted < per_class:
            raise RuntimeError(
                f"Only {accepted} patient-unique cases available for {label}; "
                f"requested {per_class}"
            )

    # Keep grouped order deterministic for plotting and add the exact targets.
    target_lookup = predictions.set_index("case_id")
    targets = np.asarray(
        [
            [int(target_lookup.loc[case_id, f"target::{name}"]) for name in label_names]
            for case_id in chosen_ids
        ],
        dtype=np.uint8,
    )
    return chosen_ids, np.asarray(chosen_group, dtype=np.int64), DISPLAY_LABELS


def select_qualitative_cases(
    predictions: pd.DataFrame,
    index_by_case: dict[str, int],
) -> list[str]:
    """Choose one deterministic, high-confidence true positive per display label."""
    selected: list[str] = []
    used_patients: set[str] = set()
    target_columns = [f"target::{name}" for name in DISPLAY_LABELS]
    for label in DISPLAY_LABELS:
        frame = predictions[
            (predictions[f"target::{label}"] == 1)
            & (predictions[target_columns].sum(axis=1) == 1)
        ].sort_values([f"probability::{label}", "case_id"], ascending=[False, True])
        for case_id in frame["case_id"].astype(str):
            patient_id = patient_id_from_case_id(case_id)
            if case_id in index_by_case and patient_id not in used_patients:
                selected.append(case_id)
                used_patients.add(patient_id)
                break
    if len(selected) != len(DISPLAY_LABELS):
        raise RuntimeError("Could not select all qualitative exemplars")
    return selected


def spaced_peaks(weights: np.ndarray, count: int = 4, distance: int = 7) -> np.ndarray:
    order = np.argsort(weights)[::-1]
    peaks: list[int] = []
    for index in order:
        if all(abs(int(index) - existing) >= distance for existing in peaks):
            peaks.append(int(index))
        if len(peaks) == count:
            break
    if len(peaks) < count:
        for index in order:
            if int(index) not in peaks:
                peaks.append(int(index))
            if len(peaks) == count:
                break
    return np.asarray(sorted(peaks), dtype=np.int64)


def build_gate_pairs(dataset, count: int, seed: int) -> list[tuple[str, str]]:
    families: dict[str, list[str]] = defaultdict(list)
    for case_id, _, _ in dataset.items:
        family = "_".join(case_id.split("_")[:3])
        families[family].append(case_id)
    candidates = [tuple(sorted(ids)[:2]) for ids in families.values() if len(ids) >= 2]
    candidates = sorted(candidates)
    rng = np.random.default_rng(seed + 17)
    rng.shuffle(candidates)
    return candidates[:count]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    selected_model_specs = OrderedDict(
        (name, MODEL_SPECS[name]) for name in args.models
    )

    base_config = MODEL_SPECS["Controlled baseline"][0]
    base_cfg, dataset, index_by_case = build_validation_dataset(base_config)
    label_names = list(base_cfg.data.module.label_cols)
    prediction_path = ROOT / "calibration/predictions/acer_v1.csv"
    predictions = pd.read_csv(prediction_path)
    selected_ids, selected_groups, display_labels = select_cases(
        predictions,
        index_by_case,
        label_names,
        args.per_class,
        args.seed,
    )
    qualitative_ids = select_qualitative_cases(predictions, index_by_case)
    qualitative_label_indices = np.asarray(
        [label_names.index(name) for name in display_labels], dtype=np.int64
    )
    selected_targets = np.stack(
        [dataset.items[index_by_case[case_id]][2] for case_id in selected_ids]
    ).astype(np.uint8)

    artifacts: dict[str, np.ndarray] = {
        "model_names": np.asarray(list(selected_model_specs), dtype="U32"),
        "label_names": np.asarray(label_names, dtype="U64"),
        "display_labels": np.asarray(display_labels, dtype="U32"),
        "case_ids": np.asarray(selected_ids, dtype="U32"),
        "group_ids": selected_groups,
        "targets": selected_targets,
        "qualitative_case_ids": np.asarray(qualitative_ids, dtype="U32"),
        "qualitative_label_indices": qualitative_label_indices,
    }

    all_embeddings: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    relation_sum = None
    relation_count = 0
    qualitative_images = []
    qualitative_attention = []
    qualitative_slices = []
    qualitative_probabilities = []

    for model_index, (model_name, (config_path, checkpoint_path)) in enumerate(
        selected_model_specs.items()
    ):
        print(f"\n=== {model_name} ===", flush=True)
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing model artifact: {config_path}, {checkpoint_path}")
        cfg, module = _load_model(config_path, checkpoint_path)
        module.to(device).eval()
        encoder = module.model.encoders[str(cfg.model.modalities[0])]
        embeddings: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []

        for case_index, case_id in enumerate(selected_ids):
            volume, _, loaded_id = dataset[index_by_case[case_id]]
            if loaded_id != case_id:
                raise RuntimeError(f"Dataset alignment failure: {loaded_id} != {case_id}")
            volume_device = volume.unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                patient_embedding, attention = encoder(volume_device)
                if module.model.use_cross_disease_relation:
                    patient_embedding, relation_attention = module.model.relation_adapter(
                        patient_embedding
                    )
                    current = relation_attention.float().mean(dim=(0, 1)).cpu().numpy()
                    relation_sum = current if relation_sum is None else relation_sum + current
                    relation_count += 1
                logits = module.model.classifier(patient_embedding)
            pooled = (
                patient_embedding.float().mean(dim=1)
                if patient_embedding.ndim == 3
                else patient_embedding.float()
            )
            embeddings.append(pooled[0].cpu().numpy().astype(np.float32))
            probabilities.append(torch.sigmoid(logits.float())[0].cpu().numpy())

            if (case_index + 1) % 10 == 0:
                print(
                    f"{model_name}: {case_index + 1}/{len(selected_ids)} cases",
                    flush=True,
                )

        all_embeddings.append(np.stack(embeddings))
        all_probabilities.append(np.stack(probabilities))

        if model_name == "Practical ACER":
            # Qualitative exemplars are selected independently of the balanced
            # embedding subset.  Run them explicitly: relying on accidental
            # overlap between the two selections previously produced empty
            # attention arrays at serialization time.
            for exemplar_index, case_id in enumerate(qualitative_ids):
                volume, _, loaded_id = dataset[index_by_case[case_id]]
                if loaded_id != case_id:
                    raise RuntimeError(
                        f"Dataset alignment failure: {loaded_id} != {case_id}"
                    )
                volume_device = volume.unsqueeze(0).to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    patient_embedding, attention = encoder(volume_device)
                    logits = module.model.classifier(patient_embedding)
                disease_index = int(qualitative_label_indices[exemplar_index])
                attn = attention[0].float().cpu().numpy()
                slice_indices = spaced_peaks(attn[disease_index])
                images = np.stack(
                    [
                        np.rint(
                            volume[:, :, :, slice_index].clamp(0, 1).numpy() * 255.0
                        ).astype(np.uint8)
                        for slice_index in slice_indices
                    ]
                )
                qualitative_images.append((exemplar_index, images))
                qualitative_attention.append((exemplar_index, attn.astype(np.float32)))
                qualitative_slices.append((exemplar_index, slice_indices))
                qualitative_probabilities.append(
                    (exemplar_index, torch.sigmoid(logits.float())[0].cpu().numpy())
                )
                print(
                    f"Practical ACER attention: {exemplar_index + 1}/"
                    f"{len(qualitative_ids)} exemplars",
                    flush=True,
                )

            # Rebuild validation from the ACER config so the 13 acquisition
            # features are populated; the common embedding dataset intentionally
            # comes from the baseline config and has no metadata side channel.
            _, gate_dataset, gate_index_by_case = build_validation_dataset(config_path)
            gate_pairs = build_gate_pairs(gate_dataset, args.gate_pairs, args.seed)
            gate_weights = []
            gate_pair_probabilities = []
            gate_metadata = []
            for pair_index, (case_a, case_b) in enumerate(gate_pairs):
                volume_a, _, _ = gate_dataset[gate_index_by_case[case_a]]
                volume_b, _, _ = gate_dataset[gate_index_by_case[case_b]]
                volume_a = volume_a.unsqueeze(0).to(device)
                volume_b = volume_b.unsqueeze(0).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    embedding_a, _ = encoder(volume_a)
                    embedding_b, _ = encoder(volume_b)
                    logits_a = module.model.classifier(embedding_a)
                    logits_b = module.model.classifier(embedding_b)
                    metadata_a = gate_dataset._metadata_tensor(case_a).unsqueeze(0).to(device)
                    metadata_b = gate_dataset._metadata_tensor(case_b).unsqueeze(0).to(device)
                    _, weights = module.acquisition_gate(
                        logits_a, logits_b, metadata_a, metadata_b
                    )
                gate_weights.append(weights[0].float().cpu().numpy())
                gate_pair_probabilities.append(
                    np.stack(
                        [
                            torch.sigmoid(logits_a.float())[0].cpu().numpy(),
                            torch.sigmoid(logits_b.float())[0].cpu().numpy(),
                        ]
                    )
                )
                gate_metadata.append(
                    np.stack(
                        [metadata_a[0].float().cpu().numpy(), metadata_b[0].float().cpu().numpy()]
                    )
                )
                print(
                    f"Practical ACER gates: {pair_index + 1}/{len(gate_pairs)} pairs",
                    flush=True,
                )
            artifacts["gate_pair_ids"] = np.asarray(gate_pairs, dtype="U32")
            artifacts["gate_weights"] = np.stack(gate_weights).astype(np.float32)
            artifacts["gate_pair_probabilities"] = np.stack(
                gate_pair_probabilities
            ).astype(np.float32)
            artifacts["gate_metadata"] = np.stack(gate_metadata).astype(np.float32)

        del module, encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    artifacts["embeddings"] = np.stack(all_embeddings).astype(np.float32)
    artifacts["probabilities"] = np.stack(all_probabilities).astype(np.float32)
    if relation_sum is not None:
        artifacts["relation_attention"] = (relation_sum / relation_count).astype(
            np.float32
        )

    def ordered(values):
        return np.stack([value for _, value in sorted(values, key=lambda item: item[0])])

    artifacts["qualitative_images"] = ordered(qualitative_images)
    artifacts["qualitative_attention"] = ordered(qualitative_attention)
    artifacts["qualitative_slice_indices"] = ordered(qualitative_slices)
    artifacts["qualitative_probabilities"] = ordered(qualitative_probabilities)
    np.savez_compressed(output, **artifacts)
    print(f"Wrote {output} ({output.stat().st_size / 2**20:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
