from argparse import Namespace

import numpy as np
import pandas as pd

from ct_rate_calibrate_ensemble import (
    LOGIT_PREFIX,
    PROB_PREFIX,
    TARGET_PREFIX,
    _sigmoid,
    best_f1_threshold,
    calibrate,
    choose_patient_split,
    fit_ensemble,
)


def test_best_f1_threshold_improves_underconfident_predictions():
    targets = np.asarray([0, 0, 0, 1, 1, 1])
    probabilities = np.asarray([0.02, 0.05, 0.10, 0.20, 0.30, 0.40])
    threshold = best_f1_threshold(targets, probabilities)
    assert threshold == 0.20


def test_patient_split_has_no_patient_leakage():
    patients = np.repeat([f"valid_{index}" for index in range(80)], 2)
    base = np.arange(80)
    patient_targets = np.stack([
        base % 2 == 0,
        base % 3 == 0,
        base % 5 < 2,
    ], axis=1).astype(np.int64)
    targets = np.repeat(patient_targets, 2, axis=0)
    calibration, evaluation = choose_patient_split(
        targets, patients, calibration_fraction=0.5, seed=2026
    )
    assert not set(patients[calibration]) & set(patients[evaluation])
    assert len(calibration) + len(evaluation) == len(targets)


def test_fitted_ensemble_and_end_to_end_outputs(tmp_path):
    rng = np.random.default_rng(2026)
    patient_count, reconstructions, class_count = 80, 2, 3
    patient_ids = np.repeat(
        [f"valid_{index}" for index in range(patient_count)], reconstructions
    )
    case_ids = [
        f"{patient}_a_{reconstruction}"
        for patient in [f"valid_{index}" for index in range(patient_count)]
        for reconstruction in range(reconstructions)
    ]
    family_ids = ["_".join(case_id.split("_")[:3]) for case_id in case_ids]
    patient_targets = rng.binomial(
        1, np.asarray([0.25, 0.35, 0.45]), size=(patient_count, class_count)
    )
    targets = np.repeat(patient_targets, reconstructions, axis=0)
    signal = (targets * 2.0 - 1.0) * 1.2
    logits = np.stack([
        signal + rng.normal(0, 0.8, signal.shape),
        signal + rng.normal(0, 1.1, signal.shape),
    ])

    weights, thresholds = fit_ensemble(targets, logits, resolution=4)
    assert weights.shape == (class_count, 2)
    assert thresholds.shape == (class_count,)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)

    labels = [f"label_{index}" for index in range(class_count)]
    paths = []
    for model_index, model_name in enumerate(("model_a", "model_b")):
        frame = pd.DataFrame({
            "case_id": case_ids,
            "patient_id": patient_ids,
            "family_id": family_ids,
        })
        probabilities = _sigmoid(logits[model_index])
        for class_index, label in enumerate(labels):
            frame[f"{TARGET_PREFIX}{label}"] = targets[:, class_index]
            frame[f"{LOGIT_PREFIX}{label}"] = logits[model_index, :, class_index]
            frame[f"{PROB_PREFIX}{label}"] = probabilities[:, class_index]
        path = tmp_path / f"{model_name}.csv"
        frame.to_csv(path, index=False)
        paths.append(str(path))

    output_dir = tmp_path / "calibration"
    calibrate(Namespace(
        predictions=paths,
        names=["model_a", "model_b"],
        output_dir=str(output_dir),
        calibration_fraction=0.5,
        seed=2026,
        weight_resolution=4,
    ))
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "ensemble_parameters.csv").is_file()
    split = pd.read_csv(output_dir / "patient_split.csv")
    subsets_per_patient = split.groupby("patient_id")["subset"].nunique()
    assert subsets_per_patient.max() == 1
