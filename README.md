# ARCD-CABM for VLM3D 2026 CT Abnormality Classification

Team **genmi**, Mohamed bin Zayed University of Artificial Intelligence (MBZUAI).

ARCD-CABM (Acquisition-Routed Consensus Distillation and Comorbidity-Aware Balanced Margins) is a parameter-efficient 3D chest CT abnormality classifier for the VLM3D 2026 CT Abnormality Classification task (VLM3D-CT-ABNORMALITY). It labels a single 3D chest CT volume with 18 abnormalities.

> **Status:** code, configuration files and model weights are being uploaded. Items marked *(to be uploaded)* will be filled in as files are added.

## Team

| Member | Affiliation |
|---|---|
| Adinath Dukre (contact) | Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) |
| Imran Razzak | Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) |
| Yifan Lu | Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) |

## Leaderboard submissions

All submissions are to the Main leaderboard of VLM3D-CT-ABNORMALITY.

| Date | Submission ID | Model | Hidden-test AUROC | F1 | CRG | Precision | Recall |
|---|---|---|---|---|---|---|---|
| 2026-08-30 | e3e44633-02d1-4a63-8a24-156b187cf5aa | ARCD-CABM (final) | 0.8363 | 0.5839 | 0.5374 | 0.4870 | 0.7527 |
| 2026-08-17 | a267863e-c973-423d-ae57-59077933a941 | CRS-ASL (precursor) | 0.8323 | 0.5788 | 0.5316 | 0.4928 | 0.7370 |

## Methodology

**Backbone and pretrained model.** For each volume, 96 uniformly spaced axial slices are resized to 224x224 with fixed CT windows and encoded by a frozen DINOv2 ViT-B/14 (public pretrained weights). Rank-8 LoRA adapters (scale 16) modify the attention and patch-projection layers. Eighteen learned disease queries attend over the slice embeddings and give one logit per abnormality. The final model has 480,882 trainable parameters.

**Cross-reconstruction selective asymmetric loss (CRS-ASL).** Training samples two reconstructions of the same examination where available. The negative term of the asymmetric loss is down-weighted only when both reconstructions contradict an observed negative; positive supervision keeps full weight.

**Acquisition-Routed Consensus Distillation (ARCD).** A small MLP takes 13 bounded acquisition descriptors (kernel, spacing, slice count, matrix, diameter, CTDI, manufacturer) plus detached branch logits and forms a disease-wise gate over the two reconstructions. The gated consensus teacher is distilled into both branches (temperature 1.5, weight 0.2), and a routing term (weight 0.05) directs confident positive evidence to the more informative reconstruction. Metadata and the second reconstruction are used only during training.

**Comorbidity-Aware Balanced Margins (CABM).** Rare-positive logit margins (maximum 0.35, scaled by n^-1/4), deferred effective-number reweighting (beta 0.9999, from 50% of training), within-scan comorbidity reweighting (exponent 0.5, ratio clip 1.75) and disease-head norm balancing (weight 0.01).

**Inference.** One NIfTI volume passes through a single branch, without metadata or a paired reconstruction, and yields 18 probabilities. Per-class thresholds interpolate F1- and CRG-oriented thresholds (weights 0.55 and 0.45) fitted on a patient-disjoint validation subset.

**Difference between submissions.** The 2026-08-17 submission is the CRS-ASL precursor (same backbone and disease queries, without ARCD and CABM). The 2026-08-30 submission is the final ARCD-CABM model.

## Data access and preprocessing

- Data: challenge-provided CT-RATE only (no private, external or synthetic data). Access through the official CT-RATE release, subject to its license.
- After the supplied no-chest exclusions: 46,397 training volumes in 24,128 scan families; 3,001 validation volumes from 1,304 patients (patient-disjoint public split).
- Targets: 18 report-derived scan-level abnormality labels.
- Preprocessing: 96 uniformly spaced axial slices, resized to 224x224, fixed CT intensity windows.
- Acquisition metadata from the CT-RATE metadata files is used for training only.

## Training and evaluation settings

| Setting | Value |
|---|---|
| Backbone | DINOv2 ViT-B/14, frozen |
| LoRA rank / scale | 8 / 16 |
| Disease queries | 18 |
| ASL (gamma+, gamma-) / negative clipping | (0, 4) / 0.05 |
| Distillation (weight, T) | (0.2, 1.5) |
| Routing (weight, delta) | (0.05, 0.25) |
| Max margin / effective beta / DRW start | 0.35 / 0.9999 / 50% |
| Comorbidity (s, r) / head balance | (0.5, 1.75) / 0.01 |
| Optimizer | AdamW (0.9, 0.999) |
| Learning rate | LoRA 5e-6, heads 5e-5 |
| Precision / batch size | bfloat16 / 1 per process |
| Schedule | Sequential warm starts: baseline, CRS-ASL, ACER, ARCD (2 epochs, seed 2033), CABM (2 epochs, seed 2053) |

Evaluation metrics: macro AUROC, AUPRC, F1, Brier score, ECE and CRG.

## Installation and usage

*(to be uploaded: environment file and exact commands)*

- Input: one 3D chest CT volume in NIfTI format (`.nii.gz`).
- Output: an 18-length vector of abnormality probabilities and the corresponding binary predictions after per-class thresholds, in the challenge's required format.
- The submitted system is a Docker container following the VLM3D submission interface.

## Configurations and weights

| Submission | Config | Weights |
|---|---|---|
| 2026-08-30 ARCD-CABM | *(to be uploaded)* | [Hugging Face](https://huggingface.co/adidukrembzuai/ARCD-CABM-VLM3D) *(to be uploaded)* |
| 2026-08-17 CRS-ASL | *(to be uploaded)* | [Hugging Face](https://huggingface.co/adidukrembzuai/ARCD-CABM-VLM3D) *(to be uploaded)* |

## Results on public validation (3,001 scans)

| Method | AUROC | AUPRC | F1@0.5 | Brier | ECE |
|---|---|---|---|---|---|
| CRS-ASL | 0.86050 | 0.62322 | 0.55428 | 0.18293 | 0.27441 |
| ACER | 0.86178 | 0.62724 | 0.55332 | 0.18548 | 0.28228 |
| ARCD-CABM | 0.86401 | 0.63181 | 0.54596 | 0.19153 | 0.29673 |

- 16 of 18 class AUROCs improved over ACER. Paired patient bootstrap (500 replicates): +0.00223 AUROC, 95% interval +0.00136 to +0.00312.
- Gain positive in all 11 retained acquisition strata (+0.00014 to +0.00369).
- Patient-disjoint balanced thresholds: macro F1 0.6096, CRG 0.5636.

## Limits and limitations

- Final ARCD and CABM training used one seed each; the development ladder is warm-started rather than a factorial ablation, so component effects are not causally isolated.
- Public validation is single-institution and was observed repeatedly during development; per-case hidden-test predictions were not available.
- At a fixed 0.5 threshold the final model is less calibrated than its precursors; threshold fitting is required.
- Gains are not uniform: cardiomegaly and lung opacity AUROC decreased slightly, and only 5 of 18 class-wise intervals exclude zero.
- ARCD does not make paired-reconstruction predictions more consistent.
- Acquisition metadata used in training may encode scanner or site shortcuts.
- Labels are report-derived and may omit findings; disease-query attention is slice-level, not lesion localization.
- Research use only. This is not a diagnostic device.

## References

- Hamamci et al. Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography. Nature Biomedical Engineering, 2026. (CT-RATE)
- Hamamci et al. CRG Score: A Distribution-Aware Clinical Metric for Radiology Report Generation. arXiv:2505.17167, 2025.
- Liu et al. Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification. CVPR 2026. (AnyMC3D)
- Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR 2024.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Ridnik et al. Asymmetric Loss for Multi-Label Classification. ICCV 2021.
- Cao et al. Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. NeurIPS 2019.
- Cui et al. Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.

## Usage terms

Models are trained on CT-RATE, which is released under CC BY-NC-SA. Code and weights in this repository are for non-commercial research use under the same terms. DINOv2 weights remain under their original license.
