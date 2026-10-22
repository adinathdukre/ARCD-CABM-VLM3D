import torch

from data_modules.ct_rate_dataset import (
    ACQUISITION_FEATURE_DIM,
    _acquisition_features,
)
from model_arch.anymc3d import (
    AcquisitionConditionedGate,
    acquisition_distillation_loss,
    positive_view_routing_loss,
)


def test_acquisition_features_are_bounded_and_fixed_width():
    row = {
        "ConvolutionKernel": "['Br60f', '3']",
        "XYSpacing": "[0.7, 0.7]",
        "ZSpacing": "1.5",
        "NumberofSlices": "320",
        "Rows": "512",
        "ReconstructionDiameter": "420",
        "CTDIvol": "7.8",
        "Manufacturer": "Siemens Healthineers",
    }
    features = _acquisition_features(row)
    assert features.shape == (ACQUISITION_FEATURE_DIM,)
    assert torch.isfinite(torch.from_numpy(features)).all()
    assert features[0] == 0.75
    assert features[-5] == 1.0


def test_gate_prefers_stronger_image_evidence_at_initialization():
    gate = AcquisitionConditionedGate(13, 2, hidden_dim=8, evidence_scale=1.0)
    logits_a = torch.tensor([[3.0, -2.0]])
    logits_b = torch.tensor([[-1.0, 2.0]])
    metadata = torch.zeros(1, 13)
    fused, weights = gate(logits_a, logits_b, metadata, metadata)
    assert weights[0, 0, 0] > weights[0, 0, 1]
    assert weights[0, 1, 1] > weights[0, 1, 0]
    assert fused.shape == logits_a.shape


def test_gate_receives_metadata_gradients():
    gate = AcquisitionConditionedGate(13, 3, hidden_dim=8, evidence_scale=0.0)
    metadata_a = torch.zeros(2, 13)
    metadata_b = torch.arange(13, dtype=torch.float32).repeat(2, 1) / 13.0
    logits_a = torch.ones(2, 3)
    logits_b = -torch.ones(2, 3)
    fused, _ = gate(logits_a, logits_b, metadata_a, metadata_b)
    fused.sum().backward()
    assert gate.scorer[-1].weight.grad is not None
    assert gate.scorer[-1].weight.grad.abs().sum() > 0


def test_distillation_ignores_singleton_pairs_and_has_gradients():
    logits_a = torch.tensor([[2.0, -1.0], [1.0, 1.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, -2.0], [-1.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[2.5, -2.5], [4.0, 4.0]])
    loss = acquisition_distillation_loss(
        logits_a, logits_b, teacher, torch.tensor([True, False])
    )
    loss.backward()
    assert loss > 0
    assert logits_a.grad[0].abs().sum() > 0
    assert logits_b.grad[0].abs().sum() > 0
    assert logits_a.grad[1].abs().sum() == 0
    assert logits_b.grad[1].abs().sum() == 0


def test_positive_routing_selects_the_stronger_positive_view():
    raw = torch.tensor([[[2.0, -2.0], [-2.0, 2.0]]], requires_grad=True)
    weights = torch.softmax(raw, dim=-1)
    logits_a = torch.tensor([[3.0, -1.0]])
    logits_b = torch.tensor([[0.0, 2.0]])
    targets = torch.tensor([[1.0, 1.0]])
    loss = positive_view_routing_loss(
        weights, logits_a, logits_b, targets, torch.tensor([True])
    )
    loss.backward()
    assert loss < 0.05
    assert raw.grad is not None
