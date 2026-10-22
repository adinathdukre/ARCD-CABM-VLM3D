import torch

from model_arch.anymc3d import comorbidity_adjusted_positive_weights


def test_single_positive_preserves_global_weight():
    targets = torch.tensor([[0.0, 1.0, 0.0]])
    weights = torch.tensor([0.5, 1.5, 2.0])
    result = comorbidity_adjusted_positive_weights(targets, weights, strength=0.5)
    torch.testing.assert_close(result, torch.tensor([[1.0, 1.5, 1.0]]))


def test_comorbidity_shifts_gradient_toward_tail_label():
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    weights = torch.tensor([0.5, 1.0, 2.0])
    result = comorbidity_adjusted_positive_weights(
        targets, weights, strength=1.0, maximum_ratio=2.0
    )
    assert result[0, 2] > weights[2]
    assert result[0, 0] < weights[0]
    assert result[0, 1] == 1.0


def test_zero_positive_scan_keeps_unit_weights():
    targets = torch.zeros(1, 3)
    weights = torch.tensor([0.5, 1.0, 2.0])
    result = comorbidity_adjusted_positive_weights(targets, weights, strength=0.5)
    torch.testing.assert_close(result, torch.ones_like(result))
