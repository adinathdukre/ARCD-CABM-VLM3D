import torch

from model_arch.anymc3d import tail_aware_class_parameters


def test_tail_aware_parameters_prioritize_rare_classes():
    margins, weights = tail_aware_class_parameters(
        [100, 1000, 10000], max_margin=0.35, effective_beta=0.9999
    )
    assert torch.isclose(margins[0], torch.tensor(0.35))
    assert margins[0] > margins[1] > margins[2]
    assert weights[0] > weights[1] > weights[2]
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1.0e-6)


def test_tail_aware_parameters_validate_inputs():
    for counts in ([], [10, 0], [10, -1]):
        try:
            tail_aware_class_parameters(counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid counts to fail: {counts}")
