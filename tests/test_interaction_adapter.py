import numpy as np
import torch

from models.InteractionAdapter import (
    PredictionInteractionAdapter,
    _geographic_adjacency,
)


def _adjacency():
    coordinates = np.asarray(
        [[48.0, 2.0, 0.0], [48.1, 2.0, 0.0], [49.0, 3.0, 0.0]],
        dtype=np.float32,
    )
    return _geographic_adjacency(coordinates, neighbors=2)


def test_all_modes_are_exact_identity_at_initialization():
    prediction = torch.randn(2, 5, 3, 4)
    for mode in ("variable", "station", "both"):
        model = PredictionInteractionAdapter(
            pred_len=5,
            num_variables=4,
            mode=mode,
            adjacency=None if mode == "variable" else _adjacency(),
        )
        assert torch.equal(model(prediction), prediction)


def test_geographic_adjacency_is_row_normalized_without_self_loops():
    adjacency = _adjacency()
    torch.testing.assert_close(adjacency.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(adjacency.diagonal(), torch.zeros(3))


def test_variable_branch_masks_self_scaling_but_keeps_cross_variable_mixing():
    prediction = torch.randn(2, 5, 3, 4)
    model = PredictionInteractionAdapter(
        pred_len=5,
        num_variables=4,
        mode="variable",
        use_calibration=False,
    )
    with torch.no_grad():
        model.variable_weight.fill_(0.0)
        diagonal = torch.arange(4)
        model.variable_weight[:, diagonal, diagonal] = 1.0
    assert torch.equal(model(prediction), prediction)

    with torch.no_grad():
        model.variable_weight[:, 0, 1] = 1.0
    assert not torch.equal(model(prediction), prediction)


def test_only_adapter_parameters_receive_gradients():
    prediction = torch.randn(2, 5, 3, 4)
    model = PredictionInteractionAdapter(
        pred_len=5,
        num_variables=4,
        mode="both",
        adjacency=_adjacency(),
    )
    model(prediction).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
