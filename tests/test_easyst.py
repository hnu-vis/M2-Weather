from types import SimpleNamespace

import torch
import torch.nn as nn

from models.EasyST import Model


class _DummyTeacher(nn.Module):
    def __init__(self, pred_len: int):
        super().__init__()
        self.pred_len = pred_len
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_enc, *_):
        batch, _, stations, variables = x_enc.shape
        return self.anchor + x_enc.new_zeros(
            batch, self.pred_len, stations, variables
        )


def _config():
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=6,
        pred_len=3,
        num_nodes=4,
        enc_in=3,
        dropout=0.0,
        easyst_embed_dim=4,
        easyst_node_dim=4,
        easyst_time_dim=4,
        easyst_transition_dim=4,
        easyst_num_layers=1,
        easyst_distill_weight=0.3,
        easyst_teacher_delta=0.1,
        easyst_ib_weight=1e-3,
    )


def _model(monkeypatch):
    monkeypatch.setattr(
        Model,
        "_build_teacher",
        lambda self, configs: _DummyTeacher(configs.pred_len),
    )
    return Model(_config())


def test_variables_are_processed_independently_and_equivariantly(monkeypatch):
    model = _model(monkeypatch).eval()
    inputs = torch.randn(2, 6, 4, 3)
    forecast_start = torch.tensor([100, 200])
    permutation = torch.tensor([2, 0, 1])

    prediction = model(inputs, forecast_start=forecast_start)
    permuted_prediction = model(
        inputs[..., permutation], forecast_start=forecast_start
    )

    torch.testing.assert_close(
        permuted_prediction, prediction[..., permutation], rtol=0.0, atol=1e-6
    )


def test_teacher_stays_frozen_and_eval_during_student_training(monkeypatch):
    model = _model(monkeypatch).train()
    assert not model.teacher.training
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())

    inputs = torch.randn(2, 6, 4, 3)
    target = torch.randn(2, 3, 12)
    prediction = model(inputs, forecast_start=torch.tensor([100, 200]))
    flat_prediction = prediction.flatten(start_dim=2)
    base_loss = nn.functional.mse_loss(flat_prediction, target)
    objective = model.training_objective(base_loss, flat_prediction, target)
    objective.backward()

    assert torch.isfinite(objective)
    assert model.regression.weight.grad is not None
    assert model.teacher.anchor.grad is None


def test_checkpoint_round_trip_is_strict(monkeypatch, tmp_path):
    model = _model(monkeypatch).eval()
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint)

    restored = _model(monkeypatch).eval()
    incompatible = restored.load_state_dict(
        torch.load(checkpoint, weights_only=True), strict=True
    )
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

