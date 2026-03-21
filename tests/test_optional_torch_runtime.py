import numpy as np

import models
import trainer


def _make_spaces():
    return trainer.SearchSpaces(
        hidden_dims=[32, 64],
        learning_rates=[0.001],
        batch_sizes=[32],
        seq_lens=[10],
        num_layers_pool=[1],
        logreg_choices=[{"penalty": "l2", "solver": "lbfgs", "C": 1.0, "class_weight": None}],
        sgd_choices=[{"alpha": 0.001, "penalty": "l2", "class_weight": "balanced"}],
        xgb_param_pool=[],
        window_pool=[[3, 5, 10]],
        xgb_available=False,
        vol_metric_pool=["std"],
        liq_transform_pool=["ratio"],
        buy_pool=[0.55],
        sell_pool=[0.45],
        confirm_pool=[1],
        min_hold_pool=[1],
        max_hold_pool=[15],
        take_profit_pool=[None],
        stop_loss_pool=[None],
        ta_window_pool=[[6, 14]],
        calibration_pool=["none"],
        input_dim=4,
        cuda_available=False,
    )


def test_sample_explore_avoids_dl_models_when_torch_unavailable(monkeypatch):
    monkeypatch.setattr(trainer, "TORCH_AVAILABLE", False)
    spaces = _make_spaces()
    rng = np.random.default_rng(42)

    sampled = {
        trainer._sample_explore(rng, spaces, {"initial_cash": 100000.0})["model_config"]["model_type"]
        for _ in range(50)
    }

    assert sampled <= {"logreg", "sgd"}


def test_has_supported_runtime_rejects_dl_without_torch(monkeypatch):
    monkeypatch.setattr(trainer, "TORCH_AVAILABLE", False)
    spaces = _make_spaces()

    assert trainer._has_supported_runtime("mlp", spaces) is False
    assert trainer._has_supported_runtime("lstm", spaces) is False
    assert trainer._has_supported_runtime("logreg", spaces) is True


def test_get_device_falls_back_to_cpu_placeholder_without_torch(monkeypatch):
    monkeypatch.setattr(models, "TORCH_AVAILABLE", False)

    device = models._get_device()

    assert getattr(device, "type", None) == "cpu"
