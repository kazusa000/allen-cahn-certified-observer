import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_three_sensor_dynamics_joint import (
    THREE_SENSOR_INTERVALS,
    VARIANTS,
    _build_variant_models,
    _coarse_success_gates,
    _identity_certificate,
    _input_direction_components,
    _ratio_summary,
    _select_joint_variant,
)

from allen_cahn_certified_observer import AllenCahnGrid, local_average_matrix


def _rollout(value: float = 1.0) -> dict[str, object]:
    return {
        "by_nu": {
            key: {"terminal_error_mass_median": value}
            for key in ("0.005", "0.010", "0.020")
        }
    }


def _result(
    defect_rms: float,
    *,
    defect_p95: float | None = None,
    margin: float = 0.01,
    terminal: float = 1.0,
) -> dict[str, object]:
    return {
        "validation": {
            "defect": {
                "rms": defect_rms,
                "p95": defect_rms if defect_p95 is None else defect_p95,
            },
            "contraction": {"requested_margin_min": margin},
        },
        "structure": {"passed": True},
        "validation_rollout": _rollout(terminal),
    }


def test_three_sensor_intervals_match_frozen_geometry_and_total_length() -> None:
    assert np.mean(THREE_SENSOR_INTERVALS, axis=1) == pytest.approx(
        [0.2, 0.5, 0.8]
    )
    assert np.sum(
        THREE_SENSOR_INTERVALS[:, 1] - THREE_SENSOR_INTERVALS[:, 0]
    ) == pytest.approx(0.2)


def test_ratio_summary_reports_rms_and_upper_tail() -> None:
    summary = _ratio_summary(np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert summary["count"] == 4
    assert summary["rms"] == pytest.approx(np.sqrt(7.5))
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["max"] == pytest.approx(4.0)


def test_identity_certificate_exactly_satisfies_input_direction_constraint() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    batch_size = 4

    class FixedGain(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(7)
            self.register_buffer(
                "value",
                torch.randn((grid.n, matrix.shape[0]), generator=generator),
            )

        def forward(self, features: object, nu_indices: object) -> object:
            del nu_indices
            return self.value[None, :, :].expand(features.shape[0], -1, -1)

    generator = torch.Generator().manual_seed(8)
    states = torch.randn((batch_size, grid.n), generator=generator)
    estimates = states + 0.1 * torch.randn(
        (batch_size, grid.n), generator=generator
    )
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32)
    samples = {
        "states": states,
        "estimates": estimates,
        "measurements": states @ matrix_tensor.T,
        "nus": torch.as_tensor([0.005, 0.010, 0.020, 0.005]),
        "nu_indices": torch.as_tensor([0, 1, 2, 0]),
    }
    loss, ratios = _input_direction_components(
        torch,
        FixedGain(),
        _identity_certificate(torch),
        samples,
        matrix_tensor,
        torch.arange(batch_size),
        grid,
        create_graph=True,
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-12)
    assert torch.max(torch.abs(ratios)).item() == pytest.approx(0.0, abs=1.0e-12)


def test_variant_builder_freezes_exact_ablation_parameters() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    base_gains = np.zeros((3, grid.n, matrix.shape[0]))
    base_transforms = np.repeat(np.eye(grid.n)[None, :, :], 3, axis=0)

    b_only_gain, b_only_certificate = _build_variant_models(
        torch,
        grid,
        matrix,
        base_gains,
        base_transforms,
        VARIANTS["train-B__identity-T"],
        gain_trust_ratio=0.25,
        certificate_log_scale=0.2231435513,
    )
    t_only_gain, t_only_certificate = _build_variant_models(
        torch,
        grid,
        matrix,
        base_gains,
        base_transforms,
        VARIANTS["fixed-B__train-T"],
        gain_trust_ratio=0.25,
        certificate_log_scale=0.2231435513,
    )

    assert all(parameter.requires_grad for parameter in b_only_gain.parameters())
    assert list(b_only_certificate.parameters()) == []
    assert all(
        not parameter.requires_grad for parameter in t_only_gain.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in t_only_certificate.parameters()
    )


def test_ode_direction_variant_requires_ten_percent_defect_reduction() -> None:
    fixed_rollout = _rollout()
    native = _result(1.0)

    selected, audit = _select_joint_variant(
        native, _result(0.91), fixed_rollout
    )
    assert selected == "joint-native"
    assert not audit["ode_direction_selected"]

    selected, audit = _select_joint_variant(
        native, _result(0.89), fixed_rollout
    )
    assert selected == "joint-ode-direction"
    assert audit["ode_direction_selected"]


def test_coarse_success_gate_requires_T_and_joint_value() -> None:
    fixed = {
        "validation": {"defect": {"rms": 1.0, "p95": 2.0}},
        "validation_rollout": _rollout(),
    }
    passed = _coarse_success_gates(
        _result(0.70, defect_p95=1.60),
        fixed,
        _result(0.90),
        _result(0.80),
    )
    failed_T_value = _coarse_success_gates(
        _result(0.70, defect_p95=1.60),
        fixed,
        _result(0.80),
        _result(0.80),
    )

    assert passed["all_passed"]
    assert not failed_T_value["value"]["T_value_vs_B_only_20pct"]
    assert not failed_T_value["all_passed"]
