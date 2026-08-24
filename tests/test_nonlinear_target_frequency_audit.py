import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

import r5_nonlinear_target_conditional_residual_joint as joint
from r5_nonlinear_target_frequency_audit import (
    _classification,
    _frequency_arrays,
    _summarize_frequency_audit,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    dirichlet_sine_basis,
    low_frequency_projector,
)


def test_frequency_arrays_preserve_defect_energy_and_contraction_power() -> None:
    grid = AllenCahnGrid(7)
    projector = low_frequency_projector(grid, 4)
    basis = dirichlet_sine_basis(grid, 6)
    low = np.asarray([basis[:, 0], 0.5 * basis[:, 1]])
    high = np.asarray([0.3 * basis[:, 4], -0.2 * basis[:, 5]])
    transformed = low + high
    transformed_rhs = -0.4 * low + 0.6 * high
    residual = 0.2 * low - 0.7 * high
    errors = 0.5 * transformed
    fields = {
        "errors": errors,
        "transformed": transformed,
        "transformed_rhs": transformed_rhs,
        "target_rhs": transformed_rhs - residual,
        "defect_residual": residual,
        "formal_margin_rate": np.asarray([-0.1, 0.2]),
    }

    arrays = _frequency_arrays(grid, fields, projector)

    assert np.allclose(
        arrays["defect_energy_total"],
        arrays["defect_energy_low"] + arrays["defect_energy_high"],
        atol=1.0e-14,
    )
    assert np.allclose(
        arrays["power_total"],
        arrays["power_low"] + arrays["power_high"],
        atol=1.0e-14,
    )
    assert np.allclose(
        arrays["margin_power_total"],
        arrays["margin_power_low"] + arrays["margin_power_high"],
        atol=1.0e-14,
    )


def test_projected_damping_counterfactual_removes_uniform_high_residual() -> None:
    grid = AllenCahnGrid(7)
    projector = low_frequency_projector(grid, 4)
    basis = dirichlet_sine_basis(grid, 5)
    z_high = np.asarray([basis[:, 4], -0.5 * basis[:, 4]])
    lambda_value = joint.LAMBDA_RATIO * joint.NU_VALUE * np.pi**2
    residual = (1.0 + lambda_value) * z_high
    fields = {
        "errors": 0.5 * z_high,
        "transformed": z_high,
        "transformed_rhs": -z_high,
        "target_rhs": -z_high - residual,
        "defect_residual": residual,
        "formal_margin_rate": np.asarray([0.1, 0.1]),
    }

    arrays = _frequency_arrays(grid, fields, projector)
    audit = _summarize_frequency_audit(
        arrays,
        ("case-a", "case-b"),
        np.asarray([0.0, 0.0]),
    )

    assert audit["defect"]["high_normalized_squared_share"] == pytest.approx(1.0)
    assert audit["defect"]["projected_damping_counterfactual"]["summary"][
        "rms"
    ] == pytest.approx(0.0, abs=1.0e-12)


def _fake_audit(
    defect_high_share: float,
    contraction_high_share: float,
    *,
    counterfactual_reduction: float = 0.2,
) -> dict[str, object]:
    defect_high = 100.0 * defect_high_share
    defect_low = 100.0 - defect_high
    margin_high = 10.0 * contraction_high_share
    margin_low = 10.0 - margin_high
    return {
        "defect": {
            "high_normalized_squared_share": defect_high_share,
            "projected_damping_counterfactual": {
                "rms_reduction_fraction": counterfactual_reduction,
                "p95_reduction_fraction": counterfactual_reduction,
            },
        },
        "contraction_power": {
            "failed_sample_negative_margin_burden": {
                "high_share": contraction_high_share
            }
        },
        "attribution_accumulators": {
            "defect_low_normalized_squared_sum": defect_low,
            "defect_high_normalized_squared_sum": defect_high,
            "negative_margin_burden_low": margin_low,
            "negative_margin_burden_high": margin_high,
        },
    }


def test_classification_requires_consensus_and_blocks_low_frequency_failure() -> None:
    high = [_fake_audit(0.7, 0.65) for _ in range(3)]
    decision = _classification(high, projected_target_proof_passed=True)

    assert decision["defect_attribution"] == "high-frequency-dominant"
    assert decision["contraction_attribution"] == "high-frequency-dominant"
    assert decision["recommend_projected_damping_as_next_primary_candidate"]

    low_contraction = [_fake_audit(0.7, 0.3) for _ in range(3)]
    blocked = _classification(
        low_contraction, projected_target_proof_passed=True
    )

    assert blocked["contraction_attribution"] == "low-frequency-dominant"
    assert not blocked["recommend_projected_damping_as_next_primary_candidate"]
