import numpy as np
import pytest

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    build_sylvester_remainder_bridge,
    exact_remainder_batch,
    local_average_matrix,
)

NU = 0.005
ALPHA = 0.1 * NU * np.pi**2
INTERVALS = np.asarray(
    [
        [1.0 / 6.0, 7.0 / 30.0],
        [7.0 / 15.0, 8.0 / 15.0],
        [23.0 / 30.0, 5.0 / 6.0],
    ]
)


@pytest.mark.parametrize("grid_size", [31, 63, 127, 191])
def test_sylvester_bridge_exactly_conjugates_the_linear_closed_loop(
    grid_size: int,
) -> None:
    grid = AllenCahnGrid(grid_size)
    observation = local_average_matrix(grid, INTERVALS)
    bridge = build_sylvester_remainder_bridge(
        grid, NU, observation, alpha=ALPHA
    )
    diagnostics = bridge.diagnostics

    assert diagnostics["positive_linear_mode_count"] == 4
    assert diagnostics["low_observability_rank"] == 4
    assert diagnostics["low_constraint_relative_residual"] < 1.0e-12
    assert np.isfinite(
        diagnostics["full_observation_preservation_relative_residual"]
    )
    assert diagnostics["low_primal_relative_residual"] < 1.0e-12
    assert diagnostics["full_modal_relative_residual"] < 1.0e-12
    assert diagnostics["full_physical_relative_residual"] < 1.0e-12
    assert diagnostics["transform_relative_residual"] < 1.0e-12
    assert diagnostics["inverse_residual_2"] < 1.0e-10
    assert diagnostics["inverse_coordinate_condition_2"] < 1.0e5
    assert diagnostics["target_linear_minimum_decay_rate"] >= ALPHA


def test_exact_remainder_batch_reconstructs_the_transformed_dynamics() -> None:
    grid = AllenCahnGrid(31)
    observation = local_average_matrix(grid, INTERVALS)
    bridge = build_sylvester_remainder_bridge(
        grid, NU, observation, alpha=ALPHA
    )
    generator = np.random.Generator(np.random.PCG64DXSM(17))
    states = 0.4 * generator.normal(size=(32, grid.n))
    errors = 0.1 * generator.normal(size=(32, grid.n))

    arrays = exact_remainder_batch(
        states,
        errors,
        coordinate_transform=bridge.coordinate_transform,
        inverse_coordinate=bridge.inverse_coordinate,
        target_generator=bridge.target_generator,
        closed_loop_generator=bridge.closed_loop_generator,
        alpha=ALPHA,
    )

    assert np.max(arrays["rhs_reconstruction_relative_error"]) < 1.0e-10
    assert np.max(arrays["inverse_reconstruction_relative_error"]) < 1.0e-10
    assert np.max(arrays["rate_additivity_error"]) < 1.0e-10
    assert np.allclose(
        arrays["actual_margin"], arrays["total_rate"] - ALPHA
    )


def test_exact_remainder_batch_identity_recovers_cubic_dissipativity() -> None:
    generator = np.random.Generator(np.random.PCG64DXSM(29))
    states = generator.normal(size=(16, 7))
    errors = generator.normal(size=(16, 7))
    identity = np.eye(7)
    zero = np.zeros((7, 7))

    arrays = exact_remainder_batch(
        states,
        errors,
        coordinate_transform=identity,
        inverse_coordinate=identity,
        target_generator=zero,
        closed_loop_generator=zero,
        alpha=0.1,
    )

    assert np.min(arrays["nonlinear_remainder_rate"]) >= 0.0
    assert np.allclose(arrays["linear_rate"], 0.0)


def test_bridge_rejects_the_wrong_unstable_dimension() -> None:
    grid = AllenCahnGrid(31)
    observation = local_average_matrix(grid, INTERVALS)
    with pytest.raises(ValueError, match="number of positive"):
        build_sylvester_remainder_bridge(
            grid,
            NU,
            observation,
            alpha=ALPHA,
            unstable_dimension=3,
        )
