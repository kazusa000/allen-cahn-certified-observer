import numpy as np
import pytest

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    contiguous_index_blocks,
    local_average_matrix,
    observation_support_mask,
    rightmost_eigenpair,
    sensor_gap_diagnostics,
    similarity_spectral_abscissa_error,
    unobserved_mass_fraction,
    wall_jacobian,
    wall_state,
)

NU = 0.005
INTERVALS = np.asarray(
    [
        [1.0 / 6.0, 7.0 / 30.0],
        [7.0 / 15.0, 8.0 / 15.0],
        [23.0 / 30.0, 5.0 / 6.0],
    ]
)


def test_contiguous_index_blocks_splits_gaps() -> None:
    blocks = contiguous_index_blocks([0, 1, 4, 5, 6, 9])
    assert [block.tolist() for block in blocks] == [[0, 1], [4, 5, 6], [9]]
    assert contiguous_index_blocks([]) == ()
    with pytest.raises(ValueError, match="strictly increasing"):
        contiguous_index_blocks([0, 2, 2])


def test_wall_state_realizes_the_frozen_cubic_damping() -> None:
    grid = AllenCahnGrid(31)
    observation = local_average_matrix(grid, INTERVALS)
    support = observation_support_mask(observation)
    state = wall_state(observation, 3.0)

    assert np.allclose(3.0 * state[support] ** 2, 3.0)
    assert np.all(state[~support] == 0.0)
    assert np.max(np.abs(observation[:, ~support])) == 0.0


def test_fine_grid_contains_a_linearly_unstable_unobserved_gap() -> None:
    grid = AllenCahnGrid(127)
    observation = local_average_matrix(grid, INTERVALS)
    diagnostics = sensor_gap_diagnostics(grid, NU, observation)
    dangerous = diagnostics["most_dangerous_block"]

    assert diagnostics["observation_zero_on_unobserved_max"] == 0.0
    assert dangerous["node_count"] == 28
    assert dangerous["principal_growth_rate"] > 0.03


def test_gap_growth_converges_monotonically_toward_continuum_value() -> None:
    rates = []
    for grid_size in (31, 63, 127, 191):
        grid = AllenCahnGrid(grid_size)
        observation = local_average_matrix(grid, INTERVALS)
        diagnostics = sensor_gap_diagnostics(grid, NU, observation)
        rates.append(
            diagnostics["most_dangerous_block"]["principal_growth_rate"]
        )
    continuum = 1.0 - NU * (np.pi / (7.0 / 30.0)) ** 2

    assert np.all(np.diff(rates) > 0.0)
    assert rates[2] > 0.0
    assert rates[3] > 0.0
    assert abs(rates[3] - continuum) < abs(rates[2] - continuum)


def test_large_sensor_wall_isolates_the_unobserved_gap() -> None:
    grid = AllenCahnGrid(127)
    observation = local_average_matrix(grid, INTERVALS)
    support = observation_support_mask(observation)
    diagnostics = sensor_gap_diagnostics(grid, NU, observation)
    gap_rate = diagnostics["most_dangerous_block"]["principal_growth_rate"]
    gain = np.zeros((grid.n, observation.shape[0]))
    jacobian = wall_jacobian(grid, NU, observation, gain, 1.0e6)
    rate, vector = rightmost_eigenpair(jacobian)

    assert rate == pytest.approx(gap_rate, abs=2.0e-5)
    assert unobserved_mass_fraction(vector, support) > 0.999999


def test_similarity_transform_does_not_change_wall_spectrum() -> None:
    grid = AllenCahnGrid(31)
    observation = local_average_matrix(grid, INTERVALS)
    gain = np.zeros((grid.n, observation.shape[0]))
    jacobian = wall_jacobian(grid, NU, observation, gain, 3.0)
    transform = np.eye(grid.n) + 0.1 * np.diag(
        np.ones(grid.n - 1),
        k=1,
    )

    assert similarity_spectral_abscissa_error(jacobian, transform) < 1.0e-10
