import numpy as np

from allen_cahn_certified_observer import AllenCahnGrid, local_average_matrix
from allen_cahn_certified_observer.observer import (
    CausalNudging,
    observer_rhs_with_truth,
)
from allen_cahn_certified_observer.solver import allen_cahn_rhs


def test_causal_nudging_has_zero_correction_at_zero_innovation() -> None:
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, np.array([[0.25, 0.35], [0.65, 0.75]]))
    observer = CausalNudging(grid, 0.01, matrix, 3.0)
    state = 0.2 * np.sin(np.pi * grid.x)

    assert np.allclose(
        observer.rhs(state, matrix @ state), allen_cahn_rhs(grid, 0.01, state)
    )


def test_offline_harness_generates_current_measurement_only() -> None:
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, np.array([[0.25, 0.35]]))
    observer = CausalNudging(grid, 0.01, matrix, 1.0)
    truth = 0.3 * np.sin(np.pi * grid.x)
    estimate = 0.1 * np.sin(np.pi * grid.x)
    combined = np.concatenate((truth, estimate))
    rhs = observer_rhs_with_truth(0.0, combined, observer=observer)

    assert rhs.shape == combined.shape
    assert np.all(np.isfinite(rhs))
