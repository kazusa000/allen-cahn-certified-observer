"""Exact low-mode Sylvester coordinates with the true nonlinear remainder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import AllenCahnGrid
from .spectral import dirichlet_laplacian_rates

Array = np.ndarray


@dataclass(frozen=True)
class SylvesterRemainderBridge:
    """One deterministic semidiscrete observer coordinate.

    The coordinate transform maps the physical error to z and its inverse maps
    z back to the physical error.
    """

    grid: AllenCahnGrid
    nu: float
    alpha: float
    unstable_dimension: int
    observation: Array
    modal_basis: Array
    modal_eigenvalues: Array
    modal_gain: Array
    gain: Array
    coordinate_transform: Array
    inverse_coordinate: Array
    target_generator: Array
    closed_loop_generator: Array
    diagnostics: dict[str, float | int]


def _checked_observation(grid: AllenCahnGrid, observation: Array) -> Array:
    matrix = np.asarray(observation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != grid.n:
        raise ValueError("observation must have shape (sensor_count, grid.n)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("observation must be finite")
    return matrix


def _full_sine_basis(grid: AllenCahnGrid) -> Array:
    modes = np.arange(1, grid.n + 1, dtype=float)
    return np.sqrt(2.0 * grid.h) * np.sin(
        np.pi * grid.x[:, None] * modes[None, :]
    )


def _relative_matrix_residual(numerator: Array, *terms: tuple[Array, Array]) -> float:
    denominator = sum(
        float(np.linalg.norm(left, ord="fro") * np.linalg.norm(right, ord=2))
        for left, right in terms
    )
    return float(np.linalg.norm(numerator, ord="fro") / max(denominator, 1.0e-30))


def _solve_low_dual(
    low_system: Array,
    low_observation: Array,
    low_target: Array,
) -> tuple[Array, Array, dict[str, float | int]]:
    dimension = low_system.shape[0]
    sensor_count = low_observation.shape[0]
    column_count = dimension * dimension + sensor_count * dimension
    columns: list[Array] = []
    for index in range(column_count):
        transform_dual = np.zeros((dimension, dimension), dtype=float)
        gain_dual = np.zeros((sensor_count, dimension), dtype=float)
        if index < dimension * dimension:
            transform_dual.flat[index] = 1.0
        else:
            gain_dual.flat[index - dimension * dimension] = 1.0
        dynamics = (
            transform_dual @ low_system.T
            + low_observation.T @ gain_dual
            - low_target.T @ transform_dual
        )
        constraint = transform_dual @ low_observation.T
        columns.append(np.concatenate((dynamics.ravel(), constraint.ravel())))

    coefficient = np.column_stack(columns)
    right_hand_side = np.concatenate(
        (
            np.zeros(dimension * dimension, dtype=float),
            low_observation.T.ravel(),
        )
    )
    solution, _, rank, singular_values = np.linalg.lstsq(
        coefficient, right_hand_side, rcond=None
    )
    split = dimension * dimension
    transform_dual = solution[:split].reshape((dimension, dimension))
    gain_dual = solution[split:].reshape((sensor_count, dimension))
    inverse_low = transform_dual.T
    low_gain = -gain_dual.T

    dynamic_residual = (
        transform_dual @ low_system.T
        + low_observation.T @ gain_dual
        - low_target.T @ transform_dual
    )
    constraint_residual = (
        transform_dual @ low_observation.T - low_observation.T
    )
    effective_minimum = float(singular_values[int(rank) - 1])
    return inverse_low, low_gain, {
        "dual_equation_count": int(coefficient.shape[0]),
        "dual_unknown_count": int(coefficient.shape[1]),
        "dual_rank": int(rank),
        "dual_nullity": int(coefficient.shape[1] - int(rank)),
        "dual_condition_proxy": float(singular_values[0] / effective_minimum),
        "dual_absolute_residual": float(
            np.linalg.norm(coefficient @ solution - right_hand_side)
        ),
        "low_dual_relative_residual": _relative_matrix_residual(
            dynamic_residual,
            (transform_dual, low_system.T),
            (low_observation.T, gain_dual),
            (low_target.T, transform_dual),
        ),
        "low_constraint_relative_residual": float(
            np.linalg.norm(constraint_residual, ord="fro")
            / max(float(np.linalg.norm(low_observation, ord="fro")), 1.0e-30)
        ),
    }


def build_sylvester_remainder_bridge(
    grid: AllenCahnGrid,
    nu: float,
    observation: Array,
    *,
    alpha: float,
    unstable_dimension: int = 4,
) -> SylvesterRemainderBridge:
    """Construct the exact low-shift/full-tail Sylvester coordinate."""

    viscosity = float(nu)
    requested_rate = float(alpha)
    if not np.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("nu must be a positive finite scalar")
    if not np.isfinite(requested_rate) or requested_rate <= 0.0:
        raise ValueError("alpha must be a positive finite scalar")
    if (
        not isinstance(unstable_dimension, int)
        or isinstance(unstable_dimension, bool)
        or not 1 <= unstable_dimension < grid.n
    ):
        raise ValueError("unstable_dimension must satisfy 1 <= value < grid.n")

    observation_matrix = _checked_observation(grid, observation)
    basis = _full_sine_basis(grid)
    basis_residual = float(
        np.linalg.norm(basis.T @ basis - np.eye(grid.n), ord=2)
    )
    eigenvalues = 1.0 - viscosity * dirichlet_laplacian_rates(grid)
    positive_count = int(np.sum(eigenvalues > 1.0e-10))
    if positive_count != unstable_dimension:
        raise ValueError(
            "unstable_dimension must equal the number of positive linear modes"
        )

    modal_observation = observation_matrix @ basis
    low_system = np.diag(eigenvalues[:unstable_dimension])
    stable_eigenvalues = eigenvalues[unstable_dimension:]
    low_observation = modal_observation[:, :unstable_dimension]
    tail_observation = modal_observation[:, unstable_dimension:]
    shift = 1.0 + requested_rate
    low_target = low_system - shift * np.eye(unstable_dimension)

    inverse_low, low_gain, low_diagnostics = _solve_low_dual(
        low_system, low_observation, low_target
    )
    low_closed_loop = low_system - low_gain @ low_observation
    cross_columns = [
        np.linalg.solve(
            low_closed_loop - stable_value * np.eye(unstable_dimension),
            low_gain @ tail_observation[:, column],
        )
        for column, stable_value in enumerate(stable_eigenvalues)
    ]
    cross = np.column_stack(cross_columns)

    tail_dimension = grid.n - unstable_dimension
    inverse_modal = np.block(
        [
            [inverse_low, cross],
            [
                np.zeros((tail_dimension, unstable_dimension), dtype=float),
                np.eye(tail_dimension, dtype=float),
            ],
        ]
    )
    target_modal = np.diag(
        np.concatenate((np.diag(low_target), stable_eigenvalues))
    )
    modal_gain = np.vstack(
        (
            low_gain,
            np.zeros(
                (tail_dimension, observation_matrix.shape[0]), dtype=float
            ),
        )
    )
    closed_loop_modal = (
        np.diag(eigenvalues) - modal_gain @ modal_observation
    )
    coordinate_modal = np.linalg.inv(inverse_modal)

    inverse_coordinate = basis @ inverse_modal @ basis.T
    coordinate_transform = basis @ coordinate_modal @ basis.T
    target_generator = basis @ target_modal @ basis.T
    gain = basis @ modal_gain
    linear_generator = (
        viscosity * grid.laplacian + np.eye(grid.n, dtype=float)
    )
    closed_loop_generator = linear_generator - gain @ observation_matrix

    full_modal_residual = (
        closed_loop_modal @ inverse_modal - inverse_modal @ target_modal
    )
    full_residual = (
        closed_loop_generator @ inverse_coordinate
        - inverse_coordinate @ target_generator
    )
    transform_residual = (
        coordinate_transform @ closed_loop_generator
        - target_generator @ coordinate_transform
    )
    inverse_residual = max(
        float(
            np.linalg.norm(
                coordinate_transform @ inverse_coordinate - np.eye(grid.n),
                ord=2,
            )
        ),
        float(
            np.linalg.norm(
                inverse_coordinate @ coordinate_transform - np.eye(grid.n),
                ord=2,
            )
        ),
    )
    observability = np.vstack(
        [
            low_observation @ np.linalg.matrix_power(low_system, power)
            for power in range(unstable_dimension)
        ]
    )
    observability_singular_values = np.linalg.svd(
        observability, compute_uv=False
    )
    full_observation_residual = float(
        np.linalg.norm(
            observation_matrix @ inverse_coordinate - observation_matrix,
            ord="fro",
        )
        / max(float(np.linalg.norm(observation_matrix, ord="fro")), 1.0e-30)
    )
    inverse_singular_values = np.linalg.svd(
        inverse_coordinate, compute_uv=False
    )
    diagnostics: dict[str, float | int] = {
        **low_diagnostics,
        "basis_orthogonality_residual": basis_residual,
        "positive_linear_mode_count": positive_count,
        "low_observability_rank": int(np.linalg.matrix_rank(observability)),
        "low_observability_min_singular_value": float(
            observability_singular_values[-1]
        ),
        "full_observation_preservation_relative_residual": (
            full_observation_residual
        ),
        "low_primal_relative_residual": _relative_matrix_residual(
            low_closed_loop @ inverse_low - inverse_low @ low_target,
            (low_closed_loop, inverse_low),
            (inverse_low, low_target),
        ),
        "cross_relative_residual": _relative_matrix_residual(
            low_closed_loop @ cross
            - cross @ np.diag(stable_eigenvalues)
            - low_gain @ tail_observation,
            (low_closed_loop, cross),
            (cross, np.diag(stable_eigenvalues)),
            (low_gain, tail_observation),
        ),
        "full_modal_relative_residual": _relative_matrix_residual(
            full_modal_residual,
            (closed_loop_modal, inverse_modal),
            (inverse_modal, target_modal),
        ),
        "full_physical_relative_residual": _relative_matrix_residual(
            full_residual,
            (closed_loop_generator, inverse_coordinate),
            (inverse_coordinate, target_generator),
        ),
        "transform_relative_residual": _relative_matrix_residual(
            transform_residual,
            (coordinate_transform, closed_loop_generator),
            (target_generator, coordinate_transform),
        ),
        "inverse_residual_2": inverse_residual,
        "inverse_coordinate_min_singular_value": float(
            inverse_singular_values[-1]
        ),
        "inverse_coordinate_max_singular_value": float(
            inverse_singular_values[0]
        ),
        "inverse_coordinate_condition_2": float(
            inverse_singular_values[0] / inverse_singular_values[-1]
        ),
        "low_gain_fro_norm": float(np.linalg.norm(low_gain, ord="fro")),
        "cross_block_fro_norm": float(np.linalg.norm(cross, ord="fro")),
        "target_linear_minimum_decay_rate": float(
            -np.max(np.diag(target_modal))
        ),
        "closed_loop_spectral_abscissa": float(
            np.max(np.real(np.linalg.eigvals(closed_loop_modal)))
        ),
    }
    return SylvesterRemainderBridge(
        grid=grid,
        nu=viscosity,
        alpha=requested_rate,
        unstable_dimension=unstable_dimension,
        observation=observation_matrix.copy(),
        modal_basis=basis,
        modal_eigenvalues=eigenvalues,
        modal_gain=modal_gain,
        gain=gain,
        coordinate_transform=coordinate_transform,
        inverse_coordinate=inverse_coordinate,
        target_generator=target_generator,
        closed_loop_generator=closed_loop_generator,
        diagnostics=diagnostics,
    )


def exact_remainder_batch(
    states: Array,
    errors: Array,
    *,
    coordinate_transform: Array,
    inverse_coordinate: Array,
    target_generator: Array,
    closed_loop_generator: Array,
    alpha: float,
) -> dict[str, Array]:
    """Evaluate an exact fixed-linear-coordinate remainder decomposition."""

    state_array = np.asarray(states, dtype=float)
    error_array = np.asarray(errors, dtype=float)
    if state_array.ndim != 2 or error_array.shape != state_array.shape:
        raise ValueError("states and errors must have one matching two-dimensional shape")
    sample_count, dimension = state_array.shape
    if sample_count == 0:
        raise ValueError("states and errors must be non-empty")
    matrices = {
        "coordinate_transform": np.asarray(coordinate_transform, dtype=float),
        "inverse_coordinate": np.asarray(inverse_coordinate, dtype=float),
        "target_generator": np.asarray(target_generator, dtype=float),
        "closed_loop_generator": np.asarray(closed_loop_generator, dtype=float),
    }
    if any(matrix.shape != (dimension, dimension) for matrix in matrices.values()):
        raise ValueError("all coordinate and generator matrices must be square")
    if not (
        np.all(np.isfinite(state_array))
        and np.all(np.isfinite(error_array))
        and all(np.all(np.isfinite(matrix)) for matrix in matrices.values())
    ):
        raise ValueError("batch inputs must be finite")

    transform = matrices["coordinate_transform"]
    inverse = matrices["inverse_coordinate"]
    target = matrices["target_generator"]
    closed_loop = matrices["closed_loop_generator"]
    transformed = error_array @ transform.T
    nonlinear = -((state_array + error_array) ** 3 - state_array**3)
    original_rhs = error_array @ closed_loop.T + nonlinear
    transformed_rhs = original_rhs @ transform.T
    linear_rhs = transformed @ target.T
    nonlinear_remainder = nonlinear @ transform.T
    reconstructed_rhs = linear_rhs + nonlinear_remainder

    transformed_energy = np.sum(transformed**2, axis=1)
    if np.any(transformed_energy <= 1.0e-30):
        raise ValueError("transformed errors must have positive energy")

    def rate(right_hand_side: Array) -> Array:
        return -np.sum(transformed * right_hand_side, axis=1) / transformed_energy

    total_rate = rate(transformed_rhs)
    linear_rate = rate(linear_rhs)
    nonlinear_rate = rate(nonlinear_remainder)
    reconstruction_scale = (
        np.sqrt(np.sum(transformed_rhs**2, axis=1))
        + np.sqrt(np.sum(linear_rhs**2, axis=1))
        + np.sqrt(np.sum(nonlinear_remainder**2, axis=1))
    )
    reconstruction_relative_error = np.sqrt(
        np.sum((transformed_rhs - reconstructed_rhs) ** 2, axis=1)
    ) / np.maximum(reconstruction_scale, 1.0e-30)
    recovered_error = transformed @ inverse.T
    inverse_relative_error = np.sqrt(
        np.sum((recovered_error - error_array) ** 2, axis=1)
        / np.maximum(np.sum(error_array**2, axis=1), 1.0e-30)
    )
    return {
        "transformed_energy": transformed_energy,
        "total_rate": total_rate,
        "actual_margin": total_rate - float(alpha),
        "linear_rate": linear_rate,
        "linear_margin": linear_rate - float(alpha),
        "nonlinear_remainder_rate": nonlinear_rate,
        "rate_additivity_error": np.abs(
            total_rate - linear_rate - nonlinear_rate
        ),
        "rhs_reconstruction_relative_error": reconstruction_relative_error,
        "inverse_reconstruction_relative_error": inverse_relative_error,
    }
