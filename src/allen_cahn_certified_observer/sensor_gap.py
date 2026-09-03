"""Sensor-gap diagnostics for fixed-gain Allen--Cahn observers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .grid import AllenCahnGrid

Array = np.ndarray


def observation_support_mask(
    observation: Array,
    *,
    tolerance: float = 0.0,
) -> Array:
    """Return the nodal support of a finite-dimensional observation."""

    matrix = np.asarray(observation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("observation must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("observation must be finite")
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return np.any(np.abs(matrix) > threshold, axis=0)


def contiguous_index_blocks(indices: Sequence[int]) -> tuple[Array, ...]:
    """Split sorted grid indices into maximal contiguous blocks."""

    values = np.asarray(indices, dtype=int)
    if values.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if values.size == 0:
        return ()
    if np.any(values < 0) or np.any(np.diff(values) <= 0):
        raise ValueError("indices must be non-negative and strictly increasing")
    split_points = np.flatnonzero(np.diff(values) > 1) + 1
    return tuple(
        np.asarray(block, dtype=int)
        for block in np.split(values, split_points)
    )


def sensor_gap_diagnostics(
    grid: AllenCahnGrid,
    nu: float,
    observation: Array,
) -> dict[str, object]:
    """Measure the linear Allen--Cahn growth on unobserved nodal blocks."""

    viscosity = float(nu)
    if not np.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("nu must be positive and finite")
    support = observation_support_mask(observation)
    if support.shape != (grid.n,):
        raise ValueError("observation has the wrong grid dimension")
    unobserved = np.flatnonzero(~support)
    blocks = contiguous_index_blocks(unobserved)
    if not blocks:
        raise ValueError("observation support covers every grid node")

    generator = viscosity * grid.laplacian + np.eye(grid.n)
    block_results: list[dict[str, object]] = []
    for block in blocks:
        principal = generator[np.ix_(block, block)]
        maximum = float(np.max(np.linalg.eigvalsh(principal)))
        block_results.append(
            {
                "start_index": int(block[0]),
                "end_index": int(block[-1]),
                "node_count": int(block.size),
                "first_node_x": float(grid.x[block[0]]),
                "last_node_x": float(grid.x[block[-1]]),
                "effective_dirichlet_length": float(
                    (block.size + 1) * grid.h
                ),
                "principal_growth_rate": maximum,
                "indices": block.tolist(),
            }
        )
    most_dangerous = max(
        block_results,
        key=lambda item: float(item["principal_growth_rate"]),
    )
    return {
        "grid_size": grid.n,
        "grid_step": grid.h,
        "observed_node_count": int(np.sum(support)),
        "unobserved_node_count": int(np.sum(~support)),
        "observation_zero_on_unobserved_max": float(
            np.max(np.abs(np.asarray(observation)[:, ~support]))
        ),
        "blocks": block_results,
        "most_dangerous_block": most_dangerous,
    }


def wall_state(observation: Array, q: float) -> Array:
    """Construct a state with cubic damping q on observation support."""

    damping = float(q)
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError("q must be finite and non-negative")
    support = observation_support_mask(observation)
    state = np.zeros(support.size, dtype=float)
    state[support] = np.sqrt(damping / 3.0)
    return state


def wall_jacobian(
    grid: AllenCahnGrid,
    nu: float,
    observation: Array,
    gain: Array,
    q: float,
) -> Array:
    """Return the frozen zero-error Jacobian at a sensor-wall state."""

    observation_matrix = np.asarray(observation, dtype=float)
    gain_matrix = np.asarray(gain, dtype=float)
    if observation_matrix.ndim != 2 or observation_matrix.shape[1] != grid.n:
        raise ValueError("observation has the wrong shape")
    if gain_matrix.shape != (grid.n, observation_matrix.shape[0]):
        raise ValueError("gain has the wrong shape")
    if not np.all(np.isfinite(gain_matrix)):
        raise ValueError("gain must be finite")
    state = wall_state(observation_matrix, q)
    return (
        float(nu) * grid.laplacian
        + np.eye(grid.n)
        - gain_matrix @ observation_matrix
        - 3.0 * np.diag(state**2)
    )


def rightmost_eigenpair(matrix: Array) -> tuple[float, Array]:
    """Return spectral abscissa and an associated right eigenvector."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite")
    eigenvalues, eigenvectors = np.linalg.eig(values)
    index = int(np.argmax(np.real(eigenvalues)))
    vector = eigenvectors[:, index]
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("rightmost eigenvector is invalid")
    return float(np.real(eigenvalues[index])), vector / norm


def unobserved_mass_fraction(vector: Array, support: Array) -> float:
    """Return the Euclidean mass fraction outside observation support."""

    values = np.asarray(vector)
    mask = np.asarray(support, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("vector and support must be matching vectors")
    squared = np.abs(values) ** 2
    total = float(np.sum(squared))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("vector must have positive finite norm")
    return float(np.sum(squared[~mask]) / total)


def similarity_spectral_abscissa_error(
    matrix: Array,
    transform: Array,
) -> float:
    """Verify numerically that a fixed coordinate cannot change the spectrum."""

    values = np.asarray(matrix, dtype=float)
    coordinate = np.asarray(transform, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if coordinate.shape != values.shape:
        raise ValueError("transform has the wrong shape")
    inverse = np.linalg.inv(coordinate)
    transformed = coordinate @ values @ inverse
    original_rate = float(np.max(np.real(np.linalg.eigvals(values))))
    transformed_rate = float(np.max(np.real(np.linalg.eigvals(transformed))))
    return abs(original_rate - transformed_rate)
