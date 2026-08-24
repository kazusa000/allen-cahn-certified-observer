"""Physical sine-mode projections and Allen--Cahn high-frequency tail audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import AllenCahnGrid


def dirichlet_sine_basis(grid: AllenCahnGrid, mode_count: int) -> np.ndarray:
    """Return Euclidean-orthonormal discrete Dirichlet sine modes.

    Dividing this matrix by ``sqrt(grid.h)`` gives the corresponding
    ``M_h``-orthonormal basis. Since ``M_h = h I``, both induce the same
    orthogonal projector.
    """

    if not isinstance(mode_count, int) or isinstance(mode_count, bool):
        raise TypeError("mode_count must be an integer")
    if not 1 <= mode_count < grid.n:
        raise ValueError("mode_count must satisfy 1 <= mode_count < grid.n")
    modes = np.arange(1, mode_count + 1, dtype=float)
    return np.sqrt(2.0 * grid.h) * np.sin(np.pi * grid.x[:, None] * modes[None, :])


def low_frequency_projector(grid: AllenCahnGrid, mode_count: int) -> np.ndarray:
    """Return the projector onto the first ``mode_count`` physical modes."""

    basis = dirichlet_sine_basis(grid, mode_count)
    return basis @ basis.T


def dirichlet_laplacian_rates(grid: AllenCahnGrid) -> np.ndarray:
    """Return positive rates ``mu_k`` satisfying ``L_h v_k = -mu_k v_k``."""

    modes = np.arange(1, grid.n + 1, dtype=float)
    return 4.0 * np.sin(0.5 * np.pi * grid.h * modes) ** 2 / grid.h**2


def split_low_tail(
    grid: AllenCahnGrid, values: np.ndarray, mode_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split arrays whose final axis is the grid axis into low and tail parts."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.shape[-1] != grid.n:
        raise ValueError(f"values must have final dimension {grid.n}")
    projector = low_frequency_projector(grid, mode_count)
    low = array @ projector
    return low, array - low


def mass_norm(grid: AllenCahnGrid, values: np.ndarray) -> np.ndarray:
    """Return the ``M_h`` norm along the final array axis."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.shape[-1] != grid.n:
        raise ValueError(f"values must have final dimension {grid.n}")
    return np.sqrt(grid.h * np.sum(array**2, axis=-1))


def sampled_forced_tail_envelope(
    times: np.ndarray,
    tail_norm: np.ndarray,
    forcing_norm: np.ndarray,
    decay_margin: float,
) -> np.ndarray:
    """Propagate a sampled envelope for ``q' <= -a q + f``.

    A non-increasing time stamp starts a new trajectory. On each saved time
    trajectory the forcing is bounded by its maximum saved value. The result
    is a reproducible sampled envelope, not a continuous-time bound between
    saved outputs.
    """

    time_array = np.asarray(times, dtype=float)
    tail_array = np.asarray(tail_norm, dtype=float)
    forcing_array = np.asarray(forcing_norm, dtype=float)
    if time_array.ndim != 1 or time_array.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    if tail_array.shape != time_array.shape or forcing_array.shape != time_array.shape:
        raise ValueError("tail_norm and forcing_norm must match times")
    if not np.isfinite(decay_margin) or decay_margin <= 0.0:
        raise ValueError("decay_margin must be a positive finite scalar")
    if not (
        np.all(np.isfinite(time_array))
        and np.all(np.isfinite(tail_array))
        and np.all(np.isfinite(forcing_array))
    ):
        raise ValueError("sampled envelope inputs must be finite")
    if np.any(tail_array < 0.0) or np.any(forcing_array < 0.0):
        raise ValueError("norm inputs must be non-negative")

    reset_indices = np.flatnonzero(np.diff(time_array) <= 0.0) + 1
    starts = np.concatenate(([0], reset_indices))
    stops = np.concatenate((reset_indices, [time_array.size]))
    envelope = np.empty_like(tail_array)
    for start, stop in zip(starts, stops, strict=True):
        trajectory_forcing = float(np.max(forcing_array[start:stop]))
        envelope[start] = tail_array[start]
        for index in range(start + 1, stop):
            dt = time_array[index] - time_array[index - 1]
            contraction = np.exp(-decay_margin * dt)
            envelope[index] = (
                contraction * envelope[index - 1]
                + (1.0 - contraction) * trajectory_forcing / decay_margin
            )
    return envelope


@dataclass(frozen=True)
class TailAudit:
    """Sampled terms in the high-frequency Allen--Cahn energy inequality."""

    mode_count: int
    diffusion_margin: float
    tail_norm: np.ndarray
    low_norm: np.ndarray
    total_norm: np.ndarray
    low_to_tail_coupling_norm: np.ndarray
    correction_tail_norm: np.ndarray
    dissipative_remainder_inner_product: np.ndarray
    energy_rate: np.ndarray
    energy_upper_bound: np.ndarray

    @property
    def inequality_residual_max(self) -> float:
        return float(np.max(self.energy_rate - self.energy_upper_bound))

    @property
    def dissipativity_violation_max(self) -> float:
        return float(np.max(self.dissipative_remainder_inner_product))


def audit_high_frequency_tail(
    grid: AllenCahnGrid,
    nu: float,
    states: np.ndarray,
    errors: np.ndarray,
    corrections: np.ndarray,
    *,
    mode_count: int,
) -> TailAudit:
    """Audit the discrete high-frequency energy inequality on samples.

    Write ``e = p + q`` with ``p`` in the first ``mode_count`` sine modes.
    Monotonicity of the cubic term makes its tail-dependent remainder
    dissipative, leaving only low-to-tail coupling and correction injection as
    forcing terms.
    """

    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu must be a positive finite scalar")
    state_array = np.asarray(states, dtype=float)
    error_array = np.asarray(errors, dtype=float)
    correction_array = np.asarray(corrections, dtype=float)
    if state_array.ndim != 2 or state_array.shape[1] != grid.n:
        raise ValueError("states must have shape (samples, grid.n)")
    if error_array.shape != state_array.shape:
        raise ValueError("errors must have the same shape as states")
    if correction_array.shape != state_array.shape:
        raise ValueError("corrections must have the same shape as states")
    if not (
        np.all(np.isfinite(state_array))
        and np.all(np.isfinite(error_array))
        and np.all(np.isfinite(correction_array))
    ):
        raise ValueError("tail-audit inputs must be finite")

    low, tail = split_low_tail(grid, error_array, mode_count)
    _, correction_tail = split_low_tail(grid, correction_array, mode_count)
    cubic_at_low = -((state_array + low) ** 3 - state_array**3)
    cubic_full = -((state_array + error_array) ** 3 - state_array**3)
    cubic_remainder = cubic_full - cubic_at_low
    _, low_to_tail_coupling = split_low_tail(grid, cubic_at_low, mode_count)

    rates = dirichlet_laplacian_rates(grid)
    diffusion_margin = float(nu * rates[mode_count] - 1.0)
    tail_norm = mass_norm(grid, tail)
    low_norm = mass_norm(grid, low)
    total_norm = mass_norm(grid, error_array)
    coupling_norm = mass_norm(grid, low_to_tail_coupling)
    correction_tail_norm = mass_norm(grid, correction_tail)
    remainder_inner = grid.h * np.sum(tail * cubic_remainder, axis=1)

    error_rhs = (
        nu * (error_array @ grid.laplacian.T)
        + error_array
        + cubic_full
        + correction_array
    )
    _, tail_rhs = split_low_tail(grid, error_rhs, mode_count)
    energy_rate = grid.h * np.sum(tail * tail_rhs, axis=1)
    energy_upper_bound = -diffusion_margin * tail_norm**2 + tail_norm * (
        coupling_norm + correction_tail_norm
    )
    return TailAudit(
        mode_count=mode_count,
        diffusion_margin=diffusion_margin,
        tail_norm=tail_norm,
        low_norm=low_norm,
        total_norm=total_norm,
        low_to_tail_coupling_norm=coupling_norm,
        correction_tail_norm=correction_tail_norm,
        dissipative_remainder_inner_product=remainder_inner,
        energy_rate=energy_rate,
        energy_upper_bound=energy_upper_bound,
    )
