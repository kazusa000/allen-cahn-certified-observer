"""Classical causal correction baselines for R5."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .grid import AllenCahnGrid
from .solver import allen_cahn_rhs


@dataclass(frozen=True)
class CausalNudging:
    """Constant-gain local-average correction.

    This baseline uses only the current estimate and current measurement. The
    true state is used by the simulation harness only to generate observations.
    """

    grid: AllenCahnGrid
    nu: float
    observation_matrix: np.ndarray
    gain: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.observation_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.grid.n or matrix.shape[0] == 0:
            raise ValueError("observation_matrix must have shape (q, n)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("observation_matrix must be finite")
        if not np.isfinite(self.gain) or self.gain < 0.0:
            raise ValueError("gain must be a non-negative finite scalar")
        object.__setattr__(self, "observation_matrix", matrix)

    def rhs(self, estimate: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        state = np.asarray(estimate, dtype=float)
        observed = np.asarray(measurement, dtype=float)
        matrix = self.observation_matrix
        if state.shape != (self.grid.n,):
            raise ValueError(
                f"expected estimate shape {(self.grid.n,)}, got {state.shape}"
            )
        if observed.shape != (matrix.shape[0],):
            raise ValueError(
                f"expected measurement shape {(matrix.shape[0],)}, got {observed.shape}"
            )
        innovation = observed - matrix @ state
        return (
            allen_cahn_rhs(self.grid, self.nu, state)
            + self.gain * matrix.T @ innovation
        )


def observer_rhs_with_truth(
    time: float,
    combined_state: np.ndarray,
    *,
    observer: CausalNudging,
    noise: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    """RHS for a truth/estimate pair, used only by the offline test harness."""

    n = observer.grid.n
    combined = np.asarray(combined_state, dtype=float)
    if combined.shape != (2 * n,):
        raise ValueError(
            f"expected combined state shape {(2 * n,)}, got {combined.shape}"
        )
    truth, estimate = combined[:n], combined[n:]
    measurement = observer.observation_matrix @ truth
    if noise is not None:
        perturbation = np.asarray(noise(float(time)), dtype=float)
        if perturbation.shape != measurement.shape:
            raise ValueError("noise function returned the wrong shape")
        measurement = measurement + perturbation
    return np.concatenate(
        (
            allen_cahn_rhs(observer.grid, observer.nu, truth),
            observer.rhs(estimate, measurement),
        )
    )
