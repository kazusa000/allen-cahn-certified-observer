"""Diagnostics for prescribed nonlinear targets of transformed error dynamics."""

from __future__ import annotations

import numpy as np


def same_form_target_components(
    torch: object,
    states: object,
    transformed_errors: object,
    laplacian: object,
    *,
    nu: float,
    alpha: float,
) -> dict[str, object]:
    """Return the pieces of the old same-form Allen--Cahn target.

    The target is ``nu * Delta z - ((u + z)**3 - u**3) - alpha * z``.
    This is algebraically identical to retaining ``F(u+z)-F(u)`` and adding
    ``-(1+alpha)z`` for ``F(v)=v-v**3``.
    """

    viscosity = float(nu)
    requested = float(alpha)
    if not np.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("nu must be positive and finite")
    if not np.isfinite(requested) or requested <= 0.0:
        raise ValueError("alpha must be positive and finite")
    if states.shape != transformed_errors.shape:
        raise ValueError("states and transformed_errors must have matching shapes")
    if states.ndim != 2 or states.shape[1] != laplacian.shape[0]:
        raise ValueError("inputs and laplacian have incompatible dimensions")
    diffusion = viscosity * (transformed_errors @ laplacian.T)
    nonlinear = -((states + transformed_errors) ** 3 - states**3)
    damping = -requested * transformed_errors
    return {
        "diffusion": diffusion,
        "nonlinear": nonlinear,
        "damping": damping,
        "total": diffusion + nonlinear + damping,
    }


def mass_rate(
    torch: object,
    transformed_errors: object,
    rhs: object,
    grid_step: float,
    *,
    epsilon: float = 1.0e-10,
) -> object:
    """Return ``-<z,rhs>_h/(||z||_h^2+epsilon)`` for a batch."""

    if transformed_errors.shape != rhs.shape or transformed_errors.ndim != 2:
        raise ValueError("transformed_errors and rhs must be matching matrices")
    if not np.isfinite(grid_step) or grid_step <= 0.0:
        raise ValueError("grid_step must be positive and finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    squared = float(grid_step) * torch.sum(transformed_errors**2, dim=1)
    inner = float(grid_step) * torch.sum(transformed_errors * rhs, dim=1)
    return -inner / (squared + float(epsilon))


def orthogonal_defect_components(
    torch: object,
    transformed_errors: object,
    defect: object,
    grid_step: float,
    *,
    epsilon: float = 1.0e-20,
) -> dict[str, object]:
    """Split a target defect into mass-inner-product radial and tangent parts."""

    if transformed_errors.shape != defect.shape or transformed_errors.ndim != 2:
        raise ValueError("transformed_errors and defect must be matching matrices")
    if not np.isfinite(grid_step) or grid_step <= 0.0:
        raise ValueError("grid_step must be positive and finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    squared = float(grid_step) * torch.sum(transformed_errors**2, dim=1)
    inner = float(grid_step) * torch.sum(transformed_errors * defect, dim=1)
    coefficient = inner / (squared + float(epsilon))
    parallel = coefficient[:, None] * transformed_errors
    perpendicular = defect - parallel
    return {
        "parallel": parallel,
        "perpendicular": perpendicular,
        "coefficient": coefficient,
    }


def normalized_mass_norm(
    torch: object,
    values: object,
    reference: object,
    grid_step: float,
    *,
    epsilon: float = 1.0e-20,
) -> object:
    """Return ``||values||_h/(||reference||_h+epsilon)`` samplewise."""

    if values.shape != reference.shape or values.ndim != 2:
        raise ValueError("values and reference must be matching matrices")
    if not np.isfinite(grid_step) or grid_step <= 0.0:
        raise ValueError("grid_step must be positive and finite")
    numerator = torch.sqrt(float(grid_step) * torch.sum(values**2, dim=1))
    denominator = torch.sqrt(float(grid_step) * torch.sum(reference**2, dim=1))
    return numerator / (denominator + float(epsilon))
