"""Mesh-shared coordinates for direct Allen--Cahn error-fiber certificates."""

from __future__ import annotations

import numpy as np


def buffered_contraction_cvar(
    torch: object,
    margins: object,
    *,
    buffer: float,
    tail_fraction: float = 0.1,
) -> object:
    """Penalize the worst buffered contraction-margin violations.

    ``margins`` are measured relative to the requested contraction rate.  A
    positive ``buffer`` keeps a gradient after the original zero-margin gate
    has been crossed, which is essential when training for unseen worst cases.
    """

    target = float(buffer)
    fraction = float(tail_fraction)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("buffer must be positive and finite")
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if margins.ndim != 1 or margins.shape[0] < 1:
        raise ValueError("margins must be a non-empty vector")
    violations = torch.relu(target - margins) ** 2
    tail_count = max(1, int(np.ceil(fraction * int(margins.shape[0]))))
    return torch.mean(torch.topk(violations, tail_count).values)


def project_physical_modal_adversaries_(
    torch: object,
    state_coefficients: object,
    error_coefficients: object,
    state_basis: object,
    grid_step: float,
    *,
    state_maximum: float = 1.25,
    error_radius_minimum: float = 0.02,
    error_radius_maximum: float = 0.8,
) -> None:
    """Project modal adversaries into the frozen compact collocation domain."""

    if state_coefficients.ndim != 2 or error_coefficients.ndim != 2:
        raise ValueError("state and error coefficients must be matrices")
    if state_coefficients.shape[0] != error_coefficients.shape[0]:
        raise ValueError("state and error batches must match")
    if state_coefficients.shape[1] != state_basis.shape[1]:
        raise ValueError("state coefficients and basis have incompatible modes")
    if not np.isfinite(grid_step) or grid_step <= 0.0:
        raise ValueError("grid_step must be positive and finite")
    maximum = float(state_maximum)
    radius_minimum = float(error_radius_minimum)
    radius_maximum = float(error_radius_maximum)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("state_maximum must be positive and finite")
    if (
        not np.isfinite(radius_minimum)
        or not np.isfinite(radius_maximum)
        or radius_minimum <= 0.0
        or radius_maximum < radius_minimum
    ):
        raise ValueError("error radii must define a positive finite interval")

    with torch.no_grad():
        states = (state_coefficients @ state_basis.T) / float(np.sqrt(grid_step))
        amplitudes = torch.amax(torch.abs(states), dim=1, keepdim=True)
        state_scales = torch.clamp(
            maximum / amplitudes.clamp_min(1.0e-12), max=1.0
        )
        state_coefficients.mul_(state_scales)

        radii = torch.linalg.vector_norm(error_coefficients, dim=1, keepdim=True)
        zero = radii[:, 0] <= 1.0e-12
        if bool(torch.any(zero)):
            error_coefficients[zero] = 0.0
            error_coefficients[zero, 0] = radius_minimum
            radii = torch.linalg.vector_norm(
                error_coefficients, dim=1, keepdim=True
            )
        target_radii = torch.clamp(
            radii, min=radius_minimum, max=radius_maximum
        )
        error_coefficients.mul_(target_radii / radii.clamp_min(1.0e-12))


def modal_residual_jacobian_bounds(rho: float) -> tuple[float, float]:
    """Return the singular-value bounds for ``I + D_b g``.

    In ``g(a, b) - g(a, 0)``, the subtracted term is constant with respect to
    ``b``.  Consequently there is no factor two in the derivative budget.
    """

    value = float(rho)
    if not 0.0 < value < 1.0:
        raise ValueError("rho must lie in (0, 1)")
    return 1.0 - value, 1.0 + value


def modal_residual_path_layer_bound(rho: float, layer_count: int) -> float:
    """Return an equal per-layer bound whose product is ``rho``."""

    value = float(rho)
    if not 0.0 < value < 1.0:
        raise ValueError("rho must lie in (0, 1)")
    if not isinstance(layer_count, int) or isinstance(layer_count, bool):
        raise TypeError("layer_count must be an integer")
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    return float(value ** (1.0 / layer_count))


def build_low_modal_conditional_residual_transform(
    torch: object,
    base_transform: np.ndarray,
    *,
    state_dimension: int,
    hidden_width: int = 64,
    hidden_layers: int = 3,
    rho: float = 0.35,
    error_scale: float = 1.0,
) -> object:
    """Build a globally invertible conditional transform in modal coordinates.

    The returned module evaluates

    ``tau(a, b) = T0 @ (b + g(a, b) - g(a, 0))``.

    Only weights on the error path enter the global Lipschitz bound.  State
    conditioning is additive at every hidden layer and may therefore be rich
    without weakening invertibility with respect to ``b``.
    """

    base = np.asarray(base_transform, dtype=float)
    if base.ndim != 2 or base.shape[0] < 1 or base.shape[0] != base.shape[1]:
        raise ValueError("base_transform must be a non-empty square matrix")
    if not np.all(np.isfinite(base)):
        raise ValueError("base_transform must be finite")
    singular = np.linalg.svd(base, compute_uv=False)
    if float(np.min(singular)) <= 0.0:
        raise ValueError("base_transform must be invertible")
    if (
        not isinstance(state_dimension, int)
        or isinstance(state_dimension, bool)
        or state_dimension < 1
    ):
        raise ValueError("state_dimension must be a positive integer")
    if (
        not isinstance(hidden_width, int)
        or isinstance(hidden_width, bool)
        or hidden_width < 1
    ):
        raise ValueError("hidden_width must be a positive integer")
    if (
        not isinstance(hidden_layers, int)
        or isinstance(hidden_layers, bool)
        or hidden_layers < 1
    ):
        raise ValueError("hidden_layers must be a positive integer")
    scale = float(error_scale)
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("error_scale must be finite and at least one")

    error_dimension = int(base.shape[0])
    normalized_lower, normalized_upper = modal_residual_jacobian_bounds(rho)
    per_layer_bound = modal_residual_path_layer_bound(rho, hidden_layers + 1)
    nn = torch.nn

    class LowModalConditionalResidualTransform(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            error_layers: list[object] = []
            condition_layers: list[object] = []
            input_width = error_dimension
            for _ in range(hidden_layers):
                error_layers.append(nn.Linear(input_width, hidden_width, bias=False))
                condition_layers.append(
                    nn.Linear(state_dimension, hidden_width, bias=True)
                )
                input_width = hidden_width
            self.error_layers = nn.ModuleList(error_layers)
            self.condition_layers = nn.ModuleList(condition_layers)
            self.output_layer = nn.Linear(hidden_width, error_dimension, bias=True)
            self.register_buffer(
                "base_transform", torch.as_tensor(base, dtype=torch.float32)
            )
            self.state_dimension = state_dimension
            self.error_dimension = error_dimension
            self.hidden_width = hidden_width
            self.hidden_layers = hidden_layers
            self.rho = float(rho)
            self.error_scale = scale
            self.per_layer_spectral_bound = per_layer_bound
            self.normalized_lower_jacobian_bound = normalized_lower
            self.normalized_upper_jacobian_bound = normalized_upper
            self.base_min_singular = float(np.min(singular))
            self.base_max_singular = float(np.max(singular))
            self.lower_jacobian_bound = self.base_min_singular * normalized_lower
            self.upper_jacobian_bound = self.base_max_singular * normalized_upper

            for layer in self.error_layers:
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
            for layer in self.condition_layers:
                nn.init.xavier_uniform_(layer.weight, gain=0.25)
                nn.init.zeros_(layer.bias)
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.output_layer.bias)
            self.project_spectral_()

        def _check_inputs(self, states: object, errors: object) -> None:
            if states.shape[:-1] != errors.shape[:-1]:
                raise ValueError("state and error batches must match")
            if states.shape[-1] != self.state_dimension:
                raise ValueError("state coefficients have the wrong dimension")
            if errors.shape[-1] != self.error_dimension:
                raise ValueError("error coefficients have the wrong dimension")

        def g(self, states: object, errors: object) -> object:
            self._check_inputs(states, errors)
            hidden = self.error_scale * errors
            for error_layer, condition_layer in zip(
                self.error_layers, self.condition_layers, strict=True
            ):
                hidden = torch.tanh(
                    error_layer(hidden)
                    + condition_layer(self.error_scale * states)
                )
            return self.output_layer(hidden) / self.error_scale

        def normalized_forward(self, states: object, errors: object) -> object:
            zero = torch.zeros_like(errors)
            return errors + self.g(states, errors) - self.g(states, zero)

        def forward(self, states: object, errors: object) -> object:
            normalized = self.normalized_forward(states, errors)
            return normalized @ self.base_transform.T

        def _normalize_target(self, transformed_errors: object) -> object:
            return torch.linalg.solve(
                self.base_transform, transformed_errors.T
            ).T

        def inverse_fixed_point(
            self,
            states: object,
            transformed_errors: object,
            *,
            iterations: int = 20,
        ) -> object:
            if iterations < 1:
                raise ValueError("iterations must be positive")
            target = self._normalize_target(transformed_errors)
            zero = torch.zeros_like(target)
            offset = self.g(states, zero)
            current = target
            for _ in range(iterations):
                current = target - self.g(states, current) + offset
            return current

        def inverse_fixed_point_diagnostics(
            self,
            states: object,
            transformed_errors: object,
            *,
            max_iterations: int = 80,
            tolerance: float = 1.0e-9,
        ) -> tuple[object, object, int]:
            if max_iterations < 1:
                raise ValueError("max_iterations must be positive")
            if tolerance <= 0.0:
                raise ValueError("tolerance must be positive")
            target = self._normalize_target(transformed_errors)
            zero = torch.zeros_like(target)
            offset = self.g(states, zero)
            current = target
            converged = torch.zeros(
                target.shape[0], dtype=torch.bool, device=target.device
            )
            iterations_used = 0
            for iteration in range(1, max_iterations + 1):
                next_value = target - self.g(states, current) + offset
                difference = torch.max(torch.abs(next_value - current), dim=1).values
                current = next_value
                converged = difference <= tolerance
                iterations_used = iteration
                if bool(torch.all(converged)):
                    break
            return current, converged, iterations_used

        def spectral_weights(self) -> tuple[object, ...]:
            return tuple(
                [layer.weight for layer in self.error_layers]
                + [self.output_layer.weight]
            )

        def spectral_norms_tensor(self) -> object:
            return torch.stack(
                [torch.linalg.svdvals(weight)[0] for weight in self.spectral_weights()]
            )

        def residual_lipschitz_bound_tensor(self) -> object:
            return torch.prod(self.spectral_norms_tensor())

        def residual_lipschitz_bound(self) -> float:
            with torch.no_grad():
                return float(self.residual_lipschitz_bound_tensor().detach().cpu())

        def project_spectral_(self) -> None:
            with torch.no_grad():
                for weight in self.spectral_weights():
                    maximum = torch.linalg.svdvals(weight)[0]
                    if maximum > self.per_layer_spectral_bound:
                        weight.mul_(
                            self.per_layer_spectral_bound
                            / maximum.clamp_min(1.0e-12)
                        )

    return LowModalConditionalResidualTransform()


def physical_modal_coordinates(
    torch: object,
    values: object,
    euclidean_basis: object,
    grid_step: float,
) -> object:
    """Project grid values onto the corresponding ``M_h``-orthonormal modes."""

    if values.shape[-1] != euclidean_basis.shape[0]:
        raise ValueError("values and basis have incompatible grid dimensions")
    if not np.isfinite(grid_step) or grid_step <= 0.0:
        raise ValueError("grid_step must be positive and finite")
    return float(np.sqrt(grid_step)) * (values @ euclidean_basis)


def reconstruct_physical_modes(
    torch: object,
    coefficients: object,
    euclidean_basis: object,
    grid_step: float,
) -> object:
    """Lift physical modal coefficients back to grid values."""

    del torch
    if coefficients.shape[-1] != euclidean_basis.shape[1]:
        raise ValueError("coefficients and basis have incompatible mode dimensions")
    if not np.isfinite(grid_step) or grid_step <= 0.0:
        raise ValueError("grid_step must be positive and finite")
    return (coefficients @ euclidean_basis.T) / float(np.sqrt(grid_step))


def mesh_shared_fiber_transform(
    torch: object,
    transform: object,
    states: object,
    errors: object,
    condition_basis: object,
    low_basis: object,
    grid_step: float,
) -> object:
    """Lift a shared low-modal transform and leave the high tail unchanged."""

    state_coefficients = physical_modal_coordinates(
        torch, states, condition_basis, grid_step
    )
    error_coefficients = physical_modal_coordinates(
        torch, errors, low_basis, grid_step
    )
    transformed_coefficients = transform(state_coefficients, error_coefficients)
    low_error = reconstruct_physical_modes(
        torch, error_coefficients, low_basis, grid_step
    )
    low_transformed = reconstruct_physical_modes(
        torch, transformed_coefficients, low_basis, grid_step
    )
    return errors - low_error + low_transformed


def physical_modal_injection(
    torch: object,
    physical_modal_gain: object,
    low_basis: object,
    grid_step: float,
) -> object:
    """Lift ``Beta`` in ``B_h = V_{4,h} Beta`` to nodal coordinates."""

    del torch
    if physical_modal_gain.shape[0] != low_basis.shape[1]:
        raise ValueError("physical_modal_gain and basis have incompatible dimensions")
    if not np.isfinite(grid_step) or grid_step <= 0.0:
        raise ValueError("grid_step must be positive and finite")
    return (low_basis @ physical_modal_gain) / float(np.sqrt(grid_step))
