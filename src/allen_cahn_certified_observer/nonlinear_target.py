"""Nonlinear Allen--Cahn target and globally invertible residual transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .grid import AllenCahnGrid


def nonlinear_target_rhs(
    grid: AllenCahnGrid,
    nu: float,
    state: np.ndarray,
    transformed_error: np.ndarray,
    *,
    lambda_ratio: float = 0.1,
) -> np.ndarray:
    """Evaluate the contractive nonlinear Allen--Cahn target vector field.

    The decomposition is fixed as A = nu * Delta_h and F(v) = v - v**3.
    The target is A z + F(u + z) - F(u) - (1 + lambda) z with
    lambda = lambda_ratio * nu * pi**2.
    """

    viscosity = float(nu)
    ratio = float(lambda_ratio)
    values = np.asarray(state, dtype=float)
    transformed = np.asarray(transformed_error, dtype=float)
    if viscosity <= 0.0 or not np.isfinite(viscosity):
        raise ValueError("nu must be positive and finite")
    if ratio <= 0.0 or not np.isfinite(ratio):
        raise ValueError("lambda_ratio must be positive and finite")
    if values.shape != (grid.n,) or transformed.shape != (grid.n,):
        raise ValueError(f"state and transformed_error must have shape {(grid.n,)}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(transformed)):
        raise ValueError("state and transformed_error must be finite")
    lam = ratio * viscosity * np.pi**2
    reaction_increment = (
        values + transformed - (values + transformed) ** 3
        - values
        + values**3
    )
    return (
        viscosity * (grid.laplacian @ transformed)
        + reaction_increment
        - (1.0 + lam) * transformed
    )


def nonlinear_target_tensor(
    torch: object,
    states: object,
    transformed_errors: object,
    nus: object,
    laplacian: object,
    *,
    lambda_ratio: float = 0.1,
) -> object:
    """Batched torch counterpart of nonlinear_target_rhs."""

    lam = float(lambda_ratio) * nus[:, None] * np.pi**2
    reaction_increment = (
        states
        + transformed_errors
        - (states + transformed_errors) ** 3
        - states
        + states**3
    )
    return (
        nus[:, None] * (transformed_errors @ laplacian.T)
        + reaction_increment
        - (1.0 + lam) * transformed_errors
    )


def residual_jacobian_bounds(rho: float) -> tuple[float, float]:
    """Return tight singular-value bounds under 2 ||D_e g|| <= rho."""

    value = float(rho)
    if not 0.0 < value < 1.0:
        raise ValueError("rho must lie in (0, 1)")
    return 1.0 - 0.5 * value, 1.0 + 0.5 * value


def residual_path_layer_bound(rho: float, layer_count: int) -> float:
    """Per-layer spectral bound whose product is at most rho / 2."""

    value = float(rho)
    if not 0.0 < value < 1.0:
        raise ValueError("rho must lie in (0, 1)")
    if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count < 1:
        raise ValueError("layer_count must be a positive integer")
    return float((0.5 * value) ** (1.0 / layer_count))


def build_conditional_residual_transform(
    torch: object,
    dimension: int,
    *,
    hidden_width: int = 128,
    hidden_layers: int = 3,
    rho: float = 0.5,
) -> object:
    """Build T(u,e)=e+g(u,e)-g(u,0) with a hard spectral path bound."""

    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
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
    lower, upper = residual_jacobian_bounds(rho)
    path_layer_count = hidden_layers + 1
    per_layer_bound = residual_path_layer_bound(rho, path_layer_count)
    nn = torch.nn

    class ConditionalResidualTransform(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            error_layers: list[object] = []
            condition_layers: list[object] = []
            input_width = dimension
            for _ in range(hidden_layers):
                error_layers.append(
                    nn.Linear(input_width, hidden_width, bias=False)
                )
                condition_layers.append(
                    nn.Linear(dimension, hidden_width, bias=True)
                )
                input_width = hidden_width
            self.error_layers = nn.ModuleList(error_layers)
            self.condition_layers = nn.ModuleList(condition_layers)
            self.output_layer = nn.Linear(hidden_width, dimension, bias=True)
            self.dimension = dimension
            self.hidden_width = hidden_width
            self.hidden_layers = hidden_layers
            self.rho = float(rho)
            self.lower_jacobian_bound = lower
            self.upper_jacobian_bound = upper
            self.per_layer_spectral_bound = per_layer_bound

            for layer in self.error_layers:
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
            for layer in self.condition_layers:
                nn.init.xavier_uniform_(layer.weight, gain=0.25)
                nn.init.zeros_(layer.bias)
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.output_layer.bias)
            self.project_spectral_()

        def g(self, states: object, errors: object) -> object:
            hidden = errors
            for error_layer, condition_layer in zip(
                self.error_layers, self.condition_layers, strict=True
            ):
                hidden = torch.tanh(
                    error_layer(hidden) + condition_layer(states)
                )
            return self.output_layer(hidden)

        def forward(self, states: object, errors: object) -> object:
            if states.shape != errors.shape or states.shape[-1] != self.dimension:
                raise ValueError(
                    "states and errors must have matching final dimension"
                )
            zero = torch.zeros_like(errors)
            return errors + self.g(states, errors) - self.g(states, zero)

        def inverse_fixed_point(
            self,
            states: object,
            transformed_errors: object,
            *,
            iterations: int = 12,
        ) -> object:
            if iterations < 1:
                raise ValueError("iterations must be positive")
            zero = torch.zeros_like(transformed_errors)
            offset = self.g(states, zero)
            current = transformed_errors
            for _ in range(iterations):
                current = (
                    transformed_errors - self.g(states, current) + offset
                )
            return current

        def spectral_weights(self) -> tuple[object, ...]:
            return tuple(
                [layer.weight for layer in self.error_layers]
                + [self.output_layer.weight]
            )

        def spectral_norms_tensor(self) -> object:
            return torch.stack(
                [
                    torch.linalg.svdvals(weight)[0]
                    for weight in self.spectral_weights()
                ]
            )

        def residual_lipschitz_bound_tensor(self) -> object:
            return torch.prod(self.spectral_norms_tensor())

        def residual_lipschitz_bound(self) -> float:
            with torch.no_grad():
                return float(
                    self.residual_lipschitz_bound_tensor().detach().cpu()
                )

        def project_spectral_(self) -> None:
            with torch.no_grad():
                for weight in self.spectral_weights():
                    maximum = torch.linalg.svdvals(weight)[0]
                    if maximum > self.per_layer_spectral_bound:
                        weight.mul_(
                            self.per_layer_spectral_bound
                            / maximum.clamp_min(1.0e-12)
                        )

    return ConditionalResidualTransform()


def build_projected_constant_gain(
    torch: object,
    base_gain: np.ndarray,
    *,
    trust_ratio: float = 0.25,
) -> object:
    """Build a constant gain with a hard Frobenius trust region around B_0."""

    base = np.asarray(base_gain, dtype=float)
    if base.ndim != 2 or base.shape[0] < 1 or base.shape[1] < 1:
        raise ValueError("base_gain must be a non-empty matrix")
    if not np.all(np.isfinite(base)):
        raise ValueError("base_gain must be finite")
    ratio = float(trust_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("trust_ratio must lie in (0, 1)")
    nn = torch.nn

    class ProjectedConstantGain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "base_gain", torch.as_tensor(base, dtype=torch.float32)
            )
            self.delta = nn.Parameter(torch.zeros_like(self.base_gain))
            self.trust_ratio = ratio

        def forward(self, batch_size: int | None = None) -> object:
            current = self.base_gain + self.delta
            if batch_size is None:
                return current
            return current[None, :, :].expand(batch_size, -1, -1)

        def project_(self) -> None:
            with torch.no_grad():
                base_norm = torch.linalg.vector_norm(self.base_gain)
                delta_norm = torch.linalg.vector_norm(self.delta)
                limit = self.trust_ratio * base_norm
                if delta_norm > limit:
                    self.delta.mul_(limit / delta_norm.clamp_min(1.0e-12))

        def relative_delta_norm(self) -> float:
            with torch.no_grad():
                return float(
                    (
                        torch.linalg.vector_norm(self.delta)
                        / torch.linalg.vector_norm(self.base_gain).clamp_min(1.0e-12)
                    )
                    .detach()
                    .cpu()
                )

    return ProjectedConstantGain()


def spectral_product(norms: Sequence[float]) -> float:
    """Validate and multiply finite non-negative spectral norms."""

    values = np.asarray(tuple(norms), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("norms must be a non-empty sequence")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("norms must be finite and non-negative")
    return float(np.prod(values))
