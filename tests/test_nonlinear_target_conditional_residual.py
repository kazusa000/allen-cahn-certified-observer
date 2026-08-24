import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_nonlinear_target_conditional_residual_joint import (
    OUTPUT_TIMES,
    _differentiable_online_loss,
    _instantaneous_components,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    build_conditional_residual_transform,
    build_projected_constant_gain,
    nonlinear_target_rhs,
    nonlinear_target_tensor,
    residual_jacobian_bounds,
    residual_path_layer_bound,
)


def test_nonlinear_target_matches_declared_allen_cahn_field() -> None:
    grid = AllenCahnGrid(15)
    rng = np.random.default_rng(17)
    state = rng.normal(size=grid.n)
    transformed = 0.2 * rng.normal(size=grid.n)
    nu = 0.005
    lam = 0.1 * nu * np.pi**2

    actual = nonlinear_target_rhs(grid, nu, state, transformed)
    expected = (
        nu * (grid.laplacian @ transformed)
        + state
        + transformed
        - (state + transformed) ** 3
        - state
        + state**3
        - (1.0 + lam) * transformed
    )

    assert np.allclose(actual, expected)


def test_nonlinear_target_has_requested_energy_decay() -> None:
    grid = AllenCahnGrid(31)
    rng = np.random.default_rng(18)
    for _ in range(20):
        state = rng.normal(size=grid.n)
        transformed = rng.normal(size=grid.n)
        rhs = nonlinear_target_rhs(grid, 0.005, state, transformed)
        derivative = grid.h * np.dot(transformed, rhs)
        requested = -0.1 * 0.005 * np.pi**2 * (
            grid.h * np.dot(transformed, transformed)
        )
        assert derivative <= requested + 1.0e-10


def test_tensor_target_matches_numpy_target() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(7)
    rng = np.random.default_rng(19)
    states = rng.normal(size=(3, grid.n))
    transformed = 0.1 * rng.normal(size=(3, grid.n))
    nus = np.asarray([0.005, 0.005, 0.005])

    actual = nonlinear_target_tensor(
        torch,
        torch.as_tensor(states, dtype=torch.float64),
        torch.as_tensor(transformed, dtype=torch.float64),
        torch.as_tensor(nus, dtype=torch.float64),
        torch.as_tensor(grid.laplacian, dtype=torch.float64),
    ).numpy()
    expected = np.asarray(
        [
            nonlinear_target_rhs(grid, nu, state, value)
            for nu, state, value in zip(nus, states, transformed, strict=True)
        ]
    )

    assert np.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_factor_two_budget_gives_tighter_jacobian_bounds() -> None:
    lower, upper = residual_jacobian_bounds(0.5)

    assert lower == pytest.approx(0.75)
    assert upper == pytest.approx(1.25)
    assert residual_path_layer_bound(0.5, 4) ** 4 == pytest.approx(0.25)


def test_conditional_residual_is_zero_fiber_invertible_and_nonlinear() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(20)
    model = build_conditional_residual_transform(
        torch, 7, hidden_width=16, hidden_layers=2, rho=0.5
    )
    states = torch.randn(5, 7)
    errors = 0.2 * torch.randn(5, 7)

    assert torch.max(torch.abs(model(states, torch.zeros_like(errors)))).item() == 0.0
    assert 2.0 * model.residual_lipschitz_bound() <= 0.5 + 1.0e-6

    transformed = model(states, errors)
    reconstructed = model.inverse_fixed_point(
        states, transformed, iterations=30
    )
    assert torch.max(torch.abs(reconstructed - errors)).item() < 1.0e-6

    nonlinear_residual = model(states, 2.0 * errors) - 2.0 * transformed
    assert torch.linalg.vector_norm(nonlinear_residual).item() > 1.0e-8

    jacobian = torch.autograd.functional.jacobian(
        lambda value: model(states[:1], value[None, :])[0],
        errors[0],
    )
    singular = torch.linalg.svdvals(jacobian)
    assert torch.min(singular).item() >= 0.75 - 1.0e-5
    assert torch.max(singular).item() <= 1.25 + 1.0e-5


def test_projected_constant_gain_respects_hard_trust_region() -> None:
    torch = pytest.importorskip("torch")
    base = np.arange(21, dtype=float).reshape(7, 3) + 1.0
    gain = build_projected_constant_gain(torch, base, trust_ratio=0.25)
    with torch.no_grad():
        gain.delta.fill_(100.0)
    gain.project_()

    assert gain.relative_delta_norm() <= 0.25 + 1.0e-6
    assert tuple(gain().shape) == (7, 3)
    assert tuple(gain(4).shape) == (4, 7, 3)


def test_spectral_projection_is_reapplied_after_precision_conversion() -> None:
    torch = pytest.importorskip("torch")
    model = build_conditional_residual_transform(
        torch, 7, hidden_width=16, hidden_layers=3, rho=0.5
    )
    with torch.no_grad():
        for weight in model.spectral_weights():
            weight.mul_(1.0001)
    model = model.to(dtype=torch.float64)
    model.project_spectral_()

    assert 2.0 * model.residual_lipschitz_bound() <= 0.5 + 1.0e-10


def test_four_term_objective_backpropagates_to_transform_and_gain() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(21)
    grid = AllenCahnGrid(5)
    matrix = torch.randn(2, grid.n, dtype=torch.float32)
    base_gain = np.random.default_rng(21).normal(size=(grid.n, 2))
    gain = build_projected_constant_gain(torch, base_gain, trust_ratio=0.25)
    transform = build_conditional_residual_transform(
        torch, grid.n, hidden_width=8, hidden_layers=2, rho=0.5
    )
    states = torch.randn(4, grid.n, dtype=torch.float32)
    estimates = states + 0.1 * torch.randn_like(states)
    samples = {
        "states": states,
        "estimates": estimates,
        "measurements": states @ matrix.T,
        "laplacian": torch.as_tensor(grid.laplacian, dtype=torch.float32),
    }
    components = _instantaneous_components(
        torch,
        gain,
        transform,
        samples,
        matrix,
        torch.arange(states.shape[0]),
        grid,
        create_graph=True,
        include_inverse=True,
        inverse_iterations=4,
    )
    truth = states[:2, None, :].repeat(1, OUTPUT_TIMES.size, 1)
    rollouts = {
        "states": truth,
        "estimate_initials": estimates[:2],
    }
    online = _differentiable_online_loss(
        torch,
        gain,
        rollouts,
        torch.arange(2),
        matrix,
        samples["laplacian"],
        grid,
    )
    total = (
        components["contraction"]
        + components["defect"]
        + components["invertibility"]
        + online
    )

    total.backward()

    transform_gradient = sum(
        float(torch.linalg.vector_norm(parameter.grad))
        for parameter in transform.parameters()
        if parameter.grad is not None
    )
    assert gain.delta.grad is not None
    assert torch.all(torch.isfinite(gain.delta.grad))
    assert torch.linalg.vector_norm(gain.delta.grad).item() > 0.0
    assert transform_gradient > 0.0
