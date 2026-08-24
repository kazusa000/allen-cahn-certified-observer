import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_direct_fiber_multigrid_joint import (
    ALPHA,
    THREE_SENSOR_INTERVALS,
    _collocation_samples,
    _fiber_components,
    _positive_control,
    _torch_context,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    build_low_modal_conditional_residual_transform,
    build_projected_constant_gain,
    dirichlet_sine_basis,
    local_average_matrix,
    mesh_shared_fiber_transform,
    modal_residual_jacobian_bounds,
    modal_residual_path_layer_bound,
    physical_modal_coordinates,
    physical_modal_injection,
    reconstruct_physical_modes,
)


def test_modal_residual_budget_has_no_spurious_factor_two() -> None:
    lower, upper = modal_residual_jacobian_bounds(0.35)

    assert lower == pytest.approx(0.65)
    assert upper == pytest.approx(1.35)
    assert modal_residual_path_layer_bound(0.35, 4) ** 4 == pytest.approx(0.35)


@pytest.mark.parametrize("grid_size", [31, 63, 127])
def test_physical_modal_coordinates_are_mesh_independent(grid_size: int) -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(grid_size)
    basis = torch.as_tensor(dirichlet_sine_basis(grid, 8), dtype=torch.float64)
    coefficients = torch.tensor(
        [[0.3, -0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    values = reconstruct_physical_modes(torch, coefficients, basis, grid.h)
    recovered = physical_modal_coordinates(torch, values, basis, grid.h)

    assert torch.allclose(recovered, coefficients, atol=1.0e-12, rtol=1.0e-12)


def test_low_modal_transform_is_zero_fiber_invertible_and_bounded() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(31)
    base = np.diag([0.8, 1.0, 1.2, 1.5])
    model = build_low_modal_conditional_residual_transform(
        torch,
        base,
        state_dimension=8,
        hidden_width=16,
        hidden_layers=2,
        rho=0.35,
    ).to(dtype=torch.float64)
    model.project_spectral_()
    states = torch.randn(7, 8, dtype=torch.float64)
    errors = 0.2 * torch.randn(7, 4, dtype=torch.float64)

    assert torch.max(torch.abs(model(states, torch.zeros_like(errors)))).item() == 0.0
    assert model.residual_lipschitz_bound() <= 0.35 + 1.0e-10

    transformed = model(states, errors)
    reconstructed, converged, _ = model.inverse_fixed_point_diagnostics(
        states, transformed, max_iterations=80, tolerance=1.0e-10
    )
    assert bool(torch.all(converged))
    assert torch.max(torch.abs(reconstructed - errors)).item() < 1.0e-8

    jacobian = torch.autograd.functional.jacobian(
        lambda value: model(states[:1], value[None, :])[0], errors[0]
    )
    singular = torch.linalg.svdvals(jacobian)
    assert torch.min(singular).item() >= model.lower_jacobian_bound - 1.0e-8
    assert torch.max(singular).item() <= model.upper_jacobian_bound + 1.0e-8


def test_mesh_shared_transform_changes_only_low_modes() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(32)
    grid = AllenCahnGrid(31)
    condition_basis = torch.as_tensor(
        dirichlet_sine_basis(grid, 8), dtype=torch.float64
    )
    low_basis = condition_basis[:, :4]
    base = np.diag([0.8, 1.0, 1.2, 1.5])
    model = build_low_modal_conditional_residual_transform(
        torch,
        base,
        state_dimension=8,
        hidden_width=16,
        hidden_layers=2,
        rho=0.35,
    ).to(dtype=torch.float64)
    states = torch.randn(5, grid.n, dtype=torch.float64)
    errors = torch.randn(5, grid.n, dtype=torch.float64)

    transformed = mesh_shared_fiber_transform(
        torch,
        model,
        states,
        errors,
        condition_basis,
        low_basis,
        grid.h,
    )
    difference = transformed - errors
    high_coordinates = physical_modal_coordinates(
        torch, difference, condition_basis[:, 4:], grid.h
    )

    assert torch.max(torch.abs(high_coordinates)).item() < 1.0e-12


def test_physical_modal_gain_lift_preserves_modal_gain() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(63)
    basis = torch.as_tensor(
        dirichlet_sine_basis(grid, 4), dtype=torch.float64
    )
    beta = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 10.0

    injection = physical_modal_injection(torch, beta, basis, grid.h)
    recovered = float(np.sqrt(grid.h)) * (basis.T @ injection)

    assert torch.allclose(recovered, beta, atol=1.0e-12, rtol=1.0e-12)


def test_collocation_generator_obeys_frozen_physical_domain() -> None:
    grid = AllenCahnGrid(31)
    observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    samples = _collocation_samples(grid, observation, seed=1801, count=256)
    error_norms = np.sqrt(grid.h * np.sum(samples["errors"] ** 2, axis=1))

    assert samples["states"].shape == (256, grid.n)
    assert samples["errors"].shape == (256, grid.n)
    assert np.max(np.abs(samples["states"])) <= 1.25 + 1.0e-12
    assert np.min(error_norms) >= 0.02 - 1.0e-12
    assert np.max(error_norms) <= 0.80 + 1.0e-12


def test_direct_fiber_rate_backpropagates_to_gain_and_transform() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(33)
    context = _torch_context(torch, 31, device="cpu", dtype=torch.float64)
    transform = build_low_modal_conditional_residual_transform(
        torch,
        np.eye(4),
        state_dimension=8,
        hidden_width=12,
        hidden_layers=2,
        rho=0.35,
    ).to(dtype=torch.float64)
    gain = build_projected_constant_gain(
        torch, np.ones((4, 3)), trust_ratio=0.25
    ).to(dtype=torch.float64)
    states = 0.2 * torch.randn(8, 31, dtype=torch.float64)
    errors = 0.1 * torch.randn(8, 31, dtype=torch.float64)

    components = _fiber_components(
        torch,
        transform,
        gain,
        states,
        errors,
        context,
        create_graph=True,
    )
    loss = torch.mean((components["rates"] - ALPHA) ** 2)
    loss.backward()

    transform_gradient = sum(
        float(torch.linalg.vector_norm(parameter.grad))
        for parameter in transform.parameters()
        if parameter.grad is not None
    )
    assert gain.delta.grad is not None
    assert torch.linalg.vector_norm(gain.delta.grad).item() > 0.0
    assert transform_gradient > 0.0


def test_four_sensor_positive_control_passes_every_grid() -> None:
    result = _positive_control()

    assert result["all_grids_passed"]
    assert all(
        item["global_semidiscrete_margin"] > 0.0
        for item in result["grids"].values()
    )
