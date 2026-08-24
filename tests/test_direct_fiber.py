import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_direct_fiber_multigrid_joint import (
    ALPHA,
    THREE_SENSOR_INTERVALS,
    _balanced_metric_sqrt,
    _base_design,
    _collocation_samples,
    _fiber_components,
    _fixed_coordinate_contraction_rate,
    _fixed_sine_basis_change,
    _positive_control,
    _torch_context,
)
from r5_direct_fiber_adversarial_repair import (
    _adversarial_low_mode_samples,
    _append_adversary_memory,
    _checkpoint_for_seed,
    _hard_point_neighborhood,
    _seed_for_epoch,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    buffered_contraction_cvar,
    build_low_modal_conditional_residual_transform,
    build_projected_constant_gain,
    dirichlet_sine_basis,
    local_average_matrix,
    lmi_modal_injection,
    mesh_shared_fiber_transform,
    modal_residual_jacobian_bounds,
    modal_residual_path_layer_bound,
    physical_modal_coordinates,
    physical_modal_injection,
    project_physical_modal_adversaries_,
    reconstruct_physical_modes,
    unstable_modal_system,
)


def test_buffered_cvar_keeps_pressure_after_zero_margin() -> None:
    torch = pytest.importorskip("torch")
    margins = torch.tensor([0.01, 0.02, 0.03, 0.05], dtype=torch.float64)

    loss = buffered_contraction_cvar(
        torch, margins, buffer=0.04, tail_fraction=0.5
    )

    assert loss.item() == pytest.approx((0.03**2 + 0.02**2) / 2.0)


def test_modal_adversary_projection_obeys_frozen_domain() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(31)
    basis = torch.as_tensor(
        dirichlet_sine_basis(grid, 8), dtype=torch.float64
    )
    states = torch.full((3, 8), 10.0, dtype=torch.float64)
    errors = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    project_physical_modal_adversaries_(
        torch, states, errors, basis, grid.h
    )
    state_values = (states @ basis.T) / np.sqrt(grid.h)
    radii = torch.linalg.vector_norm(errors, dim=1)

    assert torch.max(torch.abs(state_values)).item() <= 1.25 + 1.0e-12
    assert torch.min(radii).item() >= 0.02 - 1.0e-12
    assert torch.max(radii).item() <= 0.8 + 1.0e-12


def test_resampling_seeds_change_by_model_epoch_and_grid() -> None:
    seeds = {
        _seed_for_epoch(model_seed, epoch, grid)
        for model_seed in (1301, 1302)
        for epoch in (0, 1)
        for grid in (31, 63, 127)
    }

    assert len(seeds) == 12


def test_hard_replay_center_exactly_matches_consumed_bad_point() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(63)
    observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    consumed = _collocation_samples(
        grid, observation, seed=1851, count=2048
    )

    neighborhood = _hard_point_neighborhood(
        torch, count=4, seed=5404, device="cpu"
    )

    assert neighborhood["states"][0] == pytest.approx(
        consumed["states"][471], abs=2.0e-7, rel=2.0e-7
    )
    assert neighborhood["errors"][0] == pytest.approx(
        consumed["errors"][471], abs=2.0e-7, rel=2.0e-7
    )


def test_adversary_memory_accumulates_before_deterministic_truncation() -> None:
    first = {
        "states": np.asarray([[1.0], [2.0]]),
        "errors": np.asarray([[10.0], [20.0]]),
    }
    second = {
        "states": np.asarray([[3.0], [4.0]]),
        "errors": np.asarray([[30.0], [40.0]]),
    }

    accumulated = _append_adversary_memory(None, first, limit=3)
    accumulated = _append_adversary_memory(accumulated, second, limit=3)

    assert accumulated["states"][:, 0].tolist() == [2.0, 3.0, 4.0]
    assert accumulated["errors"][:, 0].tolist() == [20.0, 30.0, 40.0]


@pytest.mark.parametrize(
    "name",
    ["direct-fiber__seed-1303.pt", "direct-fiber-adversarial__seed-1303.pt"],
)
def test_checkpoint_resolver_accepts_original_or_repair(
    tmp_path: Path, name: str
) -> None:
    checkpoint = tmp_path / name
    checkpoint.touch()

    assert _checkpoint_for_seed(tmp_path, 1303) == checkpoint


def test_low_mode_adversarial_search_is_differentiable_and_projected() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("cvxpy")
    base_gain, base_transform, _ = _base_design()
    gain = build_projected_constant_gain(
        torch, base_gain, trust_ratio=0.25
    ).to(dtype=torch.float32)
    transform = build_low_modal_conditional_residual_transform(
        torch,
        base_transform,
        state_dimension=8,
        hidden_width=8,
        hidden_layers=1,
        rho=0.35,
    ).to(dtype=torch.float32)

    samples, diagnostics = _adversarial_low_mode_samples(
        torch,
        transform,
        gain,
        grid_size=63,
        model_seed=13,
        refresh_index=0,
        restart_count=8,
        keep_count=4,
        steps=2,
        step_size=0.001,
        device="cpu",
    )
    grid = AllenCahnGrid(63)
    error_norms = np.sqrt(grid.h * np.sum(samples["errors"] ** 2, axis=1))

    assert samples["states"].shape == (4, 63)
    assert samples["errors"].shape == (4, 63)
    assert np.max(np.abs(samples["states"])) <= 1.25 + 1.0e-5
    assert np.min(error_norms) >= 0.02 - 1.0e-5
    assert np.max(error_norms) <= 0.8 + 1.0e-5
    assert diagnostics["final_margin_min"] <= diagnostics["initial_margin_min"]


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
        error_scale=4.0,
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


def test_fixed_sine_coordinates_exactly_reproduce_lmi_design() -> None:
    pytest.importorskip("cvxpy")
    grid = AllenCahnGrid(31)
    observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    selected = lmi_modal_injection(
        grid,
        0.005,
        observation,
        decay_rate=ALPHA,
        metric_condition_bound=16.0,
    )
    modal = unstable_modal_system(grid, 0.005, observation)
    change = _fixed_sine_basis_change(grid, modal)
    fixed_basis = dirichlet_sine_basis(grid, 4)
    beta = np.sqrt(grid.h) * change @ selected.modal_gain
    modal_transform = _balanced_metric_sqrt(selected.modal_metric)
    transform = change @ modal_transform @ change.T
    lifted_injection = fixed_basis @ beta / np.sqrt(grid.h)
    reproduced_rate = _fixed_coordinate_contraction_rate(
        grid, observation, modal.eigenvalues, beta, transform
    )

    assert np.diag(change) == pytest.approx([-1.0, -1.0, 1.0, 1.0])
    assert lifted_injection == pytest.approx(
        selected.injection_matrix, abs=1.0e-10, rel=1.0e-10
    )
    assert reproduced_rate == pytest.approx(
        selected.modal_contraction_rate, abs=1.0e-10, rel=1.0e-10
    )


def test_base_design_reports_fixed_coordinate_rate_reproduction() -> None:
    pytest.importorskip("cvxpy")
    _, _, diagnostics = _base_design()

    assert diagnostics["eigh_to_fixed_sine_change"] == pytest.approx(
        np.diag([-1.0, -1.0, 1.0, 1.0])
    )
    assert diagnostics["fixed_sine_rate_absolute_error"] <= 1.0e-10
