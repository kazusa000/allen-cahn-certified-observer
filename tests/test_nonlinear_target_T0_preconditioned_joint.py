import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

import r5_nonlinear_target_T0_preconditioned_joint as preconditioned
import r5_nonlinear_target_conditional_residual_joint as joint

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    build_preconditioned_conditional_residual_transform,
    build_projected_constant_gain,
)


def _base_transform(dimension: int) -> np.ndarray:
    diagonal = np.linspace(0.6, 1.8, dimension)
    base = np.diag(diagonal)
    base[0, -1] = 0.15
    return base


def test_zero_residual_recovers_exact_fixed_T0_transform() -> None:
    torch = pytest.importorskip("torch")
    dimension = 7
    base = _base_transform(dimension)
    model = build_preconditioned_conditional_residual_transform(
        torch, base, hidden_width=12, hidden_layers=2, rho=0.5
    )
    with torch.no_grad():
        model.residual_transform.output_layer.weight.zero_()
        model.residual_transform.output_layer.bias.zero_()
    states = torch.randn(5, dimension)
    errors = 0.2 * torch.randn(5, dimension)

    expected = errors @ torch.as_tensor(base, dtype=errors.dtype).T
    assert torch.allclose(model(states, errors), expected)
    assert torch.allclose(model.normalized_forward(states, errors), errors)
    assert torch.count_nonzero(model(states, torch.zeros_like(errors))) == 0


def test_preconditioned_transform_is_invertible_and_respects_bounds() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(1201)
    dimension = 7
    base = _base_transform(dimension)
    model = build_preconditioned_conditional_residual_transform(
        torch, base, hidden_width=16, hidden_layers=3, rho=0.5
    ).to(dtype=torch.float64)
    model.project_spectral_()
    states = torch.randn(6, dimension, dtype=torch.float64)
    errors = 0.2 * torch.randn(6, dimension, dtype=torch.float64)

    transformed = model(states, errors)
    reconstructed, converged, _ = model.inverse_fixed_point_diagnostics(
        states, transformed, max_iterations=50, tolerance=1.0e-10
    )
    assert bool(torch.all(converged))
    assert torch.max(torch.abs(reconstructed - errors)).item() < 1.0e-8
    assert 2.0 * model.residual_lipschitz_bound() <= 0.5 + 1.0e-10
    assert torch.allclose(
        transformed,
        model.normalized_forward(states, errors) @ model.base_transform.T,
    )

    jacobian = torch.autograd.functional.jacobian(
        lambda value: model(states[:1], value[None, :])[0], errors[0]
    )
    singular = torch.linalg.svdvals(jacobian)
    assert torch.min(singular).item() >= model.lower_jacobian_bound - 1.0e-5
    assert torch.max(singular).item() <= model.upper_jacobian_bound + 1.0e-5


def test_preconditioned_four_term_instantaneous_losses_reach_both_models() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(1202)
    grid = AllenCahnGrid(5)
    matrix = torch.randn(2, grid.n, dtype=torch.float32)
    base_gain = np.random.default_rng(1202).normal(size=(grid.n, 2))
    gain = build_projected_constant_gain(torch, base_gain, trust_ratio=0.25)
    transform = build_preconditioned_conditional_residual_transform(
        torch,
        _base_transform(grid.n),
        hidden_width=8,
        hidden_layers=2,
        rho=0.5,
    )
    states = torch.randn(4, grid.n, dtype=torch.float32)
    estimates = states + 0.1 * torch.randn_like(states)
    samples = {
        "states": states,
        "estimates": estimates,
        "measurements": states @ matrix.T,
        "laplacian": torch.as_tensor(grid.laplacian, dtype=torch.float32),
    }

    components = joint._instantaneous_components(
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
    total = (
        components["contraction"]
        + components["defect"]
        + components["invertibility"]
    )
    total.backward()

    assert gain.delta.grad is not None
    assert torch.linalg.vector_norm(gain.delta.grad).item() > 0.0
    transform_gradient = sum(
        float(torch.linalg.vector_norm(parameter.grad))
        for parameter in transform.parameters()
        if parameter.grad is not None
    )
    assert transform_gradient > 0.0


def test_fresh_split_and_model_seeds_are_disjoint_from_prior_run() -> None:
    fresh_case_seeds = (
        set(preconditioned.TRAIN_CASE_SEEDS)
        | set(preconditioned.VALIDATION_CASE_SEEDS)
        | set(preconditioned.TEST_CASE_SEEDS)
    )
    old_case_seeds = (
        set(joint.TRAIN_CASE_SEEDS)
        | set(joint.VALIDATION_CASE_SEEDS)
        | set(joint.TEST_CASE_SEEDS)
    )

    assert not fresh_case_seeds & old_case_seeds
    assert not set(preconditioned.TRAIN_CASE_SEEDS) & set(
        preconditioned.VALIDATION_CASE_SEEDS
    )
    assert not set(preconditioned.TRAIN_CASE_SEEDS) & set(
        preconditioned.TEST_CASE_SEEDS
    )
    assert not set(preconditioned.VALIDATION_CASE_SEEDS) & set(
        preconditioned.TEST_CASE_SEEDS
    )
    assert preconditioned.MODEL_SEEDS == (1201, 1202, 1203)
