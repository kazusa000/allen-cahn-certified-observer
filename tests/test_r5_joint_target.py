import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_tk_joint_train import (
    _target_diffusion_half_maps,
    _target_generator_tensor,
    _target_operators,
    _target_step_tensor,
)

from allen_cahn_certified_observer import AllenCahnGrid


def test_r5_target_uses_stable_diffusion_shift() -> None:
    grid = AllenCahnGrid(15)
    nu_values = (0.005, 0.01, 0.02)
    lambda_ratio = 0.5

    generators, maps = _target_operators(grid, nu_values, lambda_ratio)

    identity = np.eye(grid.n)
    for nu, generator, step_map in zip(
        nu_values, generators, maps, strict=True
    ):
        expected = (
            nu * grid.laplacian
            - lambda_ratio * nu * np.pi**2 * identity
        )
        assert np.allclose(generator, expected)
        assert np.max(np.linalg.eigvalsh(generator)) < 0.0
        assert np.max(np.abs(np.linalg.eigvals(step_map))) < 1.0


def test_nonlinear_target_retains_reaction_increment_and_is_dissipative() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(15)
    generator = torch.Generator().manual_seed(45)
    states = torch.randn((6, grid.n), generator=generator)
    transformed = 0.4 * torch.randn((6, grid.n), generator=generator)
    nus = torch.as_tensor([0.005, 0.01, 0.02, 0.005, 0.01, 0.02])
    laplacian = torch.as_tensor(grid.laplacian, dtype=torch.float32)
    lambda_ratio = 0.1

    actual = _target_generator_tensor(
        torch,
        "nonlinear",
        states,
        transformed,
        nus,
        laplacian,
        lambda_ratio,
    )
    reaction_increment = (
        states + transformed - (states + transformed) ** 3 - states + states**3
    )
    lam = lambda_ratio * nus[:, None] * np.pi**2
    expected = (
        nus[:, None] * (transformed @ laplacian.T)
        + reaction_increment
        - (1.0 + lam) * transformed
    )
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)

    first_decay = -float(np.max(np.linalg.eigvalsh(grid.laplacian)))
    energy_derivative = grid.h * torch.sum(transformed * actual, dim=1)
    guaranteed_decay = nus * first_decay + lambda_ratio * nus * np.pi**2
    squared_mass = grid.h * torch.sum(transformed**2, dim=1)
    assert torch.all(energy_derivative <= -guaranteed_decay * squared_mass + 2e-6)


def test_linearized_target_uses_df_at_the_current_state() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(7)
    states = torch.as_tensor([[0.2] * grid.n], dtype=torch.float32)
    transformed = torch.as_tensor([[0.1] * grid.n], dtype=torch.float32)
    nus = torch.as_tensor([0.01])
    laplacian = torch.as_tensor(grid.laplacian, dtype=torch.float32)
    lambda_ratio = 0.5

    actual = _target_generator_tensor(
        torch,
        "linearized",
        states,
        transformed,
        nus,
        laplacian,
        lambda_ratio,
    )
    lam = lambda_ratio * nus[:, None] * np.pi**2
    expected = (
        nus[:, None] * (transformed @ laplacian.T)
        + (1.0 - 3.0 * states**2) * transformed
        - (1.0 + lam) * transformed
    )
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize("target_kind", ["linearized", "nonlinear"])
def test_state_dependent_target_step_preserves_zero_fiber(target_kind: str) -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(7)
    nu_values = (0.01,)
    states = torch.ones((2, grid.n), dtype=torch.float32)
    next_states = 0.9 * states
    transformed = torch.zeros_like(states)
    nus = torch.full((2,), 0.01)
    nu_indices = torch.zeros(2, dtype=torch.long)
    laplacian = torch.as_tensor(grid.laplacian, dtype=torch.float32)
    _, linear_maps = _target_operators(grid, nu_values, 0.1)
    half_maps = _target_diffusion_half_maps(grid, nu_values, 0.02)

    stepped = _target_step_tensor(
        torch,
        target_kind,
        states,
        next_states,
        transformed,
        nus,
        nu_indices,
        laplacian,
        torch.as_tensor(linear_maps, dtype=torch.float32),
        torch.as_tensor(half_maps, dtype=torch.float32),
        0.1,
        0.02,
    )
    assert torch.count_nonzero(stepped).item() == 0
