import numpy as np

from allen_cahn_certified_observer import AllenCahnGrid
from allen_cahn_certified_observer.spectral import (
    audit_high_frequency_tail,
    dirichlet_laplacian_rates,
    dirichlet_sine_basis,
    low_frequency_projector,
    mass_norm,
    sampled_forced_tail_envelope,
    split_low_tail,
)


def test_discrete_sine_basis_is_orthonormal_and_diagonalizes_laplacian() -> None:
    grid = AllenCahnGrid(31)
    basis = dirichlet_sine_basis(grid, 8)
    rates = dirichlet_laplacian_rates(grid)

    assert np.allclose(basis.T @ basis, np.eye(8), atol=1.0e-13)
    assert np.allclose(
        grid.laplacian @ basis,
        -basis * rates[:8][None, :],
        atol=1.0e-11,
    )


def test_low_tail_split_is_orthogonal_in_physical_mass() -> None:
    grid = AllenCahnGrid(31)
    generator = np.random.Generator(np.random.PCG64DXSM(42))
    values = generator.normal(size=(5, grid.n))

    low, tail = split_low_tail(grid, values, 8)

    assert np.allclose(low + tail, values)
    assert np.allclose(grid.h * np.sum(low * tail, axis=1), 0.0, atol=1.0e-13)
    assert np.allclose(
        mass_norm(grid, values) ** 2,
        mass_norm(grid, low) ** 2 + mass_norm(grid, tail) ** 2,
    )


def test_projector_definition_is_grid_consistent_for_physical_modes() -> None:
    for n in (31, 63, 127):
        grid = AllenCahnGrid(n)
        projector = low_frequency_projector(grid, 8)
        physical_mode = np.sin(8.0 * np.pi * grid.x)
        first_tail_mode = np.sin(9.0 * np.pi * grid.x)

        assert np.allclose(projector @ physical_mode, physical_mode, atol=1.0e-12)
        assert np.allclose(projector @ first_tail_mode, 0.0, atol=1.0e-12)


def test_tail_audit_uses_positive_diffusion_margin_and_valid_bound() -> None:
    grid = AllenCahnGrid(31)
    generator = np.random.Generator(np.random.PCG64DXSM(43))
    states = 0.2 * generator.normal(size=(12, grid.n))
    errors = 0.05 * generator.normal(size=(12, grid.n))
    corrections = 0.01 * generator.normal(size=(12, grid.n))

    audit = audit_high_frequency_tail(
        grid, 0.005, states, errors, corrections, mode_count=8
    )

    assert audit.diffusion_margin > 0.0
    assert audit.dissipativity_violation_max <= 1.0e-14
    assert audit.inequality_residual_max <= 1.0e-12
    assert np.allclose(audit.total_norm**2, audit.low_norm**2 + audit.tail_norm**2)


def test_sampled_tail_envelope_resets_and_bounds_constant_forcing_solution() -> None:
    times = np.asarray([0.0, 0.1, 0.2, 0.0, 0.1])
    margin = 2.0
    forcing = np.ones_like(times) * 0.5
    exact = 0.25 * (1.0 - np.exp(-margin * times))

    envelope = sampled_forced_tail_envelope(times, exact, forcing, margin)

    assert np.allclose(envelope, exact)
