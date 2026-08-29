import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_h_state_ood_audit import _robustness_envelope

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    STATE_OOD_FAMILIES,
    StateOODGateThresholds,
    dirichlet_sine_basis,
    mass_norm,
    state_ood_gates,
    state_ood_initial_pairs,
    state_ood_samples,
)


@pytest.mark.parametrize("family", STATE_OOD_FAMILIES)
@pytest.mark.parametrize("severity", (1, 2, 3))
def test_ood_samples_are_deterministic_finite_and_nonzero(
    family: str, severity: int
) -> None:
    grid = AllenCahnGrid(63)
    first = state_ood_samples(grid, family, severity, seed=1971, count=12)
    second = state_ood_samples(grid, family, severity, seed=1971, count=12)

    np.testing.assert_array_equal(first["states"], second["states"])
    np.testing.assert_array_equal(first["errors"], second["errors"])
    assert first["states"].shape == (12, 63)
    assert first["errors"].shape == (12, 63)
    assert np.all(np.isfinite(first["states"]))
    assert np.all(np.isfinite(first["errors"]))
    assert np.all(np.max(np.abs(first["states"]), axis=1) > 0.0)
    assert np.all(mass_norm(grid, first["errors"]) > 0.0)


@pytest.mark.parametrize(
    ("family", "severity", "state_max", "error_radius"),
    [
        ("truth_high_frequency", 2, 1.0, 0.20),
        ("error_high_frequency", 3, 0.8, 0.25),
        ("localized_pulse", 3, 1.0, 0.25),
        ("multiple_interfaces", 3, 0.9, 0.25),
        ("large_initial_error", 1, 0.8, 0.40),
        ("large_initial_error", 2, 0.8, 0.80),
        ("large_initial_error", 3, 0.8, 1.20),
    ],
)
def test_ood_samples_match_frozen_amplitude_and_radius(
    family: str, severity: int, state_max: float, error_radius: float
) -> None:
    grid = AllenCahnGrid(191)
    samples = state_ood_samples(grid, family, severity, seed=1971, count=9)

    np.testing.assert_allclose(
        np.max(np.abs(samples["states"]), axis=1), state_max, atol=1.0e-12
    )
    np.testing.assert_allclose(
        mass_norm(grid, samples["errors"]), error_radius, atol=1.0e-12
    )


@pytest.mark.parametrize("severity,band", [(1, (5, 8)), (2, (9, 12)), (3, (13, 16))])
def test_high_frequency_truth_occupies_only_frozen_band(
    severity: int, band: tuple[int, int]
) -> None:
    grid = AllenCahnGrid(63)
    states = state_ood_samples(
        grid, "truth_high_frequency", severity, seed=1971, count=7
    )["states"]
    physical_basis = dirichlet_sine_basis(grid, 20) / np.sqrt(grid.h)
    coefficients = grid.h * states @ physical_basis
    outside = coefficients.copy()
    outside[:, band[0] - 1 : band[1]] = 0.0

    assert np.max(np.abs(outside)) < 1.0e-12
    assert np.min(np.linalg.norm(coefficients[:, band[0] - 1 : band[1]], axis=1)) > 0.0


def test_trajectory_pairs_are_disjoint_and_well_formed() -> None:
    grid = AllenCahnGrid(63)
    pairs = state_ood_initial_pairs(
        grid, "localized_pulse", 2, seed=1971, count=8
    )
    collocation = state_ood_samples(
        grid, "localized_pulse", 2, seed=1971, count=8
    )

    assert len(pairs) == 8
    assert len({pair.case_id for pair in pairs}) == 8
    assert all(pair.truth_initial.shape == (63,) for pair in pairs)
    assert all(pair.estimate_initial.shape == (63,) for pair in pairs)
    assert not np.array_equal(pairs[0].truth_initial, collocation["states"][0])
    assert mass_norm(grid, pairs[0].estimate_initial - pairs[0].truth_initial) == pytest.approx(0.25)


def _rate(*, fraction: float, p05: float, worst: float) -> dict[str, float]:
    return {
        "requested_rate_fraction": fraction,
        "requested_margin_p05": p05,
        "requested_margin_min": worst,
    }


def _grid(
    *,
    collocation_fraction: float = 0.96,
    collocation_p05: float = 0.01,
    collocation_worst: float = -0.20,
    trajectory_fraction: float = 0.92,
    trajectory_p05: float = 0.01,
    trajectory_worst: float = -0.08,
    median_ratio: float = 1.20,
    max_ratio: float = 1.40,
) -> dict[str, object]:
    return {
        "collocation": _rate(
            fraction=collocation_fraction,
            p05=collocation_p05,
            worst=collocation_worst,
        ),
        "trajectory": _rate(
            fraction=trajectory_fraction,
            p05=trajectory_p05,
            worst=trajectory_worst,
        ),
        "rollout": {
            "terminal_median_ratio_to_B0": median_ratio,
            "terminal_max_ratio_to_B0": max_ratio,
        },
    }


def test_ood_gate_allows_declared_bounded_tail_but_not_strict_gate() -> None:
    gates = state_ood_gates(
        {"63": _grid(), "191": _grid()},
        {"passed": True},
        StateOODGateThresholds(),
    )

    assert gates["practical"]["all_grids_passed"]
    assert not gates["strict"]["all_grids_passed"]


def test_ood_gate_rejects_bad_quantile_or_online_regression() -> None:
    gates = state_ood_gates(
        {
            "63": _grid(collocation_p05=-0.01),
            "191": _grid(max_ratio=1.51),
        },
        {"passed": True},
    )

    assert not gates["practical"]["all_grids_passed"]
    assert not gates["practical"]["per_grid"]["63"]["collocation"]["p05_margin"]
    assert not gates["practical"]["per_grid"]["191"]["online_terminal_max"]


def test_robustness_envelope_reports_nonmonotone_cells() -> None:
    cells = {
        "1": {"gates": {"practical": {"all_grids_passed": True}}},
        "2": {"gates": {"practical": {"all_grids_passed": False}}},
        "3": {"gates": {"practical": {"all_grids_passed": True}}},
    }

    envelope = _robustness_envelope(cells, "practical")

    assert envelope["passing_severities"] == [1, 3]
    assert envelope["highest_passing_severity"] == 3
    assert envelope["contiguous_envelope"] == 1
    assert envelope["nonmonotone"]
