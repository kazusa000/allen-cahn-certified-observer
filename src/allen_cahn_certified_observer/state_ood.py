"""Frozen state-distribution shifts for the R5-H evaluation-only audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .grid import AllenCahnGrid
from .spectral import dirichlet_sine_basis, mass_norm


STATE_OOD_FAMILIES = (
    "truth_high_frequency",
    "error_high_frequency",
    "localized_pulse",
    "multiple_interfaces",
    "large_initial_error",
)
STATE_OOD_SEVERITIES = (1, 2, 3)

_HIGH_FREQUENCY_BANDS = {1: (5, 8), 2: (9, 12), 3: (13, 16)}
_PULSE_WIDTHS = {1: 0.20, 2: 0.10, 3: 0.05}
_INTERFACE_COUNTS = {1: 1, 2: 2, 3: 3}
_ERROR_RADII = {1: 0.40, 2: 0.80, 3: 1.20}


@dataclass(frozen=True)
class StateOODInitialPair:
    """One frozen initial truth/estimate pair for an online rollout."""

    case_id: str
    truth_initial: np.ndarray
    estimate_initial: np.ndarray


@dataclass(frozen=True)
class StateOODGateThresholds:
    """Tolerant OOD gates, kept separate from strict certificate checks."""

    collocation_fraction_min: float = 0.95
    collocation_p05_margin_min: float = 0.0
    collocation_worst_margin_min: float = -0.25
    trajectory_fraction_min: float = 0.90
    trajectory_p05_margin_min: float = 0.0
    trajectory_worst_margin_min: float = -0.10
    online_terminal_median_ratio_max: float = 1.25
    online_terminal_max_ratio_max: float = 1.50

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def state_ood_samples(
    grid: AllenCahnGrid,
    family: str,
    severity: int,
    *,
    seed: int,
    count: int,
) -> dict[str, np.ndarray]:
    """Generate one deterministic OOD collocation cell on a physical grid."""

    _validate_cell(family, severity, count)
    family_index = STATE_OOD_FAMILIES.index(family)
    generator = np.random.Generator(
        np.random.PCG64DXSM(int(seed) + 100_003 * family_index + 10_007 * severity)
    )

    if family == "truth_high_frequency":
        states = _random_modal_values(
            grid, generator, count, _HIGH_FREQUENCY_BANDS[severity]
        )
        states = _scale_maximum(states, 1.0)
        errors = _scale_mass(
            grid, _random_modal_values(grid, generator, count, (1, 4)), 0.20
        )
    elif family == "error_high_frequency":
        states = _scale_maximum(
            _random_modal_values(grid, generator, count, (1, 3)), 0.8
        )
        errors = _scale_mass(
            grid,
            _random_modal_values(
                grid, generator, count, _HIGH_FREQUENCY_BANDS[severity]
            ),
            0.25,
        )
    elif family == "localized_pulse":
        states, errors = _localized_pulses(
            grid, generator, count, width=_PULSE_WIDTHS[severity]
        )
    elif family == "multiple_interfaces":
        states, errors = _multiple_interfaces(
            grid,
            generator,
            count,
            interface_count=_INTERFACE_COUNTS[severity],
        )
    else:
        states = _scale_maximum(
            _random_modal_values(grid, generator, count, (1, 3)), 0.8
        )
        errors = _scale_mass(
            grid,
            _random_modal_values(grid, generator, count, (1, 4)),
            _ERROR_RADII[severity],
        )

    if states.shape != (count, grid.n) or errors.shape != (count, grid.n):
        raise RuntimeError("OOD generator returned the wrong shape")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(errors)):
        raise RuntimeError("OOD generator returned non-finite values")
    if np.any(np.max(np.abs(states), axis=1) <= 0.0):
        raise RuntimeError("OOD generator returned a zero truth state")
    if np.any(mass_norm(grid, errors) <= 0.0):
        raise RuntimeError("OOD generator returned a zero error")
    return {"states": states, "errors": errors}


def state_ood_initial_pairs(
    grid: AllenCahnGrid,
    family: str,
    severity: int,
    *,
    seed: int,
    count: int = 8,
) -> list[StateOODInitialPair]:
    """Generate deterministic rollout cases disjoint from collocation draws."""

    samples = state_ood_samples(
        grid, family, severity, seed=int(seed) + 500_009, count=count
    )
    return [
        StateOODInitialPair(
            case_id=f"ood__{family}__severity-{severity}__case-{index}__n-{grid.n}",
            truth_initial=samples["states"][index].copy(),
            estimate_initial=(samples["states"][index] + samples["errors"][index]),
        )
        for index in range(count)
    ]


def state_ood_gates(
    grids: dict[str, object],
    structure: dict[str, object],
    thresholds: StateOODGateThresholds | None = None,
) -> dict[str, object]:
    """Evaluate strict and deliberately wider practical OOD decisions."""

    limits = thresholds or StateOODGateThresholds()
    strict_per_grid: dict[str, object] = {}
    practical_per_grid: dict[str, object] = {}
    for grid_size, raw_result in grids.items():
        result = raw_result
        collocation = result["collocation"]
        trajectory = result["trajectory"]
        rollout = result["rollout"]
        online_finite = bool(
            np.isfinite(float(rollout["terminal_median_ratio_to_B0"]))
            and np.isfinite(float(rollout["terminal_max_ratio_to_B0"]))
        )

        strict = {
            "collocation": _strict_rate_gate(collocation),
            "trajectory": _strict_rate_gate(trajectory),
            "online_terminal_median": bool(
                online_finite
                and float(rollout["terminal_median_ratio_to_B0"]) <= 1.05
            ),
            "online_terminal_max": bool(
                online_finite
                and float(rollout["terminal_max_ratio_to_B0"]) <= 1.10
            ),
        }
        strict["passed"] = bool(
            strict["collocation"]["all_margins_nonnegative"]
            and strict["trajectory"]["all_margins_nonnegative"]
            and strict["online_terminal_median"]
            and strict["online_terminal_max"]
        )
        strict_per_grid[grid_size] = strict

        practical = {
            "collocation": _practical_ood_rate_gate(
                collocation,
                fraction_min=limits.collocation_fraction_min,
                p05_margin_min=limits.collocation_p05_margin_min,
                worst_margin_min=limits.collocation_worst_margin_min,
            ),
            "trajectory": _practical_ood_rate_gate(
                trajectory,
                fraction_min=limits.trajectory_fraction_min,
                p05_margin_min=limits.trajectory_p05_margin_min,
                worst_margin_min=limits.trajectory_worst_margin_min,
            ),
            "online_finite": online_finite,
            "online_terminal_median": bool(
                online_finite
                and float(rollout["terminal_median_ratio_to_B0"])
                <= limits.online_terminal_median_ratio_max
            ),
            "online_terminal_max": bool(
                online_finite
                and float(rollout["terminal_max_ratio_to_B0"])
                <= limits.online_terminal_max_ratio_max
            ),
        }
        practical["passed"] = bool(
            practical["collocation"]["passed"]
            and practical["trajectory"]["passed"]
            and practical["online_finite"]
            and practical["online_terminal_median"]
            and practical["online_terminal_max"]
        )
        practical_per_grid[grid_size] = practical

    structure_passed = bool(structure.get("passed", False))
    return {
        "thresholds": limits.to_dict(),
        "strict": {
            "structure": structure_passed,
            "per_grid": strict_per_grid,
            "all_grids_passed": bool(
                structure_passed
                and all(value["passed"] for value in strict_per_grid.values())
            ),
        },
        "practical": {
            "structure": structure_passed,
            "per_grid": practical_per_grid,
            "all_grids_passed": bool(
                structure_passed
                and all(value["passed"] for value in practical_per_grid.values())
            ),
        },
    }


def _validate_cell(family: str, severity: int, count: int) -> None:
    if family not in STATE_OOD_FAMILIES:
        raise ValueError(f"unknown OOD family: {family}")
    if severity not in STATE_OOD_SEVERITIES:
        raise ValueError("severity must be 1, 2, or 3")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")


def _physical_modal_basis(grid: AllenCahnGrid, first: int, last: int) -> np.ndarray:
    full = dirichlet_sine_basis(grid, last) / np.sqrt(grid.h)
    return full[:, first - 1 : last]


def _random_modal_values(
    grid: AllenCahnGrid,
    generator: np.random.Generator,
    count: int,
    band: tuple[int, int],
) -> np.ndarray:
    basis = _physical_modal_basis(grid, *band)
    coefficients = generator.normal(size=(count, basis.shape[1]))
    coefficients /= np.maximum(
        np.linalg.norm(coefficients, axis=1, keepdims=True), 1.0e-12
    )
    return coefficients @ basis.T


def _scale_maximum(values: np.ndarray, target: float) -> np.ndarray:
    scale = target / np.maximum(np.max(np.abs(values), axis=1), 1.0e-12)
    return values * scale[:, None]


def _scale_mass(
    grid: AllenCahnGrid, values: np.ndarray, target: float
) -> np.ndarray:
    scale = target / np.maximum(mass_norm(grid, values), 1.0e-12)
    return values * scale[:, None]


def _localized_pulses(
    grid: AllenCahnGrid,
    generator: np.random.Generator,
    count: int,
    *,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = generator.uniform(0.20, 0.80, size=count)
    signs = generator.choice(np.asarray([-1.0, 1.0]), size=count)
    envelope = np.exp(-0.5 * ((grid.x[None, :] - centers[:, None]) / width) ** 2)
    states = signs[:, None] * np.sin(np.pi * grid.x)[None, :] * envelope
    states = _scale_maximum(states, 1.0)

    offsets = generator.choice(np.asarray([-1.0, 1.0]), size=count) * 0.5 * width
    error_centers = np.clip(centers + offsets, 0.10, 0.90)
    error_envelope = np.exp(
        -0.5 * ((grid.x[None, :] - error_centers[:, None]) / width) ** 2
    )
    errors = signs[:, None] * np.sin(np.pi * grid.x)[None, :] * error_envelope
    return states, _scale_mass(grid, errors, 0.25)


def _multiple_interfaces(
    grid: AllenCahnGrid,
    generator: np.random.Generator,
    count: int,
    *,
    interface_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.linspace(
        1.0 / (interface_count + 1),
        interface_count / (interface_count + 1),
        interface_count,
    )
    centers = base[None, :] + generator.uniform(
        -0.025, 0.025, size=(count, interface_count)
    )
    directions = generator.choice(np.asarray([-1.0, 1.0]), size=count)
    shifted = np.clip(centers + 0.02 * directions[:, None], 0.06, 0.94)

    def profile(locations: np.ndarray) -> np.ndarray:
        factors = np.tanh(
            (grid.x[None, None, :] - locations[:, :, None]) / 0.03
        )
        return np.sin(np.pi * grid.x)[None, :] * np.prod(factors, axis=1)

    raw_states = profile(centers)
    states = _scale_maximum(raw_states, 0.9)
    shifted_states = _scale_maximum(profile(shifted), 0.9)
    errors = _scale_mass(grid, shifted_states - states, 0.25)
    return states, errors


def _strict_rate_gate(summary: dict[str, float | int]) -> dict[str, bool]:
    finite = _finite_ood_rate_summary(summary)
    return {
        "finite": finite,
        "all_margins_nonnegative": bool(
            finite and float(summary["requested_margin_min"]) >= -1.0e-8
        ),
    }


def _practical_ood_rate_gate(
    summary: dict[str, float | int],
    *,
    fraction_min: float,
    p05_margin_min: float,
    worst_margin_min: float,
) -> dict[str, bool]:
    finite = _finite_ood_rate_summary(summary)
    checks = {
        "finite": finite,
        "fraction": bool(
            finite and float(summary["requested_rate_fraction"]) >= fraction_min
        ),
        "p05_margin": bool(
            finite and float(summary["requested_margin_p05"]) >= p05_margin_min
        ),
        "worst_margin_floor": bool(
            finite and float(summary["requested_margin_min"]) >= worst_margin_min
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def _finite_ood_rate_summary(summary: dict[str, float | int]) -> bool:
    names = (
        "requested_margin_min",
        "requested_margin_p05",
        "requested_rate_fraction",
    )
    return bool(all(np.isfinite(float(summary[name])) for name in names))
