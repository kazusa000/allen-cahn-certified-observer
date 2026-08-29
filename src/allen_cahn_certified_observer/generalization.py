"""Decision rules for adaptive-sensor generalization audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PracticalGateThresholds:
    """Tolerant empirical gates kept separate from strict certificates."""

    collocation_fraction_min: float = 0.995
    collocation_p01_margin_min: float = 0.0
    collocation_worst_margin_min: float = -0.10
    trajectory_fraction_min: float = 0.99
    trajectory_p01_margin_min: float = 0.0
    trajectory_worst_margin_min: float = -0.02
    online_terminal_median_ratio_max: float = 1.10
    online_terminal_max_ratio_max: float = 1.20

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def unstable_mode_count(nu: float) -> int:
    """Return the number of positive zero-state Allen--Cahn modal rates."""

    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu must be a positive finite scalar")
    count = 0
    mode = 1
    while 1.0 - nu * (mode * np.pi) ** 2 > 0.0:
        count += 1
        mode += 1
    return count


def adaptive_sensor_counts(nu: float) -> tuple[int, int]:
    """Return primary and positive-control sensor counts for one ``nu``."""

    unstable = unstable_mode_count(nu)
    return max(1, unstable - 1), unstable


def strict_rate_gate(summary: dict[str, float | int]) -> dict[str, bool]:
    """Evaluate the all-sample nonnegative-margin certificate gate."""

    finite = _finite_rate_summary(summary)
    return {
        "finite": finite,
        "all_margins_nonnegative": bool(
            finite and float(summary["requested_margin_min"]) >= -1.0e-8
        ),
    }


def practical_rate_gate(
    summary: dict[str, float | int],
    *,
    fraction_min: float,
    p01_margin_min: float,
    worst_margin_min: float,
) -> dict[str, bool]:
    """Evaluate a tolerant distributional gate without hiding the worst point."""

    finite = _finite_rate_summary(summary)
    checks = {
        "finite": finite,
        "fraction": bool(
            finite and float(summary["requested_rate_fraction"]) >= fraction_min
        ),
        "p01_margin": bool(
            finite and float(summary["requested_margin_p01"]) >= p01_margin_min
        ),
        "worst_margin_floor": bool(
            finite and float(summary["requested_margin_min"]) >= worst_margin_min
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def generalization_gates(
    grids: dict[str, object],
    structure: dict[str, object],
    thresholds: PracticalGateThresholds | None = None,
) -> dict[str, object]:
    """Return strict and practical decisions for arbitrary evaluation grids."""

    limits = thresholds or PracticalGateThresholds()
    strict_per_grid: dict[str, object] = {}
    practical_per_grid: dict[str, object] = {}
    for grid_size, value in grids.items():
        result = value
        collocation = result["collocation"]
        trajectory = result["trajectory"]
        rollout = result["rollout"]
        online_finite = bool(
            np.isfinite(float(rollout["terminal_median_ratio_to_B0"]))
            and np.isfinite(float(rollout["terminal_max_ratio_to_B0"]))
        )

        strict = {
            "collocation": strict_rate_gate(collocation),
            "trajectory": strict_rate_gate(trajectory),
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
            "collocation": practical_rate_gate(
                collocation,
                fraction_min=limits.collocation_fraction_min,
                p01_margin_min=limits.collocation_p01_margin_min,
                worst_margin_min=limits.collocation_worst_margin_min,
            ),
            "trajectory": practical_rate_gate(
                trajectory,
                fraction_min=limits.trajectory_fraction_min,
                p01_margin_min=limits.trajectory_p01_margin_min,
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


def _finite_rate_summary(summary: dict[str, float | int]) -> bool:
    names = (
        "requested_margin_min",
        "requested_margin_p01",
        "requested_rate_fraction",
    )
    return bool(all(np.isfinite(float(summary[name])) for name in names))

