import json
import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_g_generalization_audit import _validate_unlock

from allen_cahn_certified_observer import (
    PracticalGateThresholds,
    generalization_gates,
)


def _rate(*, fraction: float, p01: float, worst: float) -> dict[str, float]:
    return {
        "requested_rate_fraction": fraction,
        "requested_margin_p01": p01,
        "requested_margin_min": worst,
    }


def _grid(
    *,
    collocation_fraction: float = 0.996,
    collocation_p01: float = 0.01,
    collocation_worst: float = -0.05,
    trajectory_fraction: float = 1.0,
    trajectory_p01: float = 0.02,
    trajectory_worst: float = 0.01,
    median_ratio: float = 1.05,
    max_ratio: float = 1.15,
) -> dict[str, object]:
    return {
        "collocation": _rate(
            fraction=collocation_fraction,
            p01=collocation_p01,
            worst=collocation_worst,
        ),
        "trajectory": _rate(
            fraction=trajectory_fraction,
            p01=trajectory_p01,
            worst=trajectory_worst,
        ),
        "rollout": {
            "terminal_median_ratio_to_B0": median_ratio,
            "terminal_max_ratio_to_B0": max_ratio,
        },
    }


def test_practical_gate_allows_rare_bounded_tail_failure() -> None:
    gates = generalization_gates(
        {"63": _grid()}, {"passed": True}, PracticalGateThresholds()
    )

    assert gates["practical"]["all_grids_passed"]
    assert not gates["strict"]["all_grids_passed"]


def test_practical_gate_rejects_material_tail_or_online_regression() -> None:
    grids = {
        "63": _grid(collocation_worst=-0.11),
        "127": _grid(max_ratio=1.21),
    }
    gates = generalization_gates(grids, {"passed": True})

    assert not gates["practical"]["all_grids_passed"]
    assert not gates["practical"]["per_grid"]["63"]["collocation"][
        "worst_margin_floor"
    ]
    assert not gates["practical"]["per_grid"]["127"]["online_terminal_max"]


def test_structure_failure_blocks_both_decision_levels() -> None:
    gates = generalization_gates({"31": _grid(collocation_worst=0.01)}, {"passed": False})

    assert not gates["strict"]["all_grids_passed"]
    assert not gates["practical"]["all_grids_passed"]


def test_locked_test_requires_full_practical_validation(tmp_path: Path) -> None:
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "phase": "validation",
                "evaluation_only": True,
                "split_seed": 1871,
                "checkpoint": {"sha256": "abc"},
                "frozen": {
                    "grid_sizes": [31, 47, 63, 95, 127],
                    "collocation_count_per_grid": 4096,
                },
                "gates": {"practical": {"all_grids_passed": True}},
            }
        ),
        encoding="utf-8",
    )

    payload = _validate_unlock(validation, "abc")

    assert payload["split_seed"] == 1871


def test_locked_test_rejects_smoke_validation(tmp_path: Path) -> None:
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "phase": "validation",
                "evaluation_only": True,
                "split_seed": 1871,
                "checkpoint": {"sha256": "abc"},
                "frozen": {
                    "grid_sizes": [31, 47, 63, 95, 127],
                    "collocation_count_per_grid": 128,
                },
                "gates": {"practical": {"all_grids_passed": True}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="4096"):
        _validate_unlock(validation, "abc")
