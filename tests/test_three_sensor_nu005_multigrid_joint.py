import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

import r5_oblique_joint_train as oblique
import r5_three_sensor_dynamics_joint as joint
from r5_three_sensor_nu005_multigrid_joint import (
    GRID_SIZES,
    NU_VALUE,
    _configure_nu005,
    _grid_gates,
    _trend_summary,
)


def _rollout(value: float = 1.0) -> dict[str, object]:
    return {"by_nu": {"0.005": {"terminal_error_mass_median": value}}}


def _selected(rms: float, p95: float, margin: float = 0.01) -> dict[str, object]:
    return {
        "validation": {
            "defect": {"rms": rms, "p95": p95},
            "contraction": {"requested_margin_min": margin},
        },
        "structure": {"passed": True},
        "validation_rollout": _rollout(),
    }


def _trend_record(
    ratio: float,
    defect: float,
    margin: float,
    online: float,
    *,
    passed: bool = True,
) -> dict[str, object]:
    return {
        "metrics": {
            "defect_rms_ratio": ratio,
            "defect_rms": defect,
            "requested_margin_min": margin,
            "online_terminal_ratio": online,
        },
        "gates": {"all_passed": passed},
    }


def test_configure_nu005_restricts_both_reused_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oblique, "NU_VALUES", (0.005, 0.01, 0.02))
    monkeypatch.setattr(joint, "NU_VALUES", (0.005, 0.01, 0.02))

    _configure_nu005()

    assert oblique.NU_VALUES == (NU_VALUE,)
    assert joint.NU_VALUES == (NU_VALUE,)


def test_grid_gate_uses_only_nu005_online_and_frozen_progress_thresholds() -> None:
    fixed = {
        "validation": {"defect": {"rms": 1.0, "p95": 2.0}},
        "validation_rollout": _rollout(),
    }

    passed = _grid_gates(_selected(0.75, 1.70), fixed)
    failed = _grid_gates(_selected(0.751, 1.70), fixed)

    assert passed["all_passed"]
    assert not failed["progress"]["defect_rms_vs_fixed_T_25pct"]
    assert not failed["all_passed"]


def test_trend_summary_accepts_stable_multigrid_pattern() -> None:
    records = {
        str(size): record
        for size, record in zip(
            GRID_SIZES,
            (
                _trend_record(0.70, 0.60, 0.01, 1.00),
                _trend_record(0.72, 0.58, 0.02, 0.99),
                _trend_record(0.74, 0.56, 0.03, 0.98),
            ),
            strict=True,
        )
    }

    trend = _trend_summary(records)

    assert trend["all_grids_passed"]
    assert trend["ratio_spread_at_most_0.10"]
    assert trend["no_monotonic_worsening"]
    assert trend["mesh_robust"]


def test_trend_summary_flags_monotonic_deterioration() -> None:
    records = {
        str(size): record
        for size, record in zip(
            GRID_SIZES,
            (
                _trend_record(0.70, 0.50, 0.03, 0.98),
                _trend_record(0.72, 0.55, 0.02, 0.99),
                _trend_record(0.74, 0.60, 0.01, 1.00),
            ),
            strict=True,
        )
    }

    trend = _trend_summary(records)

    assert trend["monotonic_worsening"] == {
        "defect_rms": True,
        "requested_margin_min": True,
        "online_terminal_ratio": True,
    }
    assert not trend["no_monotonic_worsening"]
    assert not trend["mesh_robust"]
