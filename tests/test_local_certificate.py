import numpy as np
import pytest

from allen_cahn_certified_observer import (
    LocalPartition,
    partition_samples,
    summarize_local_region,
    transition_counts,
)


def test_partition_is_exhaustive_and_uses_frozen_priority() -> None:
    states = np.asarray(
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.8, 0.0],
            [0.6, -0.6, 0.6, -0.6],
        ]
    )
    errors = np.asarray(
        [
            [0.02, 0.02, 0.02, 0.02],
            [0.10, 0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20, 0.20],
        ]
    )

    result = partition_samples(states, errors, h=0.25)

    assert result["state_region"].tolist() == [
        "zero-near",
        "phase-dominated",
        "interface-mixed",
    ]
    assert result["error_region"].tolist() == ["small", "medium", "large"]
    assert np.all(np.sum(result["overlap_membership"], axis=1) >= 1)


def test_partition_rejects_non_nested_overlap_thresholds() -> None:
    with pytest.raises(ValueError, match="zero overlap"):
        LocalPartition(zero_overlap_radius=0.20)


def test_local_margin_uses_worst_defect_and_minimum_singular_value() -> None:
    mask = np.asarray([True, True, False])
    result = summarize_local_region(
        mask,
        defect_ratio=np.asarray([0.1, 0.2, 9.0]),
        contraction_rate=np.asarray([0.3, 0.1, -9.0]),
        min_singular_value=np.asarray([0.8, 0.5, 0.1]),
        max_singular_value=np.asarray([1.1, 1.2, 9.0]),
        target_decay=np.asarray([0.6, 0.5, 0.1]),
        online_error=np.asarray([0.1, 0.2, 9.0]),
        baseline_error=np.asarray([0.2, 0.2, 9.0]),
        direction_residual=np.zeros(3),
        zero_fiber_residual=np.zeros(3),
        minimum_samples=2,
    )

    assert result["maximum_normalized_defect"] == pytest.approx(0.2)
    assert result["minimum_jacobian_singular_value"] == pytest.approx(0.5)
    assert result["certified_decay_margin"] == pytest.approx(0.1)
    assert result["passed"]


def test_local_margin_fails_when_only_rms_would_pass() -> None:
    result = summarize_local_region(
        np.ones(3, dtype=bool),
        defect_ratio=np.asarray([0.01, 0.01, 0.9]),
        contraction_rate=np.ones(3),
        min_singular_value=np.ones(3),
        max_singular_value=np.ones(3),
        target_decay=np.full(3, 0.5),
        online_error=np.ones(3),
        baseline_error=np.ones(3),
        direction_residual=np.zeros(3),
        zero_fiber_residual=np.zeros(3),
        minimum_samples=3,
    )

    assert result["defect_rms"] < 0.53
    assert result["certified_decay_margin"] < 0.0
    assert not result["passed"]


def test_transition_counts_do_not_cross_trajectory_boundaries() -> None:
    result = transition_counts(
        np.asarray(["a", "a", "a", "b", "b"]),
        np.asarray(
            ["zero-near", "interface-mixed", "interface-mixed", "zero-near", "phase-dominated"]
        ),
    )

    assert result == {
        "total": 2,
        "by_pair": {
            "zero-near->interface-mixed": 1,
            "zero-near->phase-dominated": 1,
        },
    }
