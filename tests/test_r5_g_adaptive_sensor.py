import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

import r5_direct_fiber_multigrid_joint as joint
from r5_g_adaptive_sensor_audit import _validate_unlock
from r5_g_adaptive_sensor_train import configure_joint

from allen_cahn_certified_observer import adaptive_sensor_family


@pytest.mark.parametrize(
    ("name", "train_nu", "evaluation_nu", "sensor_count", "low_modes"),
    [
        ("q2", 0.0075, 0.01, 2, 3),
        ("q1", 0.015, 0.02, 1, 2),
    ],
)
def test_frozen_family_tracks_parameter_dependent_sensor_count(
    name: str,
    train_nu: float,
    evaluation_nu: float,
    sensor_count: int,
    low_modes: int,
) -> None:
    family = adaptive_sensor_family(name)

    assert family.train_nu == train_nu
    assert family.evaluation_nu == evaluation_nu
    assert family.primary_sensor_count == sensor_count
    assert family.low_mode_count == low_modes
    assert np.sum(family.primary_array()[:, 1] - family.primary_array()[:, 0]) == pytest.approx(0.2)


def test_joint_configuration_changes_dimensions_without_changing_api() -> None:
    family = adaptive_sensor_family("q1")
    names = (
        "NU_VALUE",
        "GRID_SIZES",
        "LOW_MODE_COUNT",
        "CONDITION_MODE_COUNT",
        "COLLOCATION_MODE_COUNT",
        "ALPHA",
        "THREE_SENSOR_INTERVALS",
        "MODEL_SEEDS",
        "EXPERIMENT_KIND",
        "CHECKPOINT_PREFIX",
    )
    original = {name: getattr(joint, name) for name in names}
    try:
        configure_joint(family, nu=family.train_nu, grid_sizes=(31, 63))

        assert joint.NU_VALUE == 0.015
        assert joint.GRID_SIZES == (31, 63)
        assert joint.LOW_MODE_COUNT == 2
        assert joint.CONDITION_MODE_COUNT == 4
        assert joint.COLLOCATION_MODE_COUNT == 6
        assert joint.CHECKPOINT_PREFIX == "adaptive-q1"
    finally:
        for name, value in original.items():
            setattr(joint, name, value)


def test_locked_test_requires_three_of_five_matching_checkpoints(tmp_path: Path) -> None:
    family = adaptive_sensor_family("q2")
    hashes = {seed: f"hash-{seed}" for seed in family.model_seeds}
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "phase": "validation",
                "evaluation_only": True,
                "split_seed": 1871,
                "frozen": {
                    "family": "q2",
                    "grid_sizes": [47, 95],
                    "collocation_count_per_grid": 4096,
                },
                "successful_seed_count": 3,
                "passing_seeds": [1401, 1402, 1403],
                "checkpoint_hashes": {str(k): v for k, v in hashes.items()},
            }
        ),
        encoding="utf-8",
    )

    payload = _validate_unlock(validation, family, hashes)

    assert payload["passing_seeds"] == [1401, 1402, 1403]


def test_locked_test_stays_closed_at_two_of_five(tmp_path: Path) -> None:
    family = adaptive_sensor_family("q1")
    hashes = {seed: f"hash-{seed}" for seed in family.model_seeds}
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "phase": "validation",
                "evaluation_only": True,
                "split_seed": 1871,
                "frozen": {
                    "family": "q1",
                    "grid_sizes": [47, 95],
                    "collocation_count_per_grid": 4096,
                },
                "successful_seed_count": 2,
                "passing_seeds": [1501, 1502],
                "checkpoint_hashes": {str(k): v for k, v in hashes.items()},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="3/5"):
        _validate_unlock(validation, family, hashes)
