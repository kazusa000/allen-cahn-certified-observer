"""Frozen problem families for adaptive-sensor generalization experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generalization import adaptive_sensor_counts, unstable_mode_count


@dataclass(frozen=True)
class AdaptiveSensorFamily:
    """One train-to-unseen-parameter transfer problem."""

    name: str
    train_nu: float
    evaluation_nu: float
    primary_intervals: tuple[tuple[float, float], ...]
    control_intervals: tuple[tuple[float, float], ...]
    low_mode_count: int
    condition_mode_count: int
    collocation_mode_count: int
    model_seeds: tuple[int, ...]

    @property
    def primary_sensor_count(self) -> int:
        return len(self.primary_intervals)

    @property
    def control_sensor_count(self) -> int:
        return len(self.control_intervals)

    def primary_array(self) -> np.ndarray:
        return np.asarray(self.primary_intervals, dtype=float)

    def control_array(self) -> np.ndarray:
        return np.asarray(self.control_intervals, dtype=float)


_THREE_SENSOR_CONTROL = (
    (1.0 / 6.0, 7.0 / 30.0),
    (7.0 / 15.0, 8.0 / 15.0),
    (23.0 / 30.0, 5.0 / 6.0),
)

ADAPTIVE_SENSOR_FAMILIES = {
    "q2": AdaptiveSensorFamily(
        name="q2",
        train_nu=0.0075,
        evaluation_nu=0.0100,
        primary_intervals=((0.4000, 0.5000), (0.5125, 0.6125)),
        control_intervals=_THREE_SENSOR_CONTROL,
        low_mode_count=3,
        condition_mode_count=6,
        collocation_mode_count=9,
        model_seeds=(1401, 1402, 1403, 1404, 1405),
    ),
    "q1": AdaptiveSensorFamily(
        name="q1",
        train_nu=0.0150,
        evaluation_nu=0.0200,
        primary_intervals=((0.2000, 0.4000),),
        control_intervals=((0.2000, 0.3000), (0.7000, 0.8000)),
        low_mode_count=2,
        condition_mode_count=4,
        collocation_mode_count=6,
        model_seeds=(1501, 1502, 1503, 1504, 1505),
    ),
}


def adaptive_sensor_family(name: str) -> AdaptiveSensorFamily:
    """Return and validate a frozen G2 model-family configuration."""

    try:
        family = ADAPTIVE_SENSOR_FAMILIES[name]
    except KeyError as error:
        choices = ", ".join(sorted(ADAPTIVE_SENSOR_FAMILIES))
        raise ValueError(f"unknown adaptive sensor family {name!r}; choose {choices}") from error

    for nu in (family.train_nu, family.evaluation_nu):
        primary, control = adaptive_sensor_counts(nu)
        if primary != family.primary_sensor_count or control != family.control_sensor_count:
            raise RuntimeError("frozen sensor counts do not match unstable-mode counts")
        if unstable_mode_count(nu) != family.low_mode_count:
            raise RuntimeError("frozen low-mode dimension does not match the problem")
    for intervals in (family.primary_array(), family.control_array()):
        if not np.all(intervals[:, 0] < intervals[:, 1]):
            raise RuntimeError("sensor intervals must have positive width")
        if float(np.min(intervals)) < 0.0 or float(np.max(intervals)) > 1.0:
            raise RuntimeError("sensor intervals must lie in [0, 1]")
        if not np.isclose(float(np.sum(intervals[:, 1] - intervals[:, 0])), 0.20):
            raise RuntimeError("each sensor layout must have total length 0.20")
    return family
