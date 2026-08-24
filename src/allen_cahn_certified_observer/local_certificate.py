"""Pre-registered physical partitions and finite-sample local certificates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class LocalPartition:
    """Frozen thresholds for the R5 local-certificate route."""

    zero_radius: float = 0.25
    phase_amplitude: float = 0.75
    phase_fraction: float = 0.50
    small_error_radius: float = 0.05
    medium_error_radius: float = 0.15
    zero_overlap_radius: float = 0.30
    interface_overlap_radius: float = 0.20
    phase_overlap_fraction: float = 0.40
    interface_overlap_fraction: float = 0.60

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.zero_radius,
                self.phase_amplitude,
                self.phase_fraction,
                self.small_error_radius,
                self.medium_error_radius,
                self.zero_overlap_radius,
                self.interface_overlap_radius,
                self.phase_overlap_fraction,
                self.interface_overlap_fraction,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("partition thresholds must be positive and finite")
        if self.small_error_radius >= self.medium_error_radius:
            raise ValueError("error radii must be strictly increasing")
        if self.zero_overlap_radius < self.zero_radius:
            raise ValueError("zero overlap must contain the zero core")
        if self.interface_overlap_radius > self.zero_radius:
            raise ValueError("interface overlap must reach the zero core")
        if self.phase_overlap_fraction > self.phase_fraction:
            raise ValueError("phase overlap must contain the phase core")
        if self.interface_overlap_fraction < self.phase_fraction:
            raise ValueError("interface overlap must reach the phase core")


def mass_norm(values: Array, h: float) -> Array:
    """Return row-wise discrete ``M_h = h I`` norms."""

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("values must be a non-empty two-dimensional array")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be positive and finite")
    if not np.all(np.isfinite(samples)):
        raise ValueError("values must be finite")
    return np.sqrt(h * np.sum(samples**2, axis=1))


def partition_samples(
    states: Array,
    errors: Array,
    h: float,
    partition: LocalPartition = LocalPartition(),
) -> dict[str, Array]:
    """Assign exhaustive core labels and overlapping physical neighborhoods."""

    state_samples = np.asarray(states, dtype=float)
    error_samples = np.asarray(errors, dtype=float)
    if state_samples.shape != error_samples.shape:
        raise ValueError("states and errors must have the same shape")
    state_norm = mass_norm(state_samples, h)
    error_norm = mass_norm(error_samples, h)
    phase_fraction = np.mean(
        np.abs(state_samples) >= partition.phase_amplitude, axis=1
    )

    state_region = np.full(state_samples.shape[0], "interface-mixed", dtype=object)
    state_region[phase_fraction >= partition.phase_fraction] = "phase-dominated"
    state_region[state_norm <= partition.zero_radius] = "zero-near"

    error_region = np.full(state_samples.shape[0], "large", dtype=object)
    error_region[error_norm <= partition.medium_error_radius] = "medium"
    error_region[error_norm <= partition.small_error_radius] = "small"

    overlaps = np.column_stack(
        (
            state_norm <= partition.zero_overlap_radius,
            phase_fraction >= partition.phase_overlap_fraction,
            (state_norm >= partition.interface_overlap_radius)
            & (phase_fraction <= partition.interface_overlap_fraction),
        )
    )
    return {
        "state_norm": state_norm,
        "error_norm": error_norm,
        "phase_fraction": phase_fraction,
        "state_region": state_region,
        "error_region": error_region,
        "overlap_membership": overlaps,
        "overlap_names": np.asarray(
            ["zero-near", "phase-dominated", "interface-mixed"], dtype=object
        ),
    }


def summarize_local_region(
    mask: Array,
    *,
    defect_ratio: Array,
    contraction_rate: Array,
    min_singular_value: Array,
    max_singular_value: Array,
    target_decay: Array,
    online_error: Array,
    baseline_error: Array,
    direction_residual: Array,
    zero_fiber_residual: Array,
    minimum_samples: int = 30,
) -> dict[str, object]:
    """Compute the strict finite-sample margin for one declared region."""

    selected = np.asarray(mask, dtype=bool)
    arrays = {
        "defect_ratio": defect_ratio,
        "contraction_rate": contraction_rate,
        "min_singular_value": min_singular_value,
        "max_singular_value": max_singular_value,
        "target_decay": target_decay,
        "online_error": online_error,
        "baseline_error": baseline_error,
        "direction_residual": direction_residual,
        "zero_fiber_residual": zero_fiber_residual,
    }
    values = {name: np.asarray(value, dtype=float) for name, value in arrays.items()}
    if selected.ndim != 1 or any(value.shape != selected.shape for value in values.values()):
        raise ValueError("mask and metric arrays must have the same one-dimensional shape")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    count = int(np.sum(selected))
    if count == 0:
        return {"sample_count": 0, "eligible": False, "passed": False}

    picked = {name: value[selected] for name, value in values.items()}
    finite = bool(all(np.all(np.isfinite(value)) for value in picked.values()))
    defect = picked["defect_ratio"]
    contraction = picked["contraction_rate"]
    minimum_singular = float(np.min(picked["min_singular_value"]))
    epsilon = float(np.max(defect))
    alpha = float(np.min(picked["target_decay"]))
    margin = alpha - epsilon / minimum_singular if minimum_singular > 0.0 else -np.inf
    direction_max = float(np.max(picked["direction_residual"]))
    zero_max = float(np.max(picked["zero_fiber_residual"]))
    eligible = count >= minimum_samples and finite
    constraints_passed = (
        minimum_singular > 0.0
        and direction_max <= 1.0e-7
        and zero_max <= 1.0e-7
    )
    return {
        "sample_count": count,
        "eligible": eligible,
        "finite": finite,
        "defect_rms": float(np.sqrt(np.mean(defect**2))),
        "defect_median": float(np.median(defect)),
        "defect_p95": float(np.quantile(defect, 0.95)),
        "maximum_normalized_defect": epsilon,
        "contraction_rate_min": float(np.min(contraction)),
        "contraction_rate_median": float(np.median(contraction)),
        "positive_contraction_fraction": float(np.mean(contraction > 0.0)),
        "minimum_jacobian_singular_value": minimum_singular,
        "jacobian_max_singular_value": float(
            np.max(picked["max_singular_value"])
        ),
        "minimum_target_decay": alpha,
        "certified_decay_margin": float(margin),
        "online_error_rms": float(np.sqrt(np.mean(picked["online_error"] ** 2))),
        "baseline_error_rms": float(
            np.sqrt(np.mean(picked["baseline_error"] ** 2))
        ),
        "online_to_baseline_error_ratio": float(
            np.sqrt(np.mean(picked["online_error"] ** 2))
            / (np.sqrt(np.mean(picked["baseline_error"] ** 2)) + 1.0e-12)
        ),
        "max_direction_residual": direction_max,
        "max_zero_fiber_residual": zero_max,
        "constraints_passed": constraints_passed,
        "passed": bool(eligible and constraints_passed and margin > 0.0),
    }


def transition_counts(trajectory_ids: Array, labels: Array) -> dict[str, object]:
    """Count state-region changes within ordered trajectories."""

    ids = np.asarray(trajectory_ids)
    regions = np.asarray(labels, dtype=object)
    if ids.ndim != 1 or regions.shape != ids.shape:
        raise ValueError("trajectory ids and labels must have the same shape")
    pairs: dict[str, int] = {}
    total = 0
    for index in range(1, ids.size):
        if ids[index] == ids[index - 1] and regions[index] != regions[index - 1]:
            key = f"{regions[index - 1]}->{regions[index]}"
            pairs[key] = pairs.get(key, 0) + 1
            total += 1
    return {"total": total, "by_pair": dict(sorted(pairs.items()))}
