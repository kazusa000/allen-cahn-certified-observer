"""Audit the sensor-gap obstruction for every fixed linear error coordinate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from r5_direct_fiber_multigrid_joint import (
    ALPHA,
    NU_VALUE,
    THREE_SENSOR_INTERVALS,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    build_sylvester_remainder_bridge,
    dirichlet_sine_basis,
    local_average_matrix,
    observation_support_mask,
    rightmost_eigenpair,
    sensor_gap_diagnostics,
    similarity_spectral_abscissa_error,
    unobserved_mass_fraction,
    wall_jacobian,
)

GRID_SIZES = (31, 63, 127, 191)
Q_VALUES = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    4.6875,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
)
BOUNDED_Q_MAX = 4.6875
ASYMPTOTIC_Q = 10000.0
ASYMPTOTIC_RATE_TOLERANCE = 5.0e-3
ASYMPTOTIC_MASS_MINIMUM = 0.99
SIMILARITY_TOLERANCE = 1.0e-8
EXPECTED_CHECKPOINT_SHA256 = (
    "83413559a98b9bb39226763ff3dd050610557fb6ad9b0037b7e23bb682f79d92"
)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_modal_gain(
    torch: object,
    checkpoint: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    expected = {
        "seed": 1303,
        "nu": NU_VALUE,
        "grid_sizes": (31, 63, 127),
    }
    for name, value in expected.items():
        actual = payload[name]
        if name == "grid_sizes":
            actual = tuple(actual)
        if actual != value:
            raise RuntimeError(
                f"checkpoint {name}={actual!r}, expected {value!r}"
            )
    if not np.allclose(
        payload["sensor_intervals"],
        THREE_SENSOR_INTERVALS,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("checkpoint sensor intervals changed")
    state = payload["gain_state_dict"]
    base = state["base_gain"].detach().cpu().double().numpy()
    delta = state["delta"].detach().cpu().double().numpy()
    if not np.allclose(
        base,
        payload["base_gain"],
        atol=1.0e-10,
        rtol=1.0e-10,
    ):
        raise RuntimeError("checkpoint gain base is internally inconsistent")
    gain = base + delta
    if gain.shape != (4, 3) or not np.all(np.isfinite(gain)):
        raise RuntimeError("checkpoint modal gain is invalid")
    return gain, {
        "kind": payload.get("kind"),
        "seed": int(payload["seed"]),
        "rho": float(payload["rho"]),
        "gain_trust_ratio": float(payload["gain_trust_ratio"]),
        "modal_gain": gain.tolist(),
    }


def _test_transform(dimension: int) -> np.ndarray:
    transform = np.eye(dimension)
    transform += 0.1 * np.diag(np.ones(dimension - 1), k=1)
    return transform


def _scan_gain(
    grid: AllenCahnGrid,
    observation: np.ndarray,
    gain: np.ndarray,
    *,
    q_values: tuple[float, ...],
    gap_rate: float,
) -> dict[str, object]:
    support = observation_support_mask(observation)
    transform = _test_transform(grid.n)
    scan = []
    for q in q_values:
        jacobian = wall_jacobian(
            grid,
            NU_VALUE,
            observation,
            gain,
            q,
        )
        rate, vector = rightmost_eigenpair(jacobian)
        scan.append(
            {
                "q": q,
                "state_maximum_absolute_value": float(np.sqrt(q / 3.0)),
                "spectral_abscissa": rate,
                "unstable": bool(rate > 0.0),
                "unobserved_eigenvector_mass_fraction": (
                    unobserved_mass_fraction(vector, support)
                ),
                "absolute_gap_rate_error": abs(rate - gap_rate),
                "similarity_spectral_abscissa_error": (
                    similarity_spectral_abscissa_error(
                        jacobian,
                        transform,
                    )
                ),
            }
        )
    unstable = [item for item in scan if item["unstable"]]
    bounded_unstable = [
        item
        for item in unstable
        if float(item["q"]) <= BOUNDED_Q_MAX
    ]
    asymptotic = next(
        item for item in scan if float(item["q"]) == ASYMPTOTIC_Q
    )
    return {
        "gain_fro_norm": float(np.linalg.norm(gain, ord="fro")),
        "scan": scan,
        "first_unstable_q": (
            None if not unstable else float(unstable[0]["q"])
        ),
        "first_bounded_amplitude_unstable_q": (
            None
            if not bounded_unstable
            else float(bounded_unstable[0]["q"])
        ),
        "any_finite_instability": bool(unstable),
        "any_bounded_amplitude_instability": bool(bounded_unstable),
        "asymptotic": asymptotic,
        "maximum_similarity_error": float(
            max(item["similarity_spectral_abscissa_error"] for item in scan)
        ),
    }


def run(
    torch: object,
    *,
    checkpoint: Path,
    grid_sizes: tuple[int, ...],
    q_values: tuple[float, ...],
) -> dict[str, object]:
    checkpoint_hash_before = _sha256(checkpoint)
    if checkpoint_hash_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 does not match the frozen contract")
    learned_modal_gain, checkpoint_metadata = _load_frozen_modal_gain(
        torch,
        checkpoint,
    )

    continuum_gap_length = 7.0 / 30.0
    continuum_gap_rate = float(
        1.0 - NU_VALUE * (np.pi / continuum_gap_length) ** 2
    )
    grids: dict[str, object] = {}
    gap_rates = []
    for grid_size in grid_sizes:
        grid = AllenCahnGrid(grid_size)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        gap = sensor_gap_diagnostics(grid, NU_VALUE, observation)
        gap_rate = float(
            gap["most_dangerous_block"]["principal_growth_rate"]
        )
        gap_rates.append(gap_rate)
        learned_basis = dirichlet_sine_basis(grid, 4) / np.sqrt(grid.h)
        learned_gain = learned_basis @ learned_modal_gain
        sylvester_gain = build_sylvester_remainder_bridge(
            grid,
            NU_VALUE,
            observation,
            alpha=ALPHA,
            unstable_dimension=4,
        ).gain
        grids[str(grid_size)] = {
            "sensor_gap": gap,
            "gains": {
                "frozen_learned": _scan_gain(
                    grid,
                    observation,
                    learned_gain,
                    q_values=q_values,
                    gap_rate=gap_rate,
                ),
                "sylvester": _scan_gain(
                    grid,
                    observation,
                    sylvester_gain,
                    q_values=q_values,
                    gap_rate=gap_rate,
                ),
            },
        }

    fine_grids = [grids["127"], grids["191"]]
    gain_names = ("frozen_learned", "sylvester")
    checks = {
        "continuum_gap_unstable": continuum_gap_rate > 0.0,
        "both_fine_grid_gaps_unstable": all(
            float(
                result["sensor_gap"]["most_dangerous_block"][
                    "principal_growth_rate"
                ]
            )
            > 0.0
            for result in fine_grids
        ),
        "gap_rates_strictly_increase": bool(
            np.all(np.diff(np.asarray(gap_rates)) > 0.0)
        ),
        "fine_grid_moves_toward_continuum": bool(
            abs(gap_rates[-1] - continuum_gap_rate)
            < abs(gap_rates[-2] - continuum_gap_rate)
        ),
        "both_gains_have_finite_instability": all(
            any(
                bool(
                    grid_result["gains"][gain_name][
                        "any_finite_instability"
                    ]
                )
                for grid_result in grids.values()
            )
            for gain_name in gain_names
        ),
        "fine_grid_asymptotic_rates_match_gap": all(
            float(
                grid_result["gains"][gain_name]["asymptotic"][
                    "absolute_gap_rate_error"
                ]
            )
            <= ASYMPTOTIC_RATE_TOLERANCE
            for grid_result in fine_grids
            for gain_name in gain_names
        ),
        "fine_grid_asymptotic_modes_are_unobserved": all(
            float(
                grid_result["gains"][gain_name]["asymptotic"][
                    "unobserved_eigenvector_mass_fraction"
                ]
            )
            >= ASYMPTOTIC_MASS_MINIMUM
            for grid_result in fine_grids
            for gain_name in gain_names
        ),
        "similarity_invariance": all(
            float(
                grid_result["gains"][gain_name][
                    "maximum_similarity_error"
                ]
            )
            <= SIMILARITY_TOLERANCE
            for grid_result in grids.values()
            for gain_name in gain_names
        ),
    }
    checkpoint_hash_after = _sha256(checkpoint)
    checks["checkpoint_unchanged"] = (
        checkpoint_hash_after == checkpoint_hash_before
    )
    passed = bool(all(checks.values()))
    bounded_counterexample = any(
        bool(
            grid_result["gains"][gain_name][
                "any_bounded_amplitude_instability"
            ]
        )
        for grid_result in grids.values()
        for gain_name in gain_names
    )
    return {
        "kind": "r5-l-sensor-gap-obstruction",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_head(),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "material_passport": {
            "nu": NU_VALUE,
            "sensor_count": 3,
            "sensor_intervals": THREE_SENSOR_INTERVALS.tolist(),
            "grid_sizes": list(grid_sizes),
            "q_values": list(q_values),
            "bounded_q_max": BOUNDED_Q_MAX,
            "bounded_state_maximum": float(
                np.sqrt(BOUNDED_Q_MAX / 3.0)
            ),
            "asymptotic_q": ASYMPTOTIC_Q,
            "training": False,
            "locked_test_read": False,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash_before,
            "unchanged": checkpoint_hash_after == checkpoint_hash_before,
            **checkpoint_metadata,
        },
        "continuum": {
            "internal_gap_length": continuum_gap_length,
            "principal_growth_rate": continuum_gap_rate,
        },
        "grids": grids,
        "gates": {
            "thresholds": {
                "asymptotic_rate_tolerance": (
                    ASYMPTOTIC_RATE_TOLERANCE
                ),
                "asymptotic_mass_minimum": ASYMPTOTIC_MASS_MINIMUM,
                "similarity_tolerance": SIMILARITY_TOLERANCE,
            },
            "checks": checks,
            "passed": passed,
        },
        "decision": {
            "fixed_linear_global_obstruction": passed,
            "bounded_amplitude_counterexample_found": (
                bounded_counterexample
            ),
            "next_route": (
                "derive_state_class_bound_or_use_state_dependent_design"
                if passed
                else "continue_structured_fixed_metric_search"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    import torch

    result = run(
        torch,
        checkpoint=args.checkpoint,
        grid_sizes=GRID_SIZES,
        q_values=Q_VALUES,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"]), flush=True)


if __name__ == "__main__":
    main()
