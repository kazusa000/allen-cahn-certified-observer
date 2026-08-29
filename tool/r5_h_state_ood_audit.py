"""Evaluation-only state-distribution OOD audit for the frozen R5 checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    STATE_OOD_FAMILIES,
    STATE_OOD_SEVERITIES,
    StateOODGateThresholds,
    state_ood_gates,
    state_ood_initial_pairs,
    state_ood_samples,
)
from r5_direct_fiber_adversarial_repair import _load_initialized_model
from r5_direct_fiber_multigrid_joint import (
    OUTPUT_TIMES,
    THREE_SENSOR_INTERVALS,
    _base_design,
    _git_head,
    _structure_audit,
)
from r5_g_generalization_audit import _evaluate_grids, _sha256


OOD_SEED = 1971
OOD_GRIDS = (63, 191)
COLLOCATION_COUNT = 2048
TRAJECTORY_COUNT = 8
EXPECTED_CHECKPOINT_SHA256 = (
    "83413559a98b9bb39226763ff3dd050610557fb6ad9b0037b7e23bb682f79d92"
)

_PARAMETERS = {
    "truth_high_frequency": {1: "modes-5-8", 2: "modes-9-12", 3: "modes-13-16"},
    "error_high_frequency": {1: "modes-5-8", 2: "modes-9-12", 3: "modes-13-16"},
    "localized_pulse": {1: "width-0.20", 2: "width-0.10", 3: "width-0.05"},
    "multiple_interfaces": {1: "interfaces-1", 2: "interfaces-2", 3: "interfaces-3"},
    "large_initial_error": {
        1: "mass-radius-0.40",
        2: "mass-radius-0.80",
        3: "mass-radius-1.20",
    },
}


def _cell_inputs(
    grid_sizes: Sequence[int],
    family: str,
    severity: int,
    *,
    seed: int,
    collocation_count: int,
    trajectory_count: int,
) -> tuple[dict[int, dict[str, object]], dict[int, list[object]]]:
    collocation: dict[int, dict[str, object]] = {}
    trajectories: dict[int, list[object]] = {}
    for n_value in grid_sizes:
        n = int(n_value)
        grid = AllenCahnGrid(n)
        collocation[n] = state_ood_samples(
            grid, family, severity, seed=seed, count=collocation_count
        )
        trajectories[n] = state_ood_initial_pairs(
            grid, family, severity, seed=seed, count=trajectory_count
        )
    return collocation, trajectories


def _robustness_envelope(cells: dict[str, object], decision: str) -> dict[str, object]:
    passing = [
        severity
        for severity in STATE_OOD_SEVERITIES
        if bool(cells[str(severity)]["gates"][decision]["all_grids_passed"])
    ]
    contiguous = 0
    for severity in STATE_OOD_SEVERITIES:
        if severity not in passing:
            break
        contiguous = severity
    return {
        "passing_severities": passing,
        "highest_passing_severity": max(passing, default=0),
        "contiguous_envelope": contiguous,
        "nonmonotone": bool(passing and passing != list(range(1, max(passing) + 1))),
    }


def run(
    torch: object,
    *,
    checkpoint: Path,
    device: str,
    grid_sizes: Sequence[int] = OOD_GRIDS,
    seed: int = OOD_SEED,
    collocation_count: int = COLLOCATION_COUNT,
    trajectory_count: int = TRAJECTORY_COUNT,
) -> dict[str, object]:
    """Run the complete frozen matrix without training or checkpoint writeback."""

    checkpoint_hash_before = _sha256(checkpoint)
    if checkpoint_hash_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 does not match the frozen R5-H contract")

    base_gain, base_transform, base_diagnostics = _base_design()
    gain, transform, initialization = _load_initialized_model(
        torch,
        checkpoint=checkpoint,
        base_gain=base_gain,
        base_transform=base_transform,
        seed=1303,
        rho=0.9,
        hidden_width=64,
        hidden_layers=3,
        gain_trust_ratio=0.5,
        error_scale=1.0,
        device=device,
    )
    gain = gain.to(dtype=torch.float64)
    transform = transform.to(dtype=torch.float64)
    structure = _structure_audit(torch, transform, device=device, seed=seed)
    thresholds = StateOODGateThresholds()

    cells: dict[str, object] = {}
    for family in STATE_OOD_FAMILIES:
        family_cells: dict[str, object] = {}
        for severity in STATE_OOD_SEVERITIES:
            collocation, trajectories = _cell_inputs(
                grid_sizes,
                family,
                severity,
                seed=seed,
                collocation_count=collocation_count,
                trajectory_count=trajectory_count,
            )
            grids = _evaluate_grids(
                torch,
                transform,
                gain,
                base_gain,
                base_transform,
                grid_sizes,
                collocation,
                trajectories,
                device=device,
            )
            family_cells[str(severity)] = {
                "parameter": _PARAMETERS[family][severity],
                "grids": grids,
                "gates": state_ood_gates(grids, structure, thresholds),
            }
        cells[family] = family_cells

    envelopes = {
        family: {
            "strict": _robustness_envelope(cells[family], "strict"),
            "practical": _robustness_envelope(cells[family], "practical"),
        }
        for family in STATE_OOD_FAMILIES
    }
    checkpoint_hash_after = _sha256(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("checkpoint changed during evaluation-only OOD audit")

    strict_all = all(
        cells[family][str(severity)]["gates"]["strict"]["all_grids_passed"]
        for family in STATE_OOD_FAMILIES
        for severity in STATE_OOD_SEVERITIES
    )
    practical_all = all(
        cells[family][str(severity)]["gates"]["practical"]["all_grids_passed"]
        for family in STATE_OOD_FAMILIES
        for severity in STATE_OOD_SEVERITIES
    )
    return {
        "kind": "r5-h-state-distribution-ood-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "evaluation_only": True,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash_before,
            "unchanged": True,
            "initialization": initialization,
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if device.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
        },
        "frozen": {
            "nu": 0.005,
            "sensor_count": 3,
            "sensor_intervals": THREE_SENSOR_INTERVALS.tolist(),
            "seed": seed,
            "grid_sizes": [int(value) for value in grid_sizes],
            "families": list(STATE_OOD_FAMILIES),
            "severities": list(STATE_OOD_SEVERITIES),
            "collocation_count_per_cell_grid": collocation_count,
            "trajectory_count_per_cell_grid": trajectory_count,
            "trajectory_output_times": OUTPUT_TIMES.tolist(),
            "thresholds": thresholds.to_dict(),
            "no_optimizer_created": True,
            "no_backward": True,
        },
        "base_diagnostics": base_diagnostics,
        "structure": structure,
        "cells": cells,
        "robustness_envelopes": envelopes,
        "summary": {
            "strict_all_cells_passed": bool(strict_all),
            "practical_all_cells_passed": bool(practical_all),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    result = run(torch, checkpoint=args.checkpoint, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
