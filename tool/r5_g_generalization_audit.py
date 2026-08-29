"""Evaluation-only generalization audit for the frozen three-sensor checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    PracticalGateThresholds,
    build_projected_constant_gain,
    generalization_gates,
    local_average_matrix,
)
from r5_direct_fiber_adversarial_repair import _load_initialized_model
from r5_direct_fiber_multigrid_joint import (
    THREE_SENSOR_INTERVALS,
    _base_design,
    _collocation_samples,
    _fixed_low_transform,
    _gain_injection_numpy,
    _git_head,
    _rate_audit,
    _rollout_samples,
    _structure_audit,
    _trajectory_cases,
)


FRESH_VALIDATION_SEED = 1871
LOCKED_TEST_SEED = 1901
VALIDATION_GRIDS = (31, 47, 63, 95, 127)
LOCKED_TEST_GRIDS = (31, 63, 127, 191, 255)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_inputs(
    grid_sizes: Sequence[int], *, seed: int, collocation_count: int, split: str
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, list[object]]]:
    collocation: dict[int, dict[str, np.ndarray]] = {}
    trajectories: dict[int, list[object]] = {}
    for n in grid_sizes:
        grid = AllenCahnGrid(int(n))
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        collocation[int(n)] = _collocation_samples(
            grid, observation, seed=seed, count=collocation_count
        )
        trajectories[int(n)] = _trajectory_cases(
            grid, observation, split=split, seed=seed
        )
    return collocation, trajectories


def _evaluate_grids(
    torch: object,
    transform: object,
    gain: object,
    base_gain: np.ndarray,
    base_transform: np.ndarray,
    grid_sizes: Sequence[int],
    collocation: dict[int, dict[str, np.ndarray]],
    trajectories: dict[int, list[object]],
    *,
    device: str,
) -> dict[str, object]:
    fixed_transform = _fixed_low_transform(torch, base_transform).to(
        device=device, dtype=torch.float64
    )
    baseline_gain = build_projected_constant_gain(
        torch, base_gain, trust_ratio=0.25
    ).to(device=device, dtype=torch.float64)
    grids: dict[str, object] = {}
    for n_value in grid_sizes:
        n = int(n_value)
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)

        baseline_injection = _gain_injection_numpy(baseline_gain, grid)
        baseline_trajectory_samples, baseline_rollout = _rollout_samples(
            grid, observation, baseline_injection, trajectories[n]
        )
        baseline = {
            "collocation": _rate_audit(
                torch, fixed_transform, baseline_gain, collocation[n], n, device=device
            ),
            "trajectory": _rate_audit(
                torch,
                fixed_transform,
                baseline_gain,
                baseline_trajectory_samples,
                n,
                device=device,
            ),
            "rollout": baseline_rollout,
        }

        learned_injection = _gain_injection_numpy(gain, grid)
        trajectory_samples, rollout = _rollout_samples(
            grid, observation, learned_injection, trajectories[n]
        )
        rollout["terminal_median_ratio_to_B0"] = (
            rollout["terminal_error_mass_median"]
            / max(baseline_rollout["terminal_error_mass_median"], 1.0e-12)
        )
        rollout["terminal_max_ratio_to_B0"] = (
            rollout["terminal_error_mass_max"]
            / max(baseline_rollout["terminal_error_mass_max"], 1.0e-12)
        )
        grids[str(n)] = {
            "collocation": _rate_audit(
                torch, transform, gain, collocation[n], n, device=device
            ),
            "trajectory": _rate_audit(
                torch, transform, gain, trajectory_samples, n, device=device
            ),
            "learned_B_fixed_T0": {
                "collocation": _rate_audit(
                    torch, fixed_transform, gain, collocation[n], n, device=device
                ),
                "trajectory": _rate_audit(
                    torch,
                    fixed_transform,
                    gain,
                    trajectory_samples,
                    n,
                    device=device,
                ),
            },
            "rollout": rollout,
            "baseline_B0_T0": baseline,
        }
    return grids


def _validate_unlock(path: Path, checkpoint_sha256: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("phase") != "validation":
        raise RuntimeError("locked test requires a validation-phase result")
    if not payload.get("evaluation_only", False):
        raise RuntimeError("validation result was not produced by evaluation-only mode")
    if int(payload.get("split_seed", -1)) != FRESH_VALIDATION_SEED:
        raise RuntimeError("validation result does not use fresh seed 1871")
    frozen = payload.get("frozen", {})
    if tuple(frozen.get("grid_sizes", ())) != VALIDATION_GRIDS:
        raise RuntimeError("validation result does not cover the frozen grid set")
    if int(frozen.get("collocation_count_per_grid", -1)) != 4096:
        raise RuntimeError("validation result does not use 4096 points per grid")
    if payload.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise RuntimeError("validation and test checkpoint hashes differ")
    if not payload.get("gates", {}).get("practical", {}).get(
        "all_grids_passed", False
    ):
        raise RuntimeError("practical validation gate failed; locked test remains closed")
    return payload


def run(
    torch: object,
    *,
    phase: str,
    checkpoint: Path,
    grid_sizes: Sequence[int],
    split_seed: int,
    collocation_count: int,
    device: str,
    validation_result: Path | None,
) -> dict[str, object]:
    checkpoint_hash_before = _sha256(checkpoint)
    if phase == "locked-test":
        if validation_result is None:
            raise RuntimeError("locked test requires --validation-result")
        _validate_unlock(validation_result, checkpoint_hash_before)

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
    collocation, trajectories = _split_inputs(
        grid_sizes,
        seed=split_seed,
        collocation_count=collocation_count,
        split="validation" if phase == "validation" else "test",
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
    structure = _structure_audit(torch, transform, device=device, seed=split_seed)
    thresholds = PracticalGateThresholds()
    gates = generalization_gates(grids, structure, thresholds)
    checkpoint_hash_after = _sha256(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("checkpoint changed during evaluation-only audit")

    return {
        "kind": "r5-g-generalization-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "phase": phase,
        "split_seed": split_seed,
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
            "grid_sizes": [int(value) for value in grid_sizes],
            "collocation_count_per_grid": collocation_count,
            "thresholds": thresholds.to_dict(),
            "no_optimizer_created": True,
        },
        "base_diagnostics": base_diagnostics,
        "structure": structure,
        "grids": grids,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validation", "locked-test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grid-sizes", type=int, nargs="+")
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--collocation-count", type=int)
    parser.add_argument("--validation-result", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.phase == "validation":
        grid_sizes = tuple(args.grid_sizes or VALIDATION_GRIDS)
        split_seed = args.split_seed or FRESH_VALIDATION_SEED
        collocation_count = args.collocation_count or 4096
        if split_seed != FRESH_VALIDATION_SEED:
            raise SystemExit("formal validation requires seed 1871")
    else:
        grid_sizes = tuple(args.grid_sizes or LOCKED_TEST_GRIDS)
        split_seed = args.split_seed or LOCKED_TEST_SEED
        collocation_count = args.collocation_count or 8192
        if split_seed != LOCKED_TEST_SEED:
            raise SystemExit("locked test requires seed 1901")
    if len(set(grid_sizes)) != len(grid_sizes) or min(grid_sizes) < 15:
        raise SystemExit("grid sizes must be unique integers at least 15")
    if collocation_count < 128:
        raise SystemExit("collocation count must be at least 128")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    result = run(
        torch,
        phase=args.phase,
        checkpoint=args.checkpoint,
        grid_sizes=grid_sizes,
        split_seed=split_seed,
        collocation_count=collocation_count,
        device=args.device,
        validation_result=args.validation_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": result["phase"],
                "strict_passed": result["gates"]["strict"]["all_grids_passed"],
                "practical_passed": result["gates"]["practical"]["all_grids_passed"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
