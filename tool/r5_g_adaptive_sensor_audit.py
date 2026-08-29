"""Evaluation-only zero-shot parameter audit for q=2 and q=1 model families."""

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
    adaptive_sensor_family,
    build_low_modal_conditional_residual_transform,
    build_projected_constant_gain,
    generalization_gates,
    lmi_modal_injection,
    local_average_matrix,
    unstable_modal_system,
)
import r5_direct_fiber_multigrid_joint as joint
from r5_g_adaptive_sensor_train import configure_joint


VALIDATION_GRIDS = (47, 95)
LOCKED_TEST_GRIDS = (127, 191)
VALIDATION_SEED = 1871
LOCKED_TEST_SEED = 1901
REQUIRED_PASSING_SEEDS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_paths(directory: Path, family: object) -> dict[int, Path]:
    paths = {}
    for seed in family.model_seeds:
        candidates = (
            directory / f"adaptive-{family.name}__seed-{seed}.pt",
            directory / f"adaptive-{family.name}-repair__seed-{seed}.pt",
        )
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(
                f"expected one base or repaired checkpoint for seed {seed}, "
                f"found {len(existing)}"
            )
        paths[seed] = existing[0]
    return paths


def _load_checkpoint(
    torch: object, path: Path, family: object, *, device: str
) -> tuple[object, object, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    allowed_kinds = {
        f"r5-g-adaptive-sensor-{family.name}",
        f"r5-g-adaptive-sensor-{family.name}-adversarial-repair",
    }
    if payload.get("kind") not in allowed_kinds:
        raise RuntimeError(f"{path} has unexpected kind {payload.get('kind')!r}")
    expected = {
        "nu": family.train_nu,
        "grid_sizes": (31, 63),
        "low_mode_count": family.low_mode_count,
        "condition_mode_count": family.condition_mode_count,
        "collocation_mode_count": family.collocation_mode_count,
    }
    for name, value in expected.items():
        actual = payload.get(name)
        if name == "grid_sizes" and actual is not None:
            actual = tuple(actual)
        if actual != value:
            raise RuntimeError(f"{path} has {name}={actual!r}, expected {value!r}")
    if not np.allclose(
        payload["sensor_intervals"], family.primary_array(), atol=0.0, rtol=0.0
    ):
        raise RuntimeError("checkpoint sensor layout differs from the frozen family")

    base_gain = np.asarray(payload["base_gain"], dtype=float)
    base_transform = np.asarray(payload["base_transform"], dtype=float)
    gain = build_projected_constant_gain(
        torch, base_gain, trust_ratio=float(payload["gain_trust_ratio"])
    ).to(device=device, dtype=torch.float64)
    transform = build_low_modal_conditional_residual_transform(
        torch,
        base_transform,
        state_dimension=family.condition_mode_count,
        hidden_width=int(payload["hidden_width"]),
        hidden_layers=int(payload["hidden_layers"]),
        rho=float(payload["rho"]),
        error_scale=float(payload["error_scale"]),
    ).to(device=device, dtype=torch.float64)
    gain.load_state_dict(payload["gain_state_dict"])
    transform.load_state_dict(payload["transform_state_dict"])
    gain.project_()
    transform.project_spectral_()
    return gain, transform, {
        "seed": int(payload["seed"]),
        "kind": payload["kind"],
        "base_gain": base_gain,
        "base_transform": base_transform,
        "rho": float(payload["rho"]),
        "gain_trust_ratio": float(payload["gain_trust_ratio"]),
    }


def _split_inputs(
    family: object,
    grid_sizes: Sequence[int],
    *,
    seed: int,
    count: int,
    split: str,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, list[object]]]:
    samples: dict[int, dict[str, np.ndarray]] = {}
    cases: dict[int, list[object]] = {}
    for n_value in grid_sizes:
        n = int(n_value)
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, family.primary_array())
        samples[n] = joint._collocation_samples(
            grid, observation, seed=seed, count=count
        )
        cases[n] = joint._trajectory_cases(
            grid, observation, split=split, seed=seed
        )
    return samples, cases


def _evaluate(
    torch: object,
    family: object,
    gain: object,
    transform: object,
    metadata: dict[str, object],
    grid_sizes: Sequence[int],
    samples: dict[int, dict[str, np.ndarray]],
    cases: dict[int, list[object]],
    *,
    device: str,
) -> dict[str, object]:
    base_gain = np.asarray(metadata["base_gain"], dtype=float)
    base_transform = np.asarray(metadata["base_transform"], dtype=float)
    baseline_gain = build_projected_constant_gain(
        torch, base_gain, trust_ratio=float(metadata["gain_trust_ratio"])
    ).to(device=device, dtype=torch.float64)
    fixed_transform = joint._fixed_low_transform(torch, base_transform).to(
        device=device, dtype=torch.float64
    )
    grids: dict[str, object] = {}
    for n_value in grid_sizes:
        n = int(n_value)
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, family.primary_array())
        baseline_injection = joint._gain_injection_numpy(baseline_gain, grid)
        baseline_trajectory, baseline_rollout = joint._rollout_samples(
            grid, observation, baseline_injection, cases[n]
        )
        learned_injection = joint._gain_injection_numpy(gain, grid)
        learned_trajectory, rollout = joint._rollout_samples(
            grid, observation, learned_injection, cases[n]
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
            "collocation": joint._rate_audit(
                torch, transform, gain, samples[n], n, device=device
            ),
            "trajectory": joint._rate_audit(
                torch, transform, gain, learned_trajectory, n, device=device
            ),
            "learned_B_fixed_T0": {
                "collocation": joint._rate_audit(
                    torch, fixed_transform, gain, samples[n], n, device=device
                ),
                "trajectory": joint._rate_audit(
                    torch, fixed_transform, gain, learned_trajectory, n, device=device
                ),
            },
            "rollout": rollout,
            "baseline_B0_T0": {
                "collocation": joint._rate_audit(
                    torch, fixed_transform, baseline_gain, samples[n], n, device=device
                ),
                "trajectory": joint._rate_audit(
                    torch,
                    fixed_transform,
                    baseline_gain,
                    baseline_trajectory,
                    n,
                    device=device,
                ),
                "rollout": baseline_rollout,
            },
        }
    return grids


def _linear_positive_control(family: object) -> dict[str, object]:
    grid = AllenCahnGrid(31)
    observation = local_average_matrix(grid, family.control_array())
    design = lmi_modal_injection(
        grid,
        family.evaluation_nu,
        observation,
        decay_rate=0.1 * family.evaluation_nu * np.pi**2,
        metric_condition_bound=256.0,
    )
    modal = unstable_modal_system(grid, family.evaluation_nu, observation)
    return {
        "sensor_count": family.control_sensor_count,
        "sensor_intervals": family.control_array().tolist(),
        "design_grid": 31,
        "observability_rank": modal.observability_rank,
        "unstable_dimension": modal.dimension,
        "observability_min_singular_value": (
            modal.observability_min_singular_value
        ),
        "modal_contraction_rate": design.modal_contraction_rate,
        "requested_rate": 0.1 * family.evaluation_nu * np.pi**2,
        "passed": bool(
            modal.observability_rank == modal.dimension
            and design.modal_contraction_rate
            >= 0.1 * family.evaluation_nu * np.pi**2 - 1.0e-8
        ),
    }


def _validate_unlock(
    path: Path, family: object, checkpoint_hashes: dict[int, str]
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen", {})
    if payload.get("phase") != "validation" or not payload.get("evaluation_only"):
        raise RuntimeError("locked test requires an evaluation-only validation result")
    if frozen.get("family") != family.name:
        raise RuntimeError("validation result belongs to another family")
    if tuple(frozen.get("grid_sizes", ())) != VALIDATION_GRIDS:
        raise RuntimeError("validation result does not cover grids 47 and 95")
    if int(frozen.get("collocation_count_per_grid", -1)) != 4096:
        raise RuntimeError("validation result must use 4096 points per grid")
    if int(payload.get("split_seed", -1)) != VALIDATION_SEED:
        raise RuntimeError("validation result must use seed 1871")
    if int(payload.get("successful_seed_count", 0)) < REQUIRED_PASSING_SEEDS:
        raise RuntimeError("fewer than 3/5 seeds passed; locked test remains closed")
    recorded = {int(k): v for k, v in payload.get("checkpoint_hashes", {}).items()}
    if recorded != checkpoint_hashes:
        raise RuntimeError("validation and test checkpoint sets or hashes differ")
    return payload


def _selection_key(result: dict[str, object]) -> tuple[float, ...]:
    grids = result["grids"]
    return (
        float(result["gates"]["practical"]["all_grids_passed"]),
        min(float(value["collocation"]["requested_margin_p01"]) for value in grids.values()),
        min(float(value["trajectory"]["requested_margin_p01"]) for value in grids.values()),
        -max(float(value["rollout"]["terminal_median_ratio_to_B0"]) for value in grids.values()),
    )


def run(
    torch: object,
    *,
    family_name: str,
    phase: str,
    checkpoint_dir: Path,
    grid_sizes: Sequence[int],
    split_seed: int,
    collocation_count: int,
    validation_result: Path | None,
    device: str,
) -> dict[str, object]:
    family = adaptive_sensor_family(family_name)
    configure_joint(
        family, nu=family.evaluation_nu, grid_sizes=tuple(int(n) for n in grid_sizes)
    )
    checkpoint_paths = _checkpoint_paths(checkpoint_dir, family)
    checkpoint_hashes = {seed: _sha256(path) for seed, path in checkpoint_paths.items()}
    validation_payload = None
    selected_seeds = list(family.model_seeds)
    if phase == "locked-test":
        if validation_result is None:
            raise RuntimeError("locked test requires --validation-result")
        validation_payload = _validate_unlock(
            validation_result, family, checkpoint_hashes
        )
        selected_seeds = [int(value) for value in validation_payload["passing_seeds"]]

    samples, cases = _split_inputs(
        family,
        grid_sizes,
        seed=split_seed,
        count=collocation_count,
        split="validation" if phase == "validation" else "test",
    )
    thresholds = PracticalGateThresholds()
    seed_results = []
    for seed in selected_seeds:
        gain, transform, metadata = _load_checkpoint(
            torch, checkpoint_paths[seed], family, device=device
        )
        grids = _evaluate(
            torch,
            family,
            gain,
            transform,
            metadata,
            grid_sizes,
            samples,
            cases,
            device=device,
        )
        structure = joint._structure_audit(
            torch, transform, device=device, seed=split_seed + seed
        )
        seed_results.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_paths[seed]),
                "checkpoint_sha256": checkpoint_hashes[seed],
                "structure": structure,
                "grids": grids,
                "gates": generalization_gates(grids, structure, thresholds),
            }
        )

    hashes_after = {seed: _sha256(path) for seed, path in checkpoint_paths.items()}
    if hashes_after != checkpoint_hashes:
        raise RuntimeError("a checkpoint changed during evaluation-only audit")
    passing = [
        int(item["seed"])
        for item in seed_results
        if item["gates"]["practical"]["all_grids_passed"]
    ]
    strict = [
        int(item["seed"])
        for item in seed_results
        if item["gates"]["strict"]["all_grids_passed"]
    ]
    selected = max(seed_results, key=_selection_key)
    positive_control = _linear_positive_control(family)
    return {
        "kind": "r5-g-adaptive-sensor-zero-shot-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": joint._git_head(),
        "phase": phase,
        "split_seed": split_seed,
        "evaluation_only": True,
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
            "family": family.name,
            "train_nu": family.train_nu,
            "evaluation_nu": family.evaluation_nu,
            "sensor_count": family.primary_sensor_count,
            "sensor_intervals": family.primary_array().tolist(),
            "grid_sizes": [int(value) for value in grid_sizes],
            "collocation_count_per_grid": collocation_count,
            "required_passing_seeds": REQUIRED_PASSING_SEEDS,
            "thresholds": thresholds.to_dict(),
            "no_optimizer_created": True,
        },
        "checkpoint_hashes": {str(k): v for k, v in checkpoint_hashes.items()},
        "evaluated_seeds": selected_seeds,
        "seed_results": seed_results,
        "passing_seeds": passing,
        "strict_passing_seeds": strict,
        "successful_seed_count": len(passing),
        "strict_seed_count": len(strict),
        "selected_seed": int(selected["seed"]),
        "test_unlocked": bool(
            phase == "locked-test" or len(passing) >= REQUIRED_PASSING_SEEDS
        ),
        "linear_positive_control": positive_control,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("q2", "q1"), required=True)
    parser.add_argument("--phase", choices=("validation", "locked-test"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--validation-result", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.phase == "validation":
        grids, seed, count = VALIDATION_GRIDS, VALIDATION_SEED, 4096
    else:
        grids, seed, count = LOCKED_TEST_GRIDS, LOCKED_TEST_SEED, 8192

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    result = run(
        torch,
        family_name=args.family,
        phase=args.phase,
        checkpoint_dir=args.checkpoint_dir,
        grid_sizes=grids,
        split_seed=seed,
        collocation_count=count,
        validation_result=args.validation_result,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "family": args.family,
                "phase": args.phase,
                "practical_seed_count": result["successful_seed_count"],
                "strict_seed_count": result["strict_seed_count"],
                "test_unlocked": result["test_unlocked"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
