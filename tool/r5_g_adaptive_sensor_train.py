"""Train a frozen q=2 or q=1 adaptive-sensor model family without test access."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    adaptive_sensor_family,
    local_average_matrix,
)
import r5_direct_fiber_multigrid_joint as joint


TRAIN_GRIDS = (31, 63)
TRAIN_SPLIT_SEED = 1701
TUNING_SPLIT_SEED = 1851


def configure_joint(family: object, *, nu: float, grid_sizes: tuple[int, ...]) -> None:
    """Configure the reusable joint trainer for one frozen family and split."""

    joint.NU_VALUE = float(nu)
    joint.GRID_SIZES = tuple(int(value) for value in grid_sizes)
    joint.LOW_MODE_COUNT = int(family.low_mode_count)
    joint.CONDITION_MODE_COUNT = int(family.condition_mode_count)
    joint.COLLOCATION_MODE_COUNT = int(family.collocation_mode_count)
    joint.ALPHA = 0.1 * float(nu) * np.pi**2
    joint.THREE_SENSOR_INTERVALS = family.primary_array()
    joint.MODEL_SEEDS = tuple(family.model_seeds)
    joint.EXPERIMENT_KIND = f"r5-g-adaptive-sensor-{family.name}"
    joint.CHECKPOINT_PREFIX = f"adaptive-{family.name}"


def _split_inputs(
    family: object, *, seed: int, count: int, split: str
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, list[object]]]:
    samples: dict[int, dict[str, np.ndarray]] = {}
    cases: dict[int, list[object]] = {}
    for n in TRAIN_GRIDS:
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, family.primary_array())
        samples[n] = joint._collocation_samples(
            grid, observation, seed=seed, count=count
        )
        cases[n] = joint._trajectory_cases(
            grid, observation, split=split, seed=seed
        )
    return samples, cases


def run(
    torch: object,
    *,
    family_name: str,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    rollout_batch_size: int,
    train_count: int,
    tuning_count: int,
    rho: float,
    hidden_width: int,
    hidden_layers: int,
    gain_trust_ratio: float,
    gain_learning_rate: float,
    transform_learning_rate: float,
    error_scale: float,
    device: str,
    checkpoint_dir: Path,
) -> dict[str, object]:
    family = adaptive_sensor_family(family_name)
    configure_joint(family, nu=family.train_nu, grid_sizes=TRAIN_GRIDS)
    base_gain, base_transform, base_diagnostics = joint._base_design()
    train_samples, train_cases = _split_inputs(
        family, seed=TRAIN_SPLIT_SEED, count=train_count, split="train"
    )
    tuning_samples, tuning_cases = _split_inputs(
        family, seed=TUNING_SPLIT_SEED, count=tuning_count, split="validation"
    )
    train_truth_numpy = joint._truth_rollouts(
        AllenCahnGrid(31), train_cases[31]
    )
    train_truth = joint._tensorize_truth(torch, train_truth_numpy, device=device)
    tuning_baseline = joint._baseline_split(
        torch,
        base_gain,
        base_transform,
        tuning_samples,
        tuning_cases,
        device=device,
    )

    seed_results = []
    for seed in family.model_seeds:
        _, _, result = joint._train_seed(
            torch,
            base_gain,
            base_transform,
            train_samples,
            tuning_samples,
            tuning_cases,
            tuning_baseline,
            train_truth,
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            rollout_batch_size=rollout_batch_size,
            rho=rho,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            gain_trust_ratio=gain_trust_ratio,
            gain_learning_rate=gain_learning_rate,
            transform_learning_rate=transform_learning_rate,
            error_scale=error_scale,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        seed_results.append(result)

    return {
        "kind": "r5-g-adaptive-sensor-training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": joint._git_head(),
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
            "unseen_evaluation_nu_not_accessed": family.evaluation_nu,
            "sensor_count": family.primary_sensor_count,
            "sensor_intervals": family.primary_array().tolist(),
            "sensor_geometry_source": (
                "n=31 linear observability search before nonlinear evaluation"
            ),
            "grid_sizes": list(TRAIN_GRIDS),
            "model_seeds": list(family.model_seeds),
            "train_split_seed": TRAIN_SPLIT_SEED,
            "tuning_split_seed": TUNING_SPLIT_SEED,
            "train_count_per_grid": train_count,
            "tuning_count_per_grid": tuning_count,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": batch_size,
            "rollout_batch_size": rollout_batch_size,
            "rho": rho,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "gain_trust_ratio": gain_trust_ratio,
            "gain_learning_rate": gain_learning_rate,
            "transform_learning_rate": transform_learning_rate,
            "error_scale": error_scale,
            "locked_validation_and_test_generated": False,
        },
        "base_diagnostics": base_diagnostics,
        "seed_results": seed_results,
        "checkpoint_dir": str(checkpoint_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("q2", "q1"), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps-per-epoch", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=2)
    parser.add_argument("--train-count", type=int, default=4096)
    parser.add_argument("--tuning-count", type=int, default=2048)
    parser.add_argument("--rho", type=float, default=0.35)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--gain-trust-ratio", type=float, default=0.25)
    parser.add_argument("--gain-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--transform-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--error-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.checkpoint_dir.exists():
        raise SystemExit("refusing to overwrite output or checkpoint directory")
    counts = (
        args.epochs,
        args.steps_per_epoch,
        args.batch_size,
        args.rollout_batch_size,
        args.train_count,
        args.tuning_count,
        args.hidden_width,
        args.hidden_layers,
    )
    if min(counts) < 1 or min(args.train_count, args.tuning_count) < 128:
        raise SystemExit("invalid positive count; collocation pools require at least 128")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    result = run(
        torch,
        family_name=args.family,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        rollout_batch_size=args.rollout_batch_size,
        train_count=args.train_count,
        tuning_count=args.tuning_count,
        rho=args.rho,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        gain_trust_ratio=args.gain_trust_ratio,
        gain_learning_rate=args.gain_learning_rate,
        transform_learning_rate=args.transform_learning_rate,
        error_scale=args.error_scale,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"family": args.family, "trained_seeds": len(result["seed_results"])}))


if __name__ == "__main__":
    main()
