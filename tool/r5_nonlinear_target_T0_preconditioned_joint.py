"""Run the frozen T0-preconditioned nonlinear R5 joint-training experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import r5_nonlinear_target_conditional_residual_joint as joint


TRAIN_CASE_SEEDS = (711, 712, 713, 714)
VALIDATION_CASE_SEEDS = (811, 812)
TEST_CASE_SEEDS = (911, 912)
MODEL_SEEDS = (1201, 1202, 1203)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=joint.NU_VALUE)
    parser.add_argument("--grid-size", type=int, default=joint.GRID_SIZE)
    parser.add_argument("--sensor-count", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--instant-batch-size", type=int, default=512)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--refresh-interval", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--gain-trust-ratio", type=float, default=0.25)
    parser.add_argument("--gain-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--transform-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--inverse-iterations", type=int, default=12)
    parser.add_argument("--train-base-case-limit", type=int, default=16)
    parser.add_argument("--validation-base-case-limit", type=int, default=8)
    parser.add_argument("--test-base-case-limit", type=int, default=8)
    parser.add_argument("--stress-truths-per-split", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.nu != joint.NU_VALUE:
        raise SystemExit("this frozen experiment requires --nu 0.005")
    if args.grid_size != joint.GRID_SIZE:
        raise SystemExit("this frozen experiment requires --grid-size 31")
    if args.sensor_count != 3:
        raise SystemExit("this frozen experiment requires --sensor-count 3")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.checkpoint_dir.exists():
        raise SystemExit(
            f"refusing to reuse checkpoint directory: {args.checkpoint_dir}"
        )
    if min(
        args.epochs,
        args.instant_batch_size,
        args.rollout_batch_size,
        args.hidden_width,
        args.hidden_layers,
        args.inverse_iterations,
        args.train_base_case_limit,
        args.validation_base_case_limit,
        args.test_base_case_limit,
    ) < 1:
        raise SystemExit("counts and dimensions must be positive")
    if args.stress_truths_per_split < 0:
        raise SystemExit("--stress-truths-per-split cannot be negative")
    if not 0.0 < args.rho < 1.0:
        raise SystemExit("--rho must lie in (0, 1)")
    if not 0.0 < args.gain_trust_ratio < 1.0:
        raise SystemExit("--gain-trust-ratio must lie in (0, 1)")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(0)
        warmup = torch.ones((32, 32), device=args.device, requires_grad=True)
        (warmup @ warmup).sum().backward()
        torch.cuda.synchronize()

    result = joint.run(
        torch,
        seeds=list(args.seeds),
        epochs=args.epochs,
        instant_batch_size=args.instant_batch_size,
        rollout_batch_size=args.rollout_batch_size,
        refresh_interval=args.refresh_interval,
        rho=args.rho,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        gain_trust_ratio=args.gain_trust_ratio,
        gain_learning_rate=args.gain_learning_rate,
        transform_learning_rate=args.transform_learning_rate,
        inverse_iterations=args.inverse_iterations,
        train_base_case_limit=args.train_base_case_limit,
        validation_base_case_limit=args.validation_base_case_limit,
        test_base_case_limit=args.test_base_case_limit,
        stress_truths_per_split=args.stress_truths_per_split,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        transform_base="lmi",
        train_case_seeds=TRAIN_CASE_SEEDS,
        validation_case_seeds=VALIDATION_CASE_SEEDS,
        test_case_seeds=TEST_CASE_SEEDS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_seed": result["selected_seed"],
                "successful_seed_count": result["successful_seed_count"],
                "validation_gate_passed": result["validation_gate_passed"],
                "test_evaluated": result["test_evaluated"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
