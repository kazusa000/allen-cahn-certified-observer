"""Apply the authoritative adversarial-repair stage to one adaptive family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from allen_cahn_certified_observer import adaptive_sensor_family
import r5_direct_fiber_adversarial_repair as repair
from r5_g_adaptive_sensor_train import TRAIN_GRIDS, configure_joint


def configure_repair(family: object) -> None:
    """Bind the q=3 repair implementation to a frozen q=2 or q=1 family."""

    configure_joint(family, nu=family.train_nu, grid_sizes=TRAIN_GRIDS)
    repair.NU_VALUE = family.train_nu
    repair.GRID_SIZES = TRAIN_GRIDS
    repair.LOW_MODE_COUNT = family.low_mode_count
    repair.CONDITION_MODE_COUNT = family.condition_mode_count
    repair.ALPHA = 0.1 * family.train_nu * repair.np.pi**2
    repair.THREE_SENSOR_INTERVALS = family.primary_array()
    repair.INITIAL_CHECKPOINT_PREFIXES = (
        f"adaptive-{family.name}",
        f"adaptive-{family.name}-repair",
    )
    repair.REPAIR_CHECKPOINT_PREFIX = f"adaptive-{family.name}-repair"
    repair.REPAIR_KIND = f"r5-g-adaptive-sensor-{family.name}-adversarial-repair"


def run(
    torch: object,
    *,
    family_name: str,
    initial_checkpoint_dir: Path,
    checkpoint_dir: Path,
    device: str,
    epochs: int,
) -> dict[str, object]:
    family = adaptive_sensor_family(family_name)
    configure_repair(family)
    result = repair.run(
        torch,
        seeds=family.model_seeds,
        initial_checkpoint_dir=initial_checkpoint_dir,
        validation_seed=repair.FORMAL_VALIDATION_SEED,
        allow_locked_test=False,
        evaluation_only=False,
        epochs=epochs,
        steps_per_epoch=24,
        batch_size=256,
        rollout_batch_size=2,
        resample_count=2048,
        replay_count=4096,
        hard_replay_count=32,
        validation_count=4096,
        test_count=8192,
        contraction_buffer=0.04,
        contraction_tail_fraction=0.01,
        adversary_refresh_epochs=2,
        adversary_restarts=128,
        adversary_keep=32,
        adversary_steps=15,
        adversary_step_size=0.02,
        adversary_memory_limit=2048,
        rho=0.9,
        hidden_width=64,
        hidden_layers=3,
        gain_trust_ratio=0.5,
        gain_learning_rate=2.0e-5,
        transform_learning_rate=1.0e-4,
        robust_multiplier_start=50.0,
        robust_multiplier_end=500.0,
        online_weight=0.2,
        transform_teacher_weight=0.5,
        gain_teacher_weight=5.0,
        train_condition_branch=True,
        error_scale=1.0,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    result["kind"] = "r5-g-adaptive-sensor-adversarial-repair"
    result["frozen"]["family"] = family.name
    result["frozen"]["unseen_evaluation_nu_not_accessed"] = family.evaluation_nu
    result["frozen"]["locked_validation_and_test_generated"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("q2", "q1"), required=True)
    parser.add_argument("--initial-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.checkpoint_dir.exists():
        raise SystemExit("refusing to overwrite output or checkpoint directory")
    if not args.initial_checkpoint_dir.is_dir():
        raise SystemExit("missing initial checkpoint directory")
    if args.epochs < 1:
        raise SystemExit("epochs must be positive")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    result = run(
        torch,
        family_name=args.family,
        initial_checkpoint_dir=args.initial_checkpoint_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        epochs=args.epochs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "family": args.family,
                "repaired_seeds": len(result["seed_results"]),
                "test_evaluated": result["test_evaluated"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
