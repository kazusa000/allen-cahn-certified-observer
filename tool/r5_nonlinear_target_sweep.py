"""Pre-registered R5 screen for state-dependent stable targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r5_tk_joint_train import run

TARGETS = ("linear", "linearized", "nonlinear")
ABSOLUTE_PROGRESS_RMS = 0.3668


def _selected_seed_result(grid: dict[str, object]) -> dict[str, object]:
    return next(
        item
        for item in grid["seed_results"]
        if item["seed"] == grid["selected_seed"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for target_kind in TARGETS:
        print(f"[nonlinear-target-screen] target={target_kind}", flush=True)
        result = run(
            torch,
            [31],
            args.seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_limit=12,
            noise_limit=4,
            device=args.device,
            lambda_ratio=0.1,
            target_kind=target_kind,
            base_gain=0.10,
            gain_scale=0.5,
            certificate_scale=1.0,
            stable_normalization="error-time",
            stable_weight=1.0,
            defect_weight=1.0,
            bi_weight=1.0,
            lower_lipschitz=0.5,
            upper_lipschitz=2.0,
            refresh_interval=20,
            selection_limit=12,
            selection_baseline_gain=0.10,
            certificate_kind="triangular",
            mixing_layers=2,
            shear_norm_limit=0.2,
            replay_snapshots=0,
            gain_warmup_epochs=0,
            certificate_warmup_epochs=20,
            gain_learning_rate=5.0e-4,
            certificate_learning_rate=2.0e-3,
            gradient_clip_norm=1.0,
            gain_trust_ratio=0.5,
            gain_reg_weight=1.0,
            gain_kind="mass-adjoint-constant",
            selection_mode="defect-first",
            run_defect_audit=True,
            checkpoint_dir=args.checkpoint_dir / target_kind,
        )
        grid = result["results"][0]
        selected = _selected_seed_result(grid)
        audits = grid["defect_audits"]
        fixed_rms = audits["fixed_gain_validation"]["overall"]["rms"]
        current_rms = audits["current_observer_validation"]["overall"]["rms"]
        runs.append(
            {
                "target_kind": target_kind,
                "fixed_gain_validation_defect_rms": fixed_rms,
                "current_observer_validation_defect_rms": current_rms,
                "worst_validation_defect_rms": max(fixed_rms, current_rms),
                "validation_median_terminal_error_mass": selected[
                    "validation_median_terminal_error_mass"
                ],
                "selection_baseline_median_terminal_error_mass": grid[
                    "selection_baseline_median_terminal_error_mass"
                ],
                "test_median_terminal_error_mass": grid[
                    "test_median_terminal_error_mass"
                ],
                "noisy_test_median_terminal_error_mass": grid[
                    "noisy_median_terminal_error_mass"
                ],
                "certificate_constraints_passed": grid[
                    "selection_constraint_passed"
                ],
                "all_parameter_rms_gates_passed": audits[
                    "fixed_gain_validation"
                ]["all_rms_gates_passed"]
                and audits["current_observer_validation"]["all_rms_gates_passed"],
                "result": result,
            }
        )

    linear = next(item for item in runs if item["target_kind"] == "linear")
    state_dependent = [item for item in runs if item["target_kind"] != "linear"]
    eligible = [
        item
        for item in state_dependent
        if item["certificate_constraints_passed"]
        and item["validation_median_terminal_error_mass"]
        <= item["selection_baseline_median_terminal_error_mass"]
    ]
    selected = min(
        eligible or state_dependent,
        key=lambda item: (
            item["worst_validation_defect_rms"],
            item["validation_median_terminal_error_mass"],
        ),
    )
    relative_progress = (
        selected["worst_validation_defect_rms"]
        <= 0.75 * linear["worst_validation_defect_rms"]
    )
    absolute_progress = (
        selected["worst_validation_defect_rms"] <= ABSOLUTE_PROGRESS_RMS
    )
    output = {
        "kind": "r5-state-dependent-target-screen",
        "targets": TARGETS,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "selection_rule": (
            "certificate constraints and no validation rollout regression, then "
            "minimum worst fixed/current-policy validation defect RMS"
        ),
        "relative_progress_rule": "at least 25% below the same-run linear target",
        "absolute_progress_rms": ABSOLUTE_PROGRESS_RMS,
        "selected_target": selected["target_kind"],
        "relative_progress_gate_passed": bool(relative_progress),
        "absolute_progress_gate_passed": bool(absolute_progress),
        "local_rms_certificate_gate_passed": bool(
            selected["all_parameter_rms_gates_passed"]
        ),
        "runs": runs,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_target": output["selected_target"],
                "worst_validation_defect_rms": selected[
                    "worst_validation_defect_rms"
                ],
                "relative_progress_gate_passed": output[
                    "relative_progress_gate_passed"
                ],
                "absolute_progress_gate_passed": output[
                    "absolute_progress_gate_passed"
                ],
                "local_rms_certificate_gate_passed": output[
                    "local_rms_certificate_gate_passed"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
