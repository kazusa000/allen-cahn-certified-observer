"""Train only the three-sensor joint model at nu=0.005 on three grids."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import r5_oblique_joint_train as oblique
import r5_three_sensor_dynamics_joint as joint


NU_VALUE = 0.005
GRID_SIZES = (31, 63, 127)
JOINT_VARIANT = joint.VARIANTS["joint-native"]
FIXED_LMI_VARIANT = joint.VARIANTS["fixed-B__fixed-T"]


def _configure_nu005() -> None:
    """Restrict every reused design, target, and audit helper to nu=0.005."""

    frozen = (NU_VALUE,)
    oblique.NU_VALUES = frozen
    joint.NU_VALUES = frozen


def _grid_gates(
    selected: dict[str, object], fixed_lmi: dict[str, object]
) -> dict[str, object]:
    learned_terminal = float(
        selected["validation_rollout"]["by_nu"]["0.005"][
            "terminal_error_mass_median"
        ]
    )
    fixed_terminal = float(
        fixed_lmi["validation_rollout"]["by_nu"]["0.005"][
            "terminal_error_mass_median"
        ]
    )
    online_ratio = learned_terminal / max(fixed_terminal, 1.0e-12)
    finite = all(
        np.isfinite(float(value))
        for value in (
            selected["validation"]["defect"]["rms"],
            selected["validation"]["defect"]["p95"],
            selected["validation"]["contraction"]["requested_margin_min"],
            online_ratio,
        )
    )
    basic = {
        "finite": bool(finite),
        "structure": bool(selected["structure"]["passed"]),
        "requested_contraction": bool(
            selected["validation"]["contraction"]["requested_margin_min"] >= 0.0
        ),
        "online_no_regression_1.05": bool(online_ratio <= 1.05),
        "online_terminal_ratio": online_ratio,
    }
    basic["all_passed"] = bool(
        basic["finite"]
        and basic["structure"]
        and basic["requested_contraction"]
        and basic["online_no_regression_1.05"]
    )
    learned_rms = float(selected["validation"]["defect"]["rms"])
    learned_p95 = float(selected["validation"]["defect"]["p95"])
    fixed_rms = float(fixed_lmi["validation"]["defect"]["rms"])
    fixed_p95 = float(fixed_lmi["validation"]["defect"]["p95"])
    progress = {
        "defect_rms_vs_fixed_T_25pct": bool(learned_rms <= 0.75 * fixed_rms),
        "defect_p95_vs_fixed_T_15pct": bool(learned_p95 <= 0.85 * fixed_p95),
        "defect_rms_ratio": learned_rms / max(fixed_rms, 1.0e-12),
        "defect_p95_ratio": learned_p95 / max(fixed_p95, 1.0e-12),
    }
    return {
        "basic": basic,
        "progress": progress,
        "all_passed": bool(
            basic["all_passed"]
            and progress["defect_rms_vs_fixed_T_25pct"]
            and progress["defect_p95_vs_fixed_T_15pct"]
        ),
    }


def _grid_metrics(
    selected: dict[str, object], fixed_lmi: dict[str, object]
) -> dict[str, float]:
    learned_rms = float(selected["validation"]["defect"]["rms"])
    fixed_rms = float(fixed_lmi["validation"]["defect"]["rms"])
    learned_terminal = float(
        selected["validation_rollout"]["by_nu"]["0.005"][
            "terminal_error_mass_median"
        ]
    )
    fixed_terminal = float(
        fixed_lmi["validation_rollout"]["by_nu"]["0.005"][
            "terminal_error_mass_median"
        ]
    )
    return {
        "defect_rms": learned_rms,
        "fixed_defect_rms": fixed_rms,
        "defect_rms_ratio": learned_rms / max(fixed_rms, 1.0e-12),
        "defect_p95": float(selected["validation"]["defect"]["p95"]),
        "requested_margin_min": float(
            selected["validation"]["contraction"]["requested_margin_min"]
        ),
        "requested_rate_fraction": float(
            selected["validation"]["contraction"]["requested_rate_fraction"]
        ),
        "online_terminal": learned_terminal,
        "fixed_online_terminal": fixed_terminal,
        "online_terminal_ratio": learned_terminal / max(fixed_terminal, 1.0e-12),
        "jacobian_min_singular": float(
            selected["structure"]["jacobian_min_singular"]
        ),
        "jacobian_max_singular": float(
            selected["structure"]["jacobian_max_singular"]
        ),
    }


def _monotonic_worsening(values: list[float], *, larger_is_better: bool) -> bool:
    differences = np.diff(np.asarray(values, dtype=float))
    tolerance = 1.0e-12
    if larger_is_better:
        return bool(np.all(differences <= tolerance) and np.any(differences < -tolerance))
    return bool(np.all(differences >= -tolerance) and np.any(differences > tolerance))


def _trend_summary(records: dict[str, dict[str, object]]) -> dict[str, object]:
    order = [str(size) for size in GRID_SIZES]
    ratios = [float(records[key]["metrics"]["defect_rms_ratio"]) for key in order]
    defects = [float(records[key]["metrics"]["defect_rms"]) for key in order]
    margins = [
        float(records[key]["metrics"]["requested_margin_min"]) for key in order
    ]
    online = [
        float(records[key]["metrics"]["online_terminal_ratio"]) for key in order
    ]
    worsening = {
        "defect_rms": _monotonic_worsening(defects, larger_is_better=False),
        "requested_margin_min": _monotonic_worsening(
            margins, larger_is_better=True
        ),
        "online_terminal_ratio": _monotonic_worsening(
            online, larger_is_better=False
        ),
    }
    all_grids_passed = bool(
        all(bool(records[key]["gates"]["all_passed"]) for key in order)
    )
    ratio_spread = float(max(ratios) - min(ratios))
    no_monotonic_worsening = not any(worsening.values())
    return {
        "grid_order": list(GRID_SIZES),
        "defect_rms_ratios": dict(zip(order, ratios, strict=True)),
        "defect_rms_ratio_spread": ratio_spread,
        "ratio_spread_at_most_0.10": bool(ratio_spread <= 0.10),
        "monotonic_worsening": worsening,
        "no_monotonic_worsening": no_monotonic_worsening,
        "all_grids_passed": all_grids_passed,
        "mesh_robust": bool(
            all_grids_passed
            and ratio_spread <= 0.10
            and no_monotonic_worsening
        ),
    }


def run(
    torch: object,
    *,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    refresh_interval: int,
    device: str,
    train_limit_per_nu: int,
    validation_limit_per_nu: int,
    stress_truths_per_nu: int,
    gain_trust_ratio: float,
    certificate_log_scale: float,
    gain_learning_rate: float,
    certificate_learning_rate: float,
    checkpoint_dir: Path,
) -> dict[str, object]:
    _configure_nu005()
    grid_results: dict[str, dict[str, object]] = {}
    for grid_size in GRID_SIZES:
        current, _ = joint._run_grid(
            torch,
            grid_size,
            (JOINT_VARIANT,),
            baseline_variants=(FIXED_LMI_VARIANT,),
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            refresh_interval=refresh_interval,
            device=device,
            train_limit_per_nu=train_limit_per_nu,
            validation_limit_per_nu=validation_limit_per_nu,
            test_limit_per_nu=1,
            stress_truths_per_nu=stress_truths_per_nu,
            gain_trust_ratio=gain_trust_ratio,
            certificate_log_scale=certificate_log_scale,
            gain_learning_rate=gain_learning_rate,
            certificate_learning_rate=certificate_learning_rate,
            checkpoint_dir=checkpoint_dir,
        )
        selected = current["training"][JOINT_VARIANT.name]["selected"]
        fixed_lmi = current["baselines"][FIXED_LMI_VARIANT.name]
        current["selected_variant"] = JOINT_VARIANT.name
        current["gates"] = _grid_gates(selected, fixed_lmi)
        current["metrics"] = _grid_metrics(selected, fixed_lmi)
        current["test_evaluated"] = False
        current["test"] = None
        grid_results[str(grid_size)] = current
    trend = _trend_summary(grid_results)
    return {
        "kind": "r5-three-sensor-nu005-multigrid-joint",
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
            "nu_value": NU_VALUE,
            "grid_sizes": GRID_SIZES,
            "trained_variants": [JOINT_VARIANT.name],
            "reference_variants": [FIXED_LMI_VARIANT.name],
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "refresh_interval": refresh_interval,
            "train_limit_per_nu": train_limit_per_nu,
            "validation_limit_per_nu": validation_limit_per_nu,
            "stress_truths_per_nu": stress_truths_per_nu,
            "gain_trust_ratio": gain_trust_ratio,
            "certificate_log_scale": certificate_log_scale,
            "gain_learning_rate": gain_learning_rate,
            "certificate_learning_rate": certificate_learning_rate,
            "loss_weights": joint.LOSS_WEIGHTS,
            "test_evaluated": False,
        },
        "grid_results": grid_results,
        "trend": trend,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--refresh-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-limit-per-nu", type=int, default=16)
    parser.add_argument("--validation-limit-per-nu", type=int, default=8)
    parser.add_argument("--stress-truths-per-nu", type=int, default=2)
    parser.add_argument("--gain-trust-ratio", type=float, default=0.25)
    parser.add_argument("--certificate-log-scale", type=float, default=0.2231435513)
    parser.add_argument("--gain-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--certificate-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.checkpoint_dir.exists():
        raise SystemExit(
            f"refusing to reuse checkpoint directory: {args.checkpoint_dir}"
        )
    if min(
        args.epochs,
        args.batch_size,
        args.train_limit_per_nu,
        args.validation_limit_per_nu,
        args.stress_truths_per_nu,
    ) < 1:
        raise SystemExit("epochs, batch size, and case limits must be positive")
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(0)
        warmup = torch.ones((32, 32), device=args.device, requires_grad=True)
        (warmup @ warmup).sum().backward()
        torch.cuda.synchronize()
    result = run(
        torch,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        refresh_interval=args.refresh_interval,
        device=args.device,
        train_limit_per_nu=args.train_limit_per_nu,
        validation_limit_per_nu=args.validation_limit_per_nu,
        stress_truths_per_nu=args.stress_truths_per_nu,
        gain_trust_ratio=args.gain_trust_ratio,
        certificate_log_scale=args.certificate_log_scale,
        gain_learning_rate=args.gain_learning_rate,
        certificate_learning_rate=args.certificate_learning_rate,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["trend"]), flush=True)


if __name__ == "__main__":
    main()
