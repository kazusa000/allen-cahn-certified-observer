"""Audit why a frozen direct-contraction coordinate misses prescribed targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from r5_direct_fiber_adversarial_repair import _load_initialized_model
from r5_direct_fiber_multigrid_joint import (
    ALPHA,
    NU_VALUE,
    THREE_SENSOR_INTERVALS,
    _base_design,
    _collocation_samples,
    _fixed_low_transform,
    _git_head,
    _torch_context,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    build_projected_constant_gain,
    local_average_matrix,
    mesh_shared_fiber_transform,
    physical_modal_injection,
)
from allen_cahn_certified_observer.target_diagnostics import (
    mass_rate,
    normalized_mass_norm,
    orthogonal_defect_components,
    same_form_target_components,
)

AUDIT_SEED = 2111
AUDIT_GRIDS = (31, 63, 127, 191)
AUDIT_COUNT = 4096
TARGET_DEFECT_RMS_LIMIT = 0.1
GRID_TANGENTIAL_FRACTION_MINIMUM = 0.75
POOLED_TANGENTIAL_FRACTION_MINIMUM = 0.80
DECOMPOSITION_RELATIVE_TOLERANCE = 1.0e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("diagnostic values must be finite and non-empty")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "rms": float(np.sqrt(np.mean(array**2))),
    }


def _mass_energy(torch: object, values: object, grid_step: float) -> object:
    return float(grid_step) * torch.sum(values**2, dim=1)


def _relative_reconstruction_error(
    torch: object, residual: object, reference: object, grid_step: float
) -> object:
    numerator = _mass_energy(torch, residual, grid_step)
    denominator = _mass_energy(torch, reference, grid_step)
    return torch.sqrt(numerator / (denominator + 1.0e-20))


def _target_defect_arrays(
    torch: object,
    transformed: object,
    transformed_rhs: object,
    target_rhs: object,
    errors: object,
    grid_step: float,
) -> dict[str, object]:
    defect = transformed_rhs - target_rhs
    orthogonal = orthogonal_defect_components(torch, transformed, defect, grid_step)
    defect_energy = _mass_energy(torch, defect, grid_step)
    parallel_energy = _mass_energy(torch, orthogonal["parallel"], grid_step)
    perpendicular_energy = _mass_energy(torch, orthogonal["perpendicular"], grid_step)
    return {
        "normalized_total": normalized_mass_norm(torch, defect, errors, grid_step),
        "normalized_parallel": normalized_mass_norm(
            torch, orthogonal["parallel"], errors, grid_step
        ),
        "normalized_perpendicular": normalized_mass_norm(
            torch, orthogonal["perpendicular"], errors, grid_step
        ),
        "perpendicular_sample_fraction": perpendicular_energy
        / (defect_energy + 1.0e-20),
        "defect_energy": defect_energy,
        "parallel_energy": parallel_energy,
        "perpendicular_energy": perpendicular_energy,
        "rate_shift": mass_rate(torch, transformed, defect, grid_step),
        "orthogonality_error": torch.abs(
            float(grid_step)
            * torch.sum(transformed * orthogonal["perpendicular"], dim=1)
        )
        / (_mass_energy(torch, transformed, grid_step) + 1.0e-20),
        "defect": defect,
    }


def _audit_batch(
    torch: object,
    transform: object,
    gain: object,
    states: object,
    errors: object,
    context: dict[str, object],
) -> dict[str, object]:
    grid = context["grid"]
    grid_step = float(grid.h)
    laplacian = context["laplacian"]
    observation = context["observation"]
    injection = physical_modal_injection(torch, gain(), context["low_basis"], grid_step)
    zero_states = torch.zeros_like(states)
    zero_errors = torch.zeros_like(errors)
    truth_rhs = NU_VALUE * (states @ laplacian.T) + states - states**3
    innovation = -(errors @ observation.T)
    correction = innovation @ injection.T
    error_terms = {
        "diffusion": NU_VALUE * (errors @ laplacian.T),
        "linear_sensor": errors + correction,
        "nonlinear": -((states + errors) ** 3 - states**3),
    }
    error_rhs = sum(error_terms.values())

    def transform_map(state: object, error: object) -> object:
        return mesh_shared_fiber_transform(
            torch,
            transform,
            state,
            error,
            context["condition_basis"],
            context["low_basis"],
            grid_step,
        )

    transformed, transformed_rhs = torch.autograd.functional.jvp(
        transform_map,
        (states, errors),
        (truth_rhs, error_rhs),
        create_graph=False,
    )

    def pushed(state_tangent: object, error_tangent: object) -> object:
        _, tangent = torch.autograd.functional.jvp(
            transform_map,
            (states, errors),
            (state_tangent, error_tangent),
            create_graph=False,
        )
        return tangent

    pushforward = {
        "state_transport": pushed(truth_rhs, zero_errors),
        "diffusion": pushed(zero_states, error_terms["diffusion"]),
        "linear_sensor": pushed(zero_states, error_terms["linear_sensor"]),
        "nonlinear": pushed(zero_states, error_terms["nonlinear"]),
    }
    pushforward_sum = sum(pushforward.values())

    same_target = same_form_target_components(
        torch,
        states,
        transformed,
        laplacian,
        nu=NU_VALUE,
        alpha=ALPHA,
    )
    linear_target_rhs = same_target["diffusion"] + same_target["damping"]
    same_defect = _target_defect_arrays(
        torch,
        transformed,
        transformed_rhs,
        same_target["total"],
        errors,
        grid_step,
    )
    linear_defect = _target_defect_arrays(
        torch,
        transformed,
        transformed_rhs,
        linear_target_rhs,
        errors,
        grid_step,
    )

    source_defects = {
        "state_transport": pushforward["state_transport"],
        "diffusion_commutator": pushforward["diffusion"] - same_target["diffusion"],
        "linear_sensor_damping": pushforward["linear_sensor"] - same_target["damping"],
        "nonlinear_shape": pushforward["nonlinear"] - same_target["nonlinear"],
    }
    source_sum = sum(source_defects.values())

    arrays: dict[str, object] = {
        "actual_rate": mass_rate(torch, transformed, transformed_rhs, grid_step),
        "actual_margin": mass_rate(torch, transformed, transformed_rhs, grid_step)
        - ALPHA,
        "same_target_rate": mass_rate(
            torch, transformed, same_target["total"], grid_step
        ),
        "linear_target_rate": mass_rate(
            torch, transformed, linear_target_rhs, grid_step
        ),
        "pushforward_reconstruction_relative_error": (
            _relative_reconstruction_error(
                torch,
                transformed_rhs - pushforward_sum,
                transformed_rhs,
                grid_step,
            )
        ),
        "source_reconstruction_relative_error": _relative_reconstruction_error(
            torch,
            same_defect["defect"] - source_sum,
            same_defect["defect"],
            grid_step,
        ),
    }
    for target_name, diagnostics in (
        ("same_target", same_defect),
        ("linear_target", linear_defect),
    ):
        for metric in (
            "normalized_total",
            "normalized_parallel",
            "normalized_perpendicular",
            "perpendicular_sample_fraction",
            "defect_energy",
            "parallel_energy",
            "perpendicular_energy",
            "rate_shift",
            "orthogonality_error",
        ):
            arrays[f"{target_name}__{metric}"] = diagnostics[metric]
    for name, values in pushforward.items():
        arrays[f"dynamics__{name}__rate"] = mass_rate(
            torch, transformed, values, grid_step
        )
    for name, values in source_defects.items():
        arrays[f"source__{name}__normalized_norm"] = normalized_mass_norm(
            torch, values, errors, grid_step
        )
        arrays[f"source__{name}__rate_shift"] = mass_rate(
            torch, transformed, values, grid_step
        )
    return arrays


def _summarize_target(
    arrays: dict[str, np.ndarray], target_name: str
) -> dict[str, object]:
    prefix = f"{target_name}__"
    defect_energy = arrays[f"{prefix}defect_energy"]
    perpendicular_energy = arrays[f"{prefix}perpendicular_energy"]
    return {
        "normalized_defect": {
            "total": _summary(arrays[f"{prefix}normalized_total"]),
            "parallel": _summary(arrays[f"{prefix}normalized_parallel"]),
            "perpendicular": _summary(arrays[f"{prefix}normalized_perpendicular"]),
        },
        "rate_shift_actual_minus_target": _summary(arrays[f"{prefix}rate_shift"]),
        "perpendicular_sample_fraction": _summary(
            arrays[f"{prefix}perpendicular_sample_fraction"]
        ),
        "pooled_perpendicular_energy_fraction": float(
            np.sum(perpendicular_energy) / max(float(np.sum(defect_energy)), 1.0e-30)
        ),
        "energy_sums": {
            "defect": float(np.sum(defect_energy)),
            "parallel": float(np.sum(arrays[f"{prefix}parallel_energy"])),
            "perpendicular": float(np.sum(perpendicular_energy)),
        },
        "orthogonality_error": _summary(arrays[f"{prefix}orthogonality_error"]),
    }


def _audit_model_grid(
    torch: object,
    transform: object,
    gain: object,
    samples: dict[str, np.ndarray],
    grid_size: int,
    *,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    context = _torch_context(torch, grid_size, device=device, dtype=torch.float64)
    states = torch.as_tensor(samples["states"], dtype=torch.float64, device=device)
    errors = torch.as_tensor(samples["errors"], dtype=torch.float64, device=device)
    parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, int(states.shape[0]), batch_size):
        batch = _audit_batch(
            torch,
            transform,
            gain,
            states[start : start + batch_size],
            errors[start : start + batch_size],
            context,
        )
        for name, values in batch.items():
            parts.setdefault(name, []).append(values.detach().cpu().numpy())
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    same_target = _summarize_target(arrays, "same_target")
    linear_target = _summarize_target(arrays, "linear_target")
    dynamics = {
        name: _summary(arrays[f"dynamics__{name}__rate"])
        for name in ("state_transport", "diffusion", "linear_sensor", "nonlinear")
    }
    sources = {
        name: {
            "normalized_norm": _summary(arrays[f"source__{name}__normalized_norm"]),
            "rate_shift": _summary(arrays[f"source__{name}__rate_shift"]),
        }
        for name in (
            "state_transport",
            "diffusion_commutator",
            "linear_sensor_damping",
            "nonlinear_shape",
        )
    }
    checks = {
        "pushforward_reconstruction_relative_max": float(
            np.max(arrays["pushforward_reconstruction_relative_error"])
        ),
        "source_reconstruction_relative_max": float(
            np.max(arrays["source_reconstruction_relative_error"])
        ),
    }
    gates = {
        "actual_contraction": bool(np.min(arrays["actual_margin"]) >= -1.0e-8),
        "same_target_approximately_realized": bool(
            same_target["normalized_defect"]["total"]["rms"] <= TARGET_DEFECT_RMS_LIMIT
        ),
        "same_target_tangential_grid": bool(
            same_target["pooled_perpendicular_energy_fraction"]
            >= GRID_TANGENTIAL_FRACTION_MINIMUM
        ),
        "decomposition_integrity": bool(
            max(checks.values()) <= DECOMPOSITION_RELATIVE_TOLERANCE
        ),
    }
    return {
        "actual_rate": _summary(arrays["actual_rate"]),
        "actual_margin": _summary(arrays["actual_margin"]),
        "same_target_rate": _summary(arrays["same_target_rate"]),
        "linear_target_rate": _summary(arrays["linear_target_rate"]),
        "same_form_nonlinear_target": same_target,
        "linear_target": linear_target,
        "pushforward_energy_rate_components": dynamics,
        "same_target_defect_sources": sources,
        "checks": checks,
        "gates": gates,
    }


def run(
    torch: object,
    *,
    checkpoint: Path,
    grid_sizes: Sequence[int],
    split_seed: int,
    collocation_count: int,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    checkpoint_hash_before = _sha256(checkpoint)
    base_gain, base_transform, base_diagnostics = _base_design()
    learned_gain, learned_transform, initialization = _load_initialized_model(
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
    learned_gain = learned_gain.to(device=device, dtype=torch.float64).eval()
    learned_transform = learned_transform.to(device=device, dtype=torch.float64).eval()
    fixed_transform = (
        _fixed_low_transform(torch, base_transform)
        .to(device=device, dtype=torch.float64)
        .eval()
    )
    baseline_gain = (
        build_projected_constant_gain(torch, base_gain, trust_ratio=0.25)
        .to(device=device, dtype=torch.float64)
        .eval()
    )
    for module in (
        learned_gain,
        learned_transform,
        fixed_transform,
        baseline_gain,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    models = {
        "learned_B_Tphi": (learned_gain, learned_transform),
        "learned_B_fixed_T0": (learned_gain, fixed_transform),
        "baseline_B0_T0": (baseline_gain, fixed_transform),
    }
    grids: dict[str, object] = {}
    for grid_value in grid_sizes:
        grid_size = int(grid_value)
        grid = AllenCahnGrid(grid_size)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        samples = _collocation_samples(
            grid,
            observation,
            seed=split_seed,
            count=collocation_count,
        )
        grids[str(grid_size)] = {
            model_name: _audit_model_grid(
                torch,
                transform,
                gain,
                samples,
                grid_size,
                batch_size=batch_size,
                device=device,
            )
            for model_name, (gain, transform) in models.items()
        }

    learned_grid_results = [
        grids[str(int(grid_size))]["learned_B_Tphi"] for grid_size in grid_sizes
    ]
    pooled_defect = sum(
        item["same_form_nonlinear_target"]["energy_sums"]["defect"]
        for item in learned_grid_results
    )
    pooled_perpendicular = sum(
        item["same_form_nonlinear_target"]["energy_sums"]["perpendicular"]
        for item in learned_grid_results
    )
    pooled_tangential_fraction = pooled_perpendicular / max(pooled_defect, 1.0e-30)
    integrity = all(
        model_result["gates"]["decomposition_integrity"]
        for grid_result in grids.values()
        for model_result in grid_result.values()
    )
    target_close = all(
        item["gates"]["same_target_approximately_realized"]
        for item in learned_grid_results
    )
    tangential_each_grid = all(
        item["gates"]["same_target_tangential_grid"] for item in learned_grid_results
    )
    direct_contraction = all(
        item["gates"]["actual_contraction"] for item in learned_grid_results
    )
    rotation_candidate = bool(
        integrity
        and tangential_each_grid
        and pooled_tangential_fraction >= POOLED_TANGENTIAL_FRACTION_MINIMUM
    )
    if target_close:
        decision = "same_form_target_already_approximately_realized"
    elif rotation_candidate:
        decision = "test_constrained_skew_rotation_target_next"
    else:
        decision = "analyze_transformed_nonlinearity_remainder_next"

    checkpoint_hash_after = _sha256(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("checkpoint changed during evaluation-only audit")
    return {
        "kind": "r5-i-backstepping-nonlinear-remainder-audit",
        "created_at": datetime.now(UTC).isoformat(),
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
            "nu": NU_VALUE,
            "sensor_count": 3,
            "sensor_intervals": THREE_SENSOR_INTERVALS.tolist(),
            "alpha": ALPHA,
            "grid_sizes": [int(value) for value in grid_sizes],
            "split_seed": split_seed,
            "collocation_count_per_grid": collocation_count,
            "batch_size": batch_size,
            "no_optimizer_created": True,
            "same_target": "nu*Delta(z)-((u+z)^3-u^3)-alpha*z",
            "thresholds": {
                "target_defect_rms_limit": TARGET_DEFECT_RMS_LIMIT,
                "grid_tangential_fraction_minimum": (GRID_TANGENTIAL_FRACTION_MINIMUM),
                "pooled_tangential_fraction_minimum": (
                    POOLED_TANGENTIAL_FRACTION_MINIMUM
                ),
                "decomposition_relative_tolerance": (DECOMPOSITION_RELATIVE_TOLERANCE),
            },
        },
        "base_diagnostics": base_diagnostics,
        "grids": grids,
        "decision": {
            "decomposition_integrity": integrity,
            "learned_direct_contraction_all_grids": direct_contraction,
            "same_target_close_all_grids": target_close,
            "same_target_tangential_each_grid": tangential_each_grid,
            "same_target_pooled_tangential_fraction": float(pooled_tangential_fraction),
            "rotation_candidate": rotation_candidate,
            "next_route": decision,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=AUDIT_GRIDS)
    parser.add_argument("--split-seed", type=int, default=AUDIT_SEED)
    parser.add_argument("--collocation-count", type=int, default=AUDIT_COUNT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if tuple(args.grid_sizes) != AUDIT_GRIDS and args.collocation_count != 128:
        raise SystemExit(
            "formal audit requires grids 31 63 127 191; alternate grids are smoke-only"
        )
    if args.split_seed != AUDIT_SEED:
        raise SystemExit("this audit requires frozen seed 2111")
    if args.collocation_count not in (128, AUDIT_COUNT):
        raise SystemExit("collocation count must be 128 (smoke) or 4096 (formal)")
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    result = run(
        torch,
        checkpoint=args.checkpoint,
        grid_sizes=tuple(args.grid_sizes),
        split_seed=args.split_seed,
        collocation_count=args.collocation_count,
        batch_size=args.batch_size,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"]), flush=True)


if __name__ == "__main__":
    main()
