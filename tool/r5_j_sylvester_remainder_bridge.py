"""Evaluate an exact Sylvester backbone with its true nonlinear remainder."""

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
    _git_head,
    _torch_context,
)
from r5_i_backstepping_nonlinear_remainder_audit import _audit_batch

from allen_cahn_certified_observer import (
    STATE_OOD_FAMILIES,
    STATE_OOD_SEVERITIES,
    AllenCahnGrid,
    build_sylvester_remainder_bridge,
    exact_remainder_batch,
    local_average_matrix,
    state_ood_samples,
)

AUDIT_SEED = 2231
ADVERSARIAL_SEED = 2232
AUDIT_GRIDS = (31, 63, 127, 191)
IID_COUNT = 4096
OOD_COUNT = 512
SMOKE_IID_COUNT = 128
SMOKE_OOD_COUNT = 16
STRUCTURE_TOLERANCE = 1.0e-10
CONDITION_LIMIT = 1.0e5
PRACTICAL_NONNEGATIVE_FRACTION = 0.99
PRACTICAL_P01_MARGIN = 0.0
ADVERSARIAL_RESTARTS = 64
ADVERSARIAL_STEPS = 100
ADVERSARIAL_STEP_SIZE = 0.02
EXPECTED_CHECKPOINT_SHA256 = (
    "83413559a98b9bb39226763ff3dd050610557fb6ad9b0037b7e23bb682f79d92"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float | int | bool]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = bool(array.size > 0 and np.all(np.isfinite(array)))
    if not finite:
        raise RuntimeError("audit arrays must be finite and non-empty")
    return {
        "count": int(array.size),
        "finite": True,
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


def _margin_summary(values: np.ndarray) -> dict[str, float | int | bool]:
    result = _summary(values)
    array = np.asarray(values, dtype=float).reshape(-1)
    result.update(
        {
            "nonnegative_count": int(np.sum(array >= 0.0)),
            "negative_count": int(np.sum(array < 0.0)),
            "nonnegative_fraction": float(np.mean(array >= 0.0)),
        }
    )
    return result


def _concatenate(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("at least one array dictionary is required")
    keys = tuple(parts[0])
    if any(tuple(part) != keys for part in parts):
        raise RuntimeError("array dictionaries do not have matching keys")
    return {
        name: np.concatenate([np.asarray(part[name]) for part in parts])
        for name in keys
    }


def _summarize_exact(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "actual_rate": _summary(arrays["total_rate"]),
        "actual_margin": _margin_summary(arrays["actual_margin"]),
        "linear_backbone_rate": _summary(arrays["linear_rate"]),
        "linear_backbone_margin": _margin_summary(arrays["linear_margin"]),
        "nonlinear_remainder_rate": _summary(
            arrays["nonlinear_remainder_rate"]
        ),
        "nonlinear_remainder_dissipative_fraction": float(
            np.mean(arrays["nonlinear_remainder_rate"] >= 0.0)
        ),
        "integrity": {
            "rhs_reconstruction_relative_max": float(
                np.max(arrays["rhs_reconstruction_relative_error"])
            ),
            "inverse_reconstruction_relative_max": float(
                np.max(arrays["inverse_reconstruction_relative_error"])
            ),
            "rate_additivity_absolute_max": float(
                np.max(arrays["rate_additivity_error"])
            ),
        },
    }


def _learned_arrays(
    torch: object,
    transform: object,
    gain: object,
    samples: dict[str, np.ndarray],
    grid_size: int,
    *,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    context = _torch_context(
        torch, grid_size, device=device, dtype=torch.float64
    )
    states = torch.as_tensor(
        samples["states"], dtype=torch.float64, device=device
    )
    errors = torch.as_tensor(
        samples["errors"], dtype=torch.float64, device=device
    )
    selected = (
        "actual_rate",
        "actual_margin",
        "linear_backbone_rate",
        "linear_backbone_margin",
        "nonlinear_remainder_rate",
        "dynamics__state_transport__rate",
        "pushforward_reconstruction_relative_error",
        "rate_additivity_error",
    )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in selected}
    for start in range(0, int(states.shape[0]), batch_size):
        batch = _audit_batch(
            torch,
            transform,
            gain,
            states[start : start + batch_size],
            errors[start : start + batch_size],
            context,
        )
        for name in selected:
            parts[name].append(batch[name].detach().cpu().numpy())
    return {
        name: np.concatenate(values)
        for name, values in parts.items()
    }


def _summarize_learned(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "actual_rate": _summary(arrays["actual_rate"]),
        "actual_margin": _margin_summary(arrays["actual_margin"]),
        "linear_backbone_rate": _summary(arrays["linear_backbone_rate"]),
        "linear_backbone_margin": _margin_summary(
            arrays["linear_backbone_margin"]
        ),
        "state_transport_rate": _summary(
            arrays["dynamics__state_transport__rate"]
        ),
        "nonlinear_remainder_rate": _summary(
            arrays["nonlinear_remainder_rate"]
        ),
        "integrity": {
            "pushforward_reconstruction_relative_max": float(
                np.max(arrays["pushforward_reconstruction_relative_error"])
            ),
            "rate_additivity_absolute_max": float(
                np.max(arrays["rate_additivity_error"])
            ),
        },
    }


def _exact_model_arrays(
    bridge: object,
    samples: dict[str, np.ndarray],
    *,
    identity_coordinate: bool,
) -> dict[str, np.ndarray]:
    dimension = bridge.grid.n
    if identity_coordinate:
        coordinate_transform = np.eye(dimension)
        inverse_coordinate = np.eye(dimension)
        target_generator = bridge.closed_loop_generator
    else:
        coordinate_transform = bridge.coordinate_transform
        inverse_coordinate = bridge.inverse_coordinate
        target_generator = bridge.target_generator
    return exact_remainder_batch(
        samples["states"],
        samples["errors"],
        coordinate_transform=coordinate_transform,
        inverse_coordinate=inverse_coordinate,
        target_generator=target_generator,
        closed_loop_generator=bridge.closed_loop_generator,
        alpha=ALPHA,
    )


def _practical_gate(margin: dict[str, object]) -> dict[str, object]:
    passed = bool(
        margin["finite"]
        and float(margin["nonnegative_fraction"])
        >= PRACTICAL_NONNEGATIVE_FRACTION
        and float(margin["p01"]) >= PRACTICAL_P01_MARGIN
    )
    return {
        "finite": bool(margin["finite"]),
        "nonnegative_fraction": float(margin["nonnegative_fraction"]),
        "p01_margin": float(margin["p01"]),
        "passed": passed,
    }


def _structure_gate(
    diagnostics: dict[str, float | int],
    integrity: dict[str, float],
) -> dict[str, object]:
    checks = {
        "four_unstable_modes": (
            int(diagnostics["positive_linear_mode_count"]) == 4
        ),
        "low_modes_observable": (
            int(diagnostics["low_observability_rank"]) == 4
        ),
        "low_sylvester_exact": (
            float(diagnostics["low_primal_relative_residual"])
            <= STRUCTURE_TOLERANCE
        ),
        "full_similarity_exact": (
            float(diagnostics["full_physical_relative_residual"])
            <= STRUCTURE_TOLERANCE
        ),
        "transformed_generator_exact": (
            float(diagnostics["transform_relative_residual"])
            <= STRUCTURE_TOLERANCE
        ),
        "inverse_exact": (
            float(diagnostics["inverse_residual_2"])
            <= STRUCTURE_TOLERANCE
        ),
        "coordinate_condition_allowed": (
            float(diagnostics["inverse_coordinate_condition_2"])
            <= CONDITION_LIMIT
        ),
        "target_linear_rate_met": (
            float(diagnostics["target_linear_minimum_decay_rate"])
            + STRUCTURE_TOLERANCE
            >= ALPHA
        ),
        "nonlinear_rhs_reconstructed": (
            float(integrity["rhs_reconstruction_relative_max"])
            <= STRUCTURE_TOLERANCE
        ),
        "sample_inverse_reconstructed": (
            float(integrity["inverse_reconstruction_relative_max"])
            <= STRUCTURE_TOLERANCE
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _initial_adversarial_coefficients(
    grid: AllenCahnGrid,
    *,
    seed: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.Generator(
        np.random.PCG64DXSM(seed + 4001 * grid.n)
    )
    state_coefficients = generator.normal(size=(count, 8))
    state_basis = (
        np.asarray(
            bridge_basis(grid, 8),
            dtype=float,
        )
    )
    state_values = state_coefficients @ state_basis.T
    state_targets = generator.uniform(0.05, 1.25, size=count)
    state_coefficients *= (
        state_targets
        / np.maximum(np.max(np.abs(state_values), axis=1), 1.0e-12)
    )[:, None]

    error_coefficients = generator.normal(size=(count, 12))
    error_coefficients /= np.maximum(
        np.linalg.norm(error_coefficients, axis=1, keepdims=True), 1.0e-12
    )
    radii = np.exp(
        generator.uniform(np.log(0.02), np.log(0.8), size=count)
    )
    error_coefficients *= radii[:, None]
    return state_coefficients, error_coefficients


def bridge_basis(grid: AllenCahnGrid, mode_count: int) -> np.ndarray:
    modes = np.arange(1, mode_count + 1, dtype=float)
    return np.sqrt(2.0) * np.sin(
        np.pi * grid.x[:, None] * modes[None, :]
    )


def _adversarial_search(
    torch: object,
    bridge: object,
    *,
    device: str,
    seed: int,
    restarts: int,
    steps: int,
    step_size: float,
) -> dict[str, object]:
    state_initial, error_initial = _initial_adversarial_coefficients(
        bridge.grid, seed=seed, count=restarts
    )
    state_coefficients = torch.tensor(
        state_initial,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    error_coefficients = torch.tensor(
        error_initial,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    state_basis = torch.as_tensor(
        bridge_basis(bridge.grid, 8), dtype=torch.float64, device=device
    )
    error_basis = torch.as_tensor(
        bridge_basis(bridge.grid, 12), dtype=torch.float64, device=device
    )
    transform = torch.as_tensor(
        bridge.coordinate_transform, dtype=torch.float64, device=device
    )
    target = torch.as_tensor(
        bridge.target_generator, dtype=torch.float64, device=device
    )
    closed_loop = torch.as_tensor(
        bridge.closed_loop_generator, dtype=torch.float64, device=device
    )

    def margins() -> object:
        states = state_coefficients @ state_basis.T
        errors = error_coefficients @ error_basis.T
        transformed = errors @ transform.T
        nonlinear = -((states + errors) ** 3 - states**3)
        original_rhs = errors @ closed_loop.T + nonlinear
        transformed_rhs = original_rhs @ transform.T
        target_rhs = transformed @ target.T
        remainder = nonlinear @ transform.T
        reconstruction = torch.linalg.vector_norm(
            transformed_rhs - target_rhs - remainder, dim=1
        )
        scale = (
            torch.linalg.vector_norm(transformed_rhs, dim=1)
            + torch.linalg.vector_norm(target_rhs, dim=1)
            + torch.linalg.vector_norm(remainder, dim=1)
            + 1.0e-30
        )
        if float(torch.max(reconstruction / scale).detach().cpu()) > 1.0e-9:
            raise RuntimeError("adversarial decomposition lost numerical integrity")
        energy = torch.sum(transformed**2, dim=1)
        return (
            -torch.sum(transformed * transformed_rhs, dim=1)
            / (energy + 1.0e-30)
            - ALPHA
        )

    with torch.no_grad():
        initial_margins = margins().detach().cpu().numpy()
    optimizer = torch.optim.Adam(
        [state_coefficients, error_coefficients], lr=step_size
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(margins())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            state_values = state_coefficients @ state_basis.T
            state_maximum = torch.max(torch.abs(state_values), dim=1).values
            state_scale = torch.clamp(
                1.25 / torch.clamp(state_maximum, min=1.0e-30),
                max=1.0,
            )
            state_coefficients.mul_(state_scale[:, None])

            error_norm = torch.linalg.vector_norm(
                error_coefficients, dim=1
            )
            error_scale = torch.ones_like(error_norm)
            error_scale = torch.where(
                error_norm > 0.8, 0.8 / error_norm, error_scale
            )
            error_scale = torch.where(
                error_norm < 0.02, 0.02 / error_norm, error_scale
            )
            error_coefficients.mul_(error_scale[:, None])

    with torch.no_grad():
        final_margins = margins().detach().cpu().numpy()
        state_final = state_coefficients.detach().cpu().numpy()
        error_final = error_coefficients.detach().cpu().numpy()
    worst_index = int(np.argmin(final_margins))
    worst_state = state_final[worst_index]
    worst_error = error_final[worst_index]
    state_values = worst_state @ bridge_basis(bridge.grid, 8).T
    return {
        "initial_margin": _margin_summary(initial_margins),
        "final_margin": _margin_summary(final_margins),
        "worst": {
            "restart_index": worst_index,
            "margin": float(final_margins[worst_index]),
            "state_coefficients": worst_state.tolist(),
            "error_coefficients": worst_error.tolist(),
            "state_maximum_absolute_value": float(
                np.max(np.abs(state_values))
            ),
            "error_mass_norm": float(np.linalg.norm(worst_error)),
        },
        "strictly_positive": bool(np.min(final_margins) > 0.0),
    }


def _freeze_learned_model(
    torch: object,
    *,
    checkpoint: Path,
    device: str,
) -> tuple[object, object, dict[str, object], dict[str, object]]:
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
    gain = gain.to(device=device, dtype=torch.float64).eval()
    transform = transform.to(device=device, dtype=torch.float64).eval()
    for module in (gain, transform):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return gain, transform, initialization, base_diagnostics


def run(
    torch: object,
    *,
    checkpoint: Path,
    grid_sizes: Sequence[int],
    audit_seed: int,
    iid_count: int,
    ood_count: int,
    batch_size: int,
    device: str,
    adversarial_enabled: bool,
) -> dict[str, object]:
    checkpoint_hash_before = _sha256(checkpoint)
    if checkpoint_hash_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 does not match the frozen contract")
    learned_gain, learned_transform, initialization, base_diagnostics = (
        _freeze_learned_model(
            torch, checkpoint=checkpoint, device=device
        )
    )

    grids: dict[str, object] = {}
    bridge_objects: dict[int, object] = {}
    all_structure_passed = True
    all_practical_passed = True
    for grid_value in grid_sizes:
        grid_size = int(grid_value)
        grid = AllenCahnGrid(grid_size)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        bridge = build_sylvester_remainder_bridge(
            grid,
            NU_VALUE,
            observation,
            alpha=ALPHA,
            unstable_dimension=4,
        )
        bridge_objects[grid_size] = bridge
        iid_samples = _collocation_samples(
            grid, observation, seed=audit_seed, count=iid_count
        )
        iid_sylvester = _exact_model_arrays(
            bridge, iid_samples, identity_coordinate=False
        )
        iid_identity = _exact_model_arrays(
            bridge, iid_samples, identity_coordinate=True
        )
        iid_learned = _learned_arrays(
            torch,
            learned_transform,
            learned_gain,
            iid_samples,
            grid_size,
            batch_size=batch_size,
            device=device,
        )

        ood_parts_sylvester: list[dict[str, np.ndarray]] = []
        ood_parts_identity: list[dict[str, np.ndarray]] = []
        ood_parts_learned: list[dict[str, np.ndarray]] = []
        ood_cells: dict[str, object] = {}
        for family in STATE_OOD_FAMILIES:
            family_cells: dict[str, object] = {}
            for severity in STATE_OOD_SEVERITIES:
                samples = state_ood_samples(
                    grid,
                    family,
                    severity,
                    seed=audit_seed,
                    count=ood_count,
                )
                cell_sylvester = _exact_model_arrays(
                    bridge, samples, identity_coordinate=False
                )
                cell_identity = _exact_model_arrays(
                    bridge, samples, identity_coordinate=True
                )
                cell_learned = _learned_arrays(
                    torch,
                    learned_transform,
                    learned_gain,
                    samples,
                    grid_size,
                    batch_size=batch_size,
                    device=device,
                )
                ood_parts_sylvester.append(cell_sylvester)
                ood_parts_identity.append(cell_identity)
                ood_parts_learned.append(cell_learned)
                family_cells[str(severity)] = {
                    "sylvester_B_T": _summarize_exact(cell_sylvester),
                    "sylvester_B_identity": _summarize_exact(cell_identity),
                    "frozen_learned_B_Tphi": _summarize_learned(
                        cell_learned
                    ),
                }
            ood_cells[family] = family_cells

        ood_sylvester = _concatenate(ood_parts_sylvester)
        ood_identity = _concatenate(ood_parts_identity)
        ood_learned = _concatenate(ood_parts_learned)
        combined_sylvester = _concatenate(
            [iid_sylvester, ood_sylvester]
        )
        combined_summary = _summarize_exact(combined_sylvester)
        structure = _structure_gate(
            bridge.diagnostics, combined_summary["integrity"]
        )
        iid_summary = _summarize_exact(iid_sylvester)
        ood_summary = _summarize_exact(ood_sylvester)
        practical = {
            "iid": _practical_gate(iid_summary["actual_margin"]),
            "ood": _practical_gate(ood_summary["actual_margin"]),
        }
        practical["passed"] = bool(
            practical["iid"]["passed"] and practical["ood"]["passed"]
        )
        all_structure_passed = bool(
            all_structure_passed and structure["passed"]
        )
        all_practical_passed = bool(
            all_practical_passed and practical["passed"]
        )
        grids[str(grid_size)] = {
            "sylvester_diagnostics": bridge.diagnostics,
            "iid": {
                "sylvester_B_T": iid_summary,
                "sylvester_B_identity": _summarize_exact(iid_identity),
                "frozen_learned_B_Tphi": _summarize_learned(iid_learned),
            },
            "ood_aggregate": {
                "sylvester_B_T": ood_summary,
                "sylvester_B_identity": _summarize_exact(ood_identity),
                "frozen_learned_B_Tphi": _summarize_learned(ood_learned),
            },
            "ood_cells": ood_cells,
            "gates": {
                "structure": structure,
                "practical": practical,
            },
        }

    adversarial: dict[str, object] = {
        "eligible": bool(all_structure_passed and all_practical_passed),
        "enabled": bool(adversarial_enabled),
        "ran": False,
        "per_grid": {},
        "all_grids_strictly_positive": False,
    }
    if adversarial["eligible"] and adversarial_enabled:
        adversarial_results = {
            str(grid_size): _adversarial_search(
                torch,
                bridge_objects[grid_size],
                device=device,
                seed=ADVERSARIAL_SEED,
                restarts=ADVERSARIAL_RESTARTS,
                steps=ADVERSARIAL_STEPS,
                step_size=ADVERSARIAL_STEP_SIZE,
            )
            for grid_size in bridge_objects
        }
        adversarial.update(
            {
                "ran": True,
                "per_grid": adversarial_results,
                "all_grids_strictly_positive": bool(
                    all(
                        result["strictly_positive"]
                        for result in adversarial_results.values()
                    )
                ),
            }
        )

    if not all_structure_passed:
        next_route = "stop_sylvester_route_due_to_structural_failure"
    elif not all_practical_passed:
        next_route = "retain_direct_contraction_sylvester_is_control_only"
    elif not adversarial["ran"]:
        next_route = "formal_adversarial_search_required"
    elif not adversarial["all_grids_strictly_positive"]:
        next_route = "consider_small_invertible_correction_anchored_to_sylvester"
    else:
        next_route = "derive_mesh_uniform_and_continuum_remainder_bounds"

    checkpoint_hash_after = _sha256(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("checkpoint changed during evaluation-only audit")
    return {
        "kind": "r5-j-sylvester-remainder-bridge",
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
            "numpy": np.__version__,
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
            "audit_seed": audit_seed,
            "adversarial_seed": ADVERSARIAL_SEED,
            "iid_count_per_grid": iid_count,
            "ood_count_per_family_severity_grid": ood_count,
            "ood_families": list(STATE_OOD_FAMILIES),
            "ood_severities": list(STATE_OOD_SEVERITIES),
            "locked_test_seed_1901_read": False,
            "model_training": False,
            "model_optimizer_created": False,
            "condition_limit": CONDITION_LIMIT,
            "condition_limit_calibrated_before_formal_data": True,
            "structure_tolerance": STRUCTURE_TOLERANCE,
            "practical_nonnegative_fraction": (
                PRACTICAL_NONNEGATIVE_FRACTION
            ),
            "practical_p01_margin": PRACTICAL_P01_MARGIN,
            "adversarial": {
                "restarts": ADVERSARIAL_RESTARTS,
                "steps": ADVERSARIAL_STEPS,
                "step_size": ADVERSARIAL_STEP_SIZE,
                "state_modes": 8,
                "state_maximum_absolute_value": 1.25,
                "error_modes": 12,
                "error_mass_norm_interval": [0.02, 0.8],
            },
        },
        "base_learned_model_diagnostics": base_diagnostics,
        "grids": grids,
        "adversarial": adversarial,
        "decision": {
            "structure_all_grids_passed": all_structure_passed,
            "practical_all_grids_passed": all_practical_passed,
            "proof_candidate": bool(
                all_structure_passed
                and all_practical_passed
                and adversarial["ran"]
                and adversarial["all_grids_strictly_positive"]
            ),
            "next_route": next_route,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=AUDIT_GRIDS)
    parser.add_argument("--audit-seed", type=int, default=AUDIT_SEED)
    parser.add_argument("--iid-count", type=int)
    parser.add_argument("--ood-count", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    iid_count = (
        args.iid_count
        if args.iid_count is not None
        else (SMOKE_IID_COUNT if args.smoke else IID_COUNT)
    )
    ood_count = (
        args.ood_count
        if args.ood_count is not None
        else (SMOKE_OOD_COUNT if args.smoke else OOD_COUNT)
    )
    if not args.smoke:
        if tuple(args.grid_sizes) != AUDIT_GRIDS:
            raise SystemExit("formal audit requires grids 31 63 127 191")
        if iid_count != IID_COUNT or ood_count != OOD_COUNT:
            raise SystemExit("formal audit requires 4096 IID and 512 OOD samples")
    elif iid_count != SMOKE_IID_COUNT or ood_count != SMOKE_OOD_COUNT:
        raise SystemExit("smoke audit requires 128 IID and 16 OOD samples")
    if args.audit_seed != AUDIT_SEED:
        raise SystemExit("this audit requires frozen seed 2231")
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
        audit_seed=args.audit_seed,
        iid_count=iid_count,
        ood_count=ood_count,
        batch_size=args.batch_size,
        device=args.device,
        adversarial_enabled=not args.smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"]), flush=True)


if __name__ == "__main__":
    main()
