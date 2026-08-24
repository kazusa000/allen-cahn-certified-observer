"""Audit the selected R5 checkpoint on frozen physical local regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalNudging,
    LocalPartition,
    generate_pilot_cases,
    local_average_matrix,
    partition_samples,
    simulate_causal_nudging,
    summarize_local_region,
    transition_counts,
)
from r5_e_joint_train import INTERVALS, OUTPUT_TIMES
from r5_tk_joint_train import (
    _allen_cahn_rhs_tensor,
    _build_models,
    _feature_tensor,
    _policy_rollout,
    _target_operators,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(
    torch: object, checkpoint_path: Path, device: str
) -> tuple[dict[str, object], object, object, AllenCahnGrid, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    grid = AllenCahnGrid(int(checkpoint["grid_size"]))
    matrix = local_average_matrix(grid, INTERVALS)
    gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=float(checkpoint["base_gain"]),
        gain_scale=float(checkpoint["gain_scale"]),
        certificate_scale=float(checkpoint["certificate_scale"]),
        lower_lipschitz=float(checkpoint["lower_lipschitz"]),
        upper_lipschitz=float(checkpoint["upper_lipschitz"]),
        certificate_kind=str(checkpoint["certificate_kind"]),
        mixing_layers=int(checkpoint["mixing_layers"]),
        shear_norm_limit=float(checkpoint["shear_norm_limit"]),
        gain_trust_ratio=float(checkpoint["gain_trust_ratio"]),
        gain_kind=str(checkpoint["gain_kind"]),
    )
    gain.load_state_dict(checkpoint["gain_state_dict"])
    certificate.load_state_dict(checkpoint["certificate_state_dict"])
    gain.to(device).eval()
    certificate.to(device).eval()
    return checkpoint, gain, certificate, grid, matrix


def _collect_validation(
    torch: object,
    gain: object,
    device: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    case_limit: int,
) -> dict[str, np.ndarray]:
    cases = [
        case
        for case in generate_pilot_cases()
        if case.split == "validation" and case.n == grid.n
    ]
    if case_limit > 0:
        cases = cases[:case_limit]
    states: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    baseline_estimates: list[np.ndarray] = []
    nus: list[float] = []
    times: list[float] = []
    trajectory_ids: list[str] = []
    step_indices: list[int] = []
    for case in cases:
        truth, estimate, observed, status = _policy_rollout(
            torch, gain, device, grid, matrix, case
        )
        if status != 0 or truth.shape[0] != OUTPUT_TIMES.size:
            raise RuntimeError(f"current-policy rollout failed for {case.case_id}")
        baseline = simulate_causal_nudging(
            CausalNudging(grid, case.nu, matrix, gain=0.10),
            case.initial_truth(grid),
            case.initial_estimate(grid),
            output_times=OUTPUT_TIMES,
        )
        if baseline.solver_status != 0:
            raise RuntimeError(f"fixed-gain rollout failed for {case.case_id}")
        sample_count = OUTPUT_TIMES.size - 1
        states.extend(truth[:-1])
        estimates.extend(estimate[:-1])
        measurements.extend(observed[:-1])
        baseline_estimates.extend(baseline.estimate[:-1])
        nus.extend([float(case.nu)] * sample_count)
        times.extend(float(value) for value in OUTPUT_TIMES[:-1])
        trajectory_ids.extend([case.case_id] * sample_count)
        step_indices.extend(range(sample_count))
    return {
        "states": np.asarray(states, dtype=float),
        "estimates": np.asarray(estimates, dtype=float),
        "measurements": np.asarray(measurements, dtype=float),
        "baseline_estimates": np.asarray(baseline_estimates, dtype=float),
        "nus": np.asarray(nus, dtype=float),
        "times": np.asarray(times, dtype=float),
        "trajectory_ids": np.asarray(trajectory_ids, dtype=object),
        "step_indices": np.asarray(step_indices, dtype=int),
        "case_count": np.asarray([len(cases)], dtype=int),
    }


def _jacobian_singular_values(
    torch: object,
    certificate: object,
    states: object,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = int(states.shape[1])
    identity = torch.eye(dimension, dtype=states.dtype, device=states.device)
    minima: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            state_batch = states[start : start + batch_size]
            count = int(state_batch.shape[0])
            repeated_states = (
                state_batch[:, None, :]
                .expand(count, dimension, dimension)
                .reshape(count * dimension, dimension)
            )
            basis_errors = identity[None, :, :].expand(
                count, dimension, dimension
            ).reshape(count * dimension, dimension)
            rows = certificate(repeated_states, basis_errors).reshape(
                count, dimension, dimension
            )
            singular = torch.linalg.svdvals(rows)
            minima.append(singular[:, -1].cpu().numpy())
            maxima.append(singular[:, 0].cpu().numpy())
    return np.concatenate(minima), np.concatenate(maxima)


def _sample_metrics(
    torch: object,
    gain: object,
    certificate: object,
    data: dict[str, np.ndarray],
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    lambda_ratio: float,
    device: str,
    batch_size: int,
    jacobian_batch_size: int,
) -> dict[str, np.ndarray]:
    states = torch.as_tensor(data["states"], dtype=torch.float32, device=device)
    estimates = torch.as_tensor(
        data["estimates"], dtype=torch.float32, device=device
    )
    measurements = torch.as_tensor(
        data["measurements"], dtype=torch.float32, device=device
    )
    nus = torch.as_tensor(data["nus"], dtype=torch.float32, device=device)
    nu_values = tuple(sorted({float(value) for value in data["nus"]}))
    nu_lookup = {value: index for index, value in enumerate(nu_values)}
    nu_indices_numpy = np.asarray(
        [nu_lookup[float(value)] for value in data["nus"]], dtype=int
    )
    nu_indices = torch.as_tensor(nu_indices_numpy, dtype=torch.long, device=device)
    laplacian = torch.as_tensor(
        grid.laplacian, dtype=torch.float32, device=device
    )
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    target_generators, _ = _target_operators(grid, nu_values, lambda_ratio)
    generators = torch.as_tensor(
        target_generators, dtype=torch.float32, device=device
    )
    defect_parts: list[np.ndarray] = []
    contraction_parts: list[np.ndarray] = []
    direction_parts: list[np.ndarray] = []
    zero_parts: list[np.ndarray] = []
    count = states.shape[0]
    with torch.enable_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            state = states[start:stop]
            estimate = estimates[start:stop]
            measurement = measurements[start:stop]
            nu = nus[start:stop]
            indices = nu_indices[start:stop]
            features, innovations = _feature_tensor(
                torch, estimate, measurement, nu, matrix_tensor, grid.h
            )
            gains = gain(features)
            correction = torch.bmm(gains, innovations[:, :, None]).squeeze(-1)
            truth_rhs = _allen_cahn_rhs_tensor(
                torch, grid, state, nu, laplacian
            )
            estimate_rhs = _allen_cahn_rhs_tensor(
                torch, grid, estimate, nu, laplacian
            )
            error = estimate - state
            error_rhs = estimate_rhs + correction - truth_rhs

            def transform(state_value: object, error_value: object) -> object:
                return certificate(state_value, error_value)

            transformed = certificate(state, error)
            _, directional = torch.autograd.functional.jvp(
                transform,
                (state, error),
                (truth_rhs, error_rhs),
                create_graph=False,
            )
            target = torch.bmm(
                generators[indices], transformed[:, :, None]
            ).squeeze(-1)
            residual = directional - target
            error_squared = grid.h * torch.sum(error**2, dim=1)
            transformed_squared = grid.h * torch.sum(transformed**2, dim=1)
            defect = torch.sqrt(
                grid.h * torch.sum(residual**2, dim=1) / (error_squared + 1.0e-8)
            )
            contraction = -grid.h * torch.sum(
                transformed * directional, dim=1
            ) / (transformed_squared + 1.0e-8)
            direction = torch.linalg.vector_norm(
                (transformed - error) @ matrix_tensor.T, dim=1
            )
            zero = torch.linalg.vector_norm(
                certificate(state, torch.zeros_like(error)), dim=1
            )
            defect_parts.append(defect.detach().cpu().numpy())
            contraction_parts.append(contraction.detach().cpu().numpy())
            direction_parts.append(direction.detach().cpu().numpy())
            zero_parts.append(zero.detach().cpu().numpy())

    minimum_singular, maximum_singular = _jacobian_singular_values(
        torch,
        certificate,
        states,
        batch_size=jacobian_batch_size,
    )
    errors = data["estimates"] - data["states"]
    baseline_errors = data["baseline_estimates"] - data["states"]
    target_decay_by_nu = -np.max(
        np.linalg.eigvalsh(target_generators), axis=1
    )
    return {
        "defect_ratio": np.concatenate(defect_parts),
        "contraction_rate": np.concatenate(contraction_parts),
        "min_singular_value": minimum_singular,
        "max_singular_value": maximum_singular,
        "target_decay": target_decay_by_nu[nu_indices_numpy],
        "online_error": np.sqrt(grid.h * np.sum(errors**2, axis=1)),
        "baseline_error": np.sqrt(grid.h * np.sum(baseline_errors**2, axis=1)),
        "direction_residual": np.concatenate(direction_parts),
        "zero_fiber_residual": np.concatenate(zero_parts),
    }


def _region_tables(
    partitioned: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    nus: np.ndarray,
    *,
    minimum_samples: int,
) -> dict[str, object]:
    state_labels = partitioned["state_region"]
    error_labels = partitioned["error_region"]
    state_names = ("zero-near", "phase-dominated", "interface-mixed")
    error_names = ("small", "medium", "large")
    nu_values = tuple(sorted({float(value) for value in nus}))

    def summary(mask: np.ndarray) -> dict[str, object]:
        return summarize_local_region(
            mask, **metrics, minimum_samples=minimum_samples
        )

    state = {name: summary(state_labels == name) for name in state_names}
    error = {name: summary(error_labels == name) for name in error_names}
    viscosity = {
        f"{nu:.6g}": summary(np.isclose(nus, nu)) for nu in nu_values
    }
    cross = {}
    for state_name in state_names:
        for error_name in error_names:
            for nu in nu_values:
                key = f"{state_name}__{error_name}__nu-{nu:.6g}"
                cross[key] = summary(
                    (state_labels == state_name)
                    & (error_labels == error_name)
                    & np.isclose(nus, nu)
                )
    passed = [
        name
        for table in (state, error, viscosity, cross)
        for name, value in table.items()
        if value.get("passed", False)
    ]
    return {
        "by_state_region": state,
        "by_error_region": error,
        "by_viscosity": viscosity,
        "by_cross_region": cross,
        "passed_region_count": len(passed),
        "passed_regions": passed,
    }


def run(
    torch: object,
    checkpoint_path: Path,
    *,
    device: str,
    case_limit: int,
    batch_size: int,
    jacobian_batch_size: int,
    minimum_samples: int,
) -> dict[str, object]:
    checkpoint, gain, certificate, grid, matrix = _load_checkpoint(
        torch, checkpoint_path, device
    )
    data = _collect_validation(
        torch, gain, device, grid, matrix, case_limit=case_limit
    )
    errors = data["estimates"] - data["states"]
    partition = LocalPartition()
    partitioned = partition_samples(data["states"], errors, grid.h, partition)
    metrics = _sample_metrics(
        torch,
        gain,
        certificate,
        data,
        grid,
        matrix,
        lambda_ratio=float(checkpoint["lambda_ratio"]),
        device=device,
        batch_size=batch_size,
        jacobian_batch_size=jacobian_batch_size,
    )
    region_tables = _region_tables(
        partitioned,
        metrics,
        data["nus"],
        minimum_samples=minimum_samples,
    )
    overlap = partitioned["overlap_membership"]
    transitions = transition_counts(
        data["trajectory_ids"], partitioned["state_region"]
    )
    shared_overlap = []
    for index in range(1, overlap.shape[0]):
        if (
            data["trajectory_ids"][index] == data["trajectory_ids"][index - 1]
            and partitioned["state_region"][index]
            != partitioned["state_region"][index - 1]
        ):
            shared_overlap.append(bool(np.any(overlap[index - 1] & overlap[index])))
    return {
        "kind": "r5-partitioned-local-certificate-audit",
        "status": "completed",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "grid_size": int(checkpoint["grid_size"]),
            "seed": int(checkpoint["seed"]),
            "certificate_kind": str(checkpoint["certificate_kind"]),
            "lambda_ratio": float(checkpoint["lambda_ratio"]),
        },
        "data": {
            "split": "validation",
            "case_count": int(data["case_count"][0]),
            "sample_count": int(data["states"].shape[0]),
            "time_step": float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0]),
            "minimum_region_samples": minimum_samples,
        },
        "partition": {
            **partition.__dict__,
            "state_region_counts": {
                name: int(np.sum(partitioned["state_region"] == name))
                for name in ("zero-near", "phase-dominated", "interface-mixed")
            },
            "error_region_counts": {
                name: int(np.sum(partitioned["error_region"] == name))
                for name in ("small", "medium", "large")
            },
            "overlap_neighborhood_counts": {
                str(name): int(np.sum(overlap[:, index]))
                for index, name in enumerate(partitioned["overlap_names"])
            },
            "overlap_multiple_fraction": float(np.mean(np.sum(overlap, axis=1) > 1)),
            "overlap_uncovered_count": int(np.sum(np.sum(overlap, axis=1) == 0)),
        },
        "regions": region_tables,
        "trajectory_switching": {
            **transitions,
            "shared_overlap_fraction": float(np.mean(shared_overlap))
            if shared_overlap
            else None,
            "same_transform_across_regions": True,
            "coordinate_switch_jump": 0.0,
        },
        "strict_conclusion": (
            "at-least-one-pre-registered-region-passed"
            if region_tables["passed_region_count"] > 0
            else "no-pre-registered-region-passed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--jacobian-batch-size", type=int, default=64)
    parser.add_argument("--minimum-samples", type=int, default=30)
    args = parser.parse_args()
    if args.case_limit < 0:
        raise SystemExit("--case-limit must be nonnegative")
    if args.batch_size < 1 or args.jacobian_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    if args.minimum_samples < 1:
        raise SystemExit("--minimum-samples must be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    import torch

    output = run(
        torch,
        args.checkpoint,
        device=args.device,
        case_limit=args.case_limit,
        batch_size=args.batch_size,
        jacobian_batch_size=args.jacobian_batch_size,
        minimum_samples=args.minimum_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "strict_conclusion": output["strict_conclusion"],
                "passed_region_count": output["regions"]["passed_region_count"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
