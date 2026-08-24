"""Audit Pi_4 and I-Pi_4 contributions in the formal nonlinear-target checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import r5_nonlinear_target_conditional_residual_joint as joint
from allen_cahn_certified_observer import (
    AllenCahnGrid,
    build_conditional_residual_transform,
    build_projected_constant_gain,
    dirichlet_laplacian_rates,
    local_average_matrix,
    low_frequency_projector,
)


MODE_COUNT = 4
DOMINANCE_THRESHOLD = 0.60
COUNTERFACTUAL_IMPROVEMENT_THRESHOLD = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("summary values must be a finite non-empty vector")
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "rms": float(np.sqrt(np.mean(data**2))),
        "min": float(np.min(data)),
        "p05": float(np.quantile(data, 0.05)),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
    }


def _safe_share(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def _raw_checkpoint_fields(
    torch: object,
    gain: object,
    transform: object,
    samples: joint.InstantSamples,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    tensors = joint._tensorize_samples(
        torch, samples, grid, device=device, dtype=torch.float64
    )
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float64, device=device)
    fields: dict[str, list[np.ndarray]] = {
        "errors": [],
        "transformed": [],
        "transformed_rhs": [],
        "target_rhs": [],
        "defect_residual": [],
        "formal_margin_rate": [],
    }
    sample_count = int(tensors["states"].shape[0])
    for start in range(0, sample_count, batch_size):
        indices = torch.arange(
            start, min(start + batch_size, sample_count), device=device
        )
        components = joint._instantaneous_components(
            torch,
            gain,
            transform,
            tensors,
            matrix_tensor,
            indices,
            grid,
            create_graph=False,
            include_inverse=False,
            inverse_iterations=1,
        )
        for name in (
            "errors",
            "transformed",
            "transformed_rhs",
            "target_rhs",
            "defect_residual",
        ):
            fields[name].append(components[name].detach().cpu().numpy())
        fields["formal_margin_rate"].append(
            components["margins"].detach().cpu().numpy()
        )
    return {name: np.concatenate(parts) for name, parts in fields.items()}


def _frequency_arrays(
    grid: AllenCahnGrid,
    fields: dict[str, np.ndarray],
    projector: np.ndarray,
) -> dict[str, np.ndarray]:
    transformed = np.asarray(fields["transformed"], dtype=float)
    transformed_rhs = np.asarray(fields["transformed_rhs"], dtype=float)
    residual = np.asarray(fields["defect_residual"], dtype=float)
    errors = np.asarray(fields["errors"], dtype=float)
    if not (
        transformed.shape
        == transformed_rhs.shape
        == residual.shape
        == errors.shape
    ):
        raise ValueError("checkpoint fields must have matching shapes")
    if transformed.ndim != 2 or transformed.shape[1] != grid.n:
        raise ValueError(f"checkpoint fields must have shape (samples, {grid.n})")
    projection = np.asarray(projector, dtype=float)
    if projection.shape != (grid.n, grid.n):
        raise ValueError(f"projector must have shape {(grid.n, grid.n)}")

    z_low = transformed @ projection
    z_high = transformed - z_low
    rhs_low = transformed_rhs @ projection
    rhs_high = transformed_rhs - rhs_low
    residual_low = residual @ projection
    residual_high = residual - residual_low
    lambda_value = joint.LAMBDA_RATIO * joint.NU_VALUE * np.pi**2
    projected_target_residual = residual - (1.0 + lambda_value) * z_high

    error_energy = grid.h * np.sum(errors**2, axis=1)
    denominator = error_energy + 1.0e-8

    def squared_mass(values: np.ndarray) -> np.ndarray:
        return grid.h * np.sum(values**2, axis=1)

    residual_total_energy = squared_mass(residual)
    residual_low_energy = squared_mass(residual_low)
    residual_high_energy = squared_mass(residual_high)
    projected_residual_energy = squared_mass(projected_target_residual)
    z_total_energy = squared_mass(transformed)
    z_low_energy = squared_mass(z_low)
    z_high_energy = squared_mass(z_high)
    power_total = -grid.h * np.sum(transformed * transformed_rhs, axis=1)
    power_low = -grid.h * np.sum(z_low * rhs_low, axis=1)
    power_high = -grid.h * np.sum(z_high * rhs_high, axis=1)
    margin_power_total = power_total - lambda_value * z_total_energy
    margin_power_low = power_low - lambda_value * z_low_energy
    margin_power_high = power_high - lambda_value * z_high_energy
    rate_total = power_total / (z_total_energy + 1.0e-8)
    rate_low = power_low / (z_low_energy + 1.0e-8)
    rate_high = power_high / (z_high_energy + 1.0e-8)

    return {
        "defect_ratio_total": np.sqrt(residual_total_energy / denominator),
        "defect_ratio_low": np.sqrt(residual_low_energy / denominator),
        "defect_ratio_high": np.sqrt(residual_high_energy / denominator),
        "defect_ratio_projected_target": np.sqrt(
            projected_residual_energy / denominator
        ),
        "defect_energy_total": residual_total_energy,
        "defect_energy_low": residual_low_energy,
        "defect_energy_high": residual_high_energy,
        "z_energy_total": z_total_energy,
        "z_energy_low": z_low_energy,
        "z_energy_high": z_high_energy,
        "power_total": power_total,
        "power_low": power_low,
        "power_high": power_high,
        "margin_power_total": margin_power_total,
        "margin_power_low": margin_power_low,
        "margin_power_high": margin_power_high,
        "rate_total": rate_total,
        "rate_low": rate_low,
        "rate_high": rate_high,
        "margin_rate_total": rate_total - lambda_value,
        "margin_rate_low": rate_low - lambda_value,
        "margin_rate_high": rate_high - lambda_value,
        "formal_margin_rate": np.asarray(
            fields["formal_margin_rate"], dtype=float
        ),
    }


def _case_summary(
    arrays: dict[str, np.ndarray],
    case_ids: Sequence[str],
    times: np.ndarray,
) -> list[dict[str, object]]:
    identifiers = np.asarray(tuple(case_ids), dtype=object)
    time_values = np.asarray(times, dtype=float)
    if identifiers.size != time_values.size:
        raise ValueError("case_ids and times must have matching lengths")
    ordered_ids = tuple(dict.fromkeys(identifiers.tolist()))
    records: list[dict[str, object]] = []
    for case_id in ordered_ids:
        mask = identifiers == case_id
        failed = arrays["formal_margin_rate"][mask] < 0.0
        low_burden = float(
            np.sum(np.maximum(-arrays["margin_power_low"][mask][failed], 0.0))
        )
        high_burden = float(
            np.sum(np.maximum(-arrays["margin_power_high"][mask][failed], 0.0))
        )
        low_loss = float(np.sum(arrays["defect_ratio_low"][mask] ** 2))
        high_loss = float(np.sum(arrays["defect_ratio_high"][mask] ** 2))
        records.append(
            {
                "case_id": str(case_id),
                "sample_count": int(np.sum(mask)),
                "time_min": float(np.min(time_values[mask])),
                "time_max": float(np.max(time_values[mask])),
                "defect_rms": float(
                    np.sqrt(np.mean(arrays["defect_ratio_total"][mask] ** 2))
                ),
                "defect_low_rms": float(
                    np.sqrt(np.mean(arrays["defect_ratio_low"][mask] ** 2))
                ),
                "defect_high_rms": float(
                    np.sqrt(np.mean(arrays["defect_ratio_high"][mask] ** 2))
                ),
                "defect_high_squared_share": _safe_share(
                    high_loss, low_loss + high_loss
                ),
                "projected_target_defect_rms": float(
                    np.sqrt(
                        np.mean(
                            arrays["defect_ratio_projected_target"][mask] ** 2
                        )
                    )
                ),
                "contraction_failure_count": int(np.sum(failed)),
                "contraction_failure_fraction": float(np.mean(failed)),
                "formal_margin_rate_min": float(
                    np.min(arrays["formal_margin_rate"][mask])
                ),
                "negative_margin_burden_low": low_burden,
                "negative_margin_burden_high": high_burden,
                "negative_margin_burden_high_share": _safe_share(
                    high_burden, low_burden + high_burden
                ),
            }
        )
    return records


def _summarize_frequency_audit(
    arrays: dict[str, np.ndarray],
    case_ids: Sequence[str],
    times: np.ndarray,
) -> dict[str, object]:
    total_loss = float(np.sum(arrays["defect_ratio_total"] ** 2))
    low_loss = float(np.sum(arrays["defect_ratio_low"] ** 2))
    high_loss = float(np.sum(arrays["defect_ratio_high"] ** 2))
    failed = arrays["formal_margin_rate"] < 0.0
    low_negative = np.maximum(-arrays["margin_power_low"][failed], 0.0)
    high_negative = np.maximum(-arrays["margin_power_high"][failed], 0.0)
    low_burden = float(np.sum(low_negative))
    high_burden = float(np.sum(high_negative))
    projected = _summary(arrays["defect_ratio_projected_target"])
    original = _summary(arrays["defect_ratio_total"])
    only_low = (
        (arrays["margin_power_low"] < 0.0)
        & (arrays["margin_power_high"] >= 0.0)
        & failed
    )
    only_high = (
        (arrays["margin_power_high"] < 0.0)
        & (arrays["margin_power_low"] >= 0.0)
        & failed
    )
    both = (
        (arrays["margin_power_low"] < 0.0)
        & (arrays["margin_power_high"] < 0.0)
        & failed
    )
    worst_index = int(np.argmin(arrays["formal_margin_rate"]))
    return {
        "sample_count": int(arrays["defect_ratio_total"].size),
        "defect": {
            "original_target": original,
            "low": _summary(arrays["defect_ratio_low"]),
            "high": _summary(arrays["defect_ratio_high"]),
            "low_normalized_squared_sum": low_loss,
            "high_normalized_squared_sum": high_loss,
            "total_normalized_squared_sum": total_loss,
            "low_normalized_squared_share": _safe_share(
                low_loss, low_loss + high_loss
            ),
            "high_normalized_squared_share": _safe_share(
                high_loss, low_loss + high_loss
            ),
            "high_raw_energy_share": _safe_share(
                float(np.sum(arrays["defect_energy_high"])),
                float(np.sum(arrays["defect_energy_total"])),
            ),
            "orthogonal_additivity_relative_error_max": float(
                np.max(
                    np.abs(
                        arrays["defect_energy_total"]
                        - arrays["defect_energy_low"]
                        - arrays["defect_energy_high"]
                    )
                    / (arrays["defect_energy_total"] + 1.0e-12)
                )
            ),
            "projected_damping_counterfactual": {
                "summary": projected,
                "rms_reduction_fraction": float(
                    1.0 - float(projected["rms"]) / float(original["rms"])
                ),
                "p95_reduction_fraction": float(
                    1.0 - float(projected["p95"]) / float(original["p95"])
                ),
                "interpretation": (
                    "instantaneous checkpoint counterfactual; not a retrained model"
                ),
            },
        },
        "transformed_energy": {
            "low_sum": float(np.sum(arrays["z_energy_low"])),
            "high_sum": float(np.sum(arrays["z_energy_high"])),
            "high_share": _safe_share(
                float(np.sum(arrays["z_energy_high"])),
                float(np.sum(arrays["z_energy_total"])),
            ),
        },
        "contraction_power": {
            "total": _summary(arrays["power_total"]),
            "low": _summary(arrays["power_low"]),
            "high": _summary(arrays["power_high"]),
            "power_additivity_relative_error_max": float(
                np.max(
                    np.abs(
                        arrays["power_total"]
                        - arrays["power_low"]
                        - arrays["power_high"]
                    )
                    / (np.abs(arrays["power_total"]) + 1.0e-12)
                )
            ),
            "rate_total": _summary(arrays["rate_total"]),
            "rate_low": _summary(arrays["rate_low"]),
            "rate_high": _summary(arrays["rate_high"]),
            "margin_power_total": _summary(arrays["margin_power_total"]),
            "margin_power_low": _summary(arrays["margin_power_low"]),
            "margin_power_high": _summary(arrays["margin_power_high"]),
            "margin_additivity_absolute_error_max": float(
                np.max(
                    np.abs(
                        arrays["margin_power_total"]
                        - arrays["margin_power_low"]
                        - arrays["margin_power_high"]
                    )
                )
            ),
            "formal_margin_rate": _summary(arrays["formal_margin_rate"]),
            "formal_failure_count": int(np.sum(failed)),
            "formal_failure_fraction": float(np.mean(failed)),
            "failed_sample_partition": {
                "only_low_negative": int(np.sum(only_low)),
                "only_high_negative": int(np.sum(only_high)),
                "both_negative": int(np.sum(both)),
                "opposite_sign_cancellation_or_roundoff": int(
                    np.sum(failed) - np.sum(only_low) - np.sum(only_high) - np.sum(both)
                ),
            },
            "failed_sample_negative_margin_burden": {
                "low": low_burden,
                "high": high_burden,
                "low_share": _safe_share(
                    low_burden, low_burden + high_burden
                ),
                "high_share": _safe_share(
                    high_burden, low_burden + high_burden
                ),
            },
            "worst_formal_margin_sample": {
                "index": worst_index,
                "case_id": str(case_ids[worst_index]),
                "time": float(times[worst_index]),
                "formal_margin_rate": float(
                    arrays["formal_margin_rate"][worst_index]
                ),
                "low_margin_power": float(
                    arrays["margin_power_low"][worst_index]
                ),
                "high_margin_power": float(
                    arrays["margin_power_high"][worst_index]
                ),
                "low_margin_rate": float(
                    arrays["margin_rate_low"][worst_index]
                ),
                "high_margin_rate": float(
                    arrays["margin_rate_high"][worst_index]
                ),
            },
        },
        "by_case": _case_summary(arrays, case_ids, times),
        "attribution_accumulators": {
            "defect_low_normalized_squared_sum": low_loss,
            "defect_high_normalized_squared_sum": high_loss,
            "negative_margin_burden_low": low_burden,
            "negative_margin_burden_high": high_burden,
        },
    }


def _formal_replay_check(
    audit: dict[str, object], expected: dict[str, float]
) -> dict[str, object]:
    actual = {
        "defect_rms": float(audit["defect"]["original_target"]["rms"]),
        "defect_p95": float(audit["defect"]["original_target"]["p95"]),
        "requested_margin_min": float(
            audit["contraction_power"]["formal_margin_rate"]["min"]
        ),
        "requested_rate_fraction": float(
            1.0 - audit["contraction_power"]["formal_failure_fraction"]
        ),
    }
    differences = {
        name: abs(actual[name] - float(expected[name])) for name in actual
    }
    return {
        "actual": actual,
        "expected": expected,
        "absolute_differences": differences,
        "tolerance": 1.0e-8,
        "passed": bool(max(differences.values()) <= 1.0e-8),
    }


def _expected_metrics(record: dict[str, object]) -> dict[str, float]:
    return {
        "defect_rms": float(record["dynamics"]["defect"]["rms"]),
        "defect_p95": float(record["dynamics"]["defect"]["p95"]),
        "requested_margin_min": float(
            record["dynamics"]["contraction"]["requested_margin_min"]
        ),
        "requested_rate_fraction": float(
            record["dynamics"]["contraction"]["requested_rate_fraction"]
        ),
    }


def _load_checkpoint(
    torch: object, checkpoint_path: Path, *, device: str
) -> tuple[object, object, dict[str, object]]:
    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if payload.get("kind") != "r5-nonlinear-target-conditional-residual-joint":
        raise ValueError(f"unexpected checkpoint kind: {checkpoint_path}")
    if int(payload["grid_size"]) != joint.GRID_SIZE:
        raise ValueError(f"unexpected checkpoint grid: {checkpoint_path}")
    if float(payload["nu"]) != joint.NU_VALUE:
        raise ValueError(f"unexpected checkpoint nu: {checkpoint_path}")
    gain = build_projected_constant_gain(
        torch,
        np.asarray(payload["base_gain"], dtype=float),
        trust_ratio=float(payload["gain_trust_ratio"]),
    ).to(device=device, dtype=torch.float64)
    transform = build_conditional_residual_transform(
        torch,
        joint.GRID_SIZE,
        hidden_width=int(payload["hidden_width"]),
        hidden_layers=int(payload["hidden_layers"]),
        rho=float(payload["rho"]),
    ).to(device=device, dtype=torch.float64)
    gain.load_state_dict(payload["gain_state_dict"])
    transform.load_state_dict(payload["transform_state_dict"])
    gain.eval()
    transform.eval()
    return gain, transform, payload


def _audit_one(
    torch: object,
    gain: object,
    transform: object,
    cases: list[joint.ExperimentCase],
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    projector: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[str, object], joint.InstantSamples]:
    current_gain = gain().detach().cpu().numpy()
    samples, _ = joint._simulate_cases(grid, matrix, current_gain, cases)
    fields = _raw_checkpoint_fields(
        torch,
        gain,
        transform,
        samples,
        grid,
        matrix,
        device=device,
        batch_size=batch_size,
    )
    arrays = _frequency_arrays(grid, fields, projector)
    return _summarize_frequency_audit(
        arrays, samples.case_ids, samples.times
    ), samples


def _classification(
    seed_audits: Sequence[dict[str, object]],
    projected_target_proof_passed: bool,
) -> dict[str, object]:
    defect_high_shares = [
        float(item["defect"]["high_normalized_squared_share"])
        for item in seed_audits
    ]
    contraction_high_shares = [
        float(
            item["contraction_power"][
                "failed_sample_negative_margin_burden"
            ]["high_share"]
        )
        for item in seed_audits
    ]
    accumulators = [item["attribution_accumulators"] for item in seed_audits]
    pooled_defect_low = sum(
        float(item["defect_low_normalized_squared_sum"])
        for item in accumulators
    )
    pooled_defect_high = sum(
        float(item["defect_high_normalized_squared_sum"])
        for item in accumulators
    )
    pooled_margin_low = sum(
        float(item["negative_margin_burden_low"])
        for item in accumulators
    )
    pooled_margin_high = sum(
        float(item["negative_margin_burden_high"])
        for item in accumulators
    )
    pooled_defect_high_share = _safe_share(
        pooled_defect_high, pooled_defect_low + pooled_defect_high
    )
    pooled_contraction_high_share = _safe_share(
        pooled_margin_high, pooled_margin_low + pooled_margin_high
    )

    def classify(shares: Sequence[float], pooled: float) -> str:
        high_votes = sum(value >= DOMINANCE_THRESHOLD for value in shares)
        low_votes = sum(value <= 1.0 - DOMINANCE_THRESHOLD for value in shares)
        if high_votes >= 2 and pooled >= DOMINANCE_THRESHOLD:
            return "high-frequency-dominant"
        if low_votes >= 2 and pooled <= 1.0 - DOMINANCE_THRESHOLD:
            return "low-frequency-dominant"
        return "mixed-frequency"

    defect_class = classify(defect_high_shares, pooled_defect_high_share)
    contraction_class = classify(
        contraction_high_shares, pooled_contraction_high_share
    )
    rms_reductions = [
        float(
            item["defect"]["projected_damping_counterfactual"][
                "rms_reduction_fraction"
            ]
        )
        for item in seed_audits
    ]
    p95_reductions = [
        float(
            item["defect"]["projected_damping_counterfactual"][
                "p95_reduction_fraction"
            ]
        )
        for item in seed_audits
    ]
    counterfactual_passed = bool(
        all(
            value >= COUNTERFACTUAL_IMPROVEMENT_THRESHOLD
            for value in rms_reductions + p95_reductions
        )
    )
    recommend = bool(
        projected_target_proof_passed
        and counterfactual_passed
        and defect_class == "high-frequency-dominant"
        and contraction_class != "low-frequency-dominant"
    )
    return {
        "dominance_threshold": DOMINANCE_THRESHOLD,
        "defect_high_shares_by_seed": defect_high_shares,
        "defect_high_share_pooled": pooled_defect_high_share,
        "defect_attribution": defect_class,
        "contraction_high_negative_burden_shares_by_seed": (
            contraction_high_shares
        ),
        "contraction_high_negative_burden_share_pooled": (
            pooled_contraction_high_share
        ),
        "contraction_attribution": contraction_class,
        "counterfactual_rms_reductions_by_seed": rms_reductions,
        "counterfactual_p95_reductions_by_seed": p95_reductions,
        "counterfactual_improvement_threshold": (
            COUNTERFACTUAL_IMPROVEMENT_THRESHOLD
        ),
        "counterfactual_improvement_passed": counterfactual_passed,
        "projected_target_proof_passed": projected_target_proof_passed,
        "recommend_projected_damping_as_next_primary_candidate": recommend,
        "decision": (
            "recommend -(1+lambda) Pi_4 z for the next preregistered run"
            if recommend
            else "do not select projected damping as the primary repair from this audit"
        ),
    }


def run(
    torch: object,
    *,
    formal_results_path: Path,
    checkpoint_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    formal = json.loads(formal_results_path.read_text(encoding="utf-8"))
    if formal.get("test_evaluated") is not False:
        raise ValueError("formal result must have an untouched locked test")
    grid = AllenCahnGrid(joint.GRID_SIZE)
    matrix = local_average_matrix(grid, joint.THREE_SENSOR_INTERVALS)
    projector = low_frequency_projector(grid, MODE_COUNT)
    base_gain, base_transform, _ = joint._design_base(grid, matrix)
    frozen = formal["frozen"]
    validation_cases = joint._case_split(
        "validation",
        grid,
        matrix,
        seeds=tuple(int(seed) for seed in frozen["validation_case_seeds"]),
        base_case_limit=int(frozen["validation_base_case_limit"]),
        stress_truths=int(frozen["stress_truths_per_split"]),
    )

    baseline_gain = build_projected_constant_gain(
        torch, base_gain, trust_ratio=float(frozen["gain_trust_ratio"])
    ).to(device=device, dtype=torch.float64)
    baseline_transform = joint._build_fixed_linear_transform(
        torch, base_transform
    ).to(device=device, dtype=torch.float64)
    baseline_audit, _ = _audit_one(
        torch,
        baseline_gain,
        baseline_transform,
        validation_cases,
        grid,
        matrix,
        projector,
        device=device,
        batch_size=batch_size,
    )
    baseline_expected = _expected_metrics(
        formal["baselines"]["fixed-B0__fixed-T0"]
    )
    baseline_audit["formal_replay"] = _formal_replay_check(
        baseline_audit, baseline_expected
    )

    formal_by_seed = {
        int(item["seed"]): item for item in formal["seed_results"]
    }
    checkpoint_paths = sorted(checkpoint_dir.glob("joint__grid-31__seed-*.pt"))
    if len(checkpoint_paths) != 3:
        raise ValueError("exactly three formal checkpoints are required")
    checkpoints: list[dict[str, object]] = []
    seed_audits: list[dict[str, object]] = []
    for checkpoint_path in checkpoint_paths:
        gain, transform, payload = _load_checkpoint(
            torch, checkpoint_path, device=device
        )
        seed = int(payload["seed"])
        if seed not in formal_by_seed:
            raise ValueError(f"checkpoint seed {seed} is absent from formal results")
        if not np.allclose(
            np.asarray(payload["base_gain"], dtype=float), base_gain
        ):
            raise ValueError(f"checkpoint base gain mismatch for seed {seed}")
        audit, _ = _audit_one(
            torch,
            gain,
            transform,
            validation_cases,
            grid,
            matrix,
            projector,
            device=device,
            batch_size=batch_size,
        )
        audit["formal_replay"] = _formal_replay_check(
            audit, _expected_metrics(formal_by_seed[seed]["validation"])
        )
        checkpoints.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "audit": audit,
            }
        )
        seed_audits.append(audit)

    lambda_value = joint.LAMBDA_RATIO * joint.NU_VALUE * np.pi**2
    rates = dirichlet_laplacian_rates(grid)
    fifth_mode_margin = joint.NU_VALUE * float(rates[MODE_COUNT]) - 1.0
    projected_target_proof = {
        "lambda": lambda_value,
        "fifth_discrete_laplacian_rate": float(rates[MODE_COUNT]),
        "high_frequency_linear_decay_margin": fifth_mode_margin,
        "condition": "nu * mu_5,h - 1 >= lambda",
        "passed": bool(fifth_mode_margin >= lambda_value),
        "certified_target_decay_rate": float(
            min(lambda_value, fifth_mode_margin)
        ),
    }
    replay_passed = bool(
        baseline_audit["formal_replay"]["passed"]
        and all(item["audit"]["formal_replay"]["passed"] for item in checkpoints)
    )
    classification = _classification(
        seed_audits, bool(projected_target_proof["passed"])
    )
    return {
        "kind": "r5-nonlinear-target-four-mode-frequency-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_git_commit": joint._git_head(),
        "formal_git_commit": formal["git_commit"],
        "formal_results": str(formal_results_path),
        "formal_results_sha256": _sha256(formal_results_path),
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
            "nu": joint.NU_VALUE,
            "grid_size": joint.GRID_SIZE,
            "mode_count": MODE_COUNT,
            "validation_case_count": len(validation_cases),
            "validation_sample_count": len(validation_cases)
            * joint.OUTPUT_TIMES.size,
            "locked_test_accessed": False,
            "dominance_threshold": DOMINANCE_THRESHOLD,
            "counterfactual_improvement_threshold": (
                COUNTERFACTUAL_IMPROVEMENT_THRESHOLD
            ),
        },
        "projector": {
            "rank": int(np.linalg.matrix_rank(projector)),
            "symmetry_error": float(np.max(np.abs(projector - projector.T))),
            "idempotence_error": float(
                np.max(np.abs(projector @ projector - projector))
            ),
        },
        "projected_target_contraction_proof": projected_target_proof,
        "formal_replay_all_passed": replay_passed,
        "baseline_fixed_B0_T0": baseline_audit,
        "checkpoints": checkpoints,
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-results", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--mode-count", type=int, default=MODE_COUNT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode_count != MODE_COUNT:
        raise SystemExit("this frozen audit requires --mode-count 4")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if not args.formal_results.is_file():
        raise SystemExit(f"formal results not found: {args.formal_results}")
    if not args.checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint directory not found: {args.checkpoint_dir}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    result = run(
        torch,
        formal_results_path=args.formal_results,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["classification"], indent=2), flush=True)


if __name__ == "__main__":
    main()
