"""Train B and a nonlinear conditional invertible residual transform at nu=0.005."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalOutputInjection,
    allen_cahn_rhs,
    build_conditional_residual_transform,
    build_preconditioned_conditional_residual_transform,
    build_projected_constant_gain,
    lmi_modal_injection,
    local_average_matrix,
    nonlinear_target_tensor,
    normalized_modal_transform,
    simulate_causal_nudging,
    solve_allen_cahn,
    unstable_modal_system,
)


NU_VALUE = 0.005
GRID_SIZE = 31
LAMBDA_RATIO = 0.1
OUTPUT_TIMES = np.linspace(0.0, 1.0, 101)
THREE_SENSOR_INTERVALS = np.asarray(
    [
        [1.0 / 6.0, 7.0 / 30.0],
        [7.0 / 15.0, 8.0 / 15.0],
        [23.0 / 30.0, 5.0 / 6.0],
    ],
    dtype=float,
)
LMI_CONDITION_BOUNDS = (16.0, 32.0, 64.0, 128.0, 256.0)
TRAIN_CASE_SEEDS = (701, 702, 703, 704)
VALIDATION_CASE_SEEDS = (801, 802)
TEST_CASE_SEEDS = (901, 902)
NONLINEARITY_THRESHOLD = 0.02
INVERSE_RELATIVE_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    split: str
    seed: int
    draw: int
    truth_initial: np.ndarray
    estimate_initial: np.ndarray


@dataclass(frozen=True)
class InstantSamples:
    states: np.ndarray
    estimates: np.ndarray
    measurements: np.ndarray
    times: np.ndarray
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class TruthRollouts:
    states: np.ndarray
    estimate_initials: np.ndarray
    case_ids: tuple[str, ...]


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _coefficient_draw(
    seed: int, draw: int
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64DXSM(seed + 1009 * draw))
    state = generator.uniform(-0.5, 0.5, size=3)
    error_direction = generator.normal(size=3)
    error_direction /= np.linalg.norm(error_direction)
    error = error_direction * generator.uniform(0.05, 0.25)
    return state, error


def _modal_state(
    grid: AllenCahnGrid, coefficients: Sequence[float]
) -> np.ndarray:
    values = np.asarray(tuple(coefficients), dtype=float)
    return sum(
        coefficient * np.sin(mode * np.pi * grid.x)
        for mode, coefficient in enumerate(values, start=1)
    )


def _case_split(
    split: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    seeds: Sequence[int],
    base_case_limit: int,
    stress_truths: int,
) -> list[ExperimentCase]:
    if base_case_limit < 1:
        raise ValueError("base_case_limit must be positive")
    if stress_truths < 0:
        raise ValueError("stress_truths cannot be negative")
    base: list[ExperimentCase] = []
    for seed in seeds:
        for draw in range(4):
            truth_coefficients, error_coefficients = _coefficient_draw(seed, draw)
            truth = _modal_state(grid, truth_coefficients)
            error = _modal_state(grid, error_coefficients)
            base.append(
                ExperimentCase(
                    case_id=(
                        f"r5-nonlinear-target__split-{split}__seed-{seed}"
                        f"__draw-{draw}__n-{grid.n}__nu-0p005"
                    ),
                    split=split,
                    seed=int(seed),
                    draw=draw,
                    truth_initial=truth,
                    estimate_initial=truth + error,
                )
            )
    base = base[:base_case_limit]
    cases = list(base)
    if stress_truths == 0:
        return cases

    target_mass_norm = 0.25 / np.sqrt(2.0)
    fourth = np.sin(4.0 * np.pi * grid.x)
    fourth *= target_mass_norm / np.sqrt(grid.h * np.dot(fourth, fourth))
    modal = unstable_modal_system(grid, NU_VALUE, matrix)
    _, _, right = np.linalg.svd(modal.observed_modes, full_matrices=True)
    hard = modal.modes @ right[-1]
    hard *= target_mass_norm / np.sqrt(grid.h * np.dot(hard, hard))
    for index, case in enumerate(base[:stress_truths]):
        for name, direction in (("fourth", fourth), ("min-observation", hard)):
            for sign in (-1.0, 1.0):
                cases.append(
                    ExperimentCase(
                        case_id=(
                            f"r5-nonlinear-target__split-{split}"
                            f"__stress-{name}-{index}__sign-{sign:+g}"
                            f"__n-{grid.n}__nu-0p005"
                        ),
                        split=split,
                        seed=case.seed,
                        draw=case.draw,
                        truth_initial=case.truth_initial.copy(),
                        estimate_initial=(
                            case.truth_initial + sign * direction
                        ),
                    )
                )
    return cases


def _design_base(
    grid: AllenCahnGrid, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    selected = None
    selected_bound = None
    requested_rate = LAMBDA_RATIO * NU_VALUE * np.pi**2
    for condition_bound in LMI_CONDITION_BOUNDS:
        try:
            selected = lmi_modal_injection(
                grid,
                NU_VALUE,
                matrix,
                decay_rate=requested_rate,
                metric_condition_bound=condition_bound,
            )
        except RuntimeError:
            continue
        selected_bound = condition_bound
        break
    if selected is None or selected_bound is None:
        raise RuntimeError("no feasible LMI base design for nu=0.005")
    modal = unstable_modal_system(grid, NU_VALUE, matrix)
    transform = normalized_modal_transform(
        grid, modal, selected.modal_metric
    )
    singular = np.linalg.svd(transform, compute_uv=False)
    return (
        selected.injection_matrix,
        transform,
        {
            "nu": NU_VALUE,
            "condition_bound": selected_bound,
            "unstable_dimension": modal.dimension,
            "observability_rank": modal.observability_rank,
            "closed_loop_spectral_abscissa": (
                selected.closed_loop_spectral_abscissa
            ),
            "modal_contraction_rate": selected.modal_contraction_rate,
            "mass_scaled_gain_norm": selected.mass_scaled_gain_norm,
            "modal_metric_condition": selected.modal_metric_condition,
            "transform_min_singular_value": float(singular[-1]),
            "transform_max_singular_value": float(singular[0]),
        },
    )


def _simulate_cases(
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    injection: np.ndarray,
    cases: Sequence[ExperimentCase],
) -> tuple[InstantSamples, dict[str, object]]:
    states: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    times: list[float] = []
    sample_case_ids: list[str] = []
    records: list[dict[str, object]] = []
    observer = CausalOutputInjection(
        grid, NU_VALUE, matrix, np.asarray(injection, dtype=float)
    )
    for case in cases:
        rollout = simulate_causal_nudging(
            observer,
            case.truth_initial,
            case.estimate_initial,
            output_times=OUTPUT_TIMES,
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        if rollout.solver_status != 0:
            raise RuntimeError(f"observer rollout failed for {case.case_id}")
        error = rollout.estimate - rollout.truth
        norms = np.sqrt(grid.h * np.sum(error**2, axis=1))
        records.append(
            {
                "case_id": case.case_id,
                "terminal_error_mass": float(norms[-1]),
                "peak_error_mass": float(np.max(norms)),
                "time_average_error_mass": float(np.mean(norms)),
            }
        )
        states.extend(rollout.truth)
        estimates.extend(rollout.estimate)
        measurements.extend(rollout.measurements)
        times.extend(rollout.times)
        sample_case_ids.extend([case.case_id] * rollout.times.size)
    terminal = np.asarray(
        [record["terminal_error_mass"] for record in records], dtype=float
    )
    peak = np.asarray(
        [record["peak_error_mass"] for record in records], dtype=float
    )
    average = np.asarray(
        [record["time_average_error_mass"] for record in records], dtype=float
    )
    return (
        InstantSamples(
            states=np.asarray(states),
            estimates=np.asarray(estimates),
            measurements=np.asarray(measurements),
            times=np.asarray(times),
            case_ids=tuple(sample_case_ids),
        ),
        {
            "case_count": len(records),
            "terminal_error_mass_median": float(np.median(terminal)),
            "terminal_error_mass_max": float(np.max(terminal)),
            "peak_error_mass_max": float(np.max(peak)),
            "time_average_error_mass_median": float(np.median(average)),
            "records": records,
        },
    )


def _truth_rollouts(
    grid: AllenCahnGrid, cases: Sequence[ExperimentCase]
) -> TruthRollouts:
    trajectories: list[np.ndarray] = []
    initials: list[np.ndarray] = []
    ids: list[str] = []
    for case in cases:
        solution = solve_allen_cahn(
            grid,
            NU_VALUE,
            case.truth_initial,
            output_times=OUTPUT_TIMES,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        if solution.solver_status != 0:
            raise RuntimeError(f"truth rollout failed for {case.case_id}")
        trajectories.append(solution.states)
        initials.append(case.estimate_initial)
        ids.append(case.case_id)
    return TruthRollouts(
        states=np.asarray(trajectories),
        estimate_initials=np.asarray(initials),
        case_ids=tuple(ids),
    )


def _tensorize_samples(
    torch: object,
    samples: InstantSamples,
    grid: AllenCahnGrid,
    *,
    device: str,
    dtype: object,
) -> dict[str, object]:
    return {
        "states": torch.as_tensor(samples.states, dtype=dtype, device=device),
        "estimates": torch.as_tensor(
            samples.estimates, dtype=dtype, device=device
        ),
        "measurements": torch.as_tensor(
            samples.measurements, dtype=dtype, device=device
        ),
        "laplacian": torch.as_tensor(
            grid.laplacian, dtype=dtype, device=device
        ),
    }


def _tensorize_truth_rollouts(
    torch: object,
    rollouts: TruthRollouts,
    *,
    device: str,
    dtype: object,
) -> dict[str, object]:
    return {
        "states": torch.as_tensor(
            rollouts.states, dtype=dtype, device=device
        ),
        "estimate_initials": torch.as_tensor(
            rollouts.estimate_initials, dtype=dtype, device=device
        ),
    }


def _build_fixed_linear_transform(
    torch: object, matrix: np.ndarray
) -> object:
    nn = torch.nn
    value = np.asarray(matrix, dtype=float)

    class FixedLinearTransform(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "matrix", torch.as_tensor(value, dtype=torch.float64)
            )

        def forward(self, states: object, errors: object) -> object:
            del states
            return errors @ self.matrix.T

    return FixedLinearTransform()


def _instantaneous_components(
    torch: object,
    gain: object,
    transform: object,
    samples: dict[str, object],
    matrix: object,
    indices: object,
    grid: AllenCahnGrid,
    *,
    create_graph: bool,
    include_inverse: bool,
    inverse_iterations: int,
) -> dict[str, object]:
    states = samples["states"][indices]
    estimates = samples["estimates"][indices]
    measurements = samples["measurements"][indices]
    errors = estimates - states
    batch_size = int(states.shape[0])
    gains = gain(batch_size)
    innovations = measurements - estimates @ matrix.T
    corrections = torch.bmm(
        gains, innovations[:, :, None]
    ).squeeze(-1)
    rhs_truth = (
        NU_VALUE * (states @ samples["laplacian"].T)
        + states
        - states**3
    )
    rhs_estimate = (
        NU_VALUE * (estimates @ samples["laplacian"].T)
        + estimates
        - estimates**3
        + corrections
    )
    error_rhs = rhs_estimate - rhs_truth
    transformed = transform(states, errors)

    def transform_map(state: object, error: object) -> object:
        return transform(state, error)

    _, transformed_rhs = torch.autograd.functional.jvp(
        transform_map,
        (states, errors),
        (rhs_truth, error_rhs),
        create_graph=create_graph,
    )
    target_rhs = nonlinear_target_tensor(
        torch,
        states,
        transformed,
        torch.full(
            (batch_size,),
            NU_VALUE,
            dtype=states.dtype,
            device=states.device,
        ),
        samples["laplacian"],
        lambda_ratio=LAMBDA_RATIO,
    )
    error_squared = grid.h * torch.sum(errors**2, dim=1)
    transformed_squared = grid.h * torch.sum(transformed**2, dim=1)
    defect_squared = grid.h * torch.sum(
        (transformed_rhs - target_rhs) ** 2, dim=1
    )
    defect_ratio_squared = defect_squared / (error_squared + 1.0e-8)
    rates = -grid.h * torch.sum(
        transformed * transformed_rhs, dim=1
    ) / (transformed_squared + 1.0e-8)
    requested = torch.full_like(
        rates, LAMBDA_RATIO * NU_VALUE * np.pi**2
    )
    margins = rates - requested
    violations = torch.relu(-margins) ** 2
    tail_count = max(1, int(math.ceil(0.1 * batch_size)))
    contraction_loss = torch.mean(torch.topk(violations, tail_count).values)
    defect_loss = torch.mean(defect_ratio_squared)

    if include_inverse:
        inverse_count = min(16, batch_size)
        inverse_states = states[:inverse_count]
        inverse_errors = errors[:inverse_count]
        inverse_transformed = transformed[:inverse_count]
        reconstructed = transform.inverse_fixed_point(
            inverse_states,
            inverse_transformed,
            iterations=inverse_iterations,
        )
        inverse_error = grid.h * torch.sum(
            (reconstructed - inverse_errors) ** 2, dim=1
        ) / (
            grid.h * torch.sum(inverse_errors**2, dim=1) + 1.0e-8
        )
        lipschitz = transform.residual_lipschitz_bound_tensor()
        spectral_violation = torch.relu(2.0 * lipschitz - transform.rho) ** 2
        invertibility_loss = torch.mean(inverse_error) + spectral_violation
    else:
        invertibility_loss = torch.zeros(
            (), dtype=states.dtype, device=states.device
        )

    return {
        "contraction": contraction_loss,
        "defect": defect_loss,
        "invertibility": invertibility_loss,
        "errors": errors,
        "transformed": transformed,
        "transformed_rhs": transformed_rhs,
        "target_rhs": target_rhs,
        "defect_residual": transformed_rhs - target_rhs,
        "error_squared": error_squared,
        "transformed_squared": transformed_squared,
        "defect_ratio_squared": defect_ratio_squared,
        "rates": rates,
        "requested": requested,
        "margins": margins,
    }


def _observer_rhs_tensor(
    torch: object,
    estimates: object,
    truths: object,
    gain_matrix: object,
    observation_matrix: object,
    laplacian: object,
) -> object:
    innovations = truths @ observation_matrix.T - estimates @ observation_matrix.T
    return (
        NU_VALUE * (estimates @ laplacian.T)
        + estimates
        - estimates**3
        + innovations @ gain_matrix.T
    )


def _differentiable_online_loss(
    torch: object,
    gain: object,
    truth_rollouts: dict[str, object],
    case_indices: object,
    matrix: object,
    laplacian: object,
    grid: AllenCahnGrid,
) -> object:
    truths = truth_rollouts["states"][case_indices]
    estimates = truth_rollouts["estimate_initials"][case_indices]
    gain_matrix = gain()
    initial_errors = estimates - truths[:, 0, :]
    initial_squared = (
        grid.h * torch.sum(initial_errors**2, dim=1) + 1.0e-8
    )
    normalized_errors: list[object] = [
        grid.h * torch.sum(initial_errors**2, dim=1) / initial_squared
    ]
    dt = float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0])
    for step in range(OUTPUT_TIMES.size - 1):
        truth_left = truths[:, step, :]
        truth_right = truths[:, step + 1, :]
        truth_middle = 0.5 * (truth_left + truth_right)
        k1 = _observer_rhs_tensor(
            torch, estimates, truth_left, gain_matrix, matrix, laplacian
        )
        k2 = _observer_rhs_tensor(
            torch,
            estimates + 0.5 * dt * k1,
            truth_middle,
            gain_matrix,
            matrix,
            laplacian,
        )
        k3 = _observer_rhs_tensor(
            torch,
            estimates + 0.5 * dt * k2,
            truth_middle,
            gain_matrix,
            matrix,
            laplacian,
        )
        k4 = _observer_rhs_tensor(
            torch,
            estimates + dt * k3,
            truth_right,
            gain_matrix,
            matrix,
            laplacian,
        )
        estimates = estimates + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        error = estimates - truth_right
        normalized_errors.append(
            grid.h * torch.sum(error**2, dim=1) / initial_squared
        )
    stacked = torch.stack(normalized_errors, dim=1)
    return 0.5 * torch.mean(stacked) + 0.5 * torch.mean(stacked[:, -1])


def _ratio_summary(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("values must be a finite non-empty vector")
    return {
        "count": int(data.size),
        "rms": float(np.sqrt(np.mean(data**2))),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
    }


def _dynamics_audit(
    torch: object,
    gain: object,
    transform: object,
    samples: InstantSamples,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    device: str,
    batch_size: int = 256,
) -> dict[str, object]:
    tensors = _tensorize_samples(
        torch, samples, grid, device=device, dtype=torch.float64
    )
    matrix_tensor = torch.as_tensor(
        matrix, dtype=torch.float64, device=device
    )
    defect_parts: list[np.ndarray] = []
    rate_parts: list[np.ndarray] = []
    sample_count = int(tensors["states"].shape[0])
    for start in range(0, sample_count, batch_size):
        indices = torch.arange(
            start, min(start + batch_size, sample_count), device=device
        )
        components = _instantaneous_components(
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
        defect_parts.append(
            torch.sqrt(components["defect_ratio_squared"])
            .detach()
            .cpu()
            .numpy()
        )
        rate_parts.append(components["rates"].detach().cpu().numpy())
    defect = np.concatenate(defect_parts)
    rates = np.concatenate(rate_parts)
    requested = LAMBDA_RATIO * NU_VALUE * np.pi**2
    margins = rates - requested
    violations = np.maximum(-margins, 0.0) ** 2
    tail_count = max(1, int(math.ceil(0.1 * margins.size)))
    contraction_loss = float(
        np.mean(np.partition(violations, -tail_count)[-tail_count:])
    )
    return {
        "loss": {
            "dynamics_contraction": contraction_loss,
            "nonlinear_target_defect": float(np.mean(defect**2)),
        },
        "defect": _ratio_summary(defect),
        "contraction": {
            "count": int(rates.size),
            "rate_min": float(np.min(rates)),
            "rate_p05": float(np.quantile(rates, 0.05)),
            "rate_median": float(np.median(rates)),
            "requested_rate": requested,
            "requested_margin_min": float(np.min(margins)),
            "requested_margin_p05": float(np.quantile(margins, 0.05)),
            "requested_rate_fraction": float(np.mean(margins >= 0.0)),
        },
    }


def _structure_audit(
    torch: object,
    transform: object,
    samples: InstantSamples,
    grid: AllenCahnGrid,
    *,
    device: str,
    maximum_samples: int = 24,
) -> dict[str, object]:
    sample_count = samples.states.shape[0]
    indices = np.unique(
        np.linspace(
            0, sample_count - 1, min(maximum_samples, sample_count)
        ).astype(int)
    )
    states = torch.as_tensor(
        samples.states[indices], dtype=torch.float64, device=device
    )
    errors = torch.as_tensor(
        samples.estimates[indices] - samples.states[indices],
        dtype=torch.float64,
        device=device,
    )
    zero = transform(states, torch.zeros_like(errors))
    zero_fiber = float(torch.max(torch.abs(zero)).detach().cpu())
    singular_minimum = np.inf
    singular_maximum = 0.0
    for index in range(states.shape[0]):
        state = states[index : index + 1]
        error = errors[index]
        jacobian = torch.autograd.functional.jacobian(
            lambda value: transform(state, value[None, :])[0],
            error,
            create_graph=False,
        )
        singular = torch.linalg.svdvals(jacobian).detach().cpu().numpy()
        singular_minimum = min(singular_minimum, float(np.min(singular)))
        singular_maximum = max(singular_maximum, float(np.max(singular)))

    transformed = transform(states, errors)
    current, converged, iterations_used = (
        transform.inverse_fixed_point_diagnostics(
            states,
            transformed,
            max_iterations=50,
            tolerance=1.0e-8,
        )
    )
    inverse_relative = torch.sqrt(
        grid.h * torch.sum((current - errors) ** 2, dim=1)
        / (grid.h * torch.sum(errors**2, dim=1) + 1.0e-12)
    )
    inverse_maximum = float(torch.max(inverse_relative).detach().cpu())

    doubled = transform(states, 2.0 * errors)
    eta_e = float(
        (
            torch.mean(
                torch.sqrt(
                    grid.h
                    * torch.sum((doubled - 2.0 * transformed) ** 2, dim=1)
                )
            )
            / (
                torch.mean(
                    torch.sqrt(
                        grid.h * torch.sum(transformed**2, dim=1)
                    )
                )
                + 1.0e-12
            )
        )
        .detach()
        .cpu()
    )
    reversed_states = torch.flip(states, dims=(0,))
    conditioned = transform(reversed_states, errors)
    eta_u = float(
        (
            torch.mean(
                torch.sqrt(
                    grid.h
                    * torch.sum((transformed - conditioned) ** 2, dim=1)
                )
            )
            / (
                torch.mean(
                    torch.sqrt(
                        grid.h * torch.sum(transformed**2, dim=1)
                    )
                )
                + 1.0e-12
            )
        )
        .detach()
        .cpu()
    )
    normalized = transform.normalized_forward(states, errors)
    normalized_doubled = transform.normalized_forward(states, 2.0 * errors)
    eta_e_normalized = float(
        (
            torch.mean(
                torch.sqrt(
                    grid.h
                    * torch.sum(
                        (normalized_doubled - 2.0 * normalized) ** 2,
                        dim=1,
                    )
                )
            )
            / (
                torch.mean(
                    torch.sqrt(grid.h * torch.sum(normalized**2, dim=1))
                )
                + 1.0e-12
            )
        )
        .detach()
        .cpu()
    )
    normalized_conditioned = transform.normalized_forward(
        reversed_states, errors
    )
    eta_u_normalized = float(
        (
            torch.mean(
                torch.sqrt(
                    grid.h
                    * torch.sum(
                        (normalized - normalized_conditioned) ** 2,
                        dim=1,
                    )
                )
            )
            / (
                torch.mean(
                    torch.sqrt(grid.h * torch.sum(normalized**2, dim=1))
                )
                + 1.0e-12
            )
        )
        .detach()
        .cpu()
    )
    generator = torch.Generator(device="cpu").manual_seed(1441)
    directions = torch.randn(
        errors.shape,
        generator=generator,
        dtype=torch.float64,
    ).to(device)
    directions /= torch.linalg.vector_norm(
        directions, dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    step = 1.0e-3
    second = (
        transform(states, errors + step * directions)
        - 2.0 * transformed
        + transform(states, errors - step * directions)
    ) / step**2
    second_norms = torch.sqrt(grid.h * torch.sum(second**2, dim=1))
    spectral_norms = (
        transform.spectral_norms_tensor().detach().cpu().numpy()
    )
    lipschitz = float(np.prod(spectral_norms))
    finite = bool(
        np.all(np.isfinite(spectral_norms))
        and np.isfinite(singular_minimum)
        and np.isfinite(singular_maximum)
        and np.isfinite(inverse_maximum)
        and np.isfinite(eta_e)
        and np.isfinite(eta_u)
        and np.isfinite(eta_e_normalized)
        and np.isfinite(eta_u_normalized)
    )
    spectral_passed = bool(2.0 * lipschitz <= transform.rho + 1.0e-6)
    jacobian_passed = bool(
        singular_minimum >= transform.lower_jacobian_bound - 1.0e-5
        and singular_maximum <= transform.upper_jacobian_bound + 1.0e-5
    )
    inverse_passed = bool(
        bool(torch.all(converged))
        and inverse_maximum <= INVERSE_RELATIVE_TOLERANCE
    )
    nonlinear_passed = bool(
        eta_e_normalized >= NONLINEARITY_THRESHOLD
        and eta_u_normalized >= NONLINEARITY_THRESHOLD
    )
    return {
        "sample_count": int(indices.size),
        "finite": finite,
        "zero_fiber_max_abs": zero_fiber,
        "spectral_norms": spectral_norms.tolist(),
        "residual_lipschitz_bound": lipschitz,
        "twice_residual_lipschitz_bound": 2.0 * lipschitz,
        "rho": transform.rho,
        "spectral_bound_passed": spectral_passed,
        "jacobian_min_singular": singular_minimum,
        "jacobian_max_singular": singular_maximum,
        "jacobian_bounds": [
            transform.lower_jacobian_bound,
            transform.upper_jacobian_bound,
        ],
        "normalized_jacobian_bounds": [
            transform.normalized_lower_jacobian_bound,
            transform.normalized_upper_jacobian_bound,
        ],
        "base_transform_singular_bounds": [
            transform.base_min_singular,
            transform.base_max_singular,
        ],
        "jacobian_passed": jacobian_passed,
        "inverse_all_converged": bool(torch.all(converged)),
        "inverse_iterations_used": iterations_used,
        "inverse_relative_error_max": inverse_maximum,
        "inverse_passed": inverse_passed,
        "eta_e": eta_e,
        "eta_u": eta_u,
        "eta_e_normalized": eta_e_normalized,
        "eta_u_normalized": eta_u_normalized,
        "nonlinear_gate_coordinates": "normalized_residual_S",
        "nonlinear_dependence_passed": nonlinear_passed,
        "second_directional_difference_rms": float(
            torch.sqrt(torch.mean(second_norms**2)).detach().cpu()
        ),
        "passed": bool(
            finite
            and zero_fiber <= 1.0e-10
            and spectral_passed
            and jacobian_passed
            and inverse_passed
            and nonlinear_passed
        ),
    }


def _seed_gates(
    result: dict[str, object],
    fixed_lmi: dict[str, object],
) -> dict[str, object]:
    dynamics = result["validation"]["dynamics"]
    structure = result["validation"]["structure"]
    rollout = result["validation"]["rollout"]
    fixed_dynamics = fixed_lmi["dynamics"]
    fixed_rollout = fixed_lmi["rollout"]
    defect_rms_ratio = (
        float(dynamics["defect"]["rms"])
        / max(float(fixed_dynamics["defect"]["rms"]), 1.0e-12)
    )
    defect_p95_ratio = (
        float(dynamics["defect"]["p95"])
        / max(float(fixed_dynamics["defect"]["p95"]), 1.0e-12)
    )
    terminal_ratio = (
        float(rollout["terminal_error_mass_median"])
        / max(float(fixed_rollout["terminal_error_mass_median"]), 1.0e-12)
    )
    maximum_ratio = (
        float(rollout["terminal_error_mass_max"])
        / max(float(fixed_rollout["terminal_error_mass_max"]), 1.0e-12)
    )
    finite = bool(
        np.isfinite(defect_rms_ratio)
        and np.isfinite(defect_p95_ratio)
        and np.isfinite(terminal_ratio)
        and np.isfinite(maximum_ratio)
    )
    gates = {
        "finite": finite,
        "structure": bool(structure["passed"]),
        "requested_contraction": bool(
            dynamics["contraction"]["requested_margin_min"] >= 0.0
        ),
        "defect_rms_vs_fixed_lmi_25pct": bool(defect_rms_ratio <= 0.75),
        "defect_p95_vs_fixed_lmi_15pct": bool(defect_p95_ratio <= 0.85),
        "online_terminal_median_no_regression_1.05": bool(
            terminal_ratio <= 1.05
        ),
        "online_terminal_max_no_regression_1.10": bool(
            maximum_ratio <= 1.10
        ),
        "ratios": {
            "defect_rms": defect_rms_ratio,
            "defect_p95": defect_p95_ratio,
            "online_terminal_median": terminal_ratio,
            "online_terminal_max": maximum_ratio,
        },
    }
    gates["all_passed"] = bool(
        gates["finite"]
        and gates["structure"]
        and gates["requested_contraction"]
        and gates["defect_rms_vs_fixed_lmi_25pct"]
        and gates["defect_p95_vs_fixed_lmi_15pct"]
        and gates["online_terminal_median_no_regression_1.05"]
        and gates["online_terminal_max_no_regression_1.10"]
    )
    return gates


def _selection_key(result: dict[str, object]) -> tuple[float, ...]:
    gates = result["gates"]
    dynamics = result["validation"]["dynamics"]
    structure = result["validation"]["structure"]
    ratios = gates["ratios"]
    margin = float(dynamics["contraction"]["requested_margin_min"])
    return (
        float(gates["all_passed"]),
        float(margin >= 0.0),
        margin,
        float(structure["passed"]),
        -float(ratios["online_terminal_median"]),
        -float(ratios["defect_rms"]),
    )


def _train_seed(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    base_gain: np.ndarray,
    base_transform: np.ndarray,
    train_cases: list[ExperimentCase],
    validation_cases: list[ExperimentCase],
    train_truth: TruthRollouts,
    fixed_lmi_baseline: dict[str, object],
    *,
    seed: int,
    epochs: int,
    instant_batch_size: int,
    rollout_batch_size: int,
    refresh_interval: int,
    device: str,
    rho: float,
    hidden_width: int,
    hidden_layers: int,
    gain_trust_ratio: float,
    gain_learning_rate: float,
    transform_learning_rate: float,
    inverse_iterations: int,
    checkpoint_dir: Path,
    transform_base: str,
) -> tuple[object, object, dict[str, object]]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    gain = build_projected_constant_gain(
        torch, base_gain, trust_ratio=gain_trust_ratio
    ).to(device=device, dtype=torch.float32)
    if transform_base == "identity":
        transform = build_conditional_residual_transform(
            torch,
            grid.n,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            rho=rho,
        )
        experiment_kind = "r5-nonlinear-target-conditional-residual-joint"
    elif transform_base == "lmi":
        transform = build_preconditioned_conditional_residual_transform(
            torch,
            base_transform,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            rho=rho,
        )
        experiment_kind = (
            "r5-nonlinear-target-T0-preconditioned-residual-joint"
        )
    else:
        raise ValueError("transform_base must be 'identity' or 'lmi'")
    transform = transform.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        [
            {"params": gain.parameters(), "lr": gain_learning_rate},
            {"params": transform.parameters(), "lr": transform_learning_rate},
        ]
    )
    initial_samples, _ = _simulate_cases(
        grid, matrix, base_gain, train_cases
    )
    samples = _tensorize_samples(
        torch,
        initial_samples,
        grid,
        device=device,
        dtype=torch.float32,
    )
    truth_tensors = _tensorize_truth_rollouts(
        torch,
        train_truth,
        device=device,
        dtype=torch.float32,
    )
    matrix_tensor = torch.as_tensor(
        matrix, dtype=torch.float32, device=device
    )
    history: list[dict[str, float]] = []
    refresh_count = 0
    for epoch in range(epochs):
        if (
            epoch > 0
            and refresh_interval > 0
            and epoch % refresh_interval == 0
        ):
            current_gain = gain().detach().cpu().numpy()
            refreshed, _ = _simulate_cases(
                grid, matrix, current_gain, train_cases
            )
            samples = _tensorize_samples(
                torch,
                refreshed,
                grid,
                device=device,
                dtype=torch.float32,
            )
            refresh_count += 1
        gain.train()
        transform.train()
        sample_count = int(samples["states"].shape[0])
        permutation = torch.randperm(sample_count, device=device)
        totals = {
            "dynamics_contraction": 0.0,
            "nonlinear_target_defect": 0.0,
            "invertibility": 0.0,
            "online_error": 0.0,
            "total": 0.0,
        }
        batch_count = 0
        for start in range(0, sample_count, instant_batch_size):
            indices = permutation[start : start + instant_batch_size]
            case_indices = torch.randperm(
                len(train_cases), device=device
            )[: min(rollout_batch_size, len(train_cases))]
            optimizer.zero_grad(set_to_none=True)
            components = _instantaneous_components(
                torch,
                gain,
                transform,
                samples,
                matrix_tensor,
                indices,
                grid,
                create_graph=True,
                include_inverse=True,
                inverse_iterations=inverse_iterations,
            )
            online = _differentiable_online_loss(
                torch,
                gain,
                truth_tensors,
                case_indices,
                matrix_tensor,
                samples["laplacian"],
                grid,
            )
            total = (
                components["contraction"]
                + components["defect"]
                + components["invertibility"]
                + online
            )
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"non-finite loss for seed={seed}, epoch={epoch + 1}"
                )
            total.backward()
            parameters = list(gain.parameters()) + list(transform.parameters())
            if any(
                parameter.grad is not None
                and not torch.all(torch.isfinite(parameter.grad))
                for parameter in parameters
            ):
                raise RuntimeError(
                    f"non-finite gradient for seed={seed}, epoch={epoch + 1}"
                )
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            gain.project_()
            transform.project_spectral_()
            batch_values = {
                "dynamics_contraction": components["contraction"],
                "nonlinear_target_defect": components["defect"],
                "invertibility": components["invertibility"],
                "online_error": online,
                "total": total,
            }
            for name, value in batch_values.items():
                totals[name] += float(value.detach().cpu())
            batch_count += 1
        history.append(
            {name: value / batch_count for name, value in totals.items()}
        )
        if epoch % 10 == 0 or epoch == epochs - 1:
            current = history[-1]
            print(
                f"[seed={seed}] epoch={epoch + 1}/{epochs} "
                f"total={current['total']:.6g} "
                f"contraction={current['dynamics_contraction']:.6g} "
                f"defect={current['nonlinear_target_defect']:.6g} "
                f"online={current['online_error']:.6g}",
                flush=True,
            )

    gain = gain.to(dtype=torch.float64)
    transform = transform.to(dtype=torch.float64)
    gain.project_()
    transform.project_spectral_()
    current_gain = gain().detach().cpu().numpy()
    validation_samples, validation_rollout = _simulate_cases(
        grid, matrix, current_gain, validation_cases
    )
    gain.eval()
    transform.eval()
    dynamics = _dynamics_audit(
        torch,
        gain,
        transform,
        validation_samples,
        grid,
        matrix,
        device=device,
    )
    structure = _structure_audit(
        torch,
        transform,
        validation_samples,
        grid,
        device=device,
    )
    result: dict[str, object] = {
        "seed": seed,
        "refresh_count": refresh_count,
        "final_training": history[-1],
        "gain_relative_delta_norm": gain.relative_delta_norm(),
        "validation": {
            "dynamics": dynamics,
            "structure": structure,
            "rollout": validation_rollout,
        },
    }
    result["gates"] = _seed_gates(result, fixed_lmi_baseline)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"joint__grid-31__seed-{seed}.pt"
    torch.save(
        {
            "kind": experiment_kind,
            "grid_size": grid.n,
            "nu": NU_VALUE,
            "seed": seed,
            "gain_state_dict": gain.state_dict(),
            "transform_state_dict": transform.state_dict(),
            "base_gain": base_gain,
            "base_transform": base_transform,
            "transform_base": transform_base,
            "sensor_intervals": THREE_SENSOR_INTERVALS,
            "rho": rho,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "gain_trust_ratio": gain_trust_ratio,
        },
        checkpoint,
    )
    result["checkpoint"] = str(checkpoint)
    return gain, transform, result


def _baseline_audit(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    base_gain: np.ndarray,
    transform_matrix: np.ndarray,
    validation_cases: list[ExperimentCase],
    *,
    device: str,
) -> dict[str, object]:
    samples, rollout = _simulate_cases(
        grid, matrix, base_gain, validation_cases
    )
    gain = build_projected_constant_gain(
        torch, base_gain, trust_ratio=0.25
    ).to(device=device, dtype=torch.float64)
    transform = _build_fixed_linear_transform(
        torch, transform_matrix
    ).to(device=device, dtype=torch.float64)
    return {
        "dynamics": _dynamics_audit(
            torch,
            gain,
            transform,
            samples,
            grid,
            matrix,
            device=device,
        ),
        "rollout": rollout,
    }


def run(
    torch: object,
    *,
    seeds: list[int],
    epochs: int,
    instant_batch_size: int,
    rollout_batch_size: int,
    refresh_interval: int,
    rho: float,
    hidden_width: int,
    hidden_layers: int,
    gain_trust_ratio: float,
    gain_learning_rate: float,
    transform_learning_rate: float,
    inverse_iterations: int,
    train_base_case_limit: int,
    validation_base_case_limit: int,
    test_base_case_limit: int,
    stress_truths_per_split: int,
    device: str,
    checkpoint_dir: Path,
    transform_base: str = "identity",
    train_case_seeds: Sequence[int] = TRAIN_CASE_SEEDS,
    validation_case_seeds: Sequence[int] = VALIDATION_CASE_SEEDS,
    test_case_seeds: Sequence[int] = TEST_CASE_SEEDS,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one model seed is required")
    if transform_base not in {"identity", "lmi"}:
        raise ValueError("transform_base must be 'identity' or 'lmi'")
    grid = AllenCahnGrid(GRID_SIZE)
    matrix = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    base_gain, base_transform, base_diagnostics = _design_base(grid, matrix)
    train_cases = _case_split(
        "train",
        grid,
        matrix,
        seeds=train_case_seeds,
        base_case_limit=train_base_case_limit,
        stress_truths=stress_truths_per_split,
    )
    validation_cases = _case_split(
        "validation",
        grid,
        matrix,
        seeds=validation_case_seeds,
        base_case_limit=validation_base_case_limit,
        stress_truths=stress_truths_per_split,
    )
    test_cases = _case_split(
        "test",
        grid,
        matrix,
        seeds=test_case_seeds,
        base_case_limit=test_base_case_limit,
        stress_truths=stress_truths_per_split,
    )
    train_truth = _truth_rollouts(grid, train_cases)
    fixed_lmi = _baseline_audit(
        torch,
        grid,
        matrix,
        base_gain,
        base_transform,
        validation_cases,
        device=device,
    )
    identity = _baseline_audit(
        torch,
        grid,
        matrix,
        base_gain,
        np.eye(grid.n),
        validation_cases,
        device=device,
    )
    models: dict[int, tuple[object, object]] = {}
    seed_results: list[dict[str, object]] = []
    for seed in seeds:
        print(f"[joint seed={seed}]", flush=True)
        gain, transform, result = _train_seed(
            torch,
            grid,
            matrix,
            base_gain,
            base_transform,
            train_cases,
            validation_cases,
            train_truth,
            fixed_lmi,
            seed=seed,
            epochs=epochs,
            instant_batch_size=instant_batch_size,
            rollout_batch_size=rollout_batch_size,
            refresh_interval=refresh_interval,
            device=device,
            rho=rho,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            gain_trust_ratio=gain_trust_ratio,
            gain_learning_rate=gain_learning_rate,
            transform_learning_rate=transform_learning_rate,
            inverse_iterations=inverse_iterations,
            checkpoint_dir=checkpoint_dir,
            transform_base=transform_base,
        )
        models[seed] = (gain, transform)
        seed_results.append(result)
    selected = max(seed_results, key=_selection_key)
    selected_seed = int(selected["seed"])
    successful_seed_count = sum(
        bool(item["gates"]["all_passed"]) for item in seed_results
    )
    validation_gate_passed = bool(
        successful_seed_count >= 2 and selected["gates"]["all_passed"]
    )
    test = None
    test_evaluated = False
    if validation_gate_passed:
        gain, transform = models[selected_seed]
        current_gain = gain().detach().cpu().numpy()
        test_samples, test_rollout = _simulate_cases(
            grid, matrix, current_gain, test_cases
        )
        test = {
            "dynamics": _dynamics_audit(
                torch,
                gain,
                transform,
                test_samples,
                grid,
                matrix,
                device=device,
            ),
            "structure": _structure_audit(
                torch,
                transform,
                test_samples,
                grid,
                device=device,
            ),
            "rollout": test_rollout,
        }
        test_evaluated = True
    experiment_kind = (
        "r5-nonlinear-target-conditional-residual-joint"
        if transform_base == "identity"
        else "r5-nonlinear-target-T0-preconditioned-residual-joint"
    )
    transform_description = (
        "e + g_phi(u,e) - g_phi(u,0)"
        if transform_base == "identity"
        else "T0 [e + g_tilde_phi(u,e) - g_tilde_phi(u,0)]"
    )
    return {
        "kind": experiment_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
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
            "grid_size": GRID_SIZE,
            "sensor_intervals": THREE_SENSOR_INTERVALS.tolist(),
            "lambda_ratio": LAMBDA_RATIO,
            "target": (
                "A z + F(u + z) - F(u) - (1 + lambda) z"
            ),
            "transform": transform_description,
            "transform_base": transform_base,
            "loss_terms": [
                "dynamics_contraction",
                "nonlinear_target_defect",
                "invertibility",
                "online_error",
            ],
            "seeds": seeds,
            "epochs": epochs,
            "instant_batch_size": instant_batch_size,
            "rollout_batch_size": rollout_batch_size,
            "refresh_interval": refresh_interval,
            "rho": rho,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "gain_trust_ratio": gain_trust_ratio,
            "gain_learning_rate": gain_learning_rate,
            "transform_learning_rate": transform_learning_rate,
            "inverse_iterations": inverse_iterations,
            "train_case_seeds": list(train_case_seeds),
            "validation_case_seeds": list(validation_case_seeds),
            "test_case_seeds": list(test_case_seeds),
            "train_base_case_limit": train_base_case_limit,
            "validation_base_case_limit": validation_base_case_limit,
            "test_base_case_limit": test_base_case_limit,
            "stress_truths_per_split": stress_truths_per_split,
            "test_locked_until_two_seeds_pass": True,
        },
        "base_diagnostics": base_diagnostics,
        "case_counts": {
            "train": len(train_cases),
            "validation": len(validation_cases),
            "test_locked": len(test_cases),
        },
        "baselines": {
            "fixed-B0__fixed-T0": fixed_lmi,
            "fixed-B0__identity-T": identity,
        },
        "seed_results": seed_results,
        "selected_seed": selected_seed,
        "selected": selected,
        "successful_seed_count": successful_seed_count,
        "validation_gate_passed": validation_gate_passed,
        "test_evaluated": test_evaluated,
        "test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=NU_VALUE)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--sensor-count", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1101, 1102, 1103])
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
    if args.nu != NU_VALUE:
        raise SystemExit("this frozen experiment requires --nu 0.005")
    if args.grid_size != GRID_SIZE:
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
        warmup = torch.ones(
            (32, 32), device=args.device, requires_grad=True
        )
        (warmup @ warmup).sum().backward()
        torch.cuda.synchronize()
    result = run(
        torch,
        seeds=args.seeds,
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
