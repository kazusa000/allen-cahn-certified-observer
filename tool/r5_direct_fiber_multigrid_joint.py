"""Jointly train a mesh-shared gain and direct error-fiber transform."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalOutputInjection,
    build_low_modal_conditional_residual_transform,
    build_projected_constant_gain,
    dirichlet_sine_basis,
    lmi_modal_injection,
    local_average_matrix,
    mesh_shared_fiber_transform,
    physical_modal_injection,
    simulate_causal_nudging,
    solve_allen_cahn,
    symmetric_allen_cahn_margin,
    unstable_modal_system,
)


NU_VALUE = 0.005
GRID_SIZES = (31, 63, 127)
LOW_MODE_COUNT = 4
CONDITION_MODE_COUNT = 8
COLLOCATION_MODE_COUNT = 12
ALPHA = 0.1 * NU_VALUE * np.pi**2
OUTPUT_TIMES = np.linspace(0.0, 1.0, 51)
MODEL_SEEDS = (1301, 1302, 1303)
THREE_SENSOR_INTERVALS = np.asarray(
    [
        [1.0 / 6.0, 7.0 / 30.0],
        [7.0 / 15.0, 8.0 / 15.0],
        [23.0 / 30.0, 5.0 / 6.0],
    ],
    dtype=float,
)
FOUR_SENSOR_INTERVALS = np.column_stack(
    (
        np.asarray([0.20, 0.40, 0.60, 0.80]) - 0.025,
        np.asarray([0.20, 0.40, 0.60, 0.80]) + 0.025,
    )
)
LMI_CONDITION_BOUNDS = (16.0, 32.0, 64.0, 128.0, 256.0)
CALIBRATION_SPLIT_SEED = 1801
SPLIT_SEEDS = {"train": 1701, "validation": 1851, "test": 1901}


@dataclass(frozen=True)
class TrajectoryCase:
    case_id: str
    truth_initial: np.ndarray
    estimate_initial: np.ndarray


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _balanced_metric_sqrt(metric: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(metric, dtype=float) + np.asarray(metric).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError("metric must be positive definite")
    scale = 1.0 / np.sqrt(float(np.min(eigenvalues) * np.max(eigenvalues)))
    return eigenvectors @ np.diag(np.sqrt(scale * eigenvalues)) @ eigenvectors.T


def _fixed_sine_basis_change(
    grid: AllenCahnGrid, modal: object
) -> np.ndarray:
    """Return ``R`` with ``V_eigh = V_sine R`` on the unstable modes."""

    fixed = dirichlet_sine_basis(grid, modal.dimension)
    change = fixed.T @ modal.modes
    diagonal = np.diag(np.diag(change))
    if not np.allclose(change, diagonal, atol=1.0e-10, rtol=0.0):
        raise RuntimeError("eigh modes are not aligned with fixed sine ordering")
    if not np.allclose(np.abs(np.diag(change)), 1.0, atol=1.0e-10, rtol=0.0):
        raise RuntimeError("eigh-to-sine basis change is not a sign matrix")
    return diagonal


def _fixed_coordinate_contraction_rate(
    grid: AllenCahnGrid,
    observation: np.ndarray,
    eigenvalues: np.ndarray,
    physical_gain: np.ndarray,
    transform: np.ndarray,
) -> float:
    """Audit the LMI rate after moving to fixed physical sine coordinates."""

    fixed = dirichlet_sine_basis(grid, eigenvalues.size) / np.sqrt(grid.h)
    closed_loop = np.diag(eigenvalues) - physical_gain @ (observation @ fixed)
    metric = transform.T @ transform
    derivative = closed_loop.T @ metric + metric @ closed_loop
    cholesky = np.linalg.cholesky(metric)
    inverse = np.linalg.inv(cholesky)
    normalized = inverse @ derivative @ inverse.T
    return float(-0.5 * np.max(np.linalg.eigvalsh(normalized)))


def _base_design() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    grid = AllenCahnGrid(31)
    observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    selected = None
    selected_bound = None
    for condition_bound in LMI_CONDITION_BOUNDS:
        try:
            selected = lmi_modal_injection(
                grid,
                NU_VALUE,
                observation,
                decay_rate=ALPHA,
                metric_condition_bound=condition_bound,
            )
        except RuntimeError:
            continue
        selected_bound = condition_bound
        break
    if selected is None or selected_bound is None:
        raise RuntimeError("no feasible three-sensor LMI base was found")
    modal = unstable_modal_system(grid, NU_VALUE, observation)
    if modal.dimension != LOW_MODE_COUNT:
        raise RuntimeError("the frozen nu=0.005 problem must have four unstable modes")
    change = _fixed_sine_basis_change(grid, modal)
    physical_gain = np.sqrt(grid.h) * change @ selected.modal_gain
    modal_transform = _balanced_metric_sqrt(selected.modal_metric)
    transform = change @ modal_transform @ change.T
    reproduced_rate = _fixed_coordinate_contraction_rate(
        grid,
        observation,
        modal.eigenvalues,
        physical_gain,
        transform,
    )
    rate_error = abs(reproduced_rate - selected.modal_contraction_rate)
    if rate_error > 1.0e-8:
        raise RuntimeError("fixed sine coordinates do not reproduce the LMI rate")
    return physical_gain, transform, {
        "design_grid": grid.n,
        "condition_bound": selected_bound,
        "solver_status": selected.solver_status,
        "unstable_dimension": modal.dimension,
        "observability_rank": modal.observability_rank,
        "observability_min_singular_value": modal.observability_min_singular_value,
        "closed_loop_spectral_abscissa": selected.closed_loop_spectral_abscissa,
        "modal_contraction_rate": selected.modal_contraction_rate,
        "modal_metric_condition": selected.modal_metric_condition,
        "eigh_to_fixed_sine_change": change.tolist(),
        "fixed_sine_reproduced_contraction_rate": reproduced_rate,
        "fixed_sine_rate_absolute_error": rate_error,
        "physical_modal_gain_norm": float(np.linalg.norm(physical_gain)),
        "transform_singular_values": np.linalg.svd(transform, compute_uv=False).tolist(),
    }


def _modal_values(grid: AllenCahnGrid, coefficients: np.ndarray) -> np.ndarray:
    array = np.asarray(coefficients, dtype=float)
    basis = dirichlet_sine_basis(grid, array.shape[-1])
    return (array @ basis.T) / np.sqrt(grid.h)


def _collocation_samples(
    grid: AllenCahnGrid,
    observation: np.ndarray,
    *,
    seed: int,
    count: int,
) -> dict[str, np.ndarray]:
    if count < 128:
        raise ValueError("collocation count must be at least 128")
    generator = np.random.Generator(np.random.PCG64DXSM(seed + 1009 * grid.n))

    state_coefficients = generator.normal(size=(count, CONDITION_MODE_COUNT))
    raw_states = _modal_values(grid, state_coefficients)
    target_amplitudes = generator.uniform(0.05, 1.25, size=count)
    scales = target_amplitudes / np.maximum(
        np.max(np.abs(raw_states), axis=1), 1.0e-12
    )
    state_coefficients *= scales[:, None]
    states = _modal_values(grid, state_coefficients)

    error_coefficients = generator.normal(size=(count, COLLOCATION_MODE_COUNT))
    groups = np.arange(count) % 3
    error_coefficients[groups == 0, LOW_MODE_COUNT:] = 0.0
    error_coefficients[groups == 1, :LOW_MODE_COUNT] = 0.0
    error_coefficients /= np.maximum(
        np.linalg.norm(error_coefficients, axis=1, keepdims=True), 1.0e-12
    )
    radii = np.exp(generator.uniform(np.log(0.02), np.log(0.8), size=count))
    error_coefficients *= radii[:, None]

    stress: list[np.ndarray] = []
    for mode in range(COLLOCATION_MODE_COUNT):
        for radius in (0.02, 0.20, 0.80):
            for sign in (-1.0, 1.0):
                direction = np.zeros(COLLOCATION_MODE_COUNT)
                direction[mode] = sign * radius
                stress.append(direction)
    low_basis = dirichlet_sine_basis(grid, LOW_MODE_COUNT) / np.sqrt(grid.h)
    observed_low = observation @ low_basis
    _, _, right = np.linalg.svd(observed_low, full_matrices=True)
    hard = right[-1]
    for radius in (0.02, 0.20, 0.80):
        for sign in (-1.0, 1.0):
            direction = np.zeros(COLLOCATION_MODE_COUNT)
            direction[:LOW_MODE_COUNT] = sign * radius * hard
            stress.append(direction)
    stress_array = np.asarray(stress)
    error_coefficients[: stress_array.shape[0]] = stress_array
    states[: stress_array.shape[0] // 2] = 0.0
    errors = _modal_values(grid, error_coefficients)
    return {"states": states, "errors": errors}


def _trajectory_cases(
    grid: AllenCahnGrid,
    observation: np.ndarray,
    *,
    split: str,
    base_count: int = 6,
) -> list[TrajectoryCase]:
    generator = np.random.Generator(
        np.random.PCG64DXSM(SPLIT_SEEDS[split] + 2017 * grid.n)
    )
    cases: list[TrajectoryCase] = []
    for index in range(base_count):
        truth_coefficients = generator.uniform(-0.5, 0.5, size=3)
        error_coefficients = generator.normal(size=4)
        error_coefficients /= np.linalg.norm(error_coefficients)
        error_coefficients *= generator.uniform(0.05, 0.25)
        truth = _modal_values(grid, truth_coefficients[None, :])[0]
        error = _modal_values(grid, error_coefficients[None, :])[0]
        cases.append(
            TrajectoryCase(
                f"{split}__random-{index}__n-{grid.n}", truth, truth + error
            )
        )

    truth = cases[0].truth_initial
    observed_low = observation @ (
        dirichlet_sine_basis(grid, LOW_MODE_COUNT) / np.sqrt(grid.h)
    )
    _, _, right = np.linalg.svd(observed_low, full_matrices=True)
    hard = right[-1]
    fourth = np.zeros(LOW_MODE_COUNT)
    fourth[-1] = 0.25
    for name, direction in (("fourth", fourth), ("min-observation", 0.25 * hard)):
        for sign in (-1.0, 1.0):
            error = _modal_values(grid, (sign * direction)[None, :])[0]
            cases.append(
                TrajectoryCase(
                    f"{split}__{name}__sign-{sign:+g}__n-{grid.n}",
                    truth.copy(),
                    truth + error,
                )
            )
    return cases


def _torch_context(
    torch: object,
    grid_size: int,
    *,
    device: str,
    dtype: object,
) -> dict[str, object]:
    grid = AllenCahnGrid(grid_size)
    observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    condition_basis = dirichlet_sine_basis(grid, CONDITION_MODE_COUNT)
    return {
        "grid": grid,
        "observation_numpy": observation,
        "observation": torch.as_tensor(observation, dtype=dtype, device=device),
        "laplacian": torch.as_tensor(grid.laplacian, dtype=dtype, device=device),
        "condition_basis": torch.as_tensor(
            condition_basis, dtype=dtype, device=device
        ),
        "low_basis": torch.as_tensor(
            condition_basis[:, :LOW_MODE_COUNT], dtype=dtype, device=device
        ),
    }


def _fixed_low_transform(torch: object, base_transform: np.ndarray) -> object:
    nn = torch.nn
    base = np.asarray(base_transform, dtype=float)

    class FixedLowTransform(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "base_transform", torch.as_tensor(base, dtype=torch.float32)
            )

        def forward(self, states: object, errors: object) -> object:
            del states
            return errors @ self.base_transform.T

    return FixedLowTransform()


def _fiber_components(
    torch: object,
    transform: object,
    gain: object,
    states: object,
    errors: object,
    context: dict[str, object],
    *,
    create_graph: bool,
) -> dict[str, object]:
    grid = context["grid"]
    physical_gain = gain()
    injection = physical_modal_injection(
        torch, physical_gain, context["low_basis"], grid.h
    )
    observation = context["observation"]
    laplacian = context["laplacian"]
    truth_rhs = NU_VALUE * (states @ laplacian.T) + states - states**3
    innovation = -(errors @ observation.T)
    correction = innovation @ injection.T
    error_rhs = (
        NU_VALUE * (errors @ laplacian.T)
        + errors
        - ((states + errors) ** 3 - states**3)
        + correction
    )

    def transform_map(state: object, error: object) -> object:
        return mesh_shared_fiber_transform(
            torch,
            transform,
            state,
            error,
            context["condition_basis"],
            context["low_basis"],
            grid.h,
        )

    transformed, transformed_rhs = torch.autograd.functional.jvp(
        transform_map,
        (states, errors),
        (truth_rhs, error_rhs),
        create_graph=create_graph,
    )
    transformed_squared = grid.h * torch.sum(transformed**2, dim=1)
    rates = -grid.h * torch.sum(
        transformed * transformed_rhs, dim=1
    ) / (transformed_squared + 1.0e-10)
    margins = rates - ALPHA
    violations = torch.relu(-margins) ** 2
    tail_count = max(1, int(math.ceil(0.1 * int(rates.shape[0]))))
    return {
        "rates": rates,
        "margins": margins,
        "violations": violations,
        "cvar": torch.mean(torch.topk(violations, tail_count).values),
        "transformed": transformed,
        "transformed_rhs": transformed_rhs,
    }


def _truth_rollouts(
    grid: AllenCahnGrid, cases: Sequence[TrajectoryCase]
) -> dict[str, np.ndarray]:
    truths = []
    initials = []
    for case in cases:
        solution = solve_allen_cahn(
            grid,
            NU_VALUE,
            case.truth_initial,
            output_times=OUTPUT_TIMES,
            dense_output=False,
        )
        truths.append(solution.states)
        initials.append(case.estimate_initial)
    return {"states": np.asarray(truths), "estimate_initials": np.asarray(initials)}


def _online_loss(
    torch: object,
    gain: object,
    truth_rollouts: dict[str, object],
    case_indices: object,
    context: dict[str, object],
) -> object:
    truths = truth_rollouts["states"][case_indices]
    estimates = truth_rollouts["estimate_initials"][case_indices]
    grid = context["grid"]
    observation = context["observation"]
    laplacian = context["laplacian"]
    injection = physical_modal_injection(
        torch, gain(), context["low_basis"], grid.h
    )
    initial_errors = estimates - truths[:, 0]
    initial_squared = grid.h * torch.sum(initial_errors**2, dim=1) + 1.0e-8
    normalized = [grid.h * torch.sum(initial_errors**2, dim=1) / initial_squared]
    dt = float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0])

    def rhs(estimate: object, truth: object) -> object:
        innovation = truth @ observation.T - estimate @ observation.T
        return (
            NU_VALUE * (estimate @ laplacian.T)
            + estimate
            - estimate**3
            + innovation @ injection.T
        )

    for step in range(OUTPUT_TIMES.size - 1):
        truth_left = truths[:, step]
        truth_right = truths[:, step + 1]
        truth_middle = 0.5 * (truth_left + truth_right)
        k1 = rhs(estimates, truth_left)
        k2 = rhs(estimates + 0.5 * dt * k1, truth_middle)
        k3 = rhs(estimates + 0.5 * dt * k2, truth_middle)
        k4 = rhs(estimates + dt * k3, truth_right)
        estimates = estimates + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        errors = estimates - truth_right
        normalized.append(grid.h * torch.sum(errors**2, dim=1) / initial_squared)
    stacked = torch.stack(normalized, dim=1)
    return 0.5 * torch.mean(stacked) + 0.5 * torch.mean(stacked[:, -1])


def _rate_summary(rates: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(rates, dtype=float)
    margins = values - ALPHA
    return {
        "count": int(values.size),
        "rate_min": float(np.min(values)),
        "rate_p01": float(np.quantile(values, 0.01)),
        "rate_p05": float(np.quantile(values, 0.05)),
        "rate_median": float(np.median(values)),
        "requested_rate": ALPHA,
        "requested_margin_min": float(np.min(margins)),
        "requested_margin_p01": float(np.quantile(margins, 0.01)),
        "requested_rate_fraction": float(np.mean(margins >= -1.0e-8)),
    }


def _rate_audit(
    torch: object,
    transform: object,
    gain: object,
    samples: dict[str, np.ndarray],
    grid_size: int,
    *,
    device: str,
    batch_size: int = 512,
) -> dict[str, float | int]:
    context = _torch_context(
        torch, grid_size, device=device, dtype=torch.float64
    )
    states = torch.as_tensor(samples["states"], dtype=torch.float64, device=device)
    errors = torch.as_tensor(samples["errors"], dtype=torch.float64, device=device)
    parts = []
    for start in range(0, states.shape[0], batch_size):
        components = _fiber_components(
            torch,
            transform,
            gain,
            states[start : start + batch_size],
            errors[start : start + batch_size],
            context,
            create_graph=False,
        )
        parts.append(components["rates"].detach().cpu().numpy())
    return _rate_summary(np.concatenate(parts))


def _rollout_samples(
    grid: AllenCahnGrid,
    observation: np.ndarray,
    injection: np.ndarray,
    cases: Sequence[TrajectoryCase],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    observer = CausalOutputInjection(grid, NU_VALUE, observation, injection)
    states = []
    errors = []
    terminal = []
    maximum = []
    records = []
    for case in cases:
        rollout = simulate_causal_nudging(
            observer,
            case.truth_initial,
            case.estimate_initial,
            output_times=OUTPUT_TIMES,
        )
        norms = rollout.error_mass_norm
        states.append(rollout.truth)
        errors.append(rollout.error)
        terminal.append(float(norms[-1]))
        maximum.append(float(np.max(norms)))
        records.append(
            {
                "case_id": case.case_id,
                "terminal_error_mass": float(norms[-1]),
                "maximum_error_mass": float(np.max(norms)),
                "solver_status": rollout.solver_status,
            }
        )
    return {
        "states": np.concatenate(states),
        "errors": np.concatenate(errors),
    }, {
        "case_count": len(cases),
        "terminal_error_mass_median": float(np.median(terminal)),
        "terminal_error_mass_max": float(np.max(terminal)),
        "maximum_error_mass_max": float(np.max(maximum)),
        "records": records,
    }


def _structure_audit(
    torch: object,
    transform: object,
    *,
    device: str,
    seed: int,
) -> dict[str, object]:
    generator = np.random.Generator(np.random.PCG64DXSM(seed + 4049))
    states = torch.as_tensor(
        generator.normal(size=(24, CONDITION_MODE_COUNT)),
        dtype=torch.float64,
        device=device,
    )
    errors = torch.as_tensor(
        0.25 * generator.normal(size=(24, LOW_MODE_COUNT)),
        dtype=torch.float64,
        device=device,
    )
    zero = transform(states, torch.zeros_like(errors))
    transformed = transform(states, errors)
    reconstructed, converged, iterations = transform.inverse_fixed_point_diagnostics(
        states, transformed, max_iterations=80, tolerance=1.0e-9
    )
    relative_inverse = torch.linalg.vector_norm(
        reconstructed - errors, dim=1
    ) / torch.linalg.vector_norm(errors, dim=1).clamp_min(1.0e-12)
    minimum = np.inf
    maximum = 0.0
    for index in range(states.shape[0]):
        jacobian = torch.autograd.functional.jacobian(
            lambda value: transform(states[index : index + 1], value[None])[0],
            errors[index],
        )
        singular = torch.linalg.svdvals(jacobian).detach().cpu().numpy()
        minimum = min(minimum, float(np.min(singular)))
        maximum = max(maximum, float(np.max(singular)))
    doubled = transform(states, 2.0 * errors)
    eta_e = float(
        (
            torch.mean(torch.linalg.vector_norm(doubled - 2.0 * transformed, dim=1))
            / torch.mean(torch.linalg.vector_norm(transformed, dim=1)).clamp_min(1.0e-12)
        )
        .detach()
        .cpu()
    )
    conditioned = transform(torch.flip(states, dims=(0,)), errors)
    eta_u = float(
        (
            torch.mean(torch.linalg.vector_norm(conditioned - transformed, dim=1))
            / torch.mean(torch.linalg.vector_norm(transformed, dim=1)).clamp_min(1.0e-12)
        )
        .detach()
        .cpu()
    )
    lipschitz = transform.residual_lipschitz_bound()
    finite = bool(
        np.isfinite(minimum)
        and np.isfinite(maximum)
        and np.isfinite(eta_e)
        and np.isfinite(eta_u)
    )
    spectral_passed = bool(lipschitz <= transform.rho + 1.0e-8)
    jacobian_passed = bool(
        minimum >= transform.lower_jacobian_bound - 1.0e-7
        and maximum <= transform.upper_jacobian_bound + 1.0e-7
    )
    inverse_max = float(torch.max(relative_inverse).detach().cpu())
    inverse_passed = bool(bool(torch.all(converged)) and inverse_max <= 1.0e-7)
    zero_max = float(torch.max(torch.abs(zero)).detach().cpu())
    return {
        "finite": finite,
        "zero_fiber_max_abs": zero_max,
        "rho": transform.rho,
        "residual_lipschitz_bound": lipschitz,
        "spectral_bound_passed": spectral_passed,
        "sampled_low_jacobian_min_singular": minimum,
        "sampled_low_jacobian_max_singular": maximum,
        "global_low_jacobian_bounds": [
            transform.lower_jacobian_bound,
            transform.upper_jacobian_bound,
        ],
        "global_full_jacobian_bounds": [
            min(1.0, transform.lower_jacobian_bound),
            max(1.0, transform.upper_jacobian_bound),
        ],
        "jacobian_passed": jacobian_passed,
        "inverse_all_converged": bool(torch.all(converged)),
        "inverse_iterations": iterations,
        "inverse_relative_error_max": inverse_max,
        "inverse_passed": inverse_passed,
        "eta_e_diagnostic_only": eta_e,
        "eta_u_diagnostic_only": eta_u,
        "passed": bool(
            finite
            and zero_max <= 1.0e-12
            and spectral_passed
            and jacobian_passed
            and inverse_passed
        ),
    }


def _tensorize_truth(
    torch: object,
    truth: dict[str, np.ndarray],
    *,
    device: str,
) -> dict[str, object]:
    return {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in truth.items()
    }


def _train_seed(
    torch: object,
    base_gain: np.ndarray,
    base_transform: np.ndarray,
    train_samples: dict[int, dict[str, np.ndarray]],
    validation_samples: dict[int, dict[str, np.ndarray]],
    validation_cases: dict[int, list[TrajectoryCase]],
    validation_baseline: dict[str, object],
    train_truth: dict[str, object],
    *,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    rollout_batch_size: int,
    rho: float,
    hidden_width: int,
    hidden_layers: int,
    gain_trust_ratio: float,
    gain_learning_rate: float,
    transform_learning_rate: float,
    error_scale: float,
    device: str,
    checkpoint_dir: Path,
) -> tuple[object, object, dict[str, object]]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    gain = build_projected_constant_gain(
        torch, base_gain, trust_ratio=gain_trust_ratio
    ).to(device=device, dtype=torch.float32)
    transform = build_low_modal_conditional_residual_transform(
        torch,
        base_transform,
        state_dimension=CONDITION_MODE_COUNT,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        rho=rho,
        error_scale=error_scale,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        [
            {"params": gain.parameters(), "lr": gain_learning_rate},
            {"params": transform.parameters(), "lr": transform_learning_rate},
        ]
    )
    contexts = {
        n: _torch_context(torch, n, device=device, dtype=torch.float32)
        for n in GRID_SIZES
    }
    tensors = {
        n: {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in samples.items()
        }
        for n, samples in train_samples.items()
    }
    history = []
    global_step = 0
    for epoch in range(epochs):
        totals = {"contraction": 0.0, "online": 0.0, "gain": 0.0, "total": 0.0}
        multiplier = 1.0 + 19.0 * epoch / max(epochs - 1, 1)
        for step in range(steps_per_epoch):
            n = GRID_SIZES[(global_step + step) % len(GRID_SIZES)]
            sample_count = int(tensors[n]["states"].shape[0])
            indices = torch.randint(0, sample_count, (batch_size,), device=device)
            trajectory_indices = torch.randperm(
                train_truth["states"].shape[0], device=device
            )[:rollout_batch_size]
            optimizer.zero_grad(set_to_none=True)
            components = _fiber_components(
                torch,
                transform,
                gain,
                tensors[n]["states"][indices],
                tensors[n]["errors"][indices],
                contexts[n],
                create_graph=True,
            )
            online = _online_loss(
                torch, gain, train_truth, trajectory_indices, contexts[31]
            )
            gain_regularization = (
                torch.linalg.vector_norm(gain.delta) ** 2
                / torch.linalg.vector_norm(gain.base_gain).clamp_min(1.0e-12) ** 2
            )
            total = multiplier * components["cvar"] + online + 0.01 * gain_regularization
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss at seed={seed}, epoch={epoch + 1}")
            total.backward()
            parameters = list(gain.parameters()) + list(transform.parameters())
            if any(
                parameter.grad is not None
                and not bool(torch.all(torch.isfinite(parameter.grad)))
                for parameter in parameters
            ):
                raise RuntimeError(
                    f"non-finite gradient at seed={seed}, epoch={epoch + 1}"
                )
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            gain.project_()
            transform.project_spectral_()
            values = {
                "contraction": components["cvar"],
                "online": online,
                "gain": gain_regularization,
                "total": total,
            }
            for name, value in values.items():
                totals[name] += float(value.detach().cpu())
        global_step += steps_per_epoch
        record = {
            name: value / steps_per_epoch for name, value in totals.items()
        }
        record["constraint_multiplier"] = multiplier
        history.append(record)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"[seed={seed}] epoch={epoch + 1}/{epochs} "
                f"total={record['total']:.6g} cvar={record['contraction']:.6g} "
                f"online={record['online']:.6g} mu={multiplier:.3g}",
                flush=True,
            )

    gain = gain.to(dtype=torch.float64)
    transform = transform.to(dtype=torch.float64)
    gain.project_()
    transform.project_spectral_()
    validation = _evaluate_split(
        torch,
        transform,
        gain,
        base_transform,
        validation_samples,
        validation_cases,
        validation_baseline,
        device=device,
    )
    structure = _structure_audit(torch, transform, device=device, seed=seed)
    gates = _validation_gates(validation, structure)
    result: dict[str, object] = {
        "seed": seed,
        "final_training": history[-1],
        "gain_relative_delta_norm": gain.relative_delta_norm(),
        "structure": structure,
        "validation": validation,
        "gates": gates,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"direct-fiber__seed-{seed}.pt"
    torch.save(
        {
            "kind": "r5-direct-fiber-multigrid-joint",
            "seed": seed,
            "nu": NU_VALUE,
            "grid_sizes": GRID_SIZES,
            "sensor_intervals": THREE_SENSOR_INTERVALS,
            "base_gain": base_gain,
            "base_transform": base_transform,
            "gain_state_dict": gain.state_dict(),
            "transform_state_dict": transform.state_dict(),
            "rho": rho,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "gain_trust_ratio": gain_trust_ratio,
            "error_scale": error_scale,
        },
        checkpoint,
    )
    result["checkpoint"] = str(checkpoint)
    return gain, transform, result


def _gain_injection_numpy(
    gain: object, grid: AllenCahnGrid
) -> np.ndarray:
    beta = gain().detach().cpu().numpy()
    basis = dirichlet_sine_basis(grid, LOW_MODE_COUNT)
    return basis @ beta / np.sqrt(grid.h)


def _evaluate_split(
    torch: object,
    transform: object,
    gain: object,
    base_transform: np.ndarray,
    collocation: dict[int, dict[str, np.ndarray]],
    cases: dict[int, list[TrajectoryCase]],
    baseline: dict[str, object],
    *,
    device: str,
) -> dict[str, object]:
    fixed_transform = _fixed_low_transform(torch, base_transform).to(
        device=device, dtype=torch.float64
    )
    grids: dict[str, object] = {}
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        injection = _gain_injection_numpy(gain, grid)
        trajectory_samples, rollout = _rollout_samples(
            grid, observation, injection, cases[n]
        )
        collocation_rate = _rate_audit(
            torch, transform, gain, collocation[n], n, device=device
        )
        trajectory_rate = _rate_audit(
            torch, transform, gain, trajectory_samples, n, device=device
        )
        fixed_collocation = _rate_audit(
            torch, fixed_transform, gain, collocation[n], n, device=device
        )
        fixed_trajectory = _rate_audit(
            torch, fixed_transform, gain, trajectory_samples, n, device=device
        )
        base_rollout = baseline["grids"][str(n)]["rollout"]
        rollout["terminal_median_ratio_to_B0"] = (
            rollout["terminal_error_mass_median"]
            / max(base_rollout["terminal_error_mass_median"], 1.0e-12)
        )
        rollout["terminal_max_ratio_to_B0"] = (
            rollout["terminal_error_mass_max"]
            / max(base_rollout["terminal_error_mass_max"], 1.0e-12)
        )
        grids[str(n)] = {
            "collocation": collocation_rate,
            "trajectory": trajectory_rate,
            "learned_B_fixed_T0": {
                "collocation": fixed_collocation,
                "trajectory": fixed_trajectory,
            },
            "rollout": rollout,
        }
    return {"grids": grids}


def _baseline_split(
    torch: object,
    base_gain: np.ndarray,
    base_transform: np.ndarray,
    collocation: dict[int, dict[str, np.ndarray]],
    cases: dict[int, list[TrajectoryCase]],
    *,
    device: str,
) -> dict[str, object]:
    gain = build_projected_constant_gain(torch, base_gain, trust_ratio=0.25).to(
        device=device, dtype=torch.float64
    )
    transform = _fixed_low_transform(torch, base_transform).to(
        device=device, dtype=torch.float64
    )
    grids: dict[str, object] = {}
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        injection = _gain_injection_numpy(gain, grid)
        trajectory_samples, rollout = _rollout_samples(
            grid, observation, injection, cases[n]
        )
        grids[str(n)] = {
            "collocation": _rate_audit(
                torch, transform, gain, collocation[n], n, device=device
            ),
            "trajectory": _rate_audit(
                torch, transform, gain, trajectory_samples, n, device=device
            ),
            "rollout": rollout,
        }
    return {"grids": grids}


def _validation_gates(
    validation: dict[str, object], structure: dict[str, object]
) -> dict[str, object]:
    per_grid = {}
    for n in GRID_SIZES:
        result = validation["grids"][str(n)]
        finite = bool(
            np.isfinite(result["collocation"]["requested_margin_min"])
            and np.isfinite(result["trajectory"]["requested_margin_min"])
            and np.isfinite(result["rollout"]["terminal_median_ratio_to_B0"])
            and np.isfinite(result["rollout"]["terminal_max_ratio_to_B0"])
        )
        gate = {
            "finite": finite,
            "collocation_contraction": bool(
                result["collocation"]["requested_margin_min"] >= -1.0e-8
            ),
            "trajectory_contraction": bool(
                result["trajectory"]["requested_margin_min"] >= -1.0e-8
            ),
            "online_terminal_median_no_regression": bool(
                result["rollout"]["terminal_median_ratio_to_B0"] <= 1.05
            ),
            "online_terminal_max_no_regression": bool(
                result["rollout"]["terminal_max_ratio_to_B0"] <= 1.10
            ),
        }
        gate["all_passed"] = bool(all(gate.values()))
        per_grid[str(n)] = gate
    return {
        "structure": bool(structure["passed"]),
        "per_grid": per_grid,
        "all_grids_passed": bool(
            structure["passed"]
            and all(item["all_passed"] for item in per_grid.values())
        ),
    }


def _selection_key(result: dict[str, object]) -> tuple[float, ...]:
    grids = result["validation"]["grids"]
    worst_collocation = min(
        grids[str(n)]["collocation"]["requested_margin_min"] for n in GRID_SIZES
    )
    worst_trajectory = min(
        grids[str(n)]["trajectory"]["requested_margin_min"] for n in GRID_SIZES
    )
    online = max(
        grids[str(n)]["rollout"]["terminal_median_ratio_to_B0"]
        for n in GRID_SIZES
    )
    return (
        float(result["gates"]["all_grids_passed"]),
        float(worst_collocation),
        float(worst_trajectory),
        -float(online),
    )


def _positive_control() -> dict[str, object]:
    grids = {}
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, FOUR_SENSOR_INTERVALS)
        margin = symmetric_allen_cahn_margin(
            grid, NU_VALUE, observation, gain=0.5
        )
        grids[str(n)] = {"global_semidiscrete_margin": margin, "passed": margin > 0.0}
    return {
        "sensor_count": 4,
        "sensor_intervals": FOUR_SENSOR_INTERVALS.tolist(),
        "mass_adjoint_gain": 0.5,
        "grids": grids,
        "all_grids_passed": bool(all(item["passed"] for item in grids.values())),
    }


def run(
    torch: object,
    *,
    seeds: Sequence[int],
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    rollout_batch_size: int,
    train_count: int,
    validation_count: int,
    test_count: int,
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
    base_gain, base_transform, base_diagnostics = _base_design()
    train_samples = {}
    validation_samples = {}
    train_cases = {}
    validation_cases = {}
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        train_samples[n] = _collocation_samples(
            grid, observation, seed=SPLIT_SEEDS["train"], count=train_count
        )
        validation_samples[n] = _collocation_samples(
            grid,
            observation,
            seed=SPLIT_SEEDS["validation"],
            count=validation_count,
        )
        train_cases[n] = _trajectory_cases(
            grid, observation, split="train"
        )
        validation_cases[n] = _trajectory_cases(
            grid, observation, split="validation"
        )
    train_truth_numpy = _truth_rollouts(AllenCahnGrid(31), train_cases[31])
    train_truth = _tensorize_truth(torch, train_truth_numpy, device=device)
    validation_baseline = _baseline_split(
        torch,
        base_gain,
        base_transform,
        validation_samples,
        validation_cases,
        device=device,
    )

    models: dict[int, tuple[object, object]] = {}
    seed_results = []
    for seed in seeds:
        gain, transform, result = _train_seed(
            torch,
            base_gain,
            base_transform,
            train_samples,
            validation_samples,
            validation_cases,
            validation_baseline,
            train_truth,
            seed=int(seed),
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
        models[int(seed)] = (gain, transform)
        seed_results.append(result)

    selected = max(seed_results, key=_selection_key)
    selected_seed = int(selected["seed"])
    successful_seed_count = sum(
        bool(item["gates"]["all_grids_passed"]) for item in seed_results
    )
    validation_gate_passed = bool(successful_seed_count >= 2)
    test = None
    test_evaluated = False
    if validation_gate_passed:
        test_samples = {}
        test_cases = {}
        for n in GRID_SIZES:
            grid = AllenCahnGrid(n)
            observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
            test_samples[n] = _collocation_samples(
                grid, observation, seed=SPLIT_SEEDS["test"], count=test_count
            )
            test_cases[n] = _trajectory_cases(grid, observation, split="test")
        test_baseline = _baseline_split(
            torch,
            base_gain,
            base_transform,
            test_samples,
            test_cases,
            device=device,
        )
        gain, transform = models[selected_seed]
        test = {
            "baseline": test_baseline,
            "selected": _evaluate_split(
                torch,
                transform,
                gain,
                base_transform,
                test_samples,
                test_cases,
                test_baseline,
                device=device,
            ),
        }
        test_evaluated = True

    return {
        "kind": "r5-direct-fiber-multigrid-joint",
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
            "grid_sizes": list(GRID_SIZES),
            "sensor_intervals": THREE_SENSOR_INTERVALS.tolist(),
            "alpha": ALPHA,
            "transform": (
                "V4 T0[b + g_phi(a,b) - g_phi(a,0)] + (I-Pi4)e"
            ),
            "gain": "B_h = V4_h Beta",
            "loss_terms": [
                "actual_dynamics_contraction_CVaR10",
                "online_error_n31",
                "gain_deviation_regularization",
            ],
            "target_defect_in_loss": False,
            "seeds": list(seeds),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": batch_size,
            "rollout_batch_size": rollout_batch_size,
            "train_count_per_grid": train_count,
            "validation_count_per_grid": validation_count,
            "test_count_per_grid": test_count,
            "rho": rho,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "gain_trust_ratio": gain_trust_ratio,
            "error_scale": error_scale,
            "gain_learning_rate": gain_learning_rate,
            "transform_learning_rate": transform_learning_rate,
            "split_seeds": SPLIT_SEEDS,
            "consumed_calibration_split_seed": CALIBRATION_SPLIT_SEED,
            "test_locked_until_two_seeds_pass": True,
        },
        "base_diagnostics": base_diagnostics,
        "positive_control_four_sensor": _positive_control(),
        "validation_baseline": validation_baseline,
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
    parser.add_argument("--sensor-count", type=int, default=3)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=GRID_SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps-per-epoch", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=2)
    parser.add_argument("--train-count", type=int, default=4096)
    parser.add_argument("--validation-count", type=int, default=2048)
    parser.add_argument("--test-count", type=int, default=4096)
    parser.add_argument("--rho", type=float, default=0.35)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--gain-trust-ratio", type=float, default=0.25)
    parser.add_argument("--error-scale", type=float, default=1.0)
    parser.add_argument("--gain-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--transform-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.nu != NU_VALUE:
        raise SystemExit("this frozen experiment requires --nu 0.005")
    if args.sensor_count != 3:
        raise SystemExit("this frozen experiment requires --sensor-count 3")
    if tuple(args.grid_sizes) != GRID_SIZES:
        raise SystemExit("this frozen experiment requires --grid-sizes 31 63 127")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.checkpoint_dir.exists():
        raise SystemExit(
            f"refusing to reuse checkpoint directory: {args.checkpoint_dir}"
        )
    counts = (
        args.epochs,
        args.steps_per_epoch,
        args.batch_size,
        args.rollout_batch_size,
        args.train_count,
        args.validation_count,
        args.test_count,
        args.hidden_width,
        args.hidden_layers,
    )
    if min(counts) < 1:
        raise SystemExit("counts and dimensions must be positive")
    if min(args.train_count, args.validation_count, args.test_count) < 128:
        raise SystemExit("collocation counts must be at least 128")
    if not 0.0 < args.rho < 1.0:
        raise SystemExit("--rho must lie in (0, 1)")
    if not 0.0 < args.gain_trust_ratio < 1.0:
        raise SystemExit("--gain-trust-ratio must lie in (0, 1)")
    if not np.isfinite(args.error_scale) or args.error_scale < 1.0:
        raise SystemExit("--error-scale must be finite and at least one")

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
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        rollout_batch_size=args.rollout_batch_size,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
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
