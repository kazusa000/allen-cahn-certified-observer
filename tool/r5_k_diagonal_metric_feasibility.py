"""Screen a diagonal metric that preserves Allen--Cahn cubic dissipation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from r5_direct_fiber_multigrid_joint import (
    ALPHA,
    NU_VALUE,
    THREE_SENSOR_INTERVALS,
)

from allen_cahn_certified_observer import AllenCahnGrid, local_average_matrix

CONDITION_BOUNDS = (4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 1.0e6)
POSTHOC_RATE_TOLERANCE = 1.0e-7
NORMALIZED_STRICTNESS = 1.0e-9


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _nullspace(matrix: np.ndarray) -> np.ndarray:
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=True)
    tolerance = (
        singular_values[0]
        * max(matrix.shape)
        * np.finfo(np.float64).eps
    )
    rank = int(np.sum(singular_values > tolerance))
    return right[rank:].T


def _solve(
    grid: AllenCahnGrid,
    observation: np.ndarray,
    *,
    condition_bound: float,
    solver: str,
) -> dict[str, object]:
    import cvxpy as cp

    system = NU_VALUE * grid.laplacian + np.eye(grid.n)
    system_scale = max(1.0, float(np.linalg.norm(system, ord=2)))
    normalized_system = system / system_scale
    normalized_alpha = ALPHA / system_scale

    weights = cp.Variable(grid.n)
    normalized_weighted_gain = cp.Variable(
        (grid.n, observation.shape[0])
    )
    metric = cp.diag(weights)
    derivative = (
        normalized_system.T @ metric
        + metric @ normalized_system
        - observation.T @ normalized_weighted_gain.T
        - normalized_weighted_gain @ observation
        + 2.0 * normalized_alpha * metric
    )
    smoothness = cp.norm(weights[1:] - weights[:-1], 2)
    problem = cp.Problem(
        cp.Minimize(
            cp.norm(normalized_weighted_gain, "fro")
            + 1.0e-3 * smoothness
        ),
        [
            weights >= 1.0,
            weights <= float(condition_bound),
            derivative << -NORMALIZED_STRICTNESS * np.eye(grid.n),
        ],
    )
    solver_options: dict[str, object] = {}
    if solver == "SCS":
        solver_options = {
            "eps": 1.0e-7,
            "max_iters": 200_000,
            "acceleration_lookback": 10,
        }
    try:
        problem.solve(solver=solver, **solver_options)
    except cp.error.SolverError as error:
        return {
            "condition_bound": float(condition_bound),
            "solver": solver,
            "solver_status": "solver_error",
            "solver_error": str(error),
            "feasible": False,
        }
    attempt: dict[str, object] = {
        "condition_bound": float(condition_bound),
        "solver": solver,
        "solver_status": str(problem.status),
        "objective": (
            None if problem.value is None else float(problem.value)
        ),
        "solver_iterations": (
            None
            if problem.solver_stats.num_iters is None
            else int(problem.solver_stats.num_iters)
        ),
        "solver_time_seconds": (
            None
            if problem.solver_stats.solve_time is None
            else float(problem.solver_stats.solve_time)
        ),
        "feasible": False,
    }
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return attempt

    weight_values = np.asarray(weights.value, dtype=float)
    normalized_gain_value = np.asarray(
        normalized_weighted_gain.value, dtype=float
    )
    metric_value = np.diag(weight_values)
    weighted_gain = system_scale * normalized_gain_value
    gain = weighted_gain / weight_values[:, None]
    closed_loop = system - gain @ observation
    metric_sqrt = np.sqrt(weight_values)
    inverse_metric_sqrt = 1.0 / metric_sqrt
    normalized_closed_derivative = (
        inverse_metric_sqrt[:, None]
        * (
            closed_loop.T @ metric_value
            + metric_value @ closed_loop
        )
        * inverse_metric_sqrt[None, :]
    )
    normalized_closed_derivative = 0.5 * (
        normalized_closed_derivative + normalized_closed_derivative.T
    )
    contraction_rate = float(
        -0.5 * np.max(np.linalg.eigvalsh(normalized_closed_derivative))
    )
    certificate_residual = float(
        np.max(
            np.linalg.eigvalsh(
                normalized_closed_derivative
                + 2.0 * ALPHA * np.eye(grid.n)
            )
        )
    )

    kernel = _nullspace(observation)
    plant_derivative = (
        system.T @ metric_value
        + metric_value @ system
        + 2.0 * ALPHA * metric_value
    )
    kernel_metric = kernel.T @ metric_value @ kernel
    kernel_cholesky = np.linalg.cholesky(kernel_metric)
    kernel_inverse = np.linalg.inv(kernel_cholesky)
    kernel_normalized = (
        kernel_inverse
        @ (kernel.T @ plant_derivative @ kernel)
        @ kernel_inverse.T
    )
    kernel_maximum = float(
        np.max(np.linalg.eigvalsh(0.5 * (kernel_normalized + kernel_normalized.T)))
    )

    generator = np.random.Generator(np.random.PCG64DXSM(2301 + grid.n))
    states = generator.normal(size=(4096, grid.n))
    errors = generator.normal(size=(4096, grid.n))
    cubic_increment = (states + errors) ** 3 - states**3
    cubic_power = np.sum(
        weight_values[None, :] * errors * cubic_increment, axis=1
    )
    minimum_cubic_power = float(np.min(cubic_power))
    finite = bool(
        np.all(np.isfinite(weight_values))
        and np.all(np.isfinite(gain))
        and np.isfinite(contraction_rate)
    )
    feasible = bool(
        finite
        and np.min(weight_values) > 0.0
        and contraction_rate >= ALPHA - POSTHOC_RATE_TOLERANCE
        and certificate_residual <= 2.0 * POSTHOC_RATE_TOLERANCE
        and minimum_cubic_power >= -1.0e-10
    )
    attempt.update(
        {
            "feasible": feasible,
            "finite": finite,
            "metric_weight_min": float(np.min(weight_values)),
            "metric_weight_max": float(np.max(weight_values)),
            "metric_condition": float(
                np.max(weight_values) / np.min(weight_values)
            ),
            "metric_total_variation": float(
                np.sum(np.abs(np.diff(weight_values)))
            ),
            "linear_contraction_rate": contraction_rate,
            "requested_rate": ALPHA,
            "certificate_residual_max_eigenvalue": certificate_residual,
            "kernel_necessary_condition_max_eigenvalue": kernel_maximum,
            "gain_fro_norm": float(np.linalg.norm(gain, ord="fro")),
            "weighted_gain_fro_norm": float(
                np.linalg.norm(weighted_gain, ord="fro")
            ),
            "closed_loop_spectral_abscissa": float(
                np.max(np.real(np.linalg.eigvals(closed_loop)))
            ),
            "minimum_sampled_cubic_power": minimum_cubic_power,
            "weights": weight_values.tolist(),
            "gain": gain.tolist(),
        }
    )
    return attempt


def run(
    *,
    grid_size: int,
    condition_bounds: tuple[float, ...],
    solver: str,
) -> dict[str, object]:
    grid = AllenCahnGrid(grid_size)
    observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    attempts = []
    selected = None
    for condition_bound in condition_bounds:
        print(
            f"solving condition_bound={condition_bound:g} with {solver}",
            flush=True,
        )
        attempt = _solve(
            grid,
            observation,
            condition_bound=condition_bound,
            solver=solver,
        )
        attempts.append(attempt)
        print(
            "status="
            f"{attempt['solver_status']} feasible={attempt['feasible']}",
            flush=True,
        )
        if attempt["feasible"]:
            selected = attempt
            break
    return {
        "kind": "r5-k-diagonal-metric-feasibility",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_head(),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "material_passport": {
            "nu": NU_VALUE,
            "sensor_count": 3,
            "sensor_intervals": THREE_SENSOR_INTERVALS.tolist(),
            "grid_size": grid_size,
            "alpha": ALPHA,
            "condition_bounds": list(condition_bounds),
            "solver": solver,
            "training": False,
            "sample_data_used_for_design": False,
            "locked_test_read": False,
        },
        "attempts": attempts,
        "decision": {
            "diagonal_metric_feasible": selected is not None,
            "first_feasible_condition_bound": (
                None
                if selected is None
                else selected["condition_bound"]
            ),
            "next_route": (
                "expand_to_multigrid"
                if selected is not None
                else "reject_diagonal_metric_and_test_structured_block_metric"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument(
        "--condition-bounds",
        type=float,
        nargs="+",
        default=CONDITION_BOUNDS,
    )
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.grid_size != 31:
        raise SystemExit("R5-K stage one is frozen to grid 31")
    if tuple(args.condition_bounds) != CONDITION_BOUNDS:
        raise SystemExit("R5-K stage one requires the frozen condition bounds")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    result = run(
        grid_size=args.grid_size,
        condition_bounds=tuple(args.condition_bounds),
        solver=args.solver,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"]), flush=True)


if __name__ == "__main__":
    main()
