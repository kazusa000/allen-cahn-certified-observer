"""R5 T--K-style joint training against the stable target map."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from r5_e_joint_train import INTERVALS, OUTPUT_TIMES
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalNudging,
    NullspaceCertificate,
    allen_cahn_energy,
    allen_cahn_rhs,
    generate_pilot_cases,
    local_average_matrix,
    noise_waveform,
    simulate_causal_nudging,
)


@dataclass(frozen=True)
class JointSampleSet:
    states: np.ndarray
    estimates: np.ndarray
    measurements: np.ndarray
    next_states: np.ndarray
    nus: np.ndarray
    nu_indices: np.ndarray
    nu_values: tuple[float, ...]
    dt: float


ABLATION_SEEDS = (501, 502, 503, 504)


def _split_cases(split: str, grid_size: int) -> list[object]:
    return [
        case
        for case in generate_pilot_cases()
        if case.split == split and case.n == grid_size
    ]


def _collect_samples(
    cases: list[object],
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    base_gain: float,
) -> JointSampleSet:
    states: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    nus: list[float] = []
    dt = float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0])
    for case in cases:
        rollout = simulate_causal_nudging(
            CausalNudging(grid, case.nu, matrix, gain=base_gain),
            case.initial_truth(grid),
            case.initial_estimate(grid),
            output_times=OUTPUT_TIMES,
        )
        states.extend(rollout.truth[:-1])
        estimates.extend(rollout.estimate[:-1])
        measurements.extend(rollout.measurements[:-1])
        next_states.extend(rollout.truth[1:])
        nus.extend([case.nu] * (OUTPUT_TIMES.size - 1))
    nu_array = np.asarray(nus, dtype=float)
    nu_values = tuple(sorted({float(value) for value in nu_array}))
    nu_lookup = {value: index for index, value in enumerate(nu_values)}
    nu_indices = np.asarray([nu_lookup[float(value)] for value in nu_array], dtype=int)
    return JointSampleSet(
        states=np.asarray(states, dtype=float),
        estimates=np.asarray(estimates, dtype=float),
        measurements=np.asarray(measurements, dtype=float),
        next_states=np.asarray(next_states, dtype=float),
        nus=nu_array,
        nu_indices=nu_indices,
        nu_values=nu_values,
        dt=dt,
    )


def _feature_tensor(
    torch: object,
    estimates: object,
    measurements: object,
    nus: object,
    matrix: object,
    h: float,
) -> tuple[object, object]:
    innovations = measurements - estimates @ matrix.T
    scales = torch.sqrt(h * torch.sum(estimates**2, dim=1))
    viscosity = (nus - 0.01) / 0.01
    features = torch.cat(
        (
            estimates,
            measurements,
            innovations,
            viscosity[:, None],
            scales[:, None],
        ),
        dim=1,
    )
    return features, innovations


def _allen_cahn_rhs_tensor(
    torch: object, grid: AllenCahnGrid, states: object, nus: object, laplacian: object
) -> object:
    return nus[:, None] * (states @ laplacian.T) + states - states**3


def _target_maps(
    grid: AllenCahnGrid,
    nu_values: tuple[float, ...],
    lambda_ratio: float,
) -> np.ndarray:
    identity = np.eye(grid.n, dtype=float)
    maps = []
    for nu in nu_values:
        linear = nu * grid.laplacian + identity
        lam = lambda_ratio * nu * np.pi**2
        maps.append(expm(grid_step(grid) * (linear - lam * identity)))
    return np.asarray(maps, dtype=float)


def grid_step(grid: AllenCahnGrid) -> float:
    return float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0])


def _build_models(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    base_gain: float,
    gain_scale: float,
    certificate_scale: float,
) -> tuple[object, object]:
    nn = torch.nn
    n = grid.n
    q = matrix.shape[0]
    feature_dim = n + 2 * q + 2
    basis = NullspaceCertificate(matrix).null_basis

    class GainNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, n * q),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
            self.register_buffer(
                "base_gain",
                torch.as_tensor(base_gain * matrix.T / grid.h, dtype=torch.float32),
            )

        def forward(self, features: object) -> object:
            raw = self.network(features).reshape(-1, n, q)
            return self.base_gain[None, :, :] + gain_scale * torch.tanh(raw)

    class CertificateNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            null_dimension = basis.shape[1]
            self.network = nn.Sequential(
                nn.Linear(2 * n, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, null_dimension),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
            self.register_buffer(
                "null_basis", torch.as_tensor(basis, dtype=torch.float32)
            )

        def forward(self, states: object, errors: object) -> object:
            coordinates = errors @ self.null_basis
            gates = certificate_scale * torch.tanh(
                self.network(torch.cat((states, errors), dim=1))
            )
            return errors + (gates * coordinates) @ self.null_basis.T

    return GainNet(), CertificateNet()


def _stable_loss(
    torch: object,
    gain: object,
    certificate: object,
    samples: dict[str, object],
    target_maps: object,
    grid: AllenCahnGrid,
    matrix: object,
    indices: object,
) -> object:
    states = samples["states"][indices]
    estimates = samples["estimates"][indices]
    measurements = samples["measurements"][indices]
    next_states = samples["next_states"][indices]
    nus = samples["nus"][indices]
    nu_indices = samples["nu_indices"][indices]
    features, innovations = _feature_tensor(
        torch, estimates, measurements, nus, matrix, grid.h
    )
    gains = gain(features)
    correction = torch.bmm(gains, innovations[:, :, None]).squeeze(-1)
    laplacian = samples["laplacian"]
    rhs_estimate = _allen_cahn_rhs_tensor(torch, grid, estimates, nus, laplacian)
    next_estimates = estimates + samples["dt"] * (rhs_estimate + correction)
    errors = estimates - states
    next_errors = next_estimates - next_states
    transformed = certificate(states, errors)
    next_transformed = certificate(next_states, next_errors)
    stable_target = torch.bmm(target_maps[nu_indices], transformed[:, :, None]).squeeze(
        -1
    )
    residual = next_transformed - stable_target
    return grid.h * torch.mean(torch.sum(residual**2, dim=1))


def _train_one(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    train: JointSampleSet,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    base_gain: float,
    gain_scale: float,
    certificate_scale: float,
    lambda_ratio: float,
) -> tuple[object, object, dict[str, float]]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=base_gain,
        gain_scale=gain_scale,
        certificate_scale=certificate_scale,
    )
    gain.to(device)
    certificate.to(device)
    samples = {
        "states": torch.as_tensor(train.states, dtype=torch.float32, device=device),
        "estimates": torch.as_tensor(
            train.estimates, dtype=torch.float32, device=device
        ),
        "measurements": torch.as_tensor(
            train.measurements, dtype=torch.float32, device=device
        ),
        "next_states": torch.as_tensor(
            train.next_states, dtype=torch.float32, device=device
        ),
        "nus": torch.as_tensor(train.nus, dtype=torch.float32, device=device),
        "nu_indices": torch.as_tensor(
            train.nu_indices, dtype=torch.long, device=device
        ),
        "laplacian": torch.as_tensor(
            grid.laplacian, dtype=torch.float32, device=device
        ),
        "dt": train.dt,
    }
    targets = torch.as_tensor(
        _target_maps(grid, train.nu_values, lambda_ratio),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(
        list(gain.parameters()) + list(certificate.parameters()), lr=2.0e-3
    )
    history: list[float] = []
    sample_count = samples["states"].shape[0]
    for _ in range(epochs):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            indices = permutation[start : start + batch_size]
            loss = _stable_loss(
                torch,
                gain,
                certificate,
                samples,
                targets,
                grid,
                torch.as_tensor(matrix, dtype=torch.float32, device=device),
                indices,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        history.append(float(loss.detach().cpu().item()))
    gain.eval()
    certificate.eval()
    with torch.no_grad():
        indices = torch.arange(sample_count, device=device)
        final_loss = float(
            _stable_loss(
                torch,
                gain,
                certificate,
                samples,
                targets,
                grid,
                torch.as_tensor(matrix, dtype=torch.float32, device=device),
                indices,
            )
            .cpu()
            .item()
        )
    return (
        gain,
        certificate,
        {
            "stable_training_loss": final_loss,
            "stable_initial_last_batch_loss": history[0],
            "stable_final_last_batch_loss": history[-1],
        },
    )


def _validation_loss(
    torch: object,
    gain: object,
    certificate: object,
    validation: JointSampleSet,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    lambda_ratio: float,
    device: str,
) -> float:
    samples = {
        "states": torch.as_tensor(
            validation.states, dtype=torch.float32, device=device
        ),
        "estimates": torch.as_tensor(
            validation.estimates, dtype=torch.float32, device=device
        ),
        "measurements": torch.as_tensor(
            validation.measurements, dtype=torch.float32, device=device
        ),
        "next_states": torch.as_tensor(
            validation.next_states, dtype=torch.float32, device=device
        ),
        "nus": torch.as_tensor(validation.nus, dtype=torch.float32, device=device),
        "nu_indices": torch.as_tensor(
            validation.nu_indices, dtype=torch.long, device=device
        ),
        "laplacian": torch.as_tensor(
            grid.laplacian, dtype=torch.float32, device=device
        ),
        "dt": validation.dt,
    }
    targets = torch.as_tensor(
        _target_maps(grid, validation.nu_values, lambda_ratio),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        indices = torch.arange(samples["states"].shape[0], device=device)
        return float(
            _stable_loss(
                torch,
                gain,
                certificate,
                samples,
                targets,
                grid,
                torch.as_tensor(matrix, dtype=torch.float32, device=device),
                indices,
            )
            .cpu()
            .item()
        )


def _simulate(
    torch: object,
    gain: object,
    device: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    case: object,
    *,
    noise: object = None,
) -> dict[str, float | int]:
    n = grid.n

    def rhs(time: float, combined: np.ndarray) -> np.ndarray:
        truth, estimate = combined[:n], combined[n:]
        measurement = matrix @ truth
        if noise is not None:
            measurement = measurement + noise(float(time))
        features, innovation = _feature_numpy(
            grid, matrix, estimate, measurement, case.nu
        )
        with torch.no_grad():
            value = torch.as_tensor(
                features[None, :], dtype=torch.float32, device=device
            )
            gain_value = gain(value)[0].cpu().numpy()
        correction = gain_value @ innovation
        return np.concatenate(
            (
                allen_cahn_rhs(grid, case.nu, truth),
                allen_cahn_rhs(grid, case.nu, estimate) + correction,
            )
        )

    result = solve_ivp(
        rhs,
        (0.0, 1.0),
        np.concatenate((case.initial_truth(grid), case.initial_estimate(grid))),
        method="DOP853",
        t_eval=OUTPUT_TIMES,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    trajectories = result.y.T
    error = trajectories[:, n:] - trajectories[:, :n]
    error_mass = np.sqrt(grid.h * np.sum(error**2, axis=1))
    energies = np.asarray(
        [allen_cahn_energy(grid, case.nu, state) for state in trajectories[:, n:]]
    )
    return {
        "solver_status": int(result.status),
        "terminal_error_mass": float(error_mass[-1]),
        "peak_error_mass": float(np.max(error_mass)),
        "energy_defect": float(
            max(0.0, np.max(np.diff(energies, prepend=energies[0])))
        ),
    }


def _feature_numpy(
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    estimate: np.ndarray,
    measurement: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    innovation = measurement - matrix @ estimate
    features = np.concatenate(
        (
            estimate,
            measurement,
            innovation,
            np.asarray([(nu - 0.01) / 0.01]),
            np.asarray([np.sqrt(grid.h * np.dot(estimate, estimate))]),
        )
    )
    return features, innovation


def _median(records: list[dict[str, float | int]], key: str) -> float:
    return float(np.median([record[key] for record in records]))


def _audit(
    torch: object,
    certificate: object,
    matrix: np.ndarray,
    grid: AllenCahnGrid,
    device: str,
) -> dict[str, float]:
    rng = np.random.Generator(np.random.PCG64DXSM(20000 + grid.n))
    states = rng.normal(size=(3, grid.n)) * 0.1
    errors = rng.normal(size=(3, grid.n)) * 0.05
    state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    error_tensor = torch.as_tensor(errors, dtype=torch.float32, device=device)
    with torch.no_grad():
        transformed = certificate(state_tensor, error_tensor).cpu().numpy()
        zero = certificate(state_tensor, torch.zeros_like(error_tensor)).cpu().numpy()
    direction = np.linalg.norm((transformed - errors) @ matrix.T, axis=1)
    minimum_singular: list[float] = []
    maximum_singular: list[float] = []
    for index in range(states.shape[0]):
        state = state_tensor[index].detach()
        error = error_tensor[index].detach().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            lambda value, state=state: certificate(state[None, :], value[None, :])[0],
            error,
        )
        singular_values = np.linalg.svd(
            jacobian.detach().cpu().numpy(), compute_uv=False
        )
        minimum_singular.append(float(np.min(singular_values)))
        maximum_singular.append(float(np.max(singular_values)))
    return {
        "max_zero_fiber_residual": float(np.max(np.linalg.norm(zero, axis=1))),
        "max_direction_residual": float(np.max(direction)),
        "min_jacobian_singular_value": min(minimum_singular),
        "max_jacobian_singular_value": max(maximum_singular),
    }


def run(
    torch: object,
    grid_sizes: list[int],
    seeds: list[int],
    *,
    epochs: int,
    batch_size: int,
    eval_limit: int,
    noise_limit: int,
    device: str,
    lambda_ratio: float,
    base_gain: float,
    gain_scale: float,
    certificate_scale: float,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for grid_size in grid_sizes:
        grid = AllenCahnGrid(grid_size)
        matrix = local_average_matrix(grid, INTERVALS)
        train = _collect_samples(
            _split_cases("train", grid_size), grid, matrix, base_gain=base_gain
        )
        validation = _collect_samples(
            _split_cases("validation", grid_size), grid, matrix, base_gain=base_gain
        )
        test_cases = _split_cases("test", grid_size)
        models: dict[int, tuple[object, object]] = {}
        seed_results = []
        print(f"[grid={grid_size}] training {len(seeds)} seeds", flush=True)
        for seed in seeds:
            gain, certificate, losses = _train_one(
                torch,
                grid,
                matrix,
                train,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                base_gain=base_gain,
                gain_scale=gain_scale,
                certificate_scale=certificate_scale,
                lambda_ratio=lambda_ratio,
            )
            validation_loss = _validation_loss(
                torch,
                gain,
                certificate,
                validation,
                grid,
                matrix,
                lambda_ratio=lambda_ratio,
                device=device,
            )
            seed_results.append(
                {
                    "seed": seed,
                    **losses,
                    "stable_validation_loss": validation_loss,
                }
            )
            models[seed] = (gain, certificate)
        best = min(seed_results, key=lambda item: item["stable_validation_loss"])
        best_seed = int(best["seed"])
        gain, certificate = models[best_seed]
        replay = [
            _simulate(torch, gain, device, grid, matrix, case)
            for case in test_cases[:eval_limit]
        ]
        noisy = lambda time, q=matrix.shape[0]: noise_waveform(
            "common-sine", 0.01, q, time
        )
        noisy_replay = [
            _simulate(
                torch,
                gain,
                device,
                grid,
                matrix,
                case,
                noise=noisy,
            )
            for case in test_cases[:noise_limit]
        ]
        grid_result = {
            "grid_size": grid_size,
            "lambda_ratio": lambda_ratio,
            "base_gain": base_gain,
            "gain_scale": gain_scale,
            "certificate_scale": certificate_scale,
            "selected_seed": best_seed,
            "seed_results": seed_results,
            "test_case_count": len(replay),
            "test_median_terminal_error_mass": _median(replay, "terminal_error_mass"),
            "test_median_peak_error_mass": _median(replay, "peak_error_mass"),
            "noisy_case_count": len(noisy_replay),
            "noisy_median_terminal_error_mass": _median(
                noisy_replay, "terminal_error_mass"
            ),
            "certificate_audit": _audit(torch, certificate, matrix, grid, device),
        }
        results.append(grid_result)
        print(
            f"[grid={grid_size}] seed={best_seed} "
            f"stable={best['stable_validation_loss']:.6g} "
            f"test={grid_result['test_median_terminal_error_mass']:.6g} "
            f"noisy={grid_result['noisy_median_terminal_error_mass']:.6g}",
            flush=True,
        )
    return {
        "kind": "r5-tk-joint-training",
        "lambda_ratio": lambda_ratio,
        "base_gain": base_gain,
        "gain_scale": gain_scale,
        "certificate_scale": certificate_scale,
        "grid_sizes": grid_sizes,
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_limit": eval_limit,
        "noise_limit": noise_limit,
        "device": device,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[31, 63, 127])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ABLATION_SEEDS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=48)
    parser.add_argument("--noise-limit", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambda-ratio", type=float, default=0.5)
    parser.add_argument("--base-gain", type=float, default=0.02)
    parser.add_argument("--gain-scale", type=float, default=0.5)
    parser.add_argument("--certificate-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    if args.lambda_ratio <= 0.0:
        raise SystemExit("--lambda-ratio must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        torch,
        args.grid_sizes,
        args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_limit=args.eval_limit,
        noise_limit=args.noise_limit,
        device=args.device,
        lambda_ratio=args.lambda_ratio,
        base_gain=args.base_gain,
        gain_scale=args.gain_scale,
        certificate_scale=args.certificate_scale,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"grid_count": len(result["results"]), "device": args.device}))


if __name__ == "__main__":
    main()
