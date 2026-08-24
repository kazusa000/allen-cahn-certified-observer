"""Repair the final grid-63 low-mode contraction failure by min--max training."""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    buffered_contraction_cvar,
    build_low_modal_conditional_residual_transform,
    build_projected_constant_gain,
    project_physical_modal_adversaries_,
)
from r5_direct_fiber_multigrid_joint import (
    ALPHA,
    CONDITION_MODE_COUNT,
    GRID_SIZES,
    LOW_MODE_COUNT,
    NU_VALUE,
    THREE_SENSOR_INTERVALS,
    _base_design,
    _baseline_split,
    _collocation_samples,
    _evaluate_split,
    _fiber_components,
    _git_head,
    _online_loss,
    _positive_control,
    _selection_key,
    _structure_audit,
    _tensorize_truth,
    _torch_context,
    _trajectory_cases,
    _truth_rollouts,
    _validation_gates,
    dirichlet_sine_basis,
    local_average_matrix,
)


TRAIN_TRAJECTORY_SEED = 2301
RESAMPLE_SEED_BASE = 2401
FORMAL_VALIDATION_SEED = 1871
LOCKED_TEST_SEED = 1901


def _seed_for_epoch(model_seed: int, epoch: int, grid_size: int) -> int:
    """Return a deterministic, collision-free seed for online resampling."""

    return int(RESAMPLE_SEED_BASE + 100_000 * model_seed + 100 * epoch + grid_size)


def _resampled_training_samples(
    *, model_seed: int, epoch: int, count: int
) -> dict[int, dict[str, np.ndarray]]:
    samples: dict[int, dict[str, np.ndarray]] = {}
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        samples[n] = _collocation_samples(
            grid,
            observation,
            seed=_seed_for_epoch(model_seed, epoch, n),
            count=count,
        )
    return samples


def _modal_values_torch(
    coefficients: object, basis: object, grid_step: float
) -> object:
    return (coefficients @ basis.T) / float(np.sqrt(grid_step))


def _adversarial_low_mode_samples(
    torch: object,
    transform: object,
    gain: object,
    *,
    grid_size: int,
    model_seed: int,
    refresh_index: int,
    restart_count: int,
    keep_count: int,
    steps: int,
    step_size: float,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Search one frozen-grid low-mode domain for current worst cases."""

    if not 1 <= keep_count <= restart_count:
        raise ValueError("keep_count must lie between one and restart_count")
    if steps < 1 or step_size <= 0.0:
        raise ValueError("adversarial steps and step_size must be positive")
    if grid_size not in GRID_SIZES:
        raise ValueError(f"grid_size must be one of {GRID_SIZES}")
    n = int(grid_size)
    context = _torch_context(torch, n, device=device, dtype=torch.float32)
    grid = context["grid"]
    generator = torch.Generator(device=device)
    search_seed = int(
        3101 + 100_000 * model_seed + 1_000 * refresh_index + n
    )
    generator.manual_seed(search_seed)

    state_coefficients = torch.randn(
        (restart_count, CONDITION_MODE_COUNT),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    error_coefficients = torch.randn(
        (restart_count, LOW_MODE_COUNT),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    error_coefficients /= torch.linalg.vector_norm(
        error_coefficients, dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    log_radii = torch.empty(
        (restart_count, 1), device=device, dtype=torch.float32
    ).uniform_(math.log(0.02), math.log(0.8), generator=generator)
    error_coefficients *= torch.exp(log_radii)
    project_physical_modal_adversaries_(
        torch,
        state_coefficients,
        error_coefficients,
        context["condition_basis"],
        grid.h,
    )
    state_coefficients.requires_grad_(True)
    error_coefficients.requires_grad_(True)

    def current_components(*, create_graph: bool) -> dict[str, object]:
        states = _modal_values_torch(
            state_coefficients, context["condition_basis"], grid.h
        )
        errors = _modal_values_torch(
            error_coefficients, context["low_basis"], grid.h
        )
        return _fiber_components(
            torch,
            transform,
            gain,
            states,
            errors,
            context,
            create_graph=create_graph,
        )

    initial_rates = current_components(create_graph=False)["rates"].detach()
    for _ in range(steps):
        rates = current_components(create_graph=True)["rates"]
        state_gradient, error_gradient = torch.autograd.grad(
            torch.sum(rates),
            (state_coefficients, error_coefficients),
            only_inputs=True,
        )
        with torch.no_grad():
            state_scale = torch.linalg.vector_norm(
                state_gradient, dim=1, keepdim=True
            ).clamp_min(1.0e-8)
            error_scale = torch.linalg.vector_norm(
                error_gradient, dim=1, keepdim=True
            ).clamp_min(1.0e-8)
            state_coefficients.sub_(step_size * state_gradient / state_scale)
            error_coefficients.sub_(step_size * error_gradient / error_scale)
        project_physical_modal_adversaries_(
            torch,
            state_coefficients,
            error_coefficients,
            context["condition_basis"],
            grid.h,
        )

    final_rates = current_components(create_graph=False)["rates"].detach()
    indices = torch.topk(final_rates, keep_count, largest=False).indices
    selected_state_coefficients = state_coefficients.detach()[indices]
    selected_error_coefficients = error_coefficients.detach()[indices]
    selected_states = _modal_values_torch(
        selected_state_coefficients, context["condition_basis"], grid.h
    )
    selected_errors = _modal_values_torch(
        selected_error_coefficients, context["low_basis"], grid.h
    )
    worst_index = int(indices[0].detach().cpu())
    diagnostics = {
        "grid_size": n,
        "search_seed": search_seed,
        "restart_count": restart_count,
        "keep_count": keep_count,
        "steps": steps,
        "step_size": step_size,
        "initial_margin_min": float(torch.min(initial_rates).cpu()) - ALPHA,
        "final_margin_min": float(torch.min(final_rates).cpu()) - ALPHA,
        "selected_margin_max": float(torch.max(final_rates[indices]).cpu()) - ALPHA,
        "worst_state_coefficients": state_coefficients.detach()[worst_index]
        .cpu()
        .numpy()
        .tolist(),
        "worst_error_coefficients": error_coefficients.detach()[worst_index]
        .cpu()
        .numpy()
        .tolist(),
    }
    return {
        "states": selected_states.cpu().numpy().astype(float),
        "errors": selected_errors.cpu().numpy().astype(float),
    }, diagnostics


def _fixed_replay_samples(count: int) -> dict[int, dict[str, np.ndarray]]:
    """Recreate the original fixed training pool for anti-forgetting replay."""

    samples: dict[int, dict[str, np.ndarray]] = {}
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        samples[n] = _collocation_samples(
            grid, observation, seed=1701, count=count
        )
    return samples


def _hard_point_neighborhood(
    torch: object,
    *,
    count: int,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Build a projected neighborhood of the consumed grid-63 bad point."""

    if count < 1:
        raise ValueError("hard replay count must be positive")
    grid = AllenCahnGrid(63)
    observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    consumed = _collocation_samples(grid, observation, seed=1851, count=2048)
    condition_basis_numpy = dirichlet_sine_basis(grid, CONDITION_MODE_COUNT)
    low_basis_numpy = condition_basis_numpy[:, :LOW_MODE_COUNT]
    state_center = (
        np.sqrt(grid.h) * consumed["states"][471] @ condition_basis_numpy
    )
    error_center = np.sqrt(grid.h) * consumed["errors"][471] @ low_basis_numpy
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    state_coefficients = np.repeat(state_center[None, :], count, axis=0)
    error_coefficients = np.repeat(error_center[None, :], count, axis=0)
    if count > 1:
        state_coefficients[1:] += generator.normal(
            scale=0.02, size=(count - 1, CONDITION_MODE_COUNT)
        )
        error_coefficients[1:] += generator.normal(
            scale=0.01, size=(count - 1, LOW_MODE_COUNT)
        )
    state_tensor = torch.as_tensor(
        state_coefficients, dtype=torch.float32, device=device
    )
    error_tensor = torch.as_tensor(
        error_coefficients, dtype=torch.float32, device=device
    )
    condition_basis = torch.as_tensor(
        condition_basis_numpy, dtype=torch.float32, device=device
    )
    low_basis = condition_basis[:, :LOW_MODE_COUNT]
    project_physical_modal_adversaries_(
        torch, state_tensor, error_tensor, condition_basis, grid.h
    )
    return {
        "states": _modal_values_torch(
            state_tensor, condition_basis, grid.h
        ).cpu().numpy().astype(float),
        "errors": _modal_values_torch(
            error_tensor, low_basis, grid.h
        ).cpu().numpy().astype(float),
    }


def _transform_teacher_loss(
    torch: object,
    transform: object,
    teacher: object,
    states: object,
    errors: object,
    context: dict[str, object],
) -> object:
    """Anchor the transform function on replay/random points, not parameters."""

    grid = context["grid"]
    state_coefficients = float(np.sqrt(grid.h)) * (
        states @ context["condition_basis"]
    )
    error_coefficients = float(np.sqrt(grid.h)) * (
        errors @ context["low_basis"]
    )
    with torch.no_grad():
        target = teacher(state_coefficients, error_coefficients)
    actual = transform(state_coefficients, error_coefficients)
    numerator = torch.sum((actual - target) ** 2, dim=1)
    denominator = torch.sum(target**2, dim=1).clamp_min(1.0e-6)
    return torch.mean(numerator / denominator)


def _load_initialized_model(
    torch: object,
    *,
    checkpoint: Path,
    base_gain: np.ndarray,
    base_transform: np.ndarray,
    seed: int,
    rho: float,
    hidden_width: int,
    hidden_layers: int,
    gain_trust_ratio: float,
    error_scale: float,
    device: str,
) -> tuple[object, object, dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "seed": seed,
        "nu": NU_VALUE,
        "grid_sizes": GRID_SIZES,
        "rho": rho,
        "hidden_width": hidden_width,
        "hidden_layers": hidden_layers,
        "gain_trust_ratio": gain_trust_ratio,
        "error_scale": error_scale,
    }
    for name, value in expected.items():
        actual = payload[name]
        if name == "grid_sizes":
            actual = tuple(actual)
        if actual != value:
            raise RuntimeError(
                f"checkpoint {checkpoint} has {name}={actual!r}, expected {value!r}"
            )
    if not np.allclose(payload["base_gain"], base_gain, atol=1.0e-10, rtol=1.0e-10):
        raise RuntimeError("checkpoint base gain does not match corrected LMI design")
    if not np.allclose(
        payload["base_transform"], base_transform, atol=1.0e-10, rtol=1.0e-10
    ):
        raise RuntimeError("checkpoint base transform does not match corrected LMI design")

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
    gain.load_state_dict(payload["gain_state_dict"])
    transform.load_state_dict(payload["transform_state_dict"])
    gain.project_()
    transform.project_spectral_()
    return gain, transform, {
        "path": str(checkpoint),
        "kind": payload.get("kind"),
        "seed": int(payload["seed"]),
    }


def _train_seed(
    torch: object,
    base_gain: np.ndarray,
    base_transform: np.ndarray,
    validation_samples: dict[int, dict[str, np.ndarray]],
    validation_cases: dict[int, list[object]],
    validation_baseline: dict[str, object],
    train_truth: dict[str, object],
    *,
    initial_checkpoint: Path,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    rollout_batch_size: int,
    resample_count: int,
    replay_count: int,
    hard_replay_count: int,
    contraction_buffer: float,
    adversary_refresh_epochs: int,
    adversary_restarts: int,
    adversary_keep: int,
    adversary_steps: int,
    adversary_step_size: float,
    rho: float,
    hidden_width: int,
    hidden_layers: int,
    gain_trust_ratio: float,
    gain_learning_rate: float,
    transform_learning_rate: float,
    robust_multiplier_start: float,
    robust_multiplier_end: float,
    online_weight: float,
    transform_teacher_weight: float,
    gain_teacher_weight: float,
    train_condition_branch: bool,
    error_scale: float,
    device: str,
    checkpoint_dir: Path,
) -> tuple[object, object, dict[str, object]]:
    torch.manual_seed(seed + 10_000)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed + 10_000)
    gain, transform, initialization = _load_initialized_model(
        torch,
        checkpoint=initial_checkpoint,
        base_gain=base_gain,
        base_transform=base_transform,
        seed=seed,
        rho=rho,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        gain_trust_ratio=gain_trust_ratio,
        error_scale=error_scale,
        device=device,
    )
    teacher_gain, teacher_transform, _ = _load_initialized_model(
        torch,
        checkpoint=initial_checkpoint,
        base_gain=base_gain,
        base_transform=base_transform,
        seed=seed,
        rho=rho,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        gain_trust_ratio=gain_trust_ratio,
        error_scale=error_scale,
        device=device,
    )
    for parameter in list(teacher_gain.parameters()) + list(
        teacher_transform.parameters()
    ):
        parameter.requires_grad_(False)
    if not train_condition_branch:
        for parameter in transform.condition_layers.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    parameter
                    for parameter in gain.parameters()
                    if parameter.requires_grad
                ],
                "lr": gain_learning_rate,
            },
            {
                "params": [
                    parameter
                    for parameter in transform.parameters()
                    if parameter.requires_grad
                ],
                "lr": transform_learning_rate,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=0.0
    )
    contexts = {
        n: _torch_context(torch, n, device=device, dtype=torch.float32)
        for n in GRID_SIZES
    }
    replay_numpy = _fixed_replay_samples(replay_count)
    replay_tensors = {
        n: {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in values.items()
        }
        for n, values in replay_numpy.items()
    }
    hard_numpy = _hard_point_neighborhood(
        torch,
        count=hard_replay_count,
        seed=4101 + seed,
        device=device,
    )
    hard_tensors = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in hard_numpy.items()
    }
    history: list[dict[str, float]] = []
    adversary_history: list[dict[str, object]] = []
    adversarial_samples: dict[int, dict[str, np.ndarray]] = {}
    global_step = 0

    for epoch in range(epochs):
        sampled = _resampled_training_samples(
            model_seed=seed, epoch=epoch, count=resample_count
        )
        tensors = {
            n: {
                name: torch.as_tensor(value, dtype=torch.float32, device=device)
                for name, value in values.items()
            }
            for n, values in sampled.items()
        }
        if epoch % adversary_refresh_epochs == 0:
            for n in GRID_SIZES:
                values, diagnostics = _adversarial_low_mode_samples(
                    torch,
                    transform,
                    gain,
                    grid_size=n,
                    model_seed=seed,
                    refresh_index=epoch // adversary_refresh_epochs,
                    restart_count=adversary_restarts,
                    keep_count=adversary_keep,
                    steps=adversary_steps,
                    step_size=adversary_step_size,
                    device=device,
                )
                adversarial_samples[n] = values
                diagnostics["epoch"] = epoch + 1
                adversary_history.append(diagnostics)
        if set(adversarial_samples) != set(GRID_SIZES):
            raise RuntimeError("adversarial pools were not initialized")
        adversarial_tensors = {
            n: {
                name: torch.as_tensor(value, dtype=torch.float32, device=device)
                for name, value in values.items()
            }
            for n, values in adversarial_samples.items()
        }

        totals = {
            "contraction": 0.0,
            "online": 0.0,
            "gain": 0.0,
            "transform_teacher": 0.0,
            "gain_teacher": 0.0,
            "total": 0.0,
            "sampled_margin_min": 0.0,
        }
        multiplier = robust_multiplier_start + (
            robust_multiplier_end - robust_multiplier_start
        ) * epoch / max(epochs - 1, 1)
        for step in range(steps_per_epoch):
            n = GRID_SIZES[(global_step + step) % len(GRID_SIZES)]
            reserved_count = adversary_keep + (
                hard_replay_count if n == 63 else 0
            )
            random_count = batch_size - reserved_count
            dynamic_count = random_count // 2
            replay_batch_count = random_count - dynamic_count
            dynamic_indices = torch.randint(
                0,
                int(tensors[n]["states"].shape[0]),
                (dynamic_count,),
                device=device,
            )
            replay_indices = torch.randint(
                0,
                int(replay_tensors[n]["states"].shape[0]),
                (replay_batch_count,),
                device=device,
            )
            states = torch.cat(
                (
                    tensors[n]["states"][dynamic_indices],
                    replay_tensors[n]["states"][replay_indices],
                ),
                dim=0,
            )
            errors = torch.cat(
                (
                    tensors[n]["errors"][dynamic_indices],
                    replay_tensors[n]["errors"][replay_indices],
                ),
                dim=0,
            )
            anchor_states = states
            anchor_errors = errors
            states = torch.cat(
                (states, adversarial_tensors[n]["states"]), dim=0
            )
            errors = torch.cat(
                (errors, adversarial_tensors[n]["errors"]), dim=0
            )
            if n == 63:
                states = torch.cat((states, hard_tensors["states"]), dim=0)
                errors = torch.cat((errors, hard_tensors["errors"]), dim=0)
            trajectory_indices = torch.randperm(
                train_truth["states"].shape[0], device=device
            )[:rollout_batch_size]

            optimizer.zero_grad(set_to_none=True)
            components = _fiber_components(
                torch,
                transform,
                gain,
                states,
                errors,
                contexts[n],
                create_graph=True,
            )
            contraction = buffered_contraction_cvar(
                torch,
                components["margins"],
                buffer=contraction_buffer,
                tail_fraction=0.1,
            )
            online = _online_loss(
                torch, gain, train_truth, trajectory_indices, contexts[31]
            )
            transform_teacher = _transform_teacher_loss(
                torch,
                transform,
                teacher_transform,
                anchor_states,
                anchor_errors,
                contexts[n],
            )
            gain_teacher = (
                torch.linalg.vector_norm(gain() - teacher_gain()) ** 2
                / torch.linalg.vector_norm(teacher_gain()).clamp_min(1.0e-12) ** 2
            )
            gain_regularization = (
                torch.linalg.vector_norm(gain.delta) ** 2
                / torch.linalg.vector_norm(gain.base_gain).clamp_min(1.0e-12) ** 2
            )
            total = (
                multiplier * contraction
                + online_weight * online
                + transform_teacher_weight * transform_teacher
                + gain_teacher_weight * gain_teacher
                + 0.001 * gain_regularization
            )
            if not bool(torch.isfinite(total)):
                raise RuntimeError(
                    f"non-finite loss at seed={seed}, epoch={epoch + 1}"
                )
            total.backward()
            parameters = [
                parameter
                for parameter in list(gain.parameters())
                + list(transform.parameters())
                if parameter.requires_grad
            ]
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
                "contraction": contraction,
                "online": online,
                "gain": gain_regularization,
                "transform_teacher": transform_teacher,
                "gain_teacher": gain_teacher,
                "total": total,
                "sampled_margin_min": torch.min(components["margins"]),
            }
            for name, value in values.items():
                totals[name] += float(value.detach().cpu())
        global_step += steps_per_epoch
        scheduler.step()
        record = {
            name: value / steps_per_epoch for name, value in totals.items()
        }
        record["constraint_multiplier"] = multiplier
        record["gain_learning_rate"] = optimizer.param_groups[0]["lr"]
        record["transform_learning_rate"] = optimizer.param_groups[1]["lr"]
        history.append(record)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"[repair seed={seed}] epoch={epoch + 1}/{epochs} "
                f"total={record['total']:.6g} robust={record['contraction']:.6g} "
                f"sample-margin={record['sampled_margin_min']:.6g} "
                f"adv-margin={min(item['final_margin_min'] for item in adversary_history[-3:]):.6g}",
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
    structure = _structure_audit(torch, transform, device=device, seed=seed + 10_000)
    gates = _validation_gates(validation, structure)
    result: dict[str, object] = {
        "seed": seed,
        "initialization": initialization,
        "final_training": history[-1],
        "adversary_history": adversary_history,
        "gain_relative_delta_norm": gain.relative_delta_norm(),
        "structure": structure,
        "validation": validation,
        "gates": gates,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"direct-fiber-adversarial__seed-{seed}.pt"
    torch.save(
        {
            "kind": "r5-direct-fiber-adversarial-repair",
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
            "contraction_buffer": contraction_buffer,
            "replay_count": replay_count,
            "hard_replay_count": hard_replay_count,
            "robust_multiplier_start": robust_multiplier_start,
            "robust_multiplier_end": robust_multiplier_end,
            "online_weight": online_weight,
            "transform_teacher_weight": transform_teacher_weight,
            "gain_teacher_weight": gain_teacher_weight,
            "train_condition_branch": train_condition_branch,
            "initial_checkpoint": str(initial_checkpoint),
        },
        checkpoint,
    )
    result["checkpoint"] = str(checkpoint)
    return gain, transform, result


def run(
    torch: object,
    *,
    seeds: Sequence[int],
    initial_checkpoint_dir: Path,
    validation_seed: int,
    allow_locked_test: bool,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    rollout_batch_size: int,
    resample_count: int,
    replay_count: int,
    hard_replay_count: int,
    validation_count: int,
    test_count: int,
    contraction_buffer: float,
    adversary_refresh_epochs: int,
    adversary_restarts: int,
    adversary_keep: int,
    adversary_steps: int,
    adversary_step_size: float,
    rho: float,
    hidden_width: int,
    hidden_layers: int,
    gain_trust_ratio: float,
    gain_learning_rate: float,
    transform_learning_rate: float,
    robust_multiplier_start: float,
    robust_multiplier_end: float,
    online_weight: float,
    transform_teacher_weight: float,
    gain_teacher_weight: float,
    train_condition_branch: bool,
    error_scale: float,
    device: str,
    checkpoint_dir: Path,
) -> dict[str, object]:
    base_gain, base_transform, base_diagnostics = _base_design()
    validation_samples: dict[int, dict[str, np.ndarray]] = {}
    train_cases: dict[int, list[object]] = {}
    validation_cases: dict[int, list[object]] = {}
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        validation_samples[n] = _collocation_samples(
            grid, observation, seed=validation_seed, count=validation_count
        )
        train_cases[n] = _trajectory_cases(
            grid,
            observation,
            split="train",
            seed=TRAIN_TRAJECTORY_SEED,
        )
        validation_cases[n] = _trajectory_cases(
            grid,
            observation,
            split="validation",
            seed=validation_seed,
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
    for seed_value in seeds:
        seed = int(seed_value)
        initial_checkpoint = (
            initial_checkpoint_dir / f"direct-fiber__seed-{seed}.pt"
        )
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(f"missing initial checkpoint: {initial_checkpoint}")
        gain, transform, result = _train_seed(
            torch,
            base_gain,
            base_transform,
            validation_samples,
            validation_cases,
            validation_baseline,
            train_truth,
            initial_checkpoint=initial_checkpoint,
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            rollout_batch_size=rollout_batch_size,
            resample_count=resample_count,
            replay_count=replay_count,
            hard_replay_count=hard_replay_count,
            contraction_buffer=contraction_buffer,
            adversary_refresh_epochs=adversary_refresh_epochs,
            adversary_restarts=adversary_restarts,
            adversary_keep=adversary_keep,
            adversary_steps=adversary_steps,
            adversary_step_size=adversary_step_size,
            rho=rho,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            gain_trust_ratio=gain_trust_ratio,
            gain_learning_rate=gain_learning_rate,
            transform_learning_rate=transform_learning_rate,
            robust_multiplier_start=robust_multiplier_start,
            robust_multiplier_end=robust_multiplier_end,
            online_weight=online_weight,
            transform_teacher_weight=transform_teacher_weight,
            gain_teacher_weight=gain_teacher_weight,
            train_condition_branch=train_condition_branch,
            error_scale=error_scale,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        models[seed] = (gain, transform)
        seed_results.append(result)

    selected = max(seed_results, key=_selection_key)
    selected_seed = int(selected["seed"])
    successful_seed_count = sum(
        bool(item["gates"]["all_grids_passed"]) for item in seed_results
    )
    required_seed_count = 1 if len(seeds) == 1 else 2
    validation_gate_passed = bool(successful_seed_count >= required_seed_count)
    test = None
    test_evaluated = False
    if allow_locked_test and successful_seed_count >= 2:
        test_samples: dict[int, dict[str, np.ndarray]] = {}
        test_cases: dict[int, list[object]] = {}
        for n in GRID_SIZES:
            grid = AllenCahnGrid(n)
            observation = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
            test_samples[n] = _collocation_samples(
                grid, observation, seed=LOCKED_TEST_SEED, count=test_count
            )
            test_cases[n] = _trajectory_cases(
                grid,
                observation,
                split="test",
                seed=LOCKED_TEST_SEED,
            )
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
            "seed": LOCKED_TEST_SEED,
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
        "kind": "r5-direct-fiber-adversarial-repair",
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
            "transform": "V4 T0[b + g_phi(a,b) - g_phi(a,0)] + (I-Pi4)e",
            "gain": "B_h = V4_h Beta",
            "initial_checkpoint_dir": str(initial_checkpoint_dir),
            "seeds": list(seeds),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": batch_size,
            "rollout_batch_size": rollout_batch_size,
            "resample_count_per_epoch_per_grid": resample_count,
            "fixed_replay_count_per_grid": replay_count,
            "hard_replay_count_grid_63": hard_replay_count,
            "resample_seed_base": RESAMPLE_SEED_BASE,
            "contraction_buffer": contraction_buffer,
            "adversary_grids": list(GRID_SIZES),
            "adversary_refresh_epochs": adversary_refresh_epochs,
            "adversary_restarts": adversary_restarts,
            "adversary_keep": adversary_keep,
            "adversary_steps": adversary_steps,
            "adversary_step_size": adversary_step_size,
            "rho": rho,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "gain_trust_ratio": gain_trust_ratio,
            "gain_learning_rate": gain_learning_rate,
            "transform_learning_rate": transform_learning_rate,
            "robust_multiplier_start": robust_multiplier_start,
            "robust_multiplier_end": robust_multiplier_end,
            "online_weight": online_weight,
            "transform_teacher_weight": transform_teacher_weight,
            "gain_teacher_weight": gain_teacher_weight,
            "train_condition_branch": train_condition_branch,
            "learning_rate_schedule": "cosine_to_zero",
            "error_scale": error_scale,
            "validation_seed": validation_seed,
            "validation_count_per_grid": validation_count,
            "locked_test_seed": LOCKED_TEST_SEED,
            "test_count_per_grid": test_count,
            "allow_locked_test": allow_locked_test,
            "test_locked_until_two_seeds_pass": True,
            "calibration_bad_point_neighborhood_used_for_training": True,
        },
        "base_diagnostics": base_diagnostics,
        "positive_control_four_sensor": _positive_control(),
        "validation_baseline": validation_baseline,
        "seed_results": seed_results,
        "selected_seed": selected_seed,
        "selected": selected,
        "successful_seed_count": successful_seed_count,
        "required_seed_count": required_seed_count,
        "validation_gate_passed": validation_gate_passed,
        "test_evaluated": test_evaluated,
        "test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=NU_VALUE)
    parser.add_argument("--sensor-count", type=int, default=3)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=GRID_SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1301, 1302, 1303))
    parser.add_argument("--initial-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--validation-seed", type=int, default=FORMAL_VALIDATION_SEED)
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=2)
    parser.add_argument("--resample-count", type=int, default=2048)
    parser.add_argument("--replay-count", type=int, default=4096)
    parser.add_argument("--hard-replay-count", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=4096)
    parser.add_argument("--test-count", type=int, default=8192)
    parser.add_argument("--contraction-buffer", type=float, default=0.04)
    parser.add_argument("--adversary-refresh-epochs", type=int, default=2)
    parser.add_argument("--adversary-restarts", type=int, default=128)
    parser.add_argument("--adversary-keep", type=int, default=32)
    parser.add_argument("--adversary-steps", type=int, default=15)
    parser.add_argument("--adversary-step-size", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.35)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--gain-trust-ratio", type=float, default=0.25)
    parser.add_argument("--error-scale", type=float, default=1.0)
    parser.add_argument("--gain-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--transform-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--robust-multiplier-start", type=float, default=50.0)
    parser.add_argument("--robust-multiplier-end", type=float, default=500.0)
    parser.add_argument("--online-weight", type=float, default=0.2)
    parser.add_argument("--transform-teacher-weight", type=float, default=0.5)
    parser.add_argument("--gain-teacher-weight", type=float, default=5.0)
    parser.add_argument("--train-condition-branch", action="store_true")
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
    if not args.initial_checkpoint_dir.is_dir():
        raise SystemExit(
            f"missing initial checkpoint directory: {args.initial_checkpoint_dir}"
        )
    positive_counts = (
        args.epochs,
        args.steps_per_epoch,
        args.batch_size,
        args.rollout_batch_size,
        args.resample_count,
        args.replay_count,
        args.hard_replay_count,
        args.validation_count,
        args.test_count,
        args.adversary_refresh_epochs,
        args.adversary_restarts,
        args.adversary_keep,
        args.adversary_steps,
    )
    if min(positive_counts) < 1:
        raise SystemExit("counts must be positive")
    if min(
        args.resample_count,
        args.replay_count,
        args.validation_count,
        args.test_count,
    ) < 128:
        raise SystemExit("collocation counts must be at least 128")
    if args.adversary_keep + args.hard_replay_count >= args.batch_size:
        raise SystemExit(
            "adversarial and hard replay points must leave room for random replay"
        )
    if args.adversary_keep > args.adversary_restarts:
        raise SystemExit("--adversary-keep cannot exceed --adversary-restarts")
    if args.contraction_buffer <= 0.0 or args.adversary_step_size <= 0.0:
        raise SystemExit("buffer and adversarial step size must be positive")
    positive_weights = (
        args.robust_multiplier_start,
        args.robust_multiplier_end,
        args.online_weight,
        args.transform_teacher_weight,
        args.gain_teacher_weight,
    )
    if min(positive_weights) <= 0.0:
        raise SystemExit("training weights must be positive")
    if args.robust_multiplier_end < args.robust_multiplier_start:
        raise SystemExit("robust multiplier schedule must be nondecreasing")
    if not 0.0 < args.rho < 1.0:
        raise SystemExit("--rho must lie in (0, 1)")
    if not 0.0 < args.gain_trust_ratio < 1.0:
        raise SystemExit("--gain-trust-ratio must lie in (0, 1)")

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
        initial_checkpoint_dir=args.initial_checkpoint_dir,
        validation_seed=args.validation_seed,
        allow_locked_test=args.allow_locked_test,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        rollout_batch_size=args.rollout_batch_size,
        resample_count=args.resample_count,
        replay_count=args.replay_count,
        hard_replay_count=args.hard_replay_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        contraction_buffer=args.contraction_buffer,
        adversary_refresh_epochs=args.adversary_refresh_epochs,
        adversary_restarts=args.adversary_restarts,
        adversary_keep=args.adversary_keep,
        adversary_steps=args.adversary_steps,
        adversary_step_size=args.adversary_step_size,
        rho=args.rho,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        gain_trust_ratio=args.gain_trust_ratio,
        gain_learning_rate=args.gain_learning_rate,
        transform_learning_rate=args.transform_learning_rate,
        robust_multiplier_start=args.robust_multiplier_start,
        robust_multiplier_end=args.robust_multiplier_end,
        online_weight=args.online_weight,
        transform_teacher_weight=args.transform_teacher_weight,
        gain_teacher_weight=args.gain_teacher_weight,
        train_condition_branch=args.train_condition_branch,
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
