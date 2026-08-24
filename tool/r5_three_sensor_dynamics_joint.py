"""Run the frozen three-sensor B--T_phi dynamics-joint ablation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    local_average_matrix,
    noise_waveform,
)
from r5_oblique_joint_train import (
    NU_VALUES,
    _build_models,
    _case_split,
    _design_bases,
    _feature_tensor,
    _fixed_rollout_summary,
    _fixed_samples,
    _loss_components,
    _policy_samples,
    _rollout_summary,
    _target_operators,
    _tensorize,
)


THREE_SENSOR_INTERVALS = np.asarray(
    [
        [1.0 / 6.0, 7.0 / 30.0],
        [7.0 / 15.0, 8.0 / 15.0],
        [23.0 / 30.0, 5.0 / 6.0],
    ],
    dtype=float,
)
LOWER_JACOBIAN_BOUND = 0.25
UPPER_JACOBIAN_BOUND = 3.5
LOSS_WEIGHTS = {
    "stable": 1.0,
    "defect": 1.0,
    "contraction": 10.0,
    "bi": 1.0,
    "gain_reg": 0.1,
}


@dataclass(frozen=True)
class Variant:
    name: str
    train_gain: bool
    certificate_kind: str
    train_certificate: bool
    input_direction_weight: float


BASELINE_VARIANTS = (
    Variant("fixed-B__identity-T", False, "identity", False, 0.0),
    Variant("fixed-B__fixed-T", False, "lmi", False, 0.0),
)
TRAINABLE_VARIANTS = (
    Variant("train-B__identity-T", True, "identity", False, 0.0),
    Variant("fixed-B__train-T", False, "lmi", True, 0.0),
    Variant("joint-native", True, "lmi", True, 0.0),
    Variant("joint-ode-direction", True, "lmi", True, 1.0),
)
VARIANTS = {item.name: item for item in BASELINE_VARIANTS + TRAINABLE_VARIANTS}


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _identity_certificate(torch: object) -> object:
    class IdentityCertificate(torch.nn.Module):
        def forward(
            self,
            states: object,
            errors: object,
            nu_indices: object,
            nus: object,
        ) -> object:
            del states, nu_indices, nus
            return errors

    return IdentityCertificate()


def _build_variant_models(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    base_gains: np.ndarray,
    base_transforms: np.ndarray,
    variant: Variant,
    *,
    gain_trust_ratio: float,
    certificate_log_scale: float,
) -> tuple[object, object]:
    gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gains,
        base_transforms,
        gain_trust_ratio=gain_trust_ratio,
        certificate_log_scale=certificate_log_scale,
    )
    if variant.certificate_kind == "identity":
        certificate = _identity_certificate(torch)
    elif variant.certificate_kind != "lmi":
        raise ValueError(f"unsupported certificate kind: {variant.certificate_kind}")
    for parameter in gain.parameters():
        parameter.requires_grad_(variant.train_gain)
    for parameter in certificate.parameters():
        parameter.requires_grad_(variant.train_certificate)
    return gain, certificate


def _input_direction_components(
    torch: object,
    gain: object,
    certificate: object,
    samples: dict[str, object],
    matrix: object,
    indices: object,
    grid: AllenCahnGrid,
    *,
    create_graph: bool,
) -> tuple[object, object]:
    """Evaluate the nonlinear differential analogue of T B = B."""

    states = samples["states"][indices]
    estimates = samples["estimates"][indices]
    measurements = samples["measurements"][indices]
    nus = samples["nus"][indices]
    nu_indices = samples["nu_indices"][indices]
    features, _ = _feature_tensor(
        torch, estimates, measurements, nus, matrix, grid.h
    )
    gains = gain(features, nu_indices)
    errors = estimates - states

    def transform(state: object, error: object) -> object:
        return certificate(state, error, nu_indices, nus)

    differences: list[object] = []
    zero_state_direction = torch.zeros_like(states)
    for sensor_index in range(gains.shape[2]):
        direction = gains[:, :, sensor_index]
        _, transformed_direction = torch.autograd.functional.jvp(
            transform,
            (states, errors),
            (zero_state_direction, direction),
            create_graph=create_graph,
        )
        differences.append(transformed_direction - direction)
    difference = torch.stack(differences, dim=2)
    numerator = grid.h * torch.sum(difference**2, dim=(1, 2))
    denominator = grid.h * torch.sum(gains**2, dim=(1, 2)) + 1.0e-8
    ratio_squared = numerator / denominator
    return torch.mean(ratio_squared), torch.sqrt(ratio_squared)


def _combined_components(
    torch: object,
    gain: object,
    certificate: object,
    samples: dict[str, object],
    matrix: object,
    target_generators: object,
    target_maps: object,
    indices: object,
    grid: AllenCahnGrid,
    *,
    input_direction_weight: float,
    create_graph: bool,
    compute_input_direction: bool,
) -> dict[str, object]:
    components = _loss_components(
        torch,
        gain,
        certificate,
        samples,
        matrix,
        target_generators,
        target_maps,
        indices,
        grid,
        stable_weight=LOSS_WEIGHTS["stable"],
        defect_weight=LOSS_WEIGHTS["defect"],
        contraction_weight=LOSS_WEIGHTS["contraction"],
        bi_weight=LOSS_WEIGHTS["bi"],
        gain_reg_weight=LOSS_WEIGHTS["gain_reg"],
        lower_bound=LOWER_JACOBIAN_BOUND,
        upper_bound=UPPER_JACOBIAN_BOUND,
        create_graph=create_graph,
    )
    if compute_input_direction:
        input_loss, input_ratios = _input_direction_components(
            torch,
            gain,
            certificate,
            samples,
            matrix,
            indices,
            grid,
            create_graph=create_graph,
        )
    else:
        input_loss = torch.zeros_like(components["total"])
        input_ratios = torch.empty(
            0, dtype=components["total"].dtype, device=components["total"].device
        )
    components["input_direction"] = input_loss
    components["input_direction_ratios"] = input_ratios
    components["total"] = (
        components["total"] + input_direction_weight * input_loss
    )
    return components


def _ratio_summary(values: np.ndarray) -> dict[str, float | int]:
    ratios = np.asarray(values, dtype=float)
    if ratios.ndim != 1 or ratios.size == 0 or not np.all(np.isfinite(ratios)):
        raise ValueError("ratios must be a finite non-empty vector")
    return {
        "count": int(ratios.size),
        "rms": float(np.sqrt(np.mean(ratios**2))),
        "median": float(np.median(ratios)),
        "p95": float(np.quantile(ratios, 0.95)),
        "max": float(np.max(ratios)),
    }


def _component_summary(components: dict[str, object]) -> dict[str, object]:
    defect_ratios = components["defect_ratios"].detach().cpu().numpy()
    input_ratios = components["input_direction_ratios"].detach().cpu().numpy()
    rates = components["rates"].detach().cpu().numpy()
    requested = components["requested"].detach().cpu().numpy()
    margins = rates - requested
    return {
        "loss": {
            name: float(components[name].detach().cpu())
            for name in (
                "stable",
                "defect",
                "contraction",
                "bi",
                "gain_reg",
                "input_direction",
                "total",
            )
        },
        "defect": _ratio_summary(defect_ratios),
        "input_direction": _ratio_summary(input_ratios),
        "contraction": {
            "count": int(rates.size),
            "rate_min": float(np.min(rates)),
            "rate_p05": float(np.quantile(rates, 0.05)),
            "rate_median": float(np.median(rates)),
            "requested_margin_min": float(np.min(margins)),
            "requested_margin_p05": float(np.quantile(margins, 0.05)),
            "requested_rate_fraction": float(np.mean(margins >= 0.0)),
        },
    }


def _certificate_structure_audit(
    torch: object,
    certificate: object,
    samples: dict[str, object],
    *,
    maximum_samples: int = 24,
) -> dict[str, float | int | bool]:
    sample_count = int(samples["states"].shape[0])
    audit_indices = np.unique(
        np.linspace(0, sample_count - 1, min(maximum_samples, sample_count)).astype(int)
    )
    minimum = np.inf
    maximum = 0.0
    zero_fiber_max = 0.0
    for index in audit_indices:
        state = samples["states"][index : index + 1].detach()
        error = (samples["estimates"][index] - samples["states"][index]).detach()
        nu_index = samples["nu_indices"][index : index + 1]
        nu = samples["nus"][index : index + 1]
        zero = certificate(state, torch.zeros_like(error)[None, :], nu_index, nu)
        zero_fiber_max = max(
            zero_fiber_max,
            float(torch.max(torch.abs(zero)).detach().cpu()),
        )
        jacobian = torch.autograd.functional.jacobian(
            lambda value: certificate(state, value[None, :], nu_index, nu)[0],
            error,
            create_graph=False,
        )
        singular_values = torch.linalg.svdvals(jacobian).detach().cpu().numpy()
        minimum = min(minimum, float(np.min(singular_values)))
        maximum = max(maximum, float(np.max(singular_values)))
    finite = bool(np.isfinite(minimum) and np.isfinite(maximum))
    return {
        "sample_count": int(audit_indices.size),
        "zero_fiber_max_abs": zero_fiber_max,
        "jacobian_min_singular": float(minimum),
        "jacobian_max_singular": float(maximum),
        "finite": finite,
        "passed": bool(
            finite
            and zero_fiber_max <= 1.0e-6
            and minimum >= LOWER_JACOBIAN_BOUND - 1.0e-5
            and maximum <= UPPER_JACOBIAN_BOUND + 1.0e-5
        ),
    }


def _operator_tensors(
    torch: object, grid: AllenCahnGrid, matrix: np.ndarray, device: str
) -> tuple[object, object, object]:
    generators, maps = _target_operators(grid)
    return (
        torch.as_tensor(generators, dtype=torch.float32, device=device),
        torch.as_tensor(maps, dtype=torch.float32, device=device),
        torch.as_tensor(matrix, dtype=torch.float32, device=device),
    )


def _validation_audit(
    torch: object,
    gain: object,
    certificate: object,
    samples: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    device: str,
    *,
    input_direction_weight: float,
) -> tuple[dict[str, object], dict[str, object]]:
    tensors = _tensorize(torch, samples, grid, device)
    generators, maps, matrix_tensor = _operator_tensors(
        torch, grid, matrix, device
    )
    indices = torch.arange(tensors["states"].shape[0], device=device)
    gain.eval()
    certificate.eval()
    components = _combined_components(
        torch,
        gain,
        certificate,
        tensors,
        matrix_tensor,
        generators,
        maps,
        indices,
        grid,
        input_direction_weight=input_direction_weight,
        create_graph=False,
        compute_input_direction=True,
    )
    return _component_summary(components), _certificate_structure_audit(
        torch, certificate, tensors
    )


def _train_seed(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    base_gains: np.ndarray,
    base_transforms: np.ndarray,
    train_cases: list[object],
    validation_cases: list[object],
    variant: Variant,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    refresh_interval: int,
    device: str,
    gain_trust_ratio: float,
    certificate_log_scale: float,
    gain_learning_rate: float,
    certificate_learning_rate: float,
    checkpoint_dir: Path,
) -> tuple[object, object, dict[str, object]]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    gain, certificate = _build_variant_models(
        torch,
        grid,
        matrix,
        base_gains,
        base_transforms,
        variant,
        gain_trust_ratio=gain_trust_ratio,
        certificate_log_scale=certificate_log_scale,
    )
    gain.to(device)
    certificate.to(device)
    parameter_groups: list[dict[str, object]] = []
    if variant.train_gain:
        parameter_groups.append(
            {"params": list(gain.parameters()), "lr": gain_learning_rate}
        )
    if variant.train_certificate:
        parameter_groups.append(
            {
                "params": list(certificate.parameters()),
                "lr": certificate_learning_rate,
            }
        )
    if not parameter_groups:
        raise ValueError(f"variant is not trainable: {variant.name}")
    optimizer = torch.optim.Adam(parameter_groups)
    samples = _tensorize(
        torch,
        _fixed_samples(train_cases, grid, matrix, base_gains),
        grid,
        device,
    )
    generators, maps, matrix_tensor = _operator_tensors(
        torch, grid, matrix, device
    )
    history: list[dict[str, float]] = []
    refresh_count = 0
    for epoch in range(epochs):
        if (
            variant.train_gain
            and epoch > 0
            and refresh_interval > 0
            and epoch % refresh_interval == 0
        ):
            refreshed = _policy_samples(
                torch, gain, device, train_cases, grid, matrix
            )
            samples = _tensorize(torch, refreshed, grid, device)
            refresh_count += 1
        gain.train(variant.train_gain)
        certificate.train(variant.train_certificate)
        permutation = torch.randperm(samples["states"].shape[0], device=device)
        totals = {
            name: 0.0
            for name in (
                "stable",
                "defect",
                "contraction",
                "bi",
                "gain_reg",
                "input_direction",
                "total",
            )
        }
        batch_count = 0
        for start in range(0, permutation.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            components = _combined_components(
                torch,
                gain,
                certificate,
                samples,
                matrix_tensor,
                generators,
                maps,
                indices,
                grid,
                input_direction_weight=variant.input_direction_weight,
                create_graph=True,
                compute_input_direction=variant.input_direction_weight > 0.0,
            )
            if not torch.isfinite(components["total"]):
                raise RuntimeError(
                    f"non-finite loss for {variant.name}, seed={seed}, epoch={epoch}"
                )
            components["total"].backward()
            trainable_parameters = [
                parameter
                for group in parameter_groups
                for parameter in group["params"]
            ]
            if any(
                parameter.grad is not None
                and not torch.all(torch.isfinite(parameter.grad))
                for parameter in trainable_parameters
            ):
                raise RuntimeError(
                    f"non-finite gradient for {variant.name}, seed={seed}, epoch={epoch}"
                )
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            for name in totals:
                totals[name] += float(components[name].detach().cpu())
            batch_count += 1
        history.append({name: value / batch_count for name, value in totals.items()})
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"[variant={variant.name} seed={seed}] epoch={epoch + 1}/{epochs} "
                f"total={history[-1]['total']:.6g} "
                f"defect={history[-1]['defect']:.6g}",
                flush=True,
            )

    validation_samples = _policy_samples(
        torch, gain, device, validation_cases, grid, matrix
    )
    validation, structure = _validation_audit(
        torch,
        gain,
        certificate,
        validation_samples,
        grid,
        matrix,
        device,
        input_direction_weight=variant.input_direction_weight,
    )
    rollout = _rollout_summary(
        torch, gain, device, grid, matrix, validation_cases
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{variant.name}__grid-{grid.n}__seed-{seed}.pt"
    torch.save(
        {
            "variant": asdict(variant),
            "grid_size": grid.n,
            "seed": seed,
            "gain_state_dict": gain.state_dict(),
            "certificate_state_dict": certificate.state_dict(),
            "base_gains": base_gains,
            "base_transforms": base_transforms,
            "nu_values": NU_VALUES,
        },
        checkpoint,
    )
    return gain, certificate, {
        "variant": asdict(variant),
        "seed": seed,
        "refresh_count": refresh_count,
        "final_training": history[-1],
        "validation": validation,
        "structure": structure,
        "validation_rollout": rollout,
        "checkpoint": str(checkpoint),
    }


def _baseline_result(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    base_gains: np.ndarray,
    base_transforms: np.ndarray,
    validation_cases: list[object],
    variant: Variant,
    *,
    device: str,
    gain_trust_ratio: float,
    certificate_log_scale: float,
) -> dict[str, object]:
    gain, certificate = _build_variant_models(
        torch,
        grid,
        matrix,
        base_gains,
        base_transforms,
        variant,
        gain_trust_ratio=gain_trust_ratio,
        certificate_log_scale=certificate_log_scale,
    )
    gain.to(device)
    certificate.to(device)
    samples = _fixed_samples(validation_cases, grid, matrix, base_gains)
    validation, structure = _validation_audit(
        torch,
        gain,
        certificate,
        samples,
        grid,
        matrix,
        device,
        input_direction_weight=0.0,
    )
    return {
        "variant": asdict(variant),
        "validation": validation,
        "structure": structure,
        "validation_rollout": _fixed_rollout_summary(
            grid, matrix, validation_cases, base_gains
        ),
    }


def _seed_selection_key(result: dict[str, object]) -> tuple[float, ...]:
    contraction = result["validation"]["contraction"]
    defect = result["validation"]["defect"]
    terminal = result["validation_rollout"]["by_nu"]["0.005"][
        "terminal_error_mass_median"
    ]
    margin = float(contraction["requested_margin_min"])
    return (
        float(margin >= 0.0),
        margin,
        -float(defect["rms"]),
        -float(terminal),
    )


def _select_seed(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise ValueError("seed results cannot be empty")
    return max(results, key=_seed_selection_key)


def _online_gate(
    rollout: dict[str, object], baseline: dict[str, object], tolerance: float = 1.05
) -> tuple[bool, dict[str, float]]:
    ratios: dict[str, float] = {}
    for nu in NU_VALUES:
        key = f"{nu:.3f}"
        learned = float(
            rollout["by_nu"][key]["terminal_error_mass_median"]
        )
        fixed = float(
            baseline["by_nu"][key]["terminal_error_mass_median"]
        )
        ratios[key] = learned / max(fixed, 1.0e-12)
    return bool(max(ratios.values()) <= tolerance), ratios


def _basic_gates(
    result: dict[str, object], fixed_rollout: dict[str, object]
) -> dict[str, object]:
    online_passed, online_ratios = _online_gate(
        result["validation_rollout"], fixed_rollout
    )
    finite = all(
        np.isfinite(float(value))
        for value in (
            result["validation"]["defect"]["rms"],
            result["validation"]["defect"]["p95"],
            result["validation"]["contraction"]["requested_margin_min"],
        )
    )
    gates = {
        "finite": bool(finite),
        "structure": bool(result["structure"]["passed"]),
        "requested_contraction": bool(
            result["validation"]["contraction"]["requested_margin_min"] >= 0.0
        ),
        "online_no_regression_1.05": online_passed,
        "online_terminal_ratios": online_ratios,
    }
    gates["all_passed"] = bool(
        gates["finite"]
        and gates["structure"]
        and gates["requested_contraction"]
        and gates["online_no_regression_1.05"]
    )
    return gates


def _select_joint_variant(
    native: dict[str, object],
    ode_direction: dict[str, object],
    fixed_rollout: dict[str, object],
) -> tuple[str, dict[str, object]]:
    native_gates = _basic_gates(native, fixed_rollout)
    ode_gates = _basic_gates(ode_direction, fixed_rollout)
    native_rms = float(native["validation"]["defect"]["rms"])
    ode_rms = float(ode_direction["validation"]["defect"]["rms"])
    ode_reduction = 1.0 - ode_rms / max(native_rms, 1.0e-12)
    use_ode = bool(ode_gates["all_passed"] and ode_reduction >= 0.10)
    return (
        "joint-ode-direction" if use_ode else "joint-native",
        {
            "native_basic_gates": native_gates,
            "ode_direction_basic_gates": ode_gates,
            "ode_direction_defect_reduction_vs_native": ode_reduction,
            "ode_direction_selected": use_ode,
        },
    )


def _coarse_success_gates(
    chosen: dict[str, object],
    fixed_lmi: dict[str, object],
    b_only: dict[str, object],
    t_only: dict[str, object],
) -> dict[str, object]:
    basic = _basic_gates(chosen, fixed_lmi["validation_rollout"])
    chosen_rms = float(chosen["validation"]["defect"]["rms"])
    chosen_p95 = float(chosen["validation"]["defect"]["p95"])
    fixed_rms = float(fixed_lmi["validation"]["defect"]["rms"])
    fixed_p95 = float(fixed_lmi["validation"]["defect"]["p95"])
    b_only_rms = float(b_only["validation"]["defect"]["rms"])
    t_only_rms = float(t_only["validation"]["defect"]["rms"])
    value_gates = {
        "defect_rms_vs_fixed_T_25pct": bool(chosen_rms <= 0.75 * fixed_rms),
        "defect_p95_vs_fixed_T_15pct": bool(chosen_p95 <= 0.85 * fixed_p95),
        "T_value_vs_B_only_20pct": bool(chosen_rms <= 0.80 * b_only_rms),
        "joint_value_vs_T_only_10pct": bool(chosen_rms <= 0.90 * t_only_rms),
        "defect_rms_ratios": {
            "fixed-B__fixed-T": chosen_rms / max(fixed_rms, 1.0e-12),
            "train-B__identity-T": chosen_rms / max(b_only_rms, 1.0e-12),
            "fixed-B__train-T": chosen_rms / max(t_only_rms, 1.0e-12),
        },
        "defect_p95_ratio_fixed-B__fixed-T": chosen_p95
        / max(fixed_p95, 1.0e-12),
    }
    all_passed = bool(
        basic["all_passed"]
        and value_gates["defect_rms_vs_fixed_T_25pct"]
        and value_gates["defect_p95_vs_fixed_T_15pct"]
        and value_gates["T_value_vs_B_only_20pct"]
        and value_gates["joint_value_vs_T_only_10pct"]
    )
    return {"basic": basic, "value": value_gates, "all_passed": all_passed}


def _higher_grid_gates(
    chosen: dict[str, object], fixed_lmi: dict[str, object]
) -> dict[str, object]:
    basic = _basic_gates(chosen, fixed_lmi["validation_rollout"])
    chosen_rms = float(chosen["validation"]["defect"]["rms"])
    chosen_p95 = float(chosen["validation"]["defect"]["p95"])
    fixed_rms = float(fixed_lmi["validation"]["defect"]["rms"])
    fixed_p95 = float(fixed_lmi["validation"]["defect"]["p95"])
    progress = {
        "defect_rms_vs_fixed_T_25pct": bool(chosen_rms <= 0.75 * fixed_rms),
        "defect_p95_vs_fixed_T_15pct": bool(chosen_p95 <= 0.85 * fixed_p95),
        "defect_rms_ratio": chosen_rms / max(fixed_rms, 1.0e-12),
        "defect_p95_ratio": chosen_p95 / max(fixed_p95, 1.0e-12),
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


def _test_audit(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    gain: object,
    certificate: object,
    test_cases: list[object],
    variant: Variant,
    *,
    device: str,
) -> dict[str, object]:
    samples = _policy_samples(torch, gain, device, test_cases, grid, matrix)
    validation, structure = _validation_audit(
        torch,
        gain,
        certificate,
        samples,
        grid,
        matrix,
        device,
        input_direction_weight=variant.input_direction_weight,
    )
    noise = lambda time: noise_waveform(
        "common-sine", 0.01, matrix.shape[0], time
    )
    return {
        "dynamics": validation,
        "structure": structure,
        "rollout": _rollout_summary(
            torch, gain, device, grid, matrix, test_cases
        ),
        "noisy_rollout": _rollout_summary(
            torch, gain, device, grid, matrix, test_cases, noise=noise
        ),
    }


def _run_grid(
    torch: object,
    grid_size: int,
    variants: tuple[Variant, ...],
    *,
    baseline_variants: tuple[Variant, ...] = BASELINE_VARIANTS,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    refresh_interval: int,
    device: str,
    train_limit_per_nu: int,
    validation_limit_per_nu: int,
    test_limit_per_nu: int,
    stress_truths_per_nu: int,
    gain_trust_ratio: float,
    certificate_log_scale: float,
    gain_learning_rate: float,
    certificate_learning_rate: float,
    checkpoint_dir: Path,
) -> tuple[dict[str, object], dict[str, tuple[object, object]]]:
    grid = AllenCahnGrid(grid_size)
    matrix = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
    base_gains, base_transforms, base_diagnostics = _design_bases(grid, matrix)
    train_cases = _case_split(
        "train",
        grid,
        matrix,
        limit_per_nu=train_limit_per_nu,
        stress_truths_per_nu=stress_truths_per_nu,
    )
    validation_cases = _case_split(
        "validation",
        grid,
        matrix,
        limit_per_nu=validation_limit_per_nu,
        stress_truths_per_nu=stress_truths_per_nu,
    )
    test_cases = _case_split(
        "test",
        grid,
        matrix,
        limit_per_nu=test_limit_per_nu,
        stress_truths_per_nu=stress_truths_per_nu,
    )
    baselines = {
        variant.name: _baseline_result(
            torch,
            grid,
            matrix,
            base_gains,
            base_transforms,
            validation_cases,
            variant,
            device=device,
            gain_trust_ratio=gain_trust_ratio,
            certificate_log_scale=certificate_log_scale,
        )
        for variant in baseline_variants
    }
    training: dict[str, object] = {}
    selected_models: dict[str, tuple[object, object]] = {}
    for variant in variants:
        print(f"[grid={grid_size} variant={variant.name}]", flush=True)
        seed_results: list[dict[str, object]] = []
        seed_models: dict[int, tuple[object, object]] = {}
        for seed in seeds:
            gain, certificate, result = _train_seed(
                torch,
                grid,
                matrix,
                base_gains,
                base_transforms,
                train_cases,
                validation_cases,
                variant,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                refresh_interval=refresh_interval,
                device=device,
                gain_trust_ratio=gain_trust_ratio,
                certificate_log_scale=certificate_log_scale,
                gain_learning_rate=gain_learning_rate,
                certificate_learning_rate=certificate_learning_rate,
                checkpoint_dir=checkpoint_dir,
            )
            seed_results.append(result)
            seed_models[seed] = (gain, certificate)
        selected = _select_seed(seed_results)
        selected_seed = int(selected["seed"])
        selected_models[variant.name] = seed_models[selected_seed]
        training[variant.name] = {
            "variant": asdict(variant),
            "selected_seed": selected_seed,
            "seed_results": seed_results,
            "selected": selected,
        }
    return (
        {
            "grid_size": grid_size,
            "sensor_intervals": THREE_SENSOR_INTERVALS.tolist(),
            "base_diagnostics": base_diagnostics,
            "case_counts": {
                "train": len(train_cases),
                "validation": len(validation_cases),
                "test_locked": len(test_cases),
            },
            "baselines": baselines,
            "training": training,
            "test_evaluated": False,
            "test": None,
        },
        selected_models,
    )


def run(
    torch: object,
    *,
    grid_sizes: list[int],
    seeds: list[int],
    epochs: int,
    batch_size: int,
    refresh_interval: int,
    device: str,
    train_limit_per_nu: int,
    validation_limit_per_nu: int,
    test_limit_per_nu: int,
    stress_truths_per_nu: int,
    gain_trust_ratio: float,
    certificate_log_scale: float,
    gain_learning_rate: float,
    certificate_learning_rate: float,
    checkpoint_dir: Path,
) -> dict[str, object]:
    if not grid_sizes or grid_sizes[0] != 31:
        raise ValueError("the frozen screen must start at grid size 31")
    coarse, coarse_models = _run_grid(
        torch,
        31,
        TRAINABLE_VARIANTS,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        refresh_interval=refresh_interval,
        device=device,
        train_limit_per_nu=train_limit_per_nu,
        validation_limit_per_nu=validation_limit_per_nu,
        test_limit_per_nu=test_limit_per_nu,
        stress_truths_per_nu=stress_truths_per_nu,
        gain_trust_ratio=gain_trust_ratio,
        certificate_log_scale=certificate_log_scale,
        gain_learning_rate=gain_learning_rate,
        certificate_learning_rate=certificate_learning_rate,
        checkpoint_dir=checkpoint_dir,
    )
    native = coarse["training"]["joint-native"]["selected"]
    ode_direction = coarse["training"]["joint-ode-direction"]["selected"]
    fixed_lmi = coarse["baselines"]["fixed-B__fixed-T"]
    chosen_name, selection = _select_joint_variant(
        native, ode_direction, fixed_lmi["validation_rollout"]
    )
    chosen = coarse["training"][chosen_name]["selected"]
    gates = _coarse_success_gates(
        chosen,
        fixed_lmi,
        coarse["training"]["train-B__identity-T"]["selected"],
        coarse["training"]["fixed-B__train-T"]["selected"],
    )
    coarse["joint_selection"] = selection
    coarse["selected_variant"] = chosen_name
    coarse["gates"] = gates
    grid_results: dict[str, object] = {"31": coarse}

    if gates["all_passed"]:
        grid = AllenCahnGrid(31)
        matrix = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
        test_cases = _case_split(
            "test",
            grid,
            matrix,
            limit_per_nu=test_limit_per_nu,
            stress_truths_per_nu=stress_truths_per_nu,
        )
        gain, certificate = coarse_models[chosen_name]
        coarse["test"] = _test_audit(
            torch,
            grid,
            matrix,
            gain,
            certificate,
            test_cases,
            VARIANTS[chosen_name],
            device=device,
        )
        coarse["test_evaluated"] = True

        selected_variant = VARIANTS[chosen_name]
        for grid_size in grid_sizes[1:]:
            current, models = _run_grid(
                torch,
                grid_size,
                (selected_variant,),
                seeds=seeds,
                epochs=epochs,
                batch_size=batch_size,
                refresh_interval=refresh_interval,
                device=device,
                train_limit_per_nu=train_limit_per_nu,
                validation_limit_per_nu=validation_limit_per_nu,
                test_limit_per_nu=test_limit_per_nu,
                stress_truths_per_nu=stress_truths_per_nu,
                gain_trust_ratio=gain_trust_ratio,
                certificate_log_scale=certificate_log_scale,
                gain_learning_rate=gain_learning_rate,
                certificate_learning_rate=certificate_learning_rate,
                checkpoint_dir=checkpoint_dir,
            )
            selected = current["training"][chosen_name]["selected"]
            current_gates = _higher_grid_gates(
                selected, current["baselines"]["fixed-B__fixed-T"]
            )
            current["selected_variant"] = chosen_name
            current["gates"] = current_gates
            if current_gates["all_passed"]:
                grid = AllenCahnGrid(grid_size)
                matrix = local_average_matrix(grid, THREE_SENSOR_INTERVALS)
                test_cases = _case_split(
                    "test",
                    grid,
                    matrix,
                    limit_per_nu=test_limit_per_nu,
                    stress_truths_per_nu=stress_truths_per_nu,
                )
                gain, certificate = models[chosen_name]
                current["test"] = _test_audit(
                    torch,
                    grid,
                    matrix,
                    gain,
                    certificate,
                    test_cases,
                    selected_variant,
                    device=device,
                )
                current["test_evaluated"] = True
            grid_results[str(grid_size)] = current

    return {
        "kind": "r5-three-sensor-dynamics-joint",
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
            "grid_sizes": grid_sizes,
            "nu_values": NU_VALUES,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "refresh_interval": refresh_interval,
            "train_limit_per_nu": train_limit_per_nu,
            "validation_limit_per_nu": validation_limit_per_nu,
            "test_limit_per_nu": test_limit_per_nu,
            "stress_truths_per_nu": stress_truths_per_nu,
            "gain_trust_ratio": gain_trust_ratio,
            "certificate_log_scale": certificate_log_scale,
            "gain_learning_rate": gain_learning_rate,
            "certificate_learning_rate": certificate_learning_rate,
            "loss_weights": LOSS_WEIGHTS,
        },
        "coarse_gate_passed": bool(gates["all_passed"]),
        "selected_variant": chosen_name,
        "grid_results": grid_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[31, 63, 127])
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--refresh-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-limit-per-nu", type=int, default=16)
    parser.add_argument("--validation-limit-per-nu", type=int, default=8)
    parser.add_argument("--test-limit-per-nu", type=int, default=8)
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
        args.test_limit_per_nu,
        args.stress_truths_per_nu,
    ) < 1:
        raise SystemExit("epochs, batch size, and case limits must be positive")
    if not 0.0 < args.gain_trust_ratio < 1.0:
        raise SystemExit("--gain-trust-ratio must lie in (0, 1)")
    if args.certificate_log_scale < 0.0:
        raise SystemExit("--certificate-log-scale must be non-negative")
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    result = run(
        torch,
        grid_sizes=args.grid_sizes,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        refresh_interval=args.refresh_interval,
        device=args.device,
        train_limit_per_nu=args.train_limit_per_nu,
        validation_limit_per_nu=args.validation_limit_per_nu,
        test_limit_per_nu=args.test_limit_per_nu,
        stress_truths_per_nu=args.stress_truths_per_nu,
        gain_trust_ratio=args.gain_trust_ratio,
        certificate_log_scale=args.certificate_log_scale,
        gain_learning_rate=args.gain_learning_rate,
        certificate_learning_rate=args.certificate_learning_rate,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "coarse_gate_passed": result["coarse_gate_passed"],
                "selected_variant": result["selected_variant"],
                "evaluated_grids": list(result["grid_results"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
