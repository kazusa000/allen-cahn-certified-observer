import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_e_joint_train import INTERVALS
from r5_tk_joint_train import (
    JointSampleSet,
    _build_models,
    _concatenate_sample_sets,
    _ratio_summary,
)

from allen_cahn_certified_observer import AllenCahnGrid, local_average_matrix


def _sample_set(nu: float, time: float) -> JointSampleSet:
    return JointSampleSet(
        states=np.asarray([[1.0, 2.0]]),
        estimates=np.asarray([[1.1, 2.1]]),
        measurements=np.asarray([[0.2]]),
        next_states=np.asarray([[1.2, 2.2]]),
        nus=np.asarray([nu]),
        nu_indices=np.asarray([0]),
        nu_values=(nu,),
        times=np.asarray([time]),
        dt=0.02,
    )


def test_concatenate_sample_sets_reindexes_viscosity_and_keeps_time() -> None:
    combined = _concatenate_sample_sets(
        _sample_set(0.02, 0.4), _sample_set(0.005, 0.8)
    )

    assert combined.nu_values == (0.005, 0.02)
    assert combined.nu_indices.tolist() == [1, 0]
    assert combined.times.tolist() == [0.4, 0.8]
    assert combined.states.shape == (2, 2)


def test_ratio_summary_reports_rms_and_tail() -> None:
    summary = _ratio_summary(np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert summary["count"] == 4
    assert summary["rms"] == pytest.approx(np.sqrt(7.5))
    assert summary["median"] == pytest.approx(2.5)
    assert summary["max"] == pytest.approx(4.0)


def test_givens_certificate_is_identity_at_initialization_and_stays_bounded() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, INTERVALS)
    _gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=0.02,
        gain_scale=0.5,
        certificate_scale=1.0,
        lower_lipschitz=0.5,
        upper_lipschitz=2.0,
        certificate_kind="givens",
        mixing_layers=2,
    )
    generator = torch.Generator().manual_seed(42)
    states = torch.randn((3, grid.n), generator=generator)
    errors = torch.randn((3, grid.n), generator=generator)

    initial = certificate(states, errors)
    assert torch.allclose(initial, errors, atol=1.0e-6)

    torch.nn.init.normal_(certificate.network[-1].weight, std=0.1)
    torch.nn.init.normal_(certificate.network[-1].bias, std=0.1)
    transformed = certificate(states, errors)
    direction = (transformed - errors) @ torch.as_tensor(
        matrix.T, dtype=torch.float32
    )
    assert torch.max(torch.abs(direction)).item() < 1.0e-5
    assert torch.max(torch.abs(certificate(states, torch.zeros_like(errors)))).item() == 0

    error = errors[0].detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: certificate(states[0:1], value[None, :])[0], error
    )
    singular_values = torch.linalg.svdvals(jacobian)
    assert torch.min(singular_values).item() >= 0.5 - 1.0e-5
    assert torch.max(singular_values).item() <= 2.0 + 1.0e-5
