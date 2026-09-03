import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_direct_fiber_multigrid_joint import ALPHA, NU_VALUE, _torch_context
from r5_i_backstepping_nonlinear_remainder_audit import _audit_batch

from allen_cahn_certified_observer.target_diagnostics import (
    mass_rate,
    normalized_mass_norm,
    orthogonal_defect_components,
    same_form_target_components,
)


def test_same_form_target_expands_the_old_nonlinear_definition() -> None:
    torch = pytest.importorskip("torch")
    states = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    transformed = torch.tensor([[0.1, 0.4]], dtype=torch.float64)
    laplacian = torch.tensor([[-2.0, 1.0], [1.0, -2.0]], dtype=torch.float64)

    pieces = same_form_target_components(
        torch, states, transformed, laplacian, nu=0.005, alpha=0.01
    )
    reaction_increment = (
        states + transformed - (states + transformed) ** 3 - states + states**3
    )
    expected = (
        0.005 * (transformed @ laplacian.T) + reaction_increment - 1.01 * transformed
    )

    assert torch.allclose(pieces["total"], expected)


def test_defect_split_is_orthogonal_and_reconstructs() -> None:
    torch = pytest.importorskip("torch")
    transformed = torch.tensor([[1.0, 2.0], [2.0, -1.0]], dtype=torch.float64)
    defect = torch.tensor([[3.0, -1.0], [0.5, 4.0]], dtype=torch.float64)
    split = orthogonal_defect_components(torch, transformed, defect, 0.25)

    reconstructed = split["parallel"] + split["perpendicular"]
    inner = 0.25 * torch.sum(transformed * split["perpendicular"], dim=1)

    assert torch.allclose(reconstructed, defect, atol=1.0e-12, rtol=0.0)
    assert torch.allclose(inner, torch.zeros_like(inner), atol=1.0e-12, rtol=0.0)


def test_mass_rate_and_normalized_norm_have_expected_scaling() -> None:
    torch = pytest.importorskip("torch")
    transformed = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    rhs = -2.0 * transformed
    reference = 2.0 * transformed

    rate = mass_rate(torch, transformed, rhs, 0.1)
    normalized = normalized_mass_norm(torch, transformed, reference, 0.1)

    assert rate.item() == pytest.approx(2.0)
    assert normalized.item() == pytest.approx(0.5)


def test_target_diagnostics_reject_incompatible_shapes() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="matching"):
        orthogonal_defect_components(
            torch,
            torch.zeros((2, 3)),
            torch.zeros((2, 4)),
            0.1,
        )

    with pytest.raises(ValueError, match="incompatible"):
        same_form_target_components(
            torch,
            torch.zeros((2, 3)),
            torch.zeros((2, 3)),
            torch.eye(4),
            nu=0.005,
            alpha=0.01,
        )


def test_identity_coordinate_audit_recovers_the_known_radial_defect() -> None:
    torch = pytest.importorskip("torch")
    context = _torch_context(torch, 31, device="cpu", dtype=torch.float64)

    class IdentityLowTransform(torch.nn.Module):
        def forward(self, states: object, errors: object) -> object:
            del states
            return errors

    class ZeroGain(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("value", torch.zeros((4, 3), dtype=torch.float64))

        def forward(self) -> object:
            return self.value

    generator = torch.Generator().manual_seed(41)
    states = 0.2 * torch.randn((6, 31), generator=generator, dtype=torch.float64)
    errors = 0.1 * torch.randn((6, 31), generator=generator, dtype=torch.float64)

    arrays = _audit_batch(
        torch,
        IdentityLowTransform(),
        ZeroGain(),
        states,
        errors,
        context,
    )

    assert torch.max(arrays["pushforward_reconstruction_relative_error"]).item() < 1e-10
    assert torch.max(arrays["source_reconstruction_relative_error"]).item() < 1e-10
    assert torch.max(arrays["same_target__normalized_perpendicular"]).item() < 1e-10
    assert arrays["same_target__normalized_parallel"].detach().numpy() == pytest.approx(
        np.full(6, 1.0 + ALPHA), rel=1e-10, abs=1e-10
    )

    laplacian = context["laplacian"]
    expected_rhs = (
        NU_VALUE * (errors @ laplacian.T)
        + errors
        - ((states + errors) ** 3 - states**3)
    )
    expected_rate = mass_rate(torch, errors, expected_rhs, context["grid"].h)
    assert torch.allclose(arrays["actual_rate"], expected_rate, atol=1e-12, rtol=1e-12)
