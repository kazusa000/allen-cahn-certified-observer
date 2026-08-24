"""Allen–Cahn reference solver and certified-observer research tools."""

from .certificate import (
    CertificateAudit,
    IdentityCertificate,
    NullspaceCertificate,
    audit_certificate,
)
from .dataset import PilotCase, generate_pilot_cases, noise_waveform
from .grid import AllenCahnGrid
from .linearization import (
    allen_cahn_jacobian,
    incremental_remainder,
    local_incremental_rhs,
)
from .local_certificate import (
    LocalPartition,
    mass_norm,
    partition_samples,
    summarize_local_region,
    transition_counts,
)
from .observations import local_average_matrix
from .observer import CausalNudging, ObserverRollout, simulate_causal_nudging
from .solver import (
    AllenCahnSolution,
    allen_cahn_energy,
    allen_cahn_rhs,
    solve_allen_cahn,
)
from .training import (
    StateConditionedLinearCorrection,
    fit_state_conditioned_linear_correction,
    simulate_learned_correction,
)

__all__ = [
    "AllenCahnGrid",
    "AllenCahnSolution",
    "CausalNudging",
    "CertificateAudit",
    "IdentityCertificate",
    "NullspaceCertificate",
    "ObserverRollout",
    "PilotCase",
    "StateConditionedLinearCorrection",
    "allen_cahn_energy",
    "allen_cahn_jacobian",
    "allen_cahn_rhs",
    "audit_certificate",
    "fit_state_conditioned_linear_correction",
    "generate_pilot_cases",
    "incremental_remainder",
    "local_average_matrix",
    "local_incremental_rhs",
    "LocalPartition",
    "mass_norm",
    "partition_samples",
    "summarize_local_region",
    "transition_counts",
    "noise_waveform",
    "simulate_causal_nudging",
    "simulate_learned_correction",
    "solve_allen_cahn",
]
