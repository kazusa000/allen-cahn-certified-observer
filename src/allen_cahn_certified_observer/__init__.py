"""Allen–Cahn reference solver and certified-observer research tools."""

from .certificate import (
    CertificateAudit,
    IdentityCertificate,
    NullspaceCertificate,
    audit_certificate,
)
from .grid import AllenCahnGrid
from .linearization import (
    allen_cahn_jacobian,
    incremental_remainder,
    local_incremental_rhs,
)
from .observations import local_average_matrix
from .observer import CausalNudging, ObserverRollout, simulate_causal_nudging
from .solver import (
    AllenCahnSolution,
    allen_cahn_energy,
    allen_cahn_rhs,
    solve_allen_cahn,
)

__all__ = [
    "AllenCahnGrid",
    "AllenCahnSolution",
    "CausalNudging",
    "CertificateAudit",
    "IdentityCertificate",
    "NullspaceCertificate",
    "ObserverRollout",
    "allen_cahn_energy",
    "allen_cahn_jacobian",
    "allen_cahn_rhs",
    "audit_certificate",
    "incremental_remainder",
    "local_average_matrix",
    "local_incremental_rhs",
    "simulate_causal_nudging",
    "solve_allen_cahn",
]
