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
    partition_samples,
    summarize_local_region,
    transition_counts,
)
from .observations import local_average_matrix
from .observer import (
    CausalNudging,
    CausalOutputInjection,
    ObserverRollout,
    simulate_causal_nudging,
)
from .observer_design import (
    ModalInjectionDesign,
    UnstableModalSystem,
    finite_horizon_transient_amplification,
    linearized_error_matrix,
    lmi_modal_injection,
    mass_adjoint_injection,
    normalized_modal_transform,
    pole_placement_modal_injection,
    riccati_modal_injection,
    symmetric_allen_cahn_margin,
    unstable_modal_system,
)
from .solver import (
    AllenCahnSolution,
    allen_cahn_energy,
    allen_cahn_rhs,
    solve_allen_cahn,
)
from .spectral import (
    TailAudit,
    audit_high_frequency_tail,
    dirichlet_laplacian_rates,
    dirichlet_sine_basis,
    low_frequency_projector,
    mass_norm,
    sampled_forced_tail_envelope,
    split_low_tail,
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
    "CausalOutputInjection",
    "CertificateAudit",
    "IdentityCertificate",
    "NullspaceCertificate",
    "ModalInjectionDesign",
    "ObserverRollout",
    "PilotCase",
    "StateConditionedLinearCorrection",
    "UnstableModalSystem",
    "TailAudit",
    "allen_cahn_energy",
    "allen_cahn_jacobian",
    "allen_cahn_rhs",
    "audit_certificate",
    "audit_high_frequency_tail",
    "dirichlet_laplacian_rates",
    "dirichlet_sine_basis",
    "fit_state_conditioned_linear_correction",
    "finite_horizon_transient_amplification",
    "generate_pilot_cases",
    "incremental_remainder",
    "linearized_error_matrix",
    "lmi_modal_injection",
    "local_average_matrix",
    "local_incremental_rhs",
    "LocalPartition",
    "low_frequency_projector",
    "mass_adjoint_injection",
    "mass_norm",
    "normalized_modal_transform",
    "partition_samples",
    "summarize_local_region",
    "transition_counts",
    "noise_waveform",
    "pole_placement_modal_injection",
    "riccati_modal_injection",
    "sampled_forced_tail_envelope",
    "simulate_causal_nudging",
    "simulate_learned_correction",
    "solve_allen_cahn",
    "split_low_tail",
    "symmetric_allen_cahn_margin",
    "unstable_modal_system",
]
