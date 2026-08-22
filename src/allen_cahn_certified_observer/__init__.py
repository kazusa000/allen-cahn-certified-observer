"""Allen–Cahn reference solver and certified-observer research tools."""

from .grid import AllenCahnGrid
from .observations import local_average_matrix
from .solver import (
    AllenCahnSolution,
    allen_cahn_energy,
    allen_cahn_rhs,
    solve_allen_cahn,
)

__all__ = [
    "AllenCahnGrid",
    "AllenCahnSolution",
    "allen_cahn_energy",
    "allen_cahn_rhs",
    "local_average_matrix",
    "solve_allen_cahn",
]
