"""Chimera: Cross-species Harmonization via Integrated Mixing of Evolutionary Residual Alphas.

A scvi-tools model for cross-species scRNA-seq integration with learnable
per-gene evolutionary gating.
"""

__version__ = "0.1.0"

from chimera._components import (
    ConstrainedBatchOffset,
    GeneIntegrationNetwork,
    LibrarySizeEncoder,
)
from chimera._model import ChimeraModel
from chimera._module import ChimeraMinimalModule, ChimeraModule

__all__ = [
    "ChimeraModel",
    "ChimeraMinimalModule",
    "ChimeraModule",
    "GeneIntegrationNetwork",
    "ConstrainedBatchOffset",
    "LibrarySizeEncoder",
]
