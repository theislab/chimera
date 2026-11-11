"""Chimera: Cross-species Harmonization via Integrated Mixing of Evolutionary Residual Alphas.

A scvi-tools model for cross-species single-cell data with a per-gene gate that
reads expression deployment relative to protein conservation.
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
