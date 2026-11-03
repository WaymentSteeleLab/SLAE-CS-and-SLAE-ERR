# Modified for SLAE-CS (2026): imports of SLAE modules that SLAE-CS does not use, and that are
# not included in this repository, were removed. See "SLAE package" in README.md.
"""
Utility functions and helpers for SLAE.

This module provides various utility functions including:
- types: Type definitions and data structures
- base_model: Base model classes
- constants: Protein structure constants
"""

from SLAE.util._irrep_utils import _fix_irreps_dict, _irreps_compatible
from SLAE.util.types import ModelOutput
from SLAE.nn.base_model import BaseModel
from SLAE.nn.mlp_decoder import MLPDecoder, PositionDecoder

__all__ = [
    # Irrep utilities
    "_fix_irreps_dict",
    "_irreps_compatible",

    # Types
    "ModelOutput",


    # Base classes
    "BaseModel",

    # Utilities
    "MLPDecoder",
    "PositionDecoder",
]
