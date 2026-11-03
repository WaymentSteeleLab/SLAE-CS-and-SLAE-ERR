# Modified for SLAE-CS (2026): imports of SLAE modules that SLAE-CS does not use, and that are
# not included in this repository, were removed. See "SLAE package" in README.md.
"""
Model architectures for SLAE.

This module contains the core neural network architectures:
- encoder: SLAE encoder architecture
- decoder: All-atom decoder
"""

from SLAE.model.encoder import ProteinEncoder
from SLAE.model.decoder import AllAtomDecoder

__all__ = [
    "ProteinEncoder",
    "AllAtomDecoder",
]
