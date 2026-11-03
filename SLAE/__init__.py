# Modified for SLAE-CS (2026): imports of SLAE modules that SLAE-CS does not use, and that are
# not included in this repository, were removed. See "SLAE package" in README.md.
"""
SLAE: Structure-based Latent All-atom Encoder
==============================================

A deep learning framework for protein structure analysis using full-atom representations.

Main Components:
- datasets: Data loading and preprocessing
- features: Feature extraction and representation
- model: Neural network architectures (encoder, decoder)
- nn: Neural network building blocks
- util: Utility functions and helpers
"""

__version__ = "0.1.0"

# Core model imports for convenience
from SLAE.model.encoder import ProteinEncoder
from SLAE.model.decoder import AllAtomDecoder

# Dataset imports
from SLAE.datasets.dataset import ProteinDataset
from SLAE.datasets.dataloader import ProteinDataLoader
from SLAE.datasets.datamodule import PDBDataModule

# Feature imports
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
from SLAE.features.fa_representation import transform_representation_fa

__all__ = [
    # Models
    "ProteinEncoder",
    "AllAtomDecoder",

    # Datasets
    "ProteinDataset",
    "ProteinDataLoader",
    "PDBDataModule",

    # Features
    "ProteinGraphFeaturizer",
    "transform_representation_fa",
]
