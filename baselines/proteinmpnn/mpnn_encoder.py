"""ProteinMPNN structure encoder (encoder half only) for chemical-shift prediction.

The encoder sees backbone N, CA, C, O (plus a virtual CB) and never the sequence,
so its per-residue output h_V [B, L, 128] is a pure backbone-structure embedding.
Hidden size is 128, the same as SLAE's residue embedding, so ShiftNet(emb_dim=128)
is reused unchanged.

Weights and protein_mpnn_utils.py: dauparas/ProteinMPNN (MIT), downloaded into external/ProteinMPNN by
baselines/setup_third_party.py --proteinmpnn; not stored in this repository.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

VENDOR_DIR = Path(__file__).resolve().parents[2] / "external" / "ProteinMPNN"
sys.path.insert(0, str(VENDOR_DIR))
from protein_mpnn_utils import ProteinMPNN, gather_nodes  # noqa: E402

DEFAULT_WEIGHTS = os.environ.get("MPNN_WEIGHTS", str(VENDOR_DIR / "v_48_020.pt"))
SLAE_FILL = 1e-5   # SLAE fills missing atom37 slots with this value


class ProteinMPNNEncoder(nn.Module):
    """features -> W_e -> 3 EncLayers of a pretrained ProteinMPNN; decoder dropped."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, pretrained: bool = True,
                 augment_eps: float = 0.0):
        super().__init__()
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        full = ProteinMPNN(ca_only=False, num_letters=21, node_features=128,
                           edge_features=128, hidden_dim=128,
                           num_encoder_layers=3, num_decoder_layers=3,
                           augment_eps=augment_eps, k_neighbors=ckpt["num_edges"])
        if pretrained:
            full.load_state_dict(ckpt["model_state_dict"])
        self.features = full.features
        self.W_e = full.W_e
        self.encoder_layers = full.encoder_layers
        self.hidden_dim = 128

    def forward(self, X, mask, residue_idx, chain_encoding):
        """
        X              : [B, L, 4, 3]  backbone N, CA, C, O (0 where missing)
        mask           : [B, L]        1.0 where the full backbone is present
        residue_idx    : [B, L]        residue numbers (drives relative positional encoding)
        chain_encoding : [B, L]        chain labels, 0 for padding
        returns h_V    : [B, L, 128]
        """
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V


def slae_batch_to_mpnn(batch):
    """Convert a SLAE PyG batch (coords [sum_L, 37, 3], atom37 order N, CA, C, O, ...)
    into padded ProteinMPNN inputs. Returns X, mask (backbone complete), pad_mask
    (residue exists), residue_idx, chain_encoding."""
    coords = batch.coords[:, :4, :].float()                       # N, CA, C, O
    res_batch = batch.batch
    n_graphs = len(batch.id)
    lengths = torch.bincount(res_batch, minlength=n_graphs)
    L = int(lengths.max())
    offsets = torch.cumsum(lengths, 0) - lengths
    pos = torch.arange(res_batch.numel(), device=coords.device) - offsets[res_batch]

    missing = ((coords - SLAE_FILL).abs() < 1e-6).all(-1).any(-1)  # any backbone atom absent
    X = coords.new_zeros((n_graphs, L, 4, 3))
    X[res_batch, pos] = torch.where(missing[:, None, None], torch.zeros_like(coords), coords)

    mask = coords.new_zeros((n_graphs, L))
    mask[res_batch, pos] = (~missing).float()
    pad_mask = torch.zeros((n_graphs, L), dtype=torch.bool, device=coords.device)
    pad_mask[res_batch, pos] = True

    residue_idx = torch.zeros((n_graphs, L), dtype=torch.long, device=coords.device)
    if hasattr(batch, "residue_id") and torch.is_tensor(batch.residue_id):
        residue_idx[res_batch, pos] = batch.residue_id.long()
    else:
        residue_idx[res_batch, pos] = pos
    chain_encoding = pad_mask.long()                              # single chain per entry
    return X, mask, pad_mask, residue_idx, chain_encoding
