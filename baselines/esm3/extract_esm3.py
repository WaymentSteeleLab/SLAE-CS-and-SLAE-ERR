"""Extract per-residue ESM3-open embeddings for the AF2 (ColabFold) structures.

Needs the ESM3-open weights (EvolutionaryScale/esm3-sm-open-v1 on Hugging Face, non-commercial
license; ESM3.from_pretrained downloads them after the license is accepted). Image: baselines/esm3/esm3.def.
Runs used revision 47f0545b2b6daf26a93439a3cd610f4f7f3d5478 of that repository.

Two passes per structure:
    seqstruct : sequence + AF2 backbone structure (structure track filled)
    seqonly   : sequence only (structure track masked) -- sequence-only control

Output: one .npz per structure with keys
    resnum, sequence                (L,) / str
    seqstruct, seqonly              (L, 1536) float16   (BOS/EOS removed)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_chain import ProteinChain


@torch.no_grad()
def embed(model, protein):
    tokens = model.encode(protein)
    out = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
    return out.embeddings[0, 1:-1].float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb_dir", required=True)
    p.add_argument("--list", required=True, help="text file, one PDB filename per line")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--chain", default="A")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ESM3.from_pretrained("esm3-open").to(args.device).eval()

    names = [l.strip() for l in open(args.list) if l.strip()]
    for k, fname in enumerate(names):
        out_path = out_dir / (Path(fname).stem + ".npz")
        if out_path.exists():
            continue
        t0 = time.time()
        try:
            chain = ProteinChain.from_pdb(str(Path(args.pdb_dir) / fname), chain_id=args.chain)
            protein = ESMProtein.from_protein_chain(chain)
            e_struct = embed(model, protein)
            e_seq = embed(model, ESMProtein(sequence=protein.sequence))
            L = len(protein.sequence)
            assert e_struct.shape[0] == L and e_seq.shape[0] == L, (e_struct.shape, e_seq.shape, L)
            np.savez_compressed(
                out_path,
                resnum=np.asarray(chain.residue_index),
                sequence=np.array(protein.sequence),
                seqstruct=e_struct.astype(np.float16),
                seqonly=e_seq.astype(np.float16),
            )
            print(f"[{k+1}/{len(names)}] {fname}: L={L} {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[{k+1}/{len(names)}] {fname}: FAILED {e!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
