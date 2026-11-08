#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference with the shift heads trained on frozen embeddings (ESM3, MACE-OFF23 baselines).

Counterpart of slae_cs/inference/chemical_shift_predictor.py. Writes the same
predictions_{atom}_{run_tag}_{split}.json ({"pred": {id: [...]}, "mask": {id: [...]}}), which
scripts/evaluate.py and scripts/ensemble/merge_seed_predictions.py read unchanged.

Same inference rules as SLAE-CS: batch size 1, eval() mode, deterministic kernels, and output mask =
residue exists AND reliability flags AND tag mask. The forward pass is
embedding_shift_trainer.EmbCSLightning, with the z-score statistics stored in the weights.

One model per call. Run from a working directory (PYTHONPATH at the repository root):
    python $R/baselines/embedding_head/emb_shift_predictor.py --ckpt weights/seqstruct/seed01 --run_tag esm3_seqstruct_seed01 \
        --emb_npz esm3_seqstruct.npz --split test --data_dir . --mask_json $MASK --tag_mask_csv $TAG
--ckpt takes a released folder, a model.safetensors or a training .ckpt.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from safetensors.torch import load_file

from slae_cs.train import chemical_shift_trainer as cst   # mask helpers (sets CUBLAS env at import)
import embedding_shift_trainer as est         # EmbCSLightning, load_embeddings


def load_weights(path):
    """state_dict and hyper-parameters from a released folder / model.safetensors, or a training .ckpt."""
    path = Path(path)
    st = path / "model.safetensors" if path.is_dir() else path
    if st.suffix == ".safetensors":
        cfg = st.parent / "config.json"
        hp = json.load(open(cfg)).get("hyper_parameters", {}) if cfg.exists() else {}
        return load_file(str(st)), hp
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"], ckpt.get("hyper_parameters", {})


def load_model(ckpt_path: Path, device) -> est.EmbCSLightning:
    sd, hp = load_weights(ckpt_path)
    model = est.EmbCSLightning(shift_dict={}, emb_mean=sd["emb_mean"].clone(),
                               emb_std=sd["emb_std"].clone(), atom=hp.get("atom", "N"))
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="released folder, model.safetensors or .ckpt")
    p.add_argument("--run_tag", required=True, help="names the outputs: predictions_<atom>_<run_tag>_<split>.json")
    p.add_argument("--emb_npz", required=True)
    p.add_argument("--split", action="append", required=True, help="val / test (repeatable)")
    p.add_argument("--data_dir", required=True, help="dir with <split>.list")
    p.add_argument("--mask_json", required=True)
    p.add_argument("--tag_mask_csv", default=None)
    p.add_argument("--out_dir", default=".")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    # GPU like the SLAE-CS predictor (fails instead of silently falling back);
    # --device cpu is only for local testing
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- pass --device cpu only for local tests")
    device = torch.device(args.device)
    # torch defaults to one thread per visible core, which thrashes when the
    # CPU quota is smaller (e.g. a batch-scheduler slot): follow OMP_NUM_THREADS
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    # same reproducibility settings as slae_cs/inference/chemical_shift_predictor.py
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    ids = {s: sorted(l.strip().lower() for l in open(Path(args.data_dir) / f"{s}.list") if l.strip())
           for s in args.split}
    emb = est.load_embeddings(Path(args.emb_npz), set().union(*ids.values()))
    for s in args.split:
        miss = [i for i in ids[s] if i not in emb]
        print(f"{s}: {len(ids[s]) - len(miss)}/{len(ids[s])} proteins have embeddings"
              + (f" (missing e.g. {miss[:5]})" if miss else ""))
        ids[s] = [i for i in ids[s] if i in emb]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    model = load_model(ckpt, device)
    atom = model.atom
    # same masks as the SLAE-CS predictor: flags (inverted, True = keep) AND tag
    mask_dict = cst._and_mask_dicts(
        cst._load_extra_mask_dict(args.mask_json, atom=atom),
        cst._load_tag_mask_dict(Path(args.tag_mask_csv)) if args.tag_mask_csv else {})
    for s in args.split:
        out = {"pred": {}, "mask": {}}
        with torch.inference_mode():
            for pid in ids[s]:
                x = torch.from_numpy(emb[pid].astype(np.float32)).unsqueeze(0)
                batch = est.EmbBatch([pid], x.to(device), torch.tensor([x.shape[1]], device=device))
                pred, pad = model(batch)
                pos = pad[0].cpu()
                ext = mask_dict.get(pid)
                if ext is not None:
                    n = min(len(pos), len(ext))
                    gated = torch.zeros_like(pos)
                    gated[:n] = pos[:n] & ext[:n]
                    pos = gated
                out["pred"][pid] = pred[0].cpu().numpy().tolist()
                out["mask"][pid] = pos.numpy().tolist()
        path = out_dir / f"predictions_{atom}_{args.run_tag}_{s}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=4)
        print(f"{ckpt.name} [{s}]: {len(out['pred'])} proteins -> {path.name}")


if __name__ == "__main__":
    main()
