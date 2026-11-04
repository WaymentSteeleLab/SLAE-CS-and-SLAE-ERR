#!/usr/bin/env python
"""Convert a trained Lightning checkpoint into the weight files predict.py reads.

    python scripts/to_safetensors.py <in.ckpt> <out_dir>

Writes <out_dir>/model.safetensors (the state_dict only: no optimizer state, no pickled code) and
<out_dir>/config.json (the Lightning hyper-parameters, the epoch, and the name and sha256 of the source
checkpoint; the path of the pretrained encoder is reduced to its file name). The file is read back and
every tensor is compared with the checkpoint (names, dtypes, shapes, values); the script stops on any
difference. This is how the released weights were made.

For predict.py, put the 10 SLAE-CS models in <dir>/seed01 ... <dir>/seed10 and SLAE-ERR in its own
directory, e.g.
    python scripts/to_safetensors.py runs/panav_seed1/best_ckpts/top3_val_N_epoch_RMSE_*.ckpt my_models/SLAE-CS-PANAV/seed01
    python predict.py --weights my_models/SLAE-CS-PANAV --error-model my_models/SLAE-ERR --pdb x.pdb --out x
"""
import argparse
import hashlib
import json
import os

import torch
from safetensors.torch import load_file, save_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck.get("state_dict", ck).items() if isinstance(v, torch.Tensor)}
    # safetensors needs contiguous tensors without shared storage
    out = {k: v.detach().clone().contiguous() for k, v in sd.items()}
    os.makedirs(a.out_dir, exist_ok=True)
    st_path = os.path.join(a.out_dir, "model.safetensors")
    save_file(out, st_path)

    hp = json.loads(json.dumps(ck.get("hyper_parameters", {}), default=str))
    if isinstance(hp.get("ckpt"), str):
        hp["ckpt"] = os.path.basename(hp["ckpt"])          # no local paths in the released config
    cfg = {"source_ckpt": os.path.basename(a.ckpt), "source_sha256": sha256(a.ckpt), "epoch": ck.get("epoch"),
           "hyper_parameters": hp}
    with open(os.path.join(a.out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    back = load_file(st_path)                               # read back and compare
    assert set(back) == set(sd), "tensor names differ from the checkpoint"
    for k, v in sd.items():
        assert back[k].dtype == v.dtype and back[k].shape == v.shape and torch.equal(back[k], v), f"{k} differs"
    n = sum(v.numel() for v in sd.values())
    print(f"{len(sd)} tensors, {n / 1e6:.2f} M parameters -> {st_path} -- identical to {os.path.basename(a.ckpt)}")


if __name__ == "__main__":
    main()
