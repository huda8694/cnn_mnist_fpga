#!/usr/bin/env python3
"""
export_from_keras.py - use the weights trained in YOUR notebook on the FPGA
===========================================================================
Run this inside the notebook (same Python session), right after the cell that does

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

either with        %run python/export_from_keras.py          (variable `model_for_export` must exist)
or from a cell     import export_from_keras; export_from_keras.export(model_for_export)

It writes  python/model_float_pruned.npz  (Keras layout: conv_w (3,3,1,32), conv_b, d1_w (5408,128),
d1_b, d2_w (128,10), d2_b).  Then, in the project folder:

    python python/quantize_export.py      # int8 model + all FPGA data files (rtl/*.mem, rtl/*_params.v ...)

and rebuild the bitstream.  The pruned model must have ~80 % zero weights in the Dense layers
(that is what the notebook's prune_low_magnitude schedule produces): the sparse weight stream has to
fit in the FPGA's block RAM, quantize_export.py stops with an error otherwise.
"""
from pathlib import Path

import numpy as np


def export(model, path=None):
    w = model.get_weights()
    if len(w) != 6:
        raise ValueError(f"expected 6 weight tensors (conv, dense, dense_1 kernels+biases), got {len(w)}")
    names = ("conv_w", "conv_b", "d1_w", "d1_b", "d2_w", "d2_b")
    shapes = ((3, 3, 1, 32), (32,), (5408, 128), (128,), (128, 10), (10,))
    for n, a, s in zip(names, w, shapes):
        if a.shape != s:
            raise ValueError(f"{n}: expected shape {s}, got {a.shape} - is this the notebook's model?")
    path = Path(path) if path else Path("python") / "model_float_pruned.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **dict(zip(names, (a.astype(np.float32) for a in w))))
    nz = {n: f"{100 * np.count_nonzero(a) / a.size:.1f} % non-zero" for n, a in zip(names, w) if n.endswith("_w")}
    print("saved", path, nz)


if "model_for_export" in globals():          # %run inside the notebook
    export(globals()["model_for_export"])
