#!/usr/bin/env python3
"""
make_bulk_vectors.py - test vectors for the optional bulk regression sim/tb_cnn_bulk.v
======================================================================================
Takes the first N MNIST test images, runs them through the bit-exact integer model
(quantize_export.int_forward, built from python/mnist_int8_model.npz) and writes

    bulk_img.mem   784 bytes per image (hex, one per line)
    bulk_exp.mem   10 expected 32-bit logits per image
    bulk_lbl.mem   true label per image

into the output directory (default: rtl/, where the simulator is normally run).
Usage:  python3 python/make_bulk_vectors.py [N=2000] [outdir]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantize_export as qe

ROOT = Path(__file__).resolve().parent.parent


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "rtl"
    m = np.load(ROOT / "python" / "mnist_int8_model.npz")
    Q = dict(WC=m["WC"], BC=m["BC"], M1=m["M1"], S1=int(m["S1"]), W1=m["W1"].astype(np.int64),
             B1=m["B1"], M2=int(m["M2"]), S2=int(m["S2"]), W2=m["W2"].astype(np.int64), B2=m["B2"])
    d = np.load(ROOT / "python" / "mnist_u8.npz")
    x, y = d["x_test"][:n], d["y_test"][:n]
    logits = qe.int_forward(x, Q)
    pred = np.argmax(logits, 1)
    (out / "bulk_img.mem").write_text("\n".join(f"{int(v):02x}" for v in x.reshape(-1)) + "\n")
    (out / "bulk_exp.mem").write_text("\n".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in logits.reshape(-1)) + "\n")
    (out / "bulk_lbl.mem").write_text("\n".join(f"{int(v):x}" for v in y) + "\n")
    print(f"{n} images -> {out}/bulk_{{img,exp,lbl}}.mem ; integer model accuracy on them: "
          f"{100 * np.mean(pred == y):.2f} % ({int(np.sum(pred == y))}/{n})")


if __name__ == "__main__":
    main()
