#!/usr/bin/env python3
"""
get_mnist.py - create python/mnist_u8.npz (uint8 MNIST) for train_prune.py / quantize_export.py
==============================================================================================
Tries, in order:  (1) keras.datasets.mnist  (what the notebook uses)   (2) a GitHub mirror of MNIST.
Writes x_train (60000,28,28) y_train x_test (10000,28,28) y_test, all uint8.
"""
import gzip
import io
import pickle
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("mnist_u8.npz")
MIRROR = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"


def from_keras():
    try:
        from tensorflow import keras
    except ImportError:
        import keras
    (xt, yt), (xe, ye) = keras.datasets.mnist.load_data()
    return xt.astype(np.uint8), yt.astype(np.uint8), xe.astype(np.uint8), ye.astype(np.uint8)


def from_mirror():
    raw = urllib.request.urlopen(MIRROR, timeout=120).read()
    tr, va, te = pickle.load(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding="latin1")
    x = np.concatenate([tr[0], va[0]]).astype(np.float64)        # this copy stores pixel/256 (not /255)
    xe = te[0].astype(np.float64)
    u = lambda a: np.rint(a * 256.0).astype(np.uint8).reshape(-1, 28, 28)
    return u(x), np.concatenate([tr[1], va[1]]).astype(np.uint8), u(xe), te[1].astype(np.uint8)


def main():
    for name, fn in (("keras", from_keras), ("GitHub mirror", from_mirror)):
        try:
            xt, yt, xe, ye = fn()
            break
        except Exception as e:                                    # noqa: BLE001
            print(f"{name}: failed ({type(e).__name__}: {str(e)[:80]})")
    else:
        raise SystemExit("could not obtain MNIST; place mnist_u8.npz in python/ manually")
    assert xt.shape == (60000, 28, 28) and xe.shape == (10000, 28, 28)
    np.savez_compressed(OUT, x_train=xt, y_train=yt, x_test=xe, y_test=ye)
    print(f"saved {OUT}  (train mean pixel {xt.mean():.2f}, expected 33.32)")


if __name__ == "__main__":
    main()
