#!/usr/bin/env python3
"""
train_prune.py - NumPy re-implementation of notebook-qp.ipynb ("Pruning and Quantization on MNIST")
=============================================================================================
Same model, same training recipe as the notebook, written in NumPy so no TensorFlow is needed:

    Reshape(28,28,1) -> Conv2D(32,(3,3),relu) -> MaxPooling2D(2,2) -> Flatten
                     -> Dense(128,relu) -> Dense(10)              (693,962 parameters)

    * data       : MNIST, pixels / 255.0, last 10 % of the 60 000 training images = validation
    * init       : Keras defaults (glorot_uniform kernels, zero biases)
    * optimiser  : Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7)   batch 128, 10 epochs
    * loss       : SparseCategoricalCrossentropy(from_logits=True)
    * pruning    : tfmot prune_low_magnitude on Conv2D, Dense(128) and Dense(10) kernels,
                   PolynomialDecay(initial 0.50 -> final 0.80, begin 0, end_step = 469*10,
                   power 3, mask refresh every 100 steps), trained from scratch like the notebook.

Modes
    --mode baseline   plain training (the notebook's "Baseline" model)
    --mode pruned     pruning-aware training (the model the notebook exports / quantises)

Output: an .npz with the Keras-layout weights
    conv_w (3,3,1,32)  conv_b (32)  d1_w (5408,128)  d1_b (128)  d2_w (128,10)  d2_b (10)
which is exactly what  model_for_export.get_weights()  gives in the notebook, so weights trained
there can be used instead (see export_from_keras.py).
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DT = np.float32
ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- model
class CNN:
    """Conv(3x3,32)+ReLU -> MaxPool(2) -> Flatten -> Dense(128)+ReLU -> Dense(10)."""

    def __init__(self, rng):
        def glorot(shape, fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, shape).astype(DT)
        self.p = {
            "conv_w": glorot((3, 3, 1, 32), 9, 9 * 32),
            "conv_b": np.zeros(32, DT),
            "d1_w": glorot((5408, 128), 5408, 128),
            "d1_b": np.zeros(128, DT),
            "d2_w": glorot((128, 10), 128, 10),
            "d2_b": np.zeros(10, DT),
        }

    def forward(self, x):
        """x: (N,28,28) in [0,1]. Returns logits and the cache needed for backprop."""
        p, n = self.p, x.shape[0]
        P = sliding_window_view(x, (3, 3), axis=(1, 2)).reshape(n * 676, 9)        # im2col
        Z = P @ p["conv_w"].reshape(9, 32) + p["conv_b"]                           # (n*676,32)
        A = np.maximum(Z, 0)
        A6 = A.reshape(n, 13, 2, 13, 2, 32)
        pooled = A6.max(axis=(2, 4))                                               # (n,13,13,32)
        F = pooled.reshape(n, 5408)                                                # (h,w,c) order
        H = np.maximum(F @ p["d1_w"] + p["d1_b"], 0)
        L = H @ p["d2_w"] + p["d2_b"]
        return L, (P, Z, A6, pooled, F, H)

    def loss_and_grads(self, x, y):
        p, n = self.p, x.shape[0]
        L, (P, Z, A6, pooled, F, H) = self.forward(x)
        Ls = L - L.max(1, keepdims=True)
        logp = Ls - np.log(np.exp(Ls).sum(1, keepdims=True))
        loss = float(-logp[np.arange(n), y].mean())
        acc = float((L.argmax(1) == y).mean())

        dL = np.exp(logp)
        dL[np.arange(n), y] -= 1
        dL /= n
        g = {}
        g["d2_w"] = H.T @ dL
        g["d2_b"] = dL.sum(0)
        dH = (dL @ p["d2_w"].T) * (H > 0)
        g["d1_w"] = F.T @ dH
        g["d1_b"] = dH.sum(0)
        dF = dH @ p["d1_w"].T
        dpool = dF.reshape(n, 13, 1, 13, 1, 32)
        dA6 = (A6 == pooled[:, :, None, :, None, :]) * dpool                       # route to the max
        dZ = dA6.reshape(n * 676, 32) * (Z > 0)
        g["conv_w"] = (P.T @ dZ).reshape(3, 3, 1, 32)
        g["conv_b"] = dZ.sum(0)
        return loss, acc, g

    def predict_logits(self, x, bs=500):
        return np.concatenate([self.forward(x[i:i + bs])[0] for i in range(0, len(x), bs)])


class Adam:
    """Keras Adam (epsilon is added to sqrt(v) without bias correction, as in tf.keras)."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-7):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        for k, w in params.items():
            m, v, gk = self.m[k], self.v[k], grads[k]
            m *= self.b1; m += (1 - self.b1) * gk
            v *= self.b2; v += (1 - self.b2) * gk * gk
            w -= (lr_t * m / (np.sqrt(v) + self.eps)).astype(w.dtype)


# ----------------------------------------------------------------------------- pruning (tfmot semantics)
PRUNED_KERNELS = ("conv_w", "d1_w", "d2_w")


def poly_sparsity(step, initial=0.50, final=0.80, begin=0, end=4690, power=3):
    """tfmot PolynomialDecay."""
    p = min(1.0, max(0.0, (step - begin) / float(end - begin)))
    return final + (initial - final) * (1.0 - p) ** power


def update_mask(w, sparsity):
    """tfmot: keep the k = round(size*(1-sparsity)) largest |w| (ties at the threshold are kept)."""
    a = np.abs(w).reshape(-1)
    k = int(np.round(a.size * (1.0 - sparsity)))
    thr = np.sort(a)[::-1][k - 1]
    return (np.abs(w) >= thr).astype(w.dtype)


# ----------------------------------------------------------------------------- data / eval
def load_mnist(path):
    d = np.load(path)
    f = lambda a: (a.astype(np.float32) / 255.0)
    return f(d["x_train"]), d["y_train"].astype(np.int64), f(d["x_test"]), d["y_test"].astype(np.int64)


def evaluate(model, x, y):
    L = model.predict_logits(x)
    logp = L - L.max(1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean()), float((L.argmax(1) == y).mean())


def grad_check():
    """Finite-difference check of the hand-written backprop (float64, tiny batch)."""
    global DT
    old, DT = DT, np.float64
    rng = np.random.default_rng(0)
    m = CNN(rng)
    x = rng.random((3, 28, 28)); y = np.array([1, 5, 7])
    _, _, g = m.loss_and_grads(x, y)
    worst = 0.0
    for k in m.p:
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in m.p[k].shape)
            w0 = m.p[k][idx]; h = 1e-6
            m.p[k][idx] = w0 + h; lp, _, _ = m.loss_and_grads(x, y)
            m.p[k][idx] = w0 - h; lm, _, _ = m.loss_and_grads(x, y)
            m.p[k][idx] = w0
            num = (lp - lm) / (2 * h); ana = g[k][idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    DT = old
    return worst


# ----------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["baseline", "pruned"], required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "python" / "mnist_u8.npz")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--grad-check", action="store_true")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="stop (exit code 3) after the first epoch that ends beyond this many seconds; "
                         "re-run the same command to resume from the checkpoint")
    args = ap.parse_args()

    if args.grad_check:
        print(f"gradient check (float64): worst relative error = {grad_check():.2e}")
        return

    out = args.out or ROOT / "python" / f"model_float_{args.mode}.npz"
    xtr_all, ytr_all, xte, yte = load_mnist(args.data)
    n_val = int(len(xtr_all) * 0.1)                       # validation_split=0.1 -> LAST 10 %
    xtr, ytr, xva, yva = xtr_all[:-n_val], ytr_all[:-n_val], xtr_all[-n_val:], ytr_all[-n_val:]
    rng = np.random.default_rng(args.seed)
    model = CNN(rng)
    opt = Adam(model.p)
    steps_per_epoch = int(np.ceil(len(xtr) / args.batch))
    end_step = int(np.ceil(len(xtr_all) / args.batch)) * args.epochs      # notebook uses 60000/128 -> 4690
    masks = {k: np.ones_like(model.p[k]) for k in PRUNED_KERNELS}
    print(f"mode={args.mode} train={len(xtr)} val={len(xva)} test={len(xte)} steps/epoch={steps_per_epoch} "
          f"prune end_step={end_step}", flush=True)

    ckpt = out.with_suffix(".ckpt.pkl")
    hist, step, ep0, t0 = [], 0, 0, time.time()
    if ckpt.exists():                                       # resume an interrupted run
        st = pickle.load(open(ckpt, "rb"))
        model.p, masks, hist, step, ep0 = st["p"], st["masks"], st["hist"], st["step"], st["epoch"]
        opt.m, opt.v, opt.t = st["m"], st["v"], st["t"]
        rng.bit_generator.state = st["rng"]
        print(f"resumed from {ckpt.name}: epoch {ep0}/{args.epochs}, step {step}", flush=True)
    t_run = time.time()
    for ep in range(ep0, args.epochs):
        order = rng.permutation(len(xtr))
        tl = ta = 0.0
        for b in range(steps_per_epoch):
            idx = order[b * args.batch:(b + 1) * args.batch]
            if args.mode == "pruned" and step % 100 == 0 and step <= end_step:
                s = poly_sparsity(step, end=end_step)
                for k in PRUNED_KERNELS:
                    masks[k] = update_mask(model.p[k], s)
                    model.p[k] *= masks[k]
            loss, acc, g = model.loss_and_grads(xtr[idx], ytr[idx])
            opt.step(model.p, g)
            if args.mode == "pruned":
                for k in PRUNED_KERNELS:
                    model.p[k] *= masks[k]                 # tfmot weight_mask_op every step
            w = len(idx) / len(xtr)
            tl += loss * w; ta += acc * w
            step += 1
            if args.max_steps and step >= args.max_steps:
                print(f"{step} steps in {time.time() - t0:.1f}s -> {(time.time() - t0) / step * 1000:.0f} ms/step")
                return
        vl, va = evaluate(model, xva, yva)
        hist.append((tl, ta, vl, va))
        sp = ""
        if args.mode == "pruned":
            sp = " sparsity " + " ".join(f"{k[:4]}={100 * (1 - masks[k].mean()):.1f}%" for k in PRUNED_KERNELS)
        print(f"Epoch {ep + 1}/{args.epochs} - loss: {tl:.4f} - accuracy: {ta:.4f} - val_loss: {vl:.4f} "
              f"- val_accuracy: {va:.4f}{sp}  [{time.time() - t_run:.0f}s]", flush=True)
        tmp = str(ckpt) + ".tmp"
        pickle.dump(dict(p=model.p, masks=masks, hist=hist, step=step, epoch=ep + 1, m=opt.m, v=opt.v,
                         t=opt.t, rng=rng.bit_generator.state), open(tmp, "wb"))
        os.replace(tmp, ckpt)
        if args.budget and ep + 1 < args.epochs and time.time() - t_run > args.budget:
            print(f"PAUSED after epoch {ep + 1}; run the same command again to continue", flush=True)
            sys.exit(3)

    if args.mode == "pruned":                              # strip_pruning: weights are already masked
        for k in PRUNED_KERNELS:
            model.p[k] *= masks[k]
    tl, ta = evaluate(model, xte, yte)
    print(f"{args.mode} test accuracy: {ta:.4f}")
    nz = {k: int(np.count_nonzero(model.p[k])) for k in PRUNED_KERNELS}
    print("non-zero kernel weights:", {k: f"{nz[k]}/{model.p[k].size} ({100 * nz[k] / model.p[k].size:.2f}%)" for k in nz})
    np.savez(out, **{k: v.astype(np.float32) for k, v in model.p.items()},
             history=np.array(hist, dtype=np.float32), test_accuracy=np.float32(ta))
    print("saved", out)


if __name__ == "__main__":
    main()
