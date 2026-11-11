#!/usr/bin/env python3
"""
uart_demo.py - talk to the MNIST CNN on the Nexys A7 through the USB cable.

Protocol (115200 baud, 8N1)
    PC   -> FPGA : 0xA5, then 784 pixel bytes (28x28 image, row-major, 0..255, white digit on black like MNIST)
    FPGA -> PC   : 0x5A, class (0..9), then 10 logits as signed 32-bit big-endian (42 bytes)

Modes
    python python/uart_demo.py --port COM5                  send the 64 stored demo digits and compare every
                                                             logit with the bit-exact integer model
    python python/uart_demo.py --port COM5 --png digit.png  classify your own picture (any size; it is cropped,
                                                             scaled to 20x20 and centred in 28x28 like MNIST).
                                                             Use --invert for black ink on white paper.
Windows: the COM number is in Device Manager > Ports ("USB Serial Port").
Linux  : /dev/ttyUSB1 (the FT2232 exposes two ports; the UART is normally the 2nd one).
Needs  : pip install pyserial numpy         (--png also needs: pip install pillow)
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HEADER, SYNC, REPLY_LEN = 0xA5, 0x5A, 42


def classify(ser, pixels):
    """Send one 28x28 image (784 values 0..255), return (class, tuple_of_10_logits)."""
    data = bytes(int(p) & 0xFF for p in np.asarray(pixels).reshape(-1))
    if len(data) != 784:
        raise ValueError(f"need exactly 784 pixel values, got {len(data)}")
    ser.reset_input_buffer()
    ser.write(bytes([HEADER]) + data)
    ser.flush()
    rep = ser.read(REPLY_LEN)
    if len(rep) != REPLY_LEN:
        raise RuntimeError(f"timeout: got {len(rep)} of {REPLY_LEN} reply bytes - is the bitstream loaded "
                           f"(display shows L/P, not EEEE), and is this the right COM port?")
    if rep[0] != SYNC:
        raise RuntimeError(f"bad sync byte 0x{rep[0]:02X} (expected 0x{SYNC:02X})")
    return rep[1], struct.unpack(">10i", rep[2:])


def png_to_mnist(path, invert=False):
    """Picture -> 28x28 uint8, MNIST style: crop to the ink, scale to fit 20x20, centre by centre of mass."""
    from PIL import Image
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    if invert:
        im = 255.0 - im
    im = np.clip(im - im.min(), 0, None)
    if im.max() > 0:
        im = im / im.max() * 255.0
    ys, xs = np.nonzero(im > 40)
    if len(ys) == 0:
        raise ValueError("no ink found in the picture (try --invert)")
    im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = im.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = np.asarray(Image.fromarray(im.astype(np.uint8)).resize((nw, nh), Image.LANCZOS), dtype=np.float64)
    canvas = np.zeros((28, 28))
    cy, cx = np.mgrid[0:nh, 0:nw]
    m = small.sum()
    com_y, com_x = (cy * small).sum() / m, (cx * small).sum() / m
    oy, ox = int(round(14 - com_y)), int(round(14 - com_x))
    oy, ox = min(max(oy, 0), 28 - nh), min(max(ox, 0), 28 - nw)
    canvas[oy:oy + nh, ox:ox + nw] = small
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def show(img):
    for r in range(0, 28, 2):
        print("   " + "".join(" .:-=+*#%@"[min(9, int(v) // 26)] for v in img[r]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", type=Path, default=Path(__file__).with_name("mnist_int8_model.npz"))
    ap.add_argument("--png", type=Path, help="classify this picture instead of the demo set")
    ap.add_argument("--invert", action="store_true", help="with --png: black ink on a white background")
    args = ap.parse_args()

    import serial                                    # imported late so --help works without pyserial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=3.0)
    except serial.SerialException as e:
        sys.exit(f"cannot open {args.port}: {e}")

    with ser:
        if args.png:
            img = png_to_mnist(args.png, args.invert)
            print("image sent to the FPGA:")
            show(img)
            cls, logits = classify(ser, img)
            print(f"\nFPGA says: digit {cls}")
            print("logits   :", list(logits))
            return

        m = np.load(args.model)
        imgs, labels = m["demo_x"], m["demo_y"]
        gold_scores, gold_pred = m["demo_scores"], m["demo_pred"]
        n = len(imgs)
        same = right = 0
        print(" #  label  FPGA  model  logits identical")
        for i in range(n):
            cls, logits = classify(ser, imgs[i])
            ident = list(logits) == [int(s) for s in gold_scores[i]] and cls == int(gold_pred[i])
            same += ident
            right += (cls == int(labels[i]))
            print(f"{i:2d}    {int(labels[i])}      {cls}     {int(gold_pred[i])}     {'yes' if ident else 'NO  <<<'}")
        print(f"\nFPGA == bit-exact integer model : {same}/{n}")
        print(f"FPGA matches the true label     : {right}/{n}")
        sys.exit(0 if same == n else 1)


if __name__ == "__main__":
    main()
