#!/usr/bin/env python3
"""
quantize_export.py - from the trained float model to everything the FPGA project needs
=======================================================================================
Input : a float model in Keras layout (python/model_float_pruned.npz by default - written by
        train_prune.py, or by export_from_keras.py from YOUR notebook's  model_for_export)
Output: * bit-exact INTEGER golden model of the hardware (this file is the specification)
        * rtl/mnist_w.mem  mnist_d.mem      sparse weight stream  (weight byte / index-delta nibble)
        * rtl/mnist_img.mem                 64 demo MNIST test digits (1024-byte stride)
        * rtl/mnist_conv_params.v           conv weights, biases, requant multipliers
        * rtl/mnist_dense_params.v          row lengths, biases, requant multiplier, checksum
        * rtl/mnist_sparse_rom.v mnist_test_rom.v   ROM wrappers (depth baked in)
        * sim/mnist_golden.v                expected logits / classes of the demo digits
        * python/mnist_int8_model.npz       same data for uart_demo.py

Quantisation (what TFLite's Optimize.DEFAULT does to the weights, made integer-only for an FPGA)
  * weights : int8 symmetric  (conv: per output channel, dense: per tensor); pruned zeros stay 0
  * activations: u8 with static scales calibrated on training images (TFLite's dynamic-range mode
    keeps activations in float, which is not an option in an FPGA)
  * requantisation: (x * M + 2^(S-1)) >> S with a 16-bit multiplier M

Integer model (identical in rtl/cnn_conv.v and rtl/cnn_dense.v):
    acc1[c,h,w] = sum_k Wc[c,k] * x[h+i, w+j] + Bc[c]          x : u8 pixel 0..255
    p[ph,pw,c]  = max(0, max over the 2x2 window of acc1)
    a1          = min(255, (p * M1[c] + 2^(S1-1)) >> S1)        flatten order (ph, pw, c)
    acc2[n]     = sum_f W1[n,f] * a1[f] + B1[n]
    a2[n]       = min(255, (min(max(acc2,0), 2^24-1) * M2 + 2^(S2-1)) >> S2)
    acc3[k]     = sum_n W2[k,n] * a2[n] + B2[k]                 class = argmax (lowest index wins)
"""
import argparse
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
N_DEMO = 64                      # demo digits stored in the FPGA (6 switches)
IMG_STRIDE = 1024                # bytes per stored image (784 used)
MAX_ENTRIES = 160_000            # block-RAM capacity guard for the sparse stream
M_BITS = 16                      # requantisation multiplier width in hardware


# ----------------------------------------------------------------------------- float reference
def float_forward(P, x):
    """x: (N,28,28) float in [0,1] -> logits.  P: dict of Keras-layout float weights."""
    n = len(x)
    out = []
    for i in range(0, n, 500):
        xb = x[i:i + 500]
        m = len(xb)
        patches = sliding_window_view(xb, (3, 3), axis=(1, 2)).reshape(m * 676, 9)
        z = np.maximum(patches @ P["conv_w"].reshape(9, 32) + P["conv_b"], 0)
        f = z.reshape(m, 13, 2, 13, 2, 32).max(axis=(2, 4)).reshape(m, 5408)
        h = np.maximum(f @ P["d1_w"] + P["d1_b"], 0)
        out.append(h @ P["d2_w"] + P["d2_b"])
    return np.concatenate(out)


# ----------------------------------------------------------------------------- integer model
def conv_pool_int(x_u8, Q):
    """-> p (N,13,13,32) int64 : ReLU + 2x2 max of the integer conv accumulators."""
    outs = []
    for i in range(0, len(x_u8), 250):
        xb = x_u8[i:i + 250]
        m = len(xb)
        pt = sliding_window_view(xb, (3, 3), axis=(1, 2)).reshape(m * 676, 9).astype(np.float64)
        acc = pt @ Q["WC"].astype(np.float64) + Q["BC"]                    # exact (< 2^53)
        p = np.maximum(acc.reshape(m, 13, 2, 13, 2, 32).max(axis=(2, 4)), 0)
        outs.append(p.astype(np.int64))
    return np.concatenate(outs)


def requant1(p, Q):
    return np.minimum(255, (p * Q["M1"] + (1 << (Q["S1"] - 1))) >> Q["S1"])


def dense1_acc(a1, Q):
    f = a1.reshape(len(a1), 5408).astype(np.float64)
    return np.rint(f @ Q["W1"].astype(np.float64)).astype(np.int64) + Q["B1"]


def requant2(acc1, Q):
    h = np.minimum(np.maximum(acc1, 0), (1 << 24) - 1)
    return np.minimum(255, (h * Q["M2"] + (1 << (Q["S2"] - 1))) >> Q["S2"])


def dense2_acc(a2, Q):
    return a2.astype(np.int64) @ Q["W2"].astype(np.int64) + Q["B2"]


def int_forward(x_u8, Q, keep=False):
    p = conv_pool_int(x_u8, Q)
    a1 = requant1(p, Q)
    acc1 = dense1_acc(a1, Q)
    a2 = requant2(acc1, Q)
    acc2 = dense2_acc(a2, Q)
    if keep:
        return acc2, dict(p=p, a1=a1, acc1=acc1, a2=a2)
    return acc2


def accuracy_int(x_u8, y, Q):
    return float(np.mean(np.argmax(int_forward(x_u8, Q), axis=1) == y))


# ----------------------------------------------------------------------------- quantisation
def pick_shift(ratio_max, bits=M_BITS):
    """largest S such that round(ratio_max * 2^S) fits in `bits` unsigned bits."""
    s = int(np.floor(np.log2(((1 << bits) - 1) / ratio_max)))
    while int(np.rint(ratio_max * (1 << s))) > (1 << bits) - 1:
        s -= 1
    return s


def build_Q(P, x_cal, pct):
    """Static quantisation of the float model P with activation scales taken at percentile `pct`."""
    Q = {}
    cw = P["conv_w"].reshape(9, 32).astype(np.float64)
    sc = np.abs(cw).max(axis=0) / 127.0
    dead = sc == 0                                   # channel whose 9 weights were all pruned: output = ReLU(bias)
    sc[dead] = np.median(sc[~dead]) if (~dead).any() else 1.0   # any scale works; this keeps M1 / the shift sane
    Q["sc"] = sc
    Q["WC"] = np.clip(np.rint(cw / sc), -127, 127).astype(np.int64)                    # (9,32)
    Q["BC"] = np.rint(P["conv_b"].astype(np.float64) * 255.0 / sc).astype(np.int64)    # (32,)

    # --- activation scale after conv/pool (SA1): real value of one u8 LSB
    p = conv_pool_int(x_cal, dict(Q, M1=None))                                          # integers
    p_real = p * (sc / 255.0)
    sa1 = np.percentile(p_real, pct) / 255.0
    ratio1 = (sc / 255.0) / sa1                                                         # per channel
    Q["S1"] = pick_shift(ratio1.max())
    Q["M1"] = np.rint(ratio1 * (1 << Q["S1"])).astype(np.int64)
    Q["SA1"] = sa1

    # --- dense 1 (per-tensor int8)
    w1 = P["d1_w"].astype(np.float64)
    s1w = np.abs(w1).max() / 127.0
    Q["W1"] = np.clip(np.rint(w1 / s1w), -127, 127).astype(np.int64)                    # (5408,128)
    Q["B1"] = np.rint(P["d1_b"].astype(np.float64) / (s1w * sa1)).astype(np.int64)
    a1 = requant1(p, Q)
    acc1 = dense1_acc(a1, Q)
    h_real = np.maximum(acc1, 0) * (s1w * sa1)
    sa2 = np.percentile(h_real, pct) / 255.0
    ratio2 = (s1w * sa1) / sa2
    Q["S2"] = pick_shift(ratio2)
    Q["M2"] = int(np.rint(ratio2 * (1 << Q["S2"])))
    Q["SA2"] = sa2

    # --- dense 2
    w2 = P["d2_w"].astype(np.float64)
    s2w = np.abs(w2).max() / 127.0
    Q["W2"] = np.clip(np.rint(w2 / s2w), -127, 127).astype(np.int64)                    # (128,10)
    Q["B2"] = np.rint(P["d2_b"].astype(np.float64) / (s2w * sa2)).astype(np.int64)
    Q["s1w"], Q["s2w"] = s1w, s2w
    return Q


def check_ranges(Q):
    """Every value must fit the register widths of the RTL."""
    b_conv = (np.abs(Q["WC"]).sum(axis=0) * 255 + np.abs(Q["BC"])).max()
    assert b_conv < (1 << 21), f"conv accumulator bound {b_conv} >= 2^21"
    assert np.abs(Q["BC"]).max() < (1 << 21), "conv bias does not fit 22 bits"
    assert Q["M1"].max() < (1 << M_BITS) and Q["M2"] < (1 << M_BITS)
    assert 1 <= Q["S1"] <= 37 and 1 <= Q["S2"] <= 39, (Q["S1"], Q["S2"])
    b1 = (np.abs(Q["W1"]).sum(axis=0) * 255 + np.abs(Q["B1"])).max()                    # per neuron
    b2 = (np.abs(Q["W2"]).sum(axis=0) * 255 + np.abs(Q["B2"])).max()
    assert max(b1, b2) < (1 << 27), f"dense accumulator bound {max(b1, b2)} >= 2^27"
    assert max(np.abs(Q["B1"]).max(), np.abs(Q["B2"]).max()) < (1 << 27)
    return int(b_conv), int(b1), int(b2)


# ----------------------------------------------------------------------------- sparse encoding
def encode_rows(rows):
    """rows: list of (sorted index array, int weight array). Returns weights, deltas, row lengths."""
    ws, ds, lens = [], [], []
    for idx, w in rows:
        n0 = len(ws)
        prev = -1
        for f, wv in zip(idx, w):
            d = int(f) - prev - 1
            while d > 15:                       # padding entry: advances the position by 16
                ws.append(0)
                ds.append(15)
                d -= 16
            ws.append(int(wv))
            ds.append(d)
            prev = int(f)
        lens.append(len(ws) - n0)
    return ws, ds, lens


def decode_rows(ws, ds, lens, l2_base_rows):
    """Python model of the hardware address generator (used to verify the encoding)."""
    out, e = [], 0
    for r, n in enumerate(lens):
        base, entries = 0, []
        for _ in range(n):
            idx = base + ds[e]
            base = idx + 1
            if ws[e] != 0:
                entries.append((idx, ws[e]))
            e += 1
        out.append(entries)
    return out


def rotl_xor_checksum(values):
    chk = 0
    for v in values:
        chk = (((chk << 1) | (chk >> 31)) & 0xFFFFFFFF) ^ int(v)
    return chk


# ----------------------------------------------------------------------------- writers
HDR = ("`timescale 1ns / 1ps\n"
       "// ---------------------------------------------------------------------------\n"
       "// AUTO-GENERATED by python/quantize_export.py - do not edit by hand.\n"
       "// ---------------------------------------------------------------------------\n")


def hexs(v, bits):
    return f"{int(v) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def write_mem(path, values, bits):
    digits = (bits + 3) // 4
    path.write_text("\n".join(f"{int(v) & ((1 << bits) - 1):0{digits}x}" for v in values) + "\n")


def write_conv_params(path, Q):
    lines = []
    for c in range(32):
        w = "".join(hexs(Q["WC"][k, c], 8) for k in range(8, -1, -1))     # {w8,...,w0}
        lines.append(f"            5'd{c}: begin w = 72'h{w}; bias = 22'h{hexs(Q['BC'][c], 22)}; "
                     f"mult = 16'd{int(Q['M1'][c])}; end")
    path.write_text(HDR + f"""// Convolution layer parameters, indexed by output channel.
//   w     : 9 int8 weights, tap k = 3*i + j (kernel row i, column j) at w[8k +: 8]
//   bias  : int22 bias in accumulator units  (round(b * 255 / scale_c))
//   mult  : requantisation multiplier M1[c];  feature = (max * M1 + 2^(s1-1)) >> s1
module mnist_conv_params (
    input  wire [4:0]          ch,
    output reg  [71:0]         w,
    output reg  signed [21:0]  bias,
    output reg  [15:0]         mult,
    output wire [5:0]          s1
);
    assign s1 = 6'd{Q['S1']};
    always @* begin
        case (ch)
{chr(10).join(lines)}
            default: begin w = 72'd0; bias = 22'd0; mult = 16'd0; end
        endcase
    end
endmodule
""")


def write_dense_params(path, Q, lens, n_ent, w_chk):
    biases = list(Q["B1"]) + list(Q["B2"])
    lines = [f"            8'd{r}: begin row_len = 13'd{lens[r]}; row_bias = 28'h{hexs(biases[r], 28)}; end"
             for r in range(len(lens))]
    path.write_text(HDR + f"""// Dense layer parameters. Rows 0..127 = hidden neurons, rows 128..137 = output classes.
//   row_len  : number of sparse-stream entries of the row (including index padding entries)
//   row_bias : int28 bias in accumulator units
//   m2, s2   : hidden-layer requantisation  a2 = (min(relu(acc), 2^24-1) * m2 + 2^(s2-1)) >> s2
//   n_ent    : total entries in the sparse ROM,  w_chk : expected ROM checksum (self-test)
module mnist_dense_params (
    input  wire [7:0]          row,
    output reg  [12:0]         row_len,
    output reg  signed [27:0]  row_bias,
    output wire [15:0]         m2,
    output wire [5:0]          s2,
    output wire [17:0]         n_ent,
    output wire [31:0]         w_chk
);
    assign m2    = 16'd{Q['M2']};
    assign s2    = 6'd{Q['S2']};
    assign n_ent = 18'd{n_ent};
    assign w_chk = 32'h{w_chk:08x};
    always @* begin
        case (row)
{chr(10).join(lines)}
            default: begin row_len = 13'd0; row_bias = 28'd0; end
        endcase
    end
endmodule
""")


def write_sparse_rom(path, n_ent):
    path.write_text(HDR + f"""// Sparse weight stream of both dense layers: {n_ent} entries {{delta[3:0], weight[7:0]}}.
// Initialised from mnist_w.mem / mnist_d.mem (one hex value per line) - these two files MUST be in the
// Vivado project (Add Sources) or in the simulator's working directory.
module mnist_sparse_rom #(
    parameter W_FILE = "mnist_w.mem",
    parameter D_FILE = "mnist_d.mem"
)(
    input  wire        clk,
    input  wire [17:0] addr,
    output reg  [7:0]  w = 8'd0,
    output reg  [3:0]  d = 4'd0
);
    localparam integer N = {n_ent};
    (* rom_style = "block" *) reg [7:0] mem_w [0:N-1];
    (* rom_style = "block" *) reg [3:0] mem_d [0:N-1];
    initial begin
        $readmemh(W_FILE, mem_w);
        $readmemh(D_FILE, mem_d);
    end
    always @(posedge clk) begin                    // synchronous read -> block RAM
        w <= mem_w[addr];
        d <= mem_d[addr];
    end
endmodule
""")


def write_test_rom(path, labels, img_chk):
    lbl = "\n".join(f"        lbl_mem[{i}] = 4'd{int(l)};" for i, l in enumerate(labels))
    path.write_text(HDR + f"""// {len(labels)} MNIST test digits (never used for training) and their labels.
// Image n occupies addresses n*{IMG_STRIDE} .. n*{IMG_STRIDE}+783 (row-major 28x28, 0..255); the rest is padding.
// Pixels come from mnist_img.mem (one byte per line) - the file MUST be in the Vivado project / sim directory.
module mnist_test_rom #(
    parameter IMG_FILE = "mnist_img.mem"
)(
    input  wire        clk,
    input  wire [15:0] addr,                       // {{image[5:0], pixel[9:0]}}
    output reg  [7:0]  pixel = 8'd0,
    input  wire [5:0]  lbl_addr,
    output reg  [3:0]  label = 4'd0,
    output wire [31:0] img_chk                     // expected checksum of the whole image memory
);
    reg [7:0] img_mem [0:{len(labels) * IMG_STRIDE - 1}];
    reg [3:0] lbl_mem [0:{len(labels) - 1}];
    assign img_chk = 32'h{img_chk:08x};
    initial begin
        $readmemh(IMG_FILE, img_mem);
{lbl}
    end
    always @(posedge clk) begin
        pixel <= img_mem[addr];
        label <= lbl_mem[lbl_addr];
    end
endmodule
""")


def write_golden(path, scores, preds):
    flat = scores.reshape(-1)
    rows = "\n".join("        " + " ".join(f"scores[{i + j}]=32'h{hexs(v, 32)};"
                                         for j, v in enumerate(flat[i:i + 4])) for i in range(0, len(flat), 4))
    pr = "\n".join("        " + " ".join(f"pred[{i + j}]=4'd{int(v)};" for j, v in enumerate(preds[i:i + 16]))
                   for i in range(0, len(preds), 16))
    path.write_text(HDR + f"""// Expected results of the integer model for the {len(preds)} demo digits (testbench only).
//   scores[n*10 + k] = 32-bit two's-complement logit of class k for image n,  pred[n] = argmax
module mnist_golden;
    reg [31:0] scores [0:{flat.size - 1}];
    reg [3:0]  pred   [0:{len(preds) - 1}];
    initial begin
{rows}
{pr}
    end
endmodule
""")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "python" / "model_float_pruned.npz")
    ap.add_argument("--baseline", type=Path, default=ROOT / "python" / "model_float_baseline.npz")
    ap.add_argument("--data", type=Path, default=ROOT / "python" / "mnist_u8.npz")
    ap.add_argument("--cal", type=int, default=3000, help="calibration images (from the training set)")
    args = ap.parse_args()

    d = np.load(args.data)
    x_tr, y_tr, x_te, y_te = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    P = {k: v.astype(np.float64) for k, v in np.load(args.weights).items()
         if k in ("conv_w", "conv_b", "d1_w", "d1_b", "d2_w", "d2_b")}
    expect = dict(conv_w=(3, 3, 1, 32), conv_b=(32,), d1_w=(5408, 128), d1_b=(128,), d2_w=(128, 10), d2_b=(10,))
    for k, shp in expect.items():
        if k not in P or P[k].shape != shp:
            raise SystemExit(f"{args.weights}: '{k}' must have shape {shp} (Keras layout of the notebook model), "
                             f"got {None if k not in P else P[k].shape}.  See python/export_from_keras.py")

    # ---- float references
    acc_f = float(np.mean(np.argmax(float_forward(P, x_te / 255.0), 1) == y_te))
    acc_b = None
    if args.baseline.exists():
        B = {k: v.astype(np.float64) for k, v in np.load(args.baseline).items()
             if k in ("conv_w", "conv_b", "d1_w", "d1_b", "d2_w", "d2_b")}
        acc_b = float(np.mean(np.argmax(float_forward(B, x_te / 255.0), 1) == y_te))
    nnz = {k: int(np.count_nonzero(P[k])) for k in ("conv_w", "d1_w", "d2_w")}

    # ---- calibration: choose the activation percentile by accuracy on training images
    x_cal, y_cal = x_tr[:args.cal], y_tr[:args.cal]
    best = None
    for pct in (99.9, 99.99, 99.999, 100.0):
        Q = build_Q(P, x_cal, pct)
        a = accuracy_int(x_cal, y_cal, Q)
        print(f"  calibration percentile {pct:7.3f}: S1={Q['S1']} S2={Q['S2']}  accuracy on {args.cal} train images {a * 100:.2f} %")
        if best is None or a >= best[0]:
            best = (a, pct, Q)
    _, pct, Q = best
    b_conv, b1, b2 = check_ranges(Q)

    # ---- integer model on the test set
    acc2_te = int_forward(x_te, Q)
    pred_te = np.argmax(acc2_te, 1)
    acc_i = float(np.mean(pred_te == y_te))

    # ---- sparse encoding of the two dense layers
    rows = []
    for n in range(128):
        col = Q["W1"][:, n]
        idx = np.nonzero(col)[0]
        rows.append((idx, col[idx]))
    for k in range(10):
        col = Q["W2"][:, k]
        idx = np.nonzero(col)[0]
        rows.append((idx, col[idx]))
    ws, ds, lens = encode_rows(rows)
    n_ent = len(ws)
    assert n_ent <= MAX_ENTRIES, (f"{n_ent} sparse entries do not fit the block RAM budget ({MAX_ENTRIES}): "
                                  f"use the pruned (80 %) model")
    assert max(lens) < (1 << 13) and n_ent < (1 << 18)
    dec = decode_rows(ws, ds, lens, 128)                                            # verify the encoding
    for r, (idx, w) in enumerate(rows):
        assert [(int(i), int(v)) for i, v in zip(idx, w)] == [(int(i), int(v)) for i, v in dec[r]], r
    n_pad = sum(1 for w, dd in zip(ws, ds) if w == 0)
    w_chk = rotl_xor_checksum(((dd << 8) | (w & 0xFF)) for w, dd in zip(ws, ds))

    # ---- demo digits: one correct digit of each class first, then the test set in order
    chosen = [int(np.where((y_te == c) & (pred_te == c))[0][0]) for c in range(10)]
    for i in range(len(y_te)):
        if len(chosen) >= N_DEMO:
            break
        if i not in chosen:
            chosen.append(i)
    demo_x, demo_y = x_te[chosen], y_te[chosen]
    demo_scores, inter = int_forward(demo_x, Q, keep=True)
    demo_pred = np.argmax(demo_scores, 1)
    img_bytes = np.zeros(N_DEMO * IMG_STRIDE, dtype=np.uint8)
    for i in range(N_DEMO):
        img_bytes[i * IMG_STRIDE:i * IMG_STRIDE + 784] = demo_x[i].reshape(-1)
    img_chk = rotl_xor_checksum(img_bytes)

    # ---- write everything
    rtl, sim = ROOT / "rtl", ROOT / "sim"
    write_mem(rtl / "mnist_w.mem", [w & 0xFF for w in ws], 8)
    write_mem(rtl / "mnist_d.mem", ds, 4)
    write_mem(rtl / "mnist_img.mem", img_bytes, 8)
    write_conv_params(rtl / "mnist_conv_params.v", Q)
    write_dense_params(rtl / "mnist_dense_params.v", Q, lens, n_ent, w_chk)
    write_sparse_rom(rtl / "mnist_sparse_rom.v", n_ent)
    write_test_rom(rtl / "mnist_test_rom.v", demo_y, img_chk)
    write_golden(sim / "mnist_golden.v", demo_scores, demo_pred)
    np.savez(ROOT / "python" / "mnist_int8_model.npz",
             WC=Q["WC"], BC=Q["BC"], M1=Q["M1"], S1=Q["S1"], W1=Q["W1"].astype(np.int8), B1=Q["B1"],
             M2=Q["M2"], S2=Q["S2"], W2=Q["W2"].astype(np.int8), B2=Q["B2"],
             demo_x=demo_x, demo_y=demo_y.astype(np.uint8), demo_idx=np.array(chosen),
             demo_scores=demo_scores.astype(np.int64), demo_pred=demo_pred.astype(np.uint8),
             demo_a1=inter["a1"][:4].astype(np.uint8), demo_a2=inter["a2"][:4].astype(np.uint8))

    # ---- report
    print()
    print(f"model            : {args.weights.name}")
    if acc_b is not None:
        print(f"float baseline   : {acc_b * 100:.2f} % test accuracy (dense, no pruning)")
    print(f"float pruned     : {acc_f * 100:.2f} % test accuracy")
    print(f"int8 (FPGA maths): {acc_i * 100:.2f} % test accuracy   <- what the hardware computes (activation percentile {pct})")
    print(f"non-zero weights : conv {nnz['conv_w']}/288, dense1 {nnz['d1_w']}/692224 "
          f"({100 * nnz['d1_w'] / 692224:.2f} %), dense2 {nnz['d2_w']}/1280")
    print(f"sparse stream    : {n_ent} entries ({n_pad} index-padding entries), longest row {max(lens)}; "
          f"weights {n_ent * 8 / 8 / 1024:.1f} KiB + deltas {n_ent * 4 / 8 / 1024:.1f} KiB")
    print(f"requantisation   : S1={Q['S1']}  M1 in [{Q['M1'].min()}, {Q['M1'].max()}]   S2={Q['S2']}  M2={Q['M2']}")
    print(f"accumulator bound: conv {b_conv} (< 2^21), dense1 {b1}, dense2 {b2} (< 2^27)")
    print(f"demo digits      : {N_DEMO} test images, integer model gets {int(np.sum(demo_pred == demo_y))}/{N_DEMO} right")
    print(f"checksums        : weights 0x{w_chk:08x}   images 0x{img_chk:08x}")
    print("wrote rtl/mnist_{w,d,img}.mem  rtl/mnist_{conv,dense}_params.v  rtl/mnist_sparse_rom.v  "
          "rtl/mnist_test_rom.v  sim/mnist_golden.v  python/mnist_int8_model.npz")


if __name__ == "__main__":
    main()
