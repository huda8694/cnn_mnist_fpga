`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// mnist_cnn_core.v - the notebook's pruned + quantised CNN, integer-only
//
//   28x28 u8 image -> Conv 3x3x32 + ReLU + MaxPool 2x2 -> 5408 features (u8)
//                  -> Dense 128 + ReLU -> Dense 10 -> argmax
//
// Interface: write the 784 image bytes (row-major) through (img_we, img_addr, img_data)
// while idle, pulse 'start' for one clock, wait for 'done' (1-clock pulse).
// 'pred' and 'scores_flat' (10 x signed 32-bit, class k at [32k +: 32]) stay valid until
// the next run.  After reset the core scans its weight ROM once (a few ms): wait for
// scan_done, then check rom_ok.  ('busy' is high during that scan and during a run.)
//
// Latency of a run at 100 MHz is about 1.7 ms (conv ~ 27 k clocks, dense ~ 147 k clocks).
// -----------------------------------------------------------------------------
module mnist_cnn_core #(
    parameter W_FILE = "mnist_w.mem",
    parameter D_FILE = "mnist_d.mem"
)(
    input  wire         clk,
    input  wire         rst,                // synchronous, active high
    // image write port
    input  wire         img_we,
    input  wire [9:0]   img_addr,
    input  wire [7:0]   img_data,
    // control
    input  wire         start,              // 1-clock pulse
    output wire         busy,
    output reg          done = 1'b0,        // 1-clock pulse
    output wire [3:0]   pred,
    output wire [319:0] scores_flat,
    // ROM self-test
    output wire         scan_done,
    output wire         rom_ok
);
    // ------------------------------------------------------------ feature RAM (5408 pooled + 128 hidden)
    reg [7:0] fm_mem [0:5535];
    reg [7:0] fm_q = 8'd0;

    wire        c_we, d_we, c_busy, d_busy, c_done, d_done;
    wire [12:0] c_wa, d_wa, d_ra;
    wire [7:0]  c_wd, d_wd;

    wire        fm_we = c_we | d_we;                       // the two writers are never active together
    wire [12:0] fm_wa = c_we ? c_wa : d_wa;
    wire [7:0]  fm_wd = c_we ? c_wd : d_wd;

    always @(posedge clk) begin
        if (fm_we) fm_mem[fm_wa] <= fm_wd;
        fm_q <= fm_mem[d_ra];
    end

    // ------------------------------------------------------------ engines
    reg conv_start = 1'b0, dense_start = 1'b0;

    cnn_conv u_conv (
        .clk(clk), .rst(rst),
        .img_we(img_we), .img_addr(img_addr), .img_data(img_data),
        .start(conv_start), .busy(c_busy), .done(c_done),
        .fm_we(c_we), .fm_addr(c_wa), .fm_data(c_wd));

    cnn_dense #(.W_FILE(W_FILE), .D_FILE(D_FILE)) u_dense (
        .clk(clk), .rst(rst), .start(dense_start), .busy(d_busy), .done(d_done),
        .fm_raddr(d_ra), .fm_rdata(fm_q),
        .fm_we(d_we), .fm_waddr(d_wa), .fm_wdata(d_wd),
        .pred(pred), .scores_flat(scores_flat),
        .scan_done(scan_done), .rom_ok(rom_ok));

    // ------------------------------------------------------------ sequencing: conv -> dense
    localparam [1:0] K_IDLE = 2'd0, K_CONV = 2'd1, K_DENSE = 2'd2;
    reg [1:0] kst = K_IDLE;

    assign busy = (kst != K_IDLE) || c_busy || d_busy;

    always @(posedge clk) begin
        conv_start  <= 1'b0;
        dense_start <= 1'b0;
        done        <= 1'b0;
        if (rst) begin
            kst <= K_IDLE;
        end else begin
            case (kst)
                K_IDLE: if (start && scan_done) begin
                    conv_start <= 1'b1;
                    kst        <= K_CONV;
                end
                K_CONV: if (c_done) begin
                    dense_start <= 1'b1;
                    kst         <= K_DENSE;
                end
                K_DENSE: if (d_done) begin
                    done <= 1'b1;
                    kst  <= K_IDLE;
                end
                default: kst <= K_IDLE;
            endcase
        end
    end
endmodule
