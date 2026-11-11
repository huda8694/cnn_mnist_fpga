`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// cnn_conv.v - Conv2D(32, 3x3, valid) + ReLU + MaxPool(2x2) + requantisation to u8
//
//   28x28x1 --conv--> 26x26x32 --relu / maxpool--> 13x13x32      (notebook layers 1-3)
//
// One "block" = one 2x2 pooling window = 4 conv positions, processed as
//   for ph in 0..12, pw in 0..12:
//     LOAD   read the 4x4 pixel patch at image position (2ph, 2pw) into 16 registers
//     COMP   for channel c in 0..31, pos in 0..3:  one 3x3 window per clock
//            (9 multipliers in parallel) -> sum + bias -> ReLU -> running max over pos
//            -> * M1[c] >> S1 (round, saturate to 255) -> feature (ph, pw, c)
//   Features are written in Keras flatten order  addr = (ph*13 + pw)*32 + c.
//
// Integer arithmetic (mirrored exactly by python/quantize_export.py):
//   acc  = sum_k Wq[c][k] * x[k] + Bq[c]          x : u8 pixel   Wq : int8 (per-channel scale)
//   p    = max over the 2x2 window of max(acc, 0)
//   feat = min(255, (p * M1[c] + 2^(S1-1)) >> S1)
//
// Pipeline.  "Stage k" = the clock in which the registers named *_k are valid; A = issue clock.
//   A  : window mux + weight ROM        -> px_r, wk_r, tags 1
//   S1 : 9 products                      -> prod,        tags 2
//   S2 : three partial sums + bias ROM   -> sa/sb/sc/b_r, tags 3
//   S3 : total = sa+sb+sc+bias           -> total,       tags 4
//   S4 : ReLU, running max, M1 ROM       -> mx, m_r, pool-complete flag, tag 5
//   S5 : rq = mx * M1                    ->              tag 6
//   S6 : round, shift, saturate          -> RAM write registers
//   S7 : RAM write
// -----------------------------------------------------------------------------
module cnn_conv (
    input  wire        clk,
    input  wire        rst,
    // image write port (use only while busy = 0): 784 bytes, row-major
    input  wire        img_we,
    input  wire [9:0]  img_addr,
    input  wire [7:0]  img_data,
    // control
    input  wire        start,
    output reg         busy = 1'b0,
    output reg         done = 1'b0,
    // pooled-feature write port (13*13*32 = 5408 bytes)
    output reg         fm_we   = 1'b0,
    output reg  [12:0] fm_addr = 13'd0,
    output reg  [7:0]  fm_data = 8'd0
);
    // ------------------------------------------------------------ image RAM
    reg [7:0] img_mem [0:783];
    reg [7:0] px_q = 8'd0;
    reg [9:0] ra   = 10'd0;                       // image read address
    always @(posedge clk) begin
        if (img_we) img_mem[img_addr] <= img_data;
        px_q <= img_mem[ra];
    end

    // ------------------------------------------------------------ control state
    localparam [2:0] C_IDLE = 3'd0, C_LOAD = 3'd1, C_COMP = 3'd2, C_DRAIN = 3'd3, C_NEXT = 3'd4;
    reg [2:0]   state    = C_IDLE;
    reg [3:0]   pw       = 4'd0;                  // block column 0..12
    reg [3:0]   ph       = 4'd0;                  // block row    0..12
    reg [9:0]   row_base = 10'd0;                 // image address of (2*ph, 0)
    reg [9:0]   blk_base = 10'd0;                 // image address of (2*ph, 2*pw)
    reg [12:0]  fblk     = 13'd0;                 // feature address of (ph, pw, 0)
    reg [4:0]   ld_k     = 5'd0;                  // patch pixel being requested, 0..15
    reg         cap_v    = 1'b0;                  // the RAM delivers a patch pixel this clock
    reg [3:0]   cap_k    = 4'd0;
    reg [6:0]   t        = 7'd0;                  // issue counter {channel[4:0], pos[1:0]}
    reg [3:0]   drain    = 4'd0;
    reg [127:0] patch    = 128'd0;                // 4x4 patch, pixel k = [8k +: 8]

    wire [4:0] cc  = t[6:2];
    wire [1:0] pp  = t[1:0];
    wire       iss = (state == C_COMP);

    // ------------------------------------------------------------ tag registers (valid / channel / position)
    reg        v1 = 1'b0, v2 = 1'b0, v3 = 1'b0, v4 = 1'b0, v5 = 1'b0, v6 = 1'b0;
    reg [4:0]  c1 = 5'd0, c2 = 5'd0, c3 = 5'd0, c4 = 5'd0, c5 = 5'd0, c6 = 5'd0;
    reg [1:0]  p1 = 2'd0, p2 = 2'd0, p3 = 2'd0, p4 = 2'd0;

    // ------------------------------------------------------------ generated parameter ROMs (combinational)
    wire [71:0]        w_now;                      // 9 int8 weights of channel cc          (clock A)
    wire signed [21:0] b_c2;                       // bias of channel c2                    (stage S2)
    wire [15:0]        m_c4;                       // requantisation multiplier of c4       (stage S4)
    wire [5:0]         s1;                         // requantisation shift (constant)
    mnist_conv_params u_pw (.ch(cc), .w(w_now), .bias(),     .mult(),     .s1());
    mnist_conv_params u_pb (.ch(c2), .w(),      .bias(b_c2), .mult(),     .s1());
    mnist_conv_params u_pm (.ch(c4), .w(),      .bias(),     .mult(m_c4), .s1(s1));

    // ------------------------------------------------------------ clock A: window select + weights
    reg [7:0]  px_r [0:8];
    reg [71:0] wk_r = 72'd0;
    genvar gi;
    generate
        for (gi = 0; gi < 9; gi = gi + 1) begin : g_win
            localparam integer I = gi / 3, J = gi % 3;
            wire [7:0] q00 = patch[8*((0+I)*4 + 0+J) +: 8];
            wire [7:0] q01 = patch[8*((0+I)*4 + 1+J) +: 8];
            wire [7:0] q10 = patch[8*((1+I)*4 + 0+J) +: 8];
            wire [7:0] q11 = patch[8*((1+I)*4 + 1+J) +: 8];
            always @(posedge clk)
                px_r[gi] <= (pp == 2'd0) ? q00 : (pp == 2'd1) ? q01 : (pp == 2'd2) ? q10 : q11;
        end
    endgenerate
    always @(posedge clk) begin
        wk_r <= w_now;
        v1 <= iss;  c1 <= cc;  p1 <= pp;
    end

    // ------------------------------------------------------------ S1: nine products
    reg signed [16:0] prod [0:8];
    generate
        for (gi = 0; gi < 9; gi = gi + 1) begin : g_mul
            always @(posedge clk)
                prod[gi] <= $signed({1'b0, px_r[gi]}) * $signed(wk_r[8*gi +: 8]);
        end
    endgenerate
    always @(posedge clk) begin
        v2 <= v1;  c2 <= c1;  p2 <= p1;
    end

    // ------------------------------------------------------------ S2: partial sums, bias
    reg signed [18:0] sa  = 19'sd0, sb = 19'sd0, sc = 19'sd0;
    reg signed [21:0] b_r = 22'sd0;
    always @(posedge clk) begin
        sa  <= prod[0] + prod[1] + prod[2];
        sb  <= prod[3] + prod[4] + prod[5];
        sc  <= prod[6] + prod[7] + prod[8];
        b_r <= b_c2;
        v3 <= v2;  c3 <= c2;  p3 <= p2;
    end

    // ------------------------------------------------------------ S3: total
    reg signed [22:0] total = 23'sd0;
    always @(posedge clk) begin
        total <= sa + sb + sc + b_r;
        v4 <= v3;  c4 <= c3;  p4 <= p3;
    end

    // ------------------------------------------------------------ S4: ReLU + running max over the window
    wire [21:0] relu = total[22] ? 22'd0 : total[21:0];
    reg  [21:0] mx  = 22'd0;
    reg  [15:0] m_r = 16'd0;
    always @(posedge clk) begin
        if (p4 == 2'd0)     mx <= relu;
        else if (relu > mx) mx <= relu;
        m_r <= m_c4;
        v5  <= v4 && (p4 == 2'd3);                 // the 4th position has just been merged
        c5  <= c4;
    end

    // ------------------------------------------------------------ S5: requantise multiply
    reg [37:0] rq = 38'd0;
    always @(posedge clk) begin
        rq <= mx * m_r;
        v6 <= v5;  c6 <= c5;
    end

    // ------------------------------------------------------------ S6: round, shift, saturate, RAM write registers
    wire [37:0] rnd = 38'd1 << (s1 - 6'd1);
    wire [37:0] shr = (rq + rnd) >> s1;
    always @(posedge clk) begin
        fm_we   <= v6;
        fm_addr <= fblk + {8'd0, c6};
        fm_data <= (|shr[37:8]) ? 8'hFF : shr[7:0];
    end

    // ------------------------------------------------------------ control FSM
    always @(posedge clk) begin
        done  <= 1'b0;
        cap_v <= 1'b0;
        if (rst) begin
            state <= C_IDLE;
            busy  <= 1'b0;
        end else begin
            case (state)
                C_IDLE: if (start) begin
                    busy     <= 1'b1;
                    ph       <= 4'd0;
                    pw       <= 4'd0;
                    row_base <= 10'd0;
                    blk_base <= 10'd0;
                    fblk     <= 13'd0;
                    ra       <= 10'd0;
                    ld_k     <= 5'd0;
                    state    <= C_LOAD;
                end

                C_LOAD: begin                               // request the 16 patch pixels, one per clock
                    cap_v <= (ld_k < 5'd16);
                    cap_k <= ld_k[3:0];
                    if (ld_k < 5'd16) begin
                        ld_k <= ld_k + 1'b1;
                        ra   <= (ld_k[1:0] == 2'd3) ? ra + 10'd25 : ra + 10'd1;
                    end else if (!cap_v) begin              // the last pixel has been captured
                        t     <= 7'd0;
                        state <= C_COMP;
                    end
                end

                C_COMP: begin
                    t <= t + 1'b1;
                    if (t == 7'd127) begin
                        drain <= 4'd0;
                        state <= C_DRAIN;
                    end
                end

                C_DRAIN: begin                              // let the pipeline write the last features
                    drain <= drain + 1'b1;
                    if (drain == 4'd8) state <= C_NEXT;
                end

                C_NEXT: begin
                    fblk <= fblk + 13'd32;
                    ld_k <= 5'd0;
                    if (pw == 4'd12) begin
                        pw <= 4'd0;
                        if (ph == 4'd12) begin
                            state <= C_IDLE;
                            busy  <= 1'b0;
                            done  <= 1'b1;
                        end else begin
                            ph       <= ph + 1'b1;
                            row_base <= row_base + 10'd56;
                            blk_base <= row_base + 10'd56;
                            ra       <= row_base + 10'd56;
                            state    <= C_LOAD;
                        end
                    end else begin
                        pw       <= pw + 1'b1;
                        blk_base <= blk_base + 10'd2;
                        ra       <= blk_base + 10'd2;
                        state    <= C_LOAD;
                    end
                end

                default: state <= C_IDLE;
            endcase
        end
    end

    // the RAM delivers each requested pixel one clock after the request
    always @(posedge clk)
        if (cap_v) patch[8*cap_k +: 8] <= px_q;
endmodule
