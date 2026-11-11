`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// mnist_top.v - Nexys A7-100T board wrapper for the pruned + quantised MNIST CNN
//
// Only 43 package pins are used (clock, reset, 1 button, 6 switches, 16 LEDs, 7-segment display,
// USB-UART); the network's data stays INSIDE the FPGA, so place_design has no I/O problem.
//
// DEMO WITHOUT A PC
//   SW[5:0]   picks one of 64 stored MNIST test digits (never used for training).
//   The digit is classified automatically after boot and whenever the switches change;
//   BTNC classifies it again.
//   7-seg:  [7:6] image number   L<d> true label   P<d> predicted digit   ('-' = no result)
//   LED[9:0]  one-hot predicted class      LED[15] result valid
//   LED[14]   prediction == label          LED[13] busy
//   LED[12]   last result came from UART  LED[11] heart-beat
//   LED[10]   BOOT SELF-TEST FAILED: a .mem file did not load (display shows EEEE;
//             LED[0] = weight ROM ok, LED[1] = image ROM ok)
//
// UART (115200 8N1 through the USB cable, see python/uart_demo.py)
//   PC -> FPGA : 0xA5, then 784 pixel bytes (28x28, row-major, 0..255, like MNIST)
//   FPGA -> PC : 0x5A, class (0..9), then 10 logits, each signed 32-bit big-endian (42 bytes)
// -----------------------------------------------------------------------------
module mnist_top #(
    parameter integer CLK_HZ       = 100_000_000,
    parameter integer BAUD         = 115_200,
    parameter integer DEBOUNCE_CYC = 1_000_000,    // 10 ms
    parameter integer REFRESH_BITS = 16,           // 7-seg: 2^16 clocks per digit
    parameter integer UART_TIMEOUT = 10_000_000,   // 100 ms without a byte aborts a frame
    parameter         W_FILE       = "mnist_w.mem",   // data files (must be found by the tool)
    parameter         D_FILE       = "mnist_d.mem",
    parameter         IMG_FILE     = "mnist_img.mem"
)(
    input  wire        CLK100MHZ,                  // E3
    input  wire        CPU_RESETN,                 // C12, active low
    input  wire        BTNC,                       // N17
    input  wire [5:0]  SW,                         // SW[5:0]
    output wire [15:0] LED,
    output wire [6:0]  SEG,                        // {CG,CF,CE,CD,CC,CB,CA}, active low
    output wire        DP,
    output wire [7:0]  AN,                         // active low
    input  wire        UART_TXD_IN,                // C4 : serial data coming FROM the PC
    output wire        UART_RXD_OUT                // D4 : serial data going TO the PC
);
    wire clk = CLK100MHZ;

    // ------------------------------------------------------------ reset (async assert, sync release)
    reg [2:0] rst_pipe = 3'b111;
    always @(posedge clk or negedge CPU_RESETN)
        if (!CPU_RESETN) rst_pipe <= 3'b111;
        else             rst_pipe <= {rst_pipe[1:0], 1'b0};
    wire rst = rst_pipe[2];

    // ------------------------------------------------------------ user inputs
    wire btn_run;
    debounce #(.STABLE_CYCLES(DEBOUNCE_CYC)) u_btn (
        .clk(clk), .rst(rst), .noisy(BTNC), .clean(), .rise(btn_run));

    reg [5:0] sw_s1 = 6'd0, sw_s2 = 6'd0;
    always @(posedge clk) begin
        sw_s1 <= SW;
        sw_s2 <= sw_s1;
    end

    // ------------------------------------------------------------ UART
    localparam integer CPB = CLK_HZ / BAUD;        // clocks per bit (868)

    wire [7:0] rx_data;
    wire       rx_valid;
    uart_rx #(.CLKS_PER_BIT(CPB)) u_rx (
        .clk(clk), .rst(rst), .rx(UART_TXD_IN), .data(rx_data), .valid(rx_valid));

    reg  [7:0] tx_data  = 8'd0;
    reg        tx_start = 1'b0;
    wire       tx_busy;
    uart_tx #(.CLKS_PER_BIT(CPB)) u_tx (
        .clk(clk), .rst(rst), .start(tx_start), .data(tx_data), .tx(UART_RXD_OUT), .busy(tx_busy));

    // ------------------------------------------------------------ demo images in block RAM
    reg  [5:0]  sel_run  = 6'd0;                   // image index of the last / current ROM run
    reg  [9:0]  ld_cnt   = 10'd0;                  // pixel counter (loader)
    reg         iscan_run = 1'b1;                  // boot: image-ROM checksum scan in progress
    reg  [16:0] iscan_cnt = 17'd0;
    wire [15:0] rom_addr = iscan_run ? iscan_cnt[15:0] : {sel_run, ld_cnt};
    wire [7:0]  rom_pix;
    wire [3:0]  rom_lbl;
    wire [31:0] rom_chk;
    mnist_test_rom #(.IMG_FILE(IMG_FILE)) u_test_rom (
        .clk(clk), .addr(rom_addr), .pixel(rom_pix),
        .lbl_addr(sel_run), .label(rom_lbl), .img_chk(rom_chk));

    // boot self-test of the image ROM (same checksum rule as the weight ROM)
    reg         iscan_v    = 1'b0;
    reg  [31:0] ichk       = 32'd0;
    reg         iscan_done = 1'b0;
    reg         img_ok     = 1'b0;
    always @(posedge clk) begin
        iscan_v <= 1'b0;
        if (rst) begin
            iscan_run  <= 1'b1;
            iscan_cnt  <= 17'd0;
            ichk       <= 32'd0;
            iscan_done <= 1'b0;
            img_ok     <= 1'b0;
        end else begin
            if (iscan_run) begin
                iscan_v   <= 1'b1;
                iscan_cnt <= iscan_cnt + 17'd1;
                if (iscan_cnt == 17'd65535) iscan_run <= 1'b0;
            end
            if (iscan_v) ichk <= {ichk[30:0], ichk[31]} ^ {24'd0, rom_pix};
            if (!iscan_run && !iscan_v && !iscan_done) begin
                iscan_done <= 1'b1;
                img_ok     <= (ichk == rom_chk);
            end
        end
    end

    // ------------------------------------------------------------ neural network
    reg          core_start = 1'b0;
    wire         core_done, scan_done, rom_ok;
    wire [3:0]   core_pred;
    wire [319:0] core_scores;

    reg  [2:0]   st = 3'd0;
    localparam [2:0] T_BOOT = 3'd0, T_IDLE = 3'd1, T_ROMLD = 3'd2, T_ROMWAIT = 3'd3,
                     T_UARTLD = 3'd4, T_RUN = 3'd5, T_REPLY = 3'd6, T_ERR = 3'd7;

    reg         ld_v = 1'b0;                       // ROM loader: write strobe (1 clk after the address)
    reg  [9:0]  ld_a = 10'd0;
    wire        img_we   = (st == T_UARTLD) ? rx_valid : ld_v;
    wire [9:0]  img_addr = (st == T_UARTLD) ? ld_cnt   : ld_a;
    wire [7:0]  img_data = (st == T_UARTLD) ? rx_data  : rom_pix;

    mnist_cnn_core #(.W_FILE(W_FILE), .D_FILE(D_FILE)) u_core (
        .clk(clk), .rst(rst),
        .img_we(img_we), .img_addr(img_addr), .img_data(img_data),
        .start(core_start), .busy(), .done(core_done),
        .pred(core_pred), .scores_flat(core_scores),
        .scan_done(scan_done), .rom_ok(rom_ok));

    // ------------------------------------------------------------ system controller
    reg        ran_once    = 1'b0;                 // forces one automatic run after boot
    reg        run_pending = 1'b0;
    reg        src_uart    = 1'b0;                 // last/current image came from the UART
    reg        res_valid   = 1'b0;
    reg  [3:0] res_pred    = 4'd0;
    reg [31:0] to_cnt      = 32'd0;
    reg  [5:0] tx_idx      = 6'd0;
    reg  [1:0] tx_ph       = 2'd0;

    wire boot_done = iscan_done && scan_done;
    wire sw_changed = !ran_once || (sw_s2 != sel_run);

    // byte n of the 42-byte reply frame
    wire [5:0]  m  = tx_idx - 6'd2;
    wire [31:0] sc = core_scores[32*m[5:2] +: 32];
    reg  [7:0]  tx_byte;
    always @* begin
        if (tx_idx == 6'd0)      tx_byte = 8'h5A;
        else if (tx_idx == 6'd1) tx_byte = {4'd0, res_pred};
        else case (m[1:0])
            2'd0:    tx_byte = sc[31:24];
            2'd1:    tx_byte = sc[23:16];
            2'd2:    tx_byte = sc[15:8];
            default: tx_byte = sc[7:0];
        endcase
    end

    always @(posedge clk) begin
        core_start <= 1'b0;                        // defaults: single-clock strobes
        tx_start   <= 1'b0;
        ld_v       <= 1'b0;
        if (btn_run) run_pending <= 1'b1;

        if (rst) begin
            st          <= T_BOOT;
            ran_once    <= 1'b0;
            run_pending <= 1'b0;
            src_uart    <= 1'b0;
            res_valid   <= 1'b0;
        end else begin
            case (st)
                T_BOOT: if (boot_done)                           // both ROM self-tests finished
                    st <= (rom_ok && img_ok) ? T_IDLE : T_ERR;

                T_ERR: ;                                         // stay: a .mem file did not load

                T_IDLE: begin
                    if (rx_valid && rx_data == 8'hA5) begin      // UART frame header
                        st        <= T_UARTLD;
                        ld_cnt    <= 10'd0;
                        to_cnt    <= 32'd0;
                        src_uart  <= 1'b1;
                        res_valid <= 1'b0;
                    end else if (run_pending || sw_changed) begin // run a stored digit
                        run_pending <= 1'b0;
                        ran_once    <= 1'b1;
                        sel_run     <= sw_s2;
                        ld_cnt      <= 10'd0;
                        src_uart    <= 1'b0;
                        res_valid   <= 1'b0;
                        st          <= T_ROMLD;
                    end
                end

                T_ROMLD: begin                                   // copy 784 pixels ROM -> core RAM
                    ld_v   <= 1'b1;
                    ld_a   <= ld_cnt;
                    ld_cnt <= ld_cnt + 1'b1;
                    if (ld_cnt == 10'd783) st <= T_ROMWAIT;
                end

                T_ROMWAIT: begin                                 // last pixel is written this clock
                    core_start <= 1'b1;
                    st         <= T_RUN;
                end

                T_UARTLD: begin                                  // 784 bytes straight into core RAM
                    if (rx_valid) begin
                        to_cnt <= 32'd0;
                        ld_cnt <= ld_cnt + 1'b1;
                        if (ld_cnt == 10'd783) begin
                            core_start <= 1'b1;
                            st         <= T_RUN;
                        end
                    end else if (to_cnt == UART_TIMEOUT) begin
                        st <= T_IDLE;                            // frame aborted
                    end else begin
                        to_cnt <= to_cnt + 1'b1;
                    end
                end

                T_RUN: if (core_done) begin
                    res_valid <= 1'b1;
                    res_pred  <= core_pred;
                    if (src_uart) begin
                        tx_idx <= 6'd0;
                        tx_ph  <= 2'd0;
                        st     <= T_REPLY;
                    end else begin
                        st <= T_IDLE;
                    end
                end

                T_REPLY: case (tx_ph)                            // stream 42 bytes to the PC
                    2'd0: if (!tx_busy) begin
                        tx_data  <= tx_byte;
                        tx_start <= 1'b1;
                        tx_ph    <= 2'd1;
                    end
                    2'd1: tx_ph <= 2'd2;                         // uart_tx raises busy now
                    default: if (!tx_busy) begin
                        if (tx_idx == 6'd41) begin
                            st <= T_IDLE;
                        end else begin
                            tx_idx <= tx_idx + 1'b1;
                            tx_ph  <= 2'd0;
                        end
                    end
                endcase

                default: st <= T_BOOT;
            endcase
        end
    end

    // ------------------------------------------------------------ display
    localparam [4:0] C_BLANK = 5'd16, C_DASH = 5'd17, C_P = 5'd18, C_L = 5'd19, C_E = 5'd20;

    wire [5:0] idx_t6 = sel_run / 6'd10;            // 0..6
    wire [5:0] idx_o6 = sel_run % 6'd10;            // 0..9
    wire [3:0] idx_t  = idx_t6[3:0];
    wire [3:0] idx_o  = idx_o6[3:0];
    wire       rom_mode = !src_uart;
    wire       boot_err = (st == T_ERR);

    wire [39:0] codes_run = {
        rom_mode  ? {1'b0, idx_t}   : C_DASH,       // digit 7
        rom_mode  ? {1'b0, idx_o}   : C_DASH,       // digit 6
        C_BLANK,                                    // digit 5
        C_L,                                        // digit 4
        rom_mode  ? {1'b0, rom_lbl} : C_DASH,       // digit 3  true label
        C_BLANK,                                    // digit 2
        C_P,                                        // digit 1
        res_valid ? {1'b0, res_pred} : C_DASH       // digit 0  prediction
    };
    wire [39:0] codes_err = {C_BLANK, C_BLANK, C_BLANK, C_BLANK, C_E, C_E, C_E, C_E};

    seg7_driver #(.REFRESH_BITS(REFRESH_BITS)) u_seg (
        .clk(clk), .codes(boot_err ? codes_err : codes_run), .seg(SEG), .an(AN), .dp(DP));

    // ------------------------------------------------------------ LEDs
    reg [26:0] hb = 27'd0;
    always @(posedge clk) hb <= hb + 1'b1;

    wire [9:0] onehot = res_valid ? (10'd1 << res_pred) : 10'd0;
    wire       match  = res_valid && rom_mode && (res_pred == rom_lbl);
    wire [9:0] led_lo = boot_err ? {8'd0, img_ok, rom_ok} : onehot;

    assign LED = {res_valid, match, (st != T_IDLE), src_uart, hb[26], boot_err, led_lo};
endmodule
