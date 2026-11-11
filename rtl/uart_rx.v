`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// uart_rx.v - 8N1 UART receiver (LSB first)
//   * 'valid' is a one-clock pulse, 'data' stays stable until the next byte.
//   * the bit is sampled in the middle of its period; a bad stop bit drops the byte.
// -----------------------------------------------------------------------------
module uart_rx #(
    parameter integer CLKS_PER_BIT = 868            // 100 MHz / 115200 baud
)(
    input  wire       clk,
    input  wire       rst,                          // synchronous, active high
    input  wire       rx,                           // asynchronous serial input, idle = 1
    output reg  [7:0] data  = 8'd0,
    output reg        valid = 1'b0
);
    localparam integer CW = $clog2(CLKS_PER_BIT + 1);

    // 2-FF synchroniser for the asynchronous input
    reg rx_m = 1'b1, rx_s = 1'b1;
    always @(posedge clk) begin
        rx_m <= rx;
        rx_s <= rx_m;
    end

    localparam [1:0] ST_IDLE = 2'd0, ST_START = 2'd1, ST_DATA = 2'd2, ST_STOP = 2'd3;
    reg [1:0]    st    = ST_IDLE;
    reg [CW-1:0] cnt   = {CW{1'b0}};
    reg [2:0]    bit_i = 3'd0;
    reg [7:0]    sh    = 8'd0;

    always @(posedge clk) begin
        valid <= 1'b0;
        if (rst) begin
            st    <= ST_IDLE;
            cnt   <= {CW{1'b0}};
            bit_i <= 3'd0;
        end else begin
            case (st)
                ST_IDLE: if (!rx_s) begin                      // falling edge = start bit
                    st  <= ST_START;
                    cnt <= {CW{1'b0}};
                end

                ST_START: if (cnt == (CLKS_PER_BIT / 2) - 1) begin   // middle of start bit
                    cnt <= {CW{1'b0}};
                    if (!rx_s) begin
                        st    <= ST_DATA;
                        bit_i <= 3'd0;
                    end else begin
                        st <= ST_IDLE;                         // glitch, not a real start bit
                    end
                end else begin
                    cnt <= cnt + 1'b1;
                end

                ST_DATA: if (cnt == CLKS_PER_BIT - 1) begin    // middle of data bit
                    cnt <= {CW{1'b0}};
                    sh  <= {rx_s, sh[7:1]};
                    if (bit_i == 3'd7) st <= ST_STOP;
                    else               bit_i <= bit_i + 1'b1;
                end else begin
                    cnt <= cnt + 1'b1;
                end

                ST_STOP: if (cnt == CLKS_PER_BIT - 1) begin    // middle of stop bit
                    cnt <= {CW{1'b0}};
                    st  <= ST_IDLE;
                    if (rx_s) begin
                        data  <= sh;
                        valid <= 1'b1;
                    end
                end else begin
                    cnt <= cnt + 1'b1;
                end
            endcase
        end
    end
endmodule
