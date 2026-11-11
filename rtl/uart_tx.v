`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// uart_tx.v - 8N1 UART transmitter (LSB first)
//   * pulse 'start' for one clock while 'busy' = 0; 'data' is captured on that clock.
//   * 'busy' goes high on the clock *after* 'start' and drops when the stop bit ends.
// -----------------------------------------------------------------------------
module uart_tx #(
    parameter integer CLKS_PER_BIT = 868            // 100 MHz / 115200 baud
)(
    input  wire       clk,
    input  wire       rst,                          // synchronous, active high
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx   = 1'b1,
    output reg        busy = 1'b0
);
    localparam integer CW = $clog2(CLKS_PER_BIT + 1);

    reg [CW-1:0] cnt   = {CW{1'b0}};
    reg [3:0]    bit_i = 4'd0;                      // 0 = start bit, 1..8 = data, 9 = stop
    reg [9:0]    frame = 10'h3FF;                   // {stop, data[7:0], start}

    always @(posedge clk) begin
        if (rst) begin
            tx    <= 1'b1;
            busy  <= 1'b0;
            cnt   <= {CW{1'b0}};
            bit_i <= 4'd0;
        end else if (!busy) begin
            if (start) begin
                frame <= {1'b1, data, 1'b0};
                tx    <= 1'b0;                      // start bit
                busy  <= 1'b1;
                cnt   <= {CW{1'b0}};
                bit_i <= 4'd0;
            end
        end else if (cnt == CLKS_PER_BIT - 1) begin
            cnt <= {CW{1'b0}};
            if (bit_i == 4'd9) begin
                busy <= 1'b0;                       // stop bit has been on the wire for a full period
            end else begin
                tx    <= frame[bit_i + 4'd1];
                bit_i <= bit_i + 1'b1;
            end
        end else begin
            cnt <= cnt + 1'b1;
        end
    end
endmodule
