`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// debounce.v - synchroniser + debouncer + rising-edge pulse for a push button
//   'clean' follows 'noisy' only after it has been stable for STABLE_CYCLES clocks
//   (1_000_000 clocks = 10 ms @ 100 MHz). 'rise' is a one-clock pulse on 0 -> 1.
// -----------------------------------------------------------------------------
module debounce #(
    parameter integer STABLE_CYCLES = 1_000_000
)(
    input  wire clk,
    input  wire rst,
    input  wire noisy,
    output reg  clean = 1'b0,
    output wire rise
);
    localparam integer CW = $clog2(STABLE_CYCLES + 1);

    reg          s1 = 1'b0, s2 = 1'b0, clean_d = 1'b0;
    reg [CW-1:0] cnt = {CW{1'b0}};

    always @(posedge clk) begin
        s1      <= noisy;
        s2      <= s1;
        clean_d <= clean;
        if (rst) begin
            cnt   <= {CW{1'b0}};
            clean <= 1'b0;
        end else if (s2 == clean) begin
            cnt <= {CW{1'b0}};
        end else if (cnt == STABLE_CYCLES - 1) begin
            clean <= s2;
            cnt   <= {CW{1'b0}};
        end else begin
            cnt <= cnt + 1'b1;
        end
    end

    assign rise = clean & ~clean_d;
endmodule
