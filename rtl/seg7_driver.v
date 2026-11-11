`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// seg7_driver.v - time-multiplexed driver for the Nexys A7 8-digit 7-segment display
//   codes : eight 5-bit symbols, digit 0 (rightmost) = codes[4:0], digit 7 = codes[39:35]
//             0..9 = decimal digit   16 = blank   17 = '-'   18 = 'P'   19 = 'L'   20 = 'E'
//   seg   : {CG,CF,CE,CD,CC,CB,CA}, ACTIVE LOW   (seg[0] = segment A)
//   an    : AN7..AN0, ACTIVE LOW
//   Each digit is lit for 2^REFRESH_BITS clocks (2^16 = 0.66 ms -> 190 Hz full refresh).
// -----------------------------------------------------------------------------
module seg7_driver #(
    parameter integer REFRESH_BITS = 16
)(
    input  wire        clk,
    input  wire [39:0] codes,
    output reg  [6:0]  seg = 7'h7F,
    output reg  [7:0]  an  = 8'hFF,
    output wire        dp
);
    reg [REFRESH_BITS+2:0] cnt = 0;
    always @(posedge clk) cnt <= cnt + 1'b1;

    wire [2:0] sel  = cnt[REFRESH_BITS+2 -: 3];
    wire [4:0] code = codes[5*sel +: 5];

    reg [6:0] pat;                                   //  gfedcba, 0 = segment on
    always @* begin
        case (code)
            5'd0:    pat = 7'b1000000;
            5'd1:    pat = 7'b1111001;
            5'd2:    pat = 7'b0100100;
            5'd3:    pat = 7'b0110000;
            5'd4:    pat = 7'b0011001;
            5'd5:    pat = 7'b0010010;
            5'd6:    pat = 7'b0000010;
            5'd7:    pat = 7'b1111000;
            5'd8:    pat = 7'b0000000;
            5'd9:    pat = 7'b0010000;
            5'd17:   pat = 7'b0111111;               // '-'
            5'd18:   pat = 7'b0001100;               // 'P'
            5'd19:   pat = 7'b1000111;               // 'L'
            5'd20:   pat = 7'b0000110;               // 'E'
            default: pat = 7'b1111111;               // blank
        endcase
    end

    always @(posedge clk) begin                      // register both so seg/an change together
        seg <= pat;
        an  <= ~(8'b0000_0001 << sel);
    end

    assign dp = 1'b1;                                // decimal point off (active low)
endmodule
