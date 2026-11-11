# Quantized MNIST CNN on Nexys A7-100T FPGA

This project implements a fully quantized (int8) and pruned (80% sparse) Convolutional Neural Network on a Xilinx Artix-7 FPGA (Nexys A7-100T). It performs integer-only inference to classify 28x28 MNIST digits in 1.72 ms.

## Prerequisites

*   **Software:** Python 3.x, Vivado (2019.1 or newer recommended).
*   **Hardware:** Nexys A7-100T FPGA board, Micro-USB cable.
*   **Python Packages:** `pip install numpy pyserial pillow`

---

## Step 1: Software Workflow (Python)

Before opening Vivado, you must generate the necessary memory (`.mem`) and Verilog parameter (`.v`) files using the provided Python scripts. 

1. Open your terminal or PowerShell and navigate to the project root directory:
   ```bash
   cd path/to/nexys_a7_mnist_cnn
   ```
2. Download the MNIST dataset and convert it into the required `.npz` format:
   ```bash
   python python/get_mnist.py
   ```
3. Quantize the model and export the hardware files to the `rtl/` directory:
   ```bash
   python python/quantize_export.py
   ```
   *(Note: This step generates `mnist_w.mem`, `mnist_d.mem`, `mnist_img.mem`, and parameter files needed for synthesis).*

---

## Step 2: Hardware Workflow (Vivado)

You can build the project using the automated batch script or manually through the Vivado GUI.

### Option A: Automated Build (Windows)
1. Double-click the `RUN_ME.bat` script in the root directory. 
2. It will automatically locate Vivado, build the bitstream in a short-path directory (to avoid Windows path length errors), and program the board.

### Option B: Manual GUI Build
1. Open Vivado and create a new RTL project. Select the part `xc7a100tcsg324-1`.
2. **Add Design Sources:** Add *every* file inside the `rtl/` folder. **Crucially, you must include the `.mem` files alongside the `.v` files.**
3. **Add Constraints:** Add `constraints/nexys_a7_100t.xdc`.
4. **Set Top Module:** Ensure `mnist_top` is set as the top module.
5. Click **Generate Bitstream**. Once completed, open the Hardware Manager and program your board.

---

## Step 3: Running on the Board

Once programmed, the FPGA operates in a standalone, PC-independent mode. 64 MNIST test digits are pre-loaded into the board's Block RAM.

*   **Select an Image:** Use the first six slide switches (`SW[5:0]`) to cycle through the 64 stored test images. 
*   **Read the Display:** The 7-segment display shows the real-time inference result. 
    *   *Example:* `10 L1 P1` means you are looking at image index `10`. The true ground label (L) is `1`, and the FPGA's prediction (P) is `1`.
*   **Read the LEDs:** 
    *   `LED[9:0]`: One-hot encoded prediction (e.g., if the network predicts '1', the 2nd LED is illuminated).
    *   `LED[11]`: System heartbeat (blinking).
    *   `LED[14]`: Illuminates if the prediction is correct.
    *   `LED[15]`: Illuminates when the inference result is valid.
*   **Buttons:** Press **BTNC** (center button) to re-run classification. Press **CPU RESET** (red button) to reset the system.

---

## Step 4: Testing via UART (Optional)

You can send your own custom digits to the FPGA over the USB-UART cable (115200 baud).

1. Send the 64 demo digits over UART to verify PC communication:
   ```bash
   python python/uart_demo.py --port COM5
   ```
2. Send your own drawn image (white digit on black background):
   ```bash
   python python/uart_demo.py --port COM5 --png my_digit.png
   ```
   *(Note: Change `COM5` to the actual port assigned to your Nexys board in Device Manager).*