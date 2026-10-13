# A2 Parallelism Calculations

## 8.1 Alternate Ring All-Reduce

Let `N` be the number of devices, `D` the size in bytes of each device's full input tensor, and `B` each device's egress bandwidth.

The alternate algorithm performs `N - 1` ring steps. In each step, every device sends one **full `D`-byte tensor** to its next neighbor and receives one full tensor from its previous neighbor. All devices send concurrently, so one step takes `D / B` seconds under the assignment's bandwidth model. Therefore,

\[
T_{\text{alternate}} = (N - 1)\frac{D}{B}.
\]

**Deliverable:** `T = (N - 1)D/B`.

**Justification:** The algorithm performs `N - 1` concurrent ring exchanges, and each device transmits a full `D`-byte tensor in every exchange.

### Why this differs from the standard ring All-Reduce

The standard algorithm in the handout first performs ring reduce-scatter and then ring all-gather. It splits the tensor into `N` chunks, so each device sends a `D/N`-byte chunk in each of `N - 1` steps in each phase:

\[
T_{\text{standard}} = 2(N - 1)\frac{D/N}{B}
                     = 2\frac{N - 1}{N}\frac{D}{B}.
\]

The `/N` appears because each ring step transfers a chunk of the tensor. In the alternate problem, each step transfers a full tensor, so there is no `/N` factor. For `N = 4` and `D = 120 MB`, the alternate algorithm sends `3 × 120 = 360 MB` per device, while the standard algorithm sends `3 × 30 MB` in each of its two phases, or `180 MB` total per device.

Here `B` is per-device egress bandwidth; startup latency and contention beyond the idealized model are ignored, as in the handout.

## 8.2 Data Parallel Calculations

Consider the FFN in the handout:

\[
X_1 = XW_1,\quad X_2 = XW_2,\quad A = \sigma(X_1)\odot X_2,\quad Y = AW_3,
\]

where `X` has shape `(B, H)` and `W_1`, `W_2`, and `W_3` each contain `H H_FF` elements. With `N_DP` data-parallel devices, each device processes a local batch of `B/N_DP` while keeping full copies of the weights.

### (a) Backward-pass FLOPs per device

The backward pass has six matrix multiplications:

1. `dA = dY W_3^T`
2. `dX_1 W_1^T`, the first contribution to `dX`
3. `dX_2 W_2^T`, the second contribution to `dX`
4. `dW_3 = A^T dY`
5. `dW_2 = X^T dX_2`
6. `dW_1 = X^T dX_1`

Each multiplication costs `2(B/N_DP) H H_FF` FLOPs. Elementwise activation derivatives, Hadamard products, and the sum forming `dX` are ignored, as requested. Thus,

\[
F_{\text{backward, per device}}
= 6\cdot 2\frac{B}{N_{\text{DP}}}HH_{\text{FF}}
= \frac{12BHH_{\text{FF}}}{N_{\text{DP}}}.
\]

**Deliverable:** `12 B H H_FF / N_DP` FLOPs per device.

**Justification:** The FFN backward pass performs six matrix multiplications, each over the device's local batch of `B/N_DP` examples.

### (b) Backward-pass communication time

After backward, each device has partial gradients for the three replicated weight matrices. Together they contain `3 H H_FF` FP16 values, or

\[
D_{\text{grad}} = 3HH_{\text{FF}}\cdot 2 = 6HH_{\text{FF}}\text{ bytes}.
\]

Use the **standard** ring All-Reduce from Section 8.1: reduce-scatter followed by all-gather. For `N_DP` devices and egress bandwidth `W`, its idealized time for these gradients is

\[
T_{\text{comm,bwd}}
= 2\frac{N_{\text{DP}}-1}{N_{\text{DP}}}\frac{D_{\text{grad}}}{W}
= \frac{12(N_{\text{DP}}-1)HH_{\text{FF}}}{N_{\text{DP}}W}.
\]

**Deliverable:** `12 (N_DP - 1) H H_FF / (N_DP W)` seconds.

**Justification:** All-Reduce communicates the three FP16 weight-gradient tensors, whose combined size is `6 H H_FF` bytes, using the standard ring's reduce-scatter and all-gather phases.

### (c) Largest compute-bound data-parallel degree

The local backward compute time is

\[
T_{\text{compute,bwd}} = \frac{F_{\text{backward, per device}}}{C}
= \frac{12BHH_{\text{FF}}}{N_{\text{DP}}C},
\]

where `C` is the device's compute throughput in FLOPs/s. Since computation and communication may overlap, the backward pass remains compute-bound when `T_comm,bwd \le T_compute,bwd`:

\[
\frac{12(N_{\text{DP}}-1)HH_{\text{FF}}}{N_{\text{DP}}W}
\le
\frac{12BHH_{\text{FF}}}{N_{\text{DP}}C}
\quad\Longrightarrow\quad
N_{\text{DP}} \le 1+\frac{BW}{C}.
\]

**Deliverable:** `N_DP <= 1 + B W / C`.

**Justification:** The communication time grows with the ring's `N_DP - 1` transfers while the local compute time shrinks with the per-device batch `B/N_DP`; setting communication no larger than computation gives the threshold above.

This calculation uses the standard ring All-Reduce, not the alternate full-tensor algorithm in Problem 8.1. It also ignores communication startup latency and assumes each device can use its egress bandwidth concurrently, matching the handout's idealized model.
