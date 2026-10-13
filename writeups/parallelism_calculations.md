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
