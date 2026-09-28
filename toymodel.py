"""Minimal ToyModel mixed-precision experiment for Assignment 2.

The experiment intentionally performs exactly one forward pass, loss
calculation, and backward pass.  Model parameters remain FP32; only the
operations selected by CUDA autocast use FP16.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class ToyModel(nn.Module):
    """The exact minimal chain required by the assignment."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        self.fc1_output = x
        x = self.relu(x)
        self.relu_output = x
        x = self.ln(x)
        self.ln_output = x
        x = self.fc2(x)
        self.logits = x
        return x


@dataclass(frozen=True)
class DtypeReport:
    """Dtypes observed during one mixed-precision forward/backward pass."""

    parameter: dict[str, torch.dtype]
    fc1_output: torch.dtype
    relu_output: torch.dtype
    layer_norm_output: torch.dtype
    logits: torch.dtype
    loss: torch.dtype
    gradients: dict[str, torch.dtype]

    def as_dict(self) -> dict[str, object]:
        return {
            "parameters_in_autocast": self.parameter,
            "fc1_output": self.fc1_output,
            "relu_output": self.relu_output,
            "layer_norm_output": self.layer_norm_output,
            "logits": self.logits,
            "loss": self.loss,
            "gradients_after_backward": self.gradients,
        }


def run_once(
    in_features: int = 4,
    out_features: int = 3,
    batch_size: int = 8,
    device: str = "cuda",
) -> DtypeReport:
    """Run one FP32-parameter, CUDA-FP16-autocast forward/backward pass."""

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This assignment experiment requires a CUDA GPU.")

    torch.manual_seed(0)
    model = ToyModel(in_features, out_features).to(device=device, dtype=torch.float32)
    inputs = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)
    targets = torch.randint(0, out_features, (batch_size,), device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        parameter_dtypes = {name: parameter.dtype for name, parameter in model.named_parameters()}
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss_dtype = loss.dtype

    loss.backward()
    gradient_dtypes = {
        name: parameter.grad.dtype
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    return DtypeReport(
        parameter=parameter_dtypes,
        fc1_output=model.fc1_output.dtype,
        relu_output=model.relu_output.dtype,
        layer_norm_output=model.ln_output.dtype,
        logits=logits.dtype,
        loss=loss_dtype,
        gradients=gradient_dtypes,
    )


def print_report(report: DtypeReport) -> None:
    """Print a stable, human-readable report for submission/debugging."""

    print("Mixed precision: CUDA autocast dtype=torch.float16")
    print("Model parameter dtypes inside autocast:")
    for name, dtype in report.parameter.items():
        print(f"  {name}: {dtype}")
    print(f"fc1 output: {report.fc1_output}")
    print(f"ReLU output: {report.relu_output}")
    print(f"LayerNorm output: {report.layer_norm_output}")
    print(f"fc2 output / logits: {report.logits}")
    print(f"loss: {report.loss}")
    print("Gradient dtypes after backward:")
    for name, dtype in report.gradients.items():
        print(f"  {name}.grad: {dtype}")


THEORY_ANSWER = (
    "LayerNorm is sensitive in the mean and variance reductions, the subtraction "
    "of the mean, and the reciprocal square root of variance plus epsilon: low "
    "precision can lose small differences or overflow/underflow. It is therefore "
    "usually kept in FP32 (or uses FP32 accumulations) to preserve stable statistics "
    "and gradients. BF16 has the same exponent range as FP32, so it is much less "
    "prone to FP16-style overflow/underflow; however, its lower mantissa precision "
    "can still make sensitive reductions less accurate, so frameworks may still "
    "compute LayerNorm statistics in FP32 even though special handling is less "
    "critical than for FP16."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-features", type=int, default=4)
    parser.add_argument("--out-features", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    report = run_once(args.in_features, args.out_features, args.batch_size)
    print_report(report)
    print("\nTheory answer:")
    print(THEORY_ANSWER)


if __name__ == "__main__":
    main()