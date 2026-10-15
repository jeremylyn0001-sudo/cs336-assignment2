"""H20 substitute run for the CS336 A2 full-step leaderboard workload.

This is explicitly an H20 measurement, not a B200 leaderboard result. It uses
two ranks, BF16 model parameters, the repository's custom asynchronous DDP, and
its Triton FlashAttention implementation. Optional per-block activation
checkpointing is enabled with --checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from cs336_basics.model import Transformerlm
from cs336_basics.nn_utils import MultiHeadSelfAttention
from cs336_systems.DDP import DDP
from cs336_systems.flashattention_triton import FlashAttention_Triton


def _flash_attention_forward(
    self: MultiHeadSelfAttention,
    x: torch.Tensor,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, sequence_length, _ = x.shape
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)
    q, k, v = (
        rearrange(t, "b s (h d) -> b h s d", h=self.num_heads).contiguous()
        for t in (q, k, v)
    )
    if token_positions is not None:
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

    q = q.reshape(batch * self.num_heads, sequence_length, self.d_k).contiguous()
    k = k.reshape(batch * self.num_heads, sequence_length, self.d_k).contiguous()
    v = v.reshape(batch * self.num_heads, sequence_length, self.d_k).contiguous()
    out = FlashAttention_Triton.apply(q, k, v, True)
    out = out.reshape(batch, self.num_heads, sequence_length, self.d_k)
    out = rearrange(out, "b h s d -> b s (h d)").contiguous()
    return self.output_proj(out)


def _compute_loss(model, inputs, targets, checkpoint_activations, chunk_tokens):
    # The full 32K x 151,936 logits tensor exceeds the H20 memory budget when
    # combined with the 8B model, gradients, and AdamW states. Compute the
    # mathematically identical token-mean CE in chunks and checkpoint each
    # vocabulary projection so only one chunk of logits is live at a time.
    base_model = model.module
    hidden = base_model.token_embeddings(inputs)
    token_positions = torch.arange(hidden.size(1), device=hidden.device)
    for layer in base_model.layers:
        if checkpoint_activations:
            hidden = checkpoint(
                layer, hidden, token_positions, use_reentrant=False
            )
        else:
            hidden = layer(hidden, token_positions)
    hidden = base_model.ln_final(hidden)

    def projection_cross_entropy(hidden_chunk, target_chunk):
        logits = base_model.lm_head(hidden_chunk)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_chunk.reshape(-1),
            reduction="sum",
        )

    total_loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.size(1), chunk_tokens):
        end = min(start + chunk_tokens, hidden.size(1))
        hidden_chunk = hidden[:, start:end, :]
        target_chunk = targets[:, start:end]
        if checkpoint_activations:
            chunk_loss = checkpoint(
                projection_cross_entropy,
                hidden_chunk,
                target_chunk,
                use_reentrant=False,
            )
        else:
            chunk_loss = projection_cross_entropy(hidden_chunk, target_chunk)
        total_loss = total_loss + chunk_loss
    return total_loss / targets.numel()


def _train_step(
    model, optimizer, inputs, targets, checkpoint_activations, chunk_tokens
) -> float:
    optimizer.zero_grad(set_to_none=True)
    # Match the assignment's last-dimension cross entropy; DDP averages local
    # batch-mean gradients across ranks, reproducing the global-batch mean.
    loss = _compute_loss(
        model, inputs, targets, checkpoint_activations, chunk_tokens
    )
    loss.backward()
    model.finish_gradient_synchronization()
    optimizer.step()
    return float(loss.detach().float().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32_768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--d-ff", type=int, default=11_008)
    parser.add_argument("--layers", type=int, default=34)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--loss-chunk-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("leaderboard_h20_result.json"))
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.batch_size % world_size:
        raise ValueError("Global batch size must be divisible by world size")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False

    MultiHeadSelfAttention.forward = _flash_attention_forward
    model = Transformerlm(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        d_ff=args.d_ff,
        rope_theta=10_000.0,
    ).to(device=device, dtype=torch.bfloat16)
    model = DDP(model)
    optimizer = AdamW(model.parameters())

    generator = torch.Generator().manual_seed(4321)
    all_inputs = torch.randint(
        args.vocab_size,
        (args.batch_size, args.seq_len),
        generator=generator,
        dtype=torch.long,
    )
    all_targets = torch.randint(
        args.vocab_size,
        (args.batch_size, args.seq_len),
        generator=generator,
        dtype=torch.long,
    )
    local_batch = args.batch_size // world_size
    begin = rank * local_batch
    end = begin + local_batch
    inputs = all_inputs[begin:end].to(device)
    targets = all_targets[begin:end].to(device)

    torch.cuda.synchronize(device)
    memory_after_init = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    first_start = time.perf_counter()
    first_loss = _train_step(
        model,
        optimizer,
        inputs,
        targets,
        args.checkpoint,
        args.loss_chunk_tokens,
    )
    torch.cuda.synchronize(device)
    first_step_seconds = time.perf_counter() - first_start
    peak_first_step = torch.cuda.max_memory_allocated(device)

    for _ in range(args.warmups):
        _train_step(
            model,
            optimizer,
            inputs,
            targets,
            args.checkpoint,
            args.loss_chunk_tokens,
        )
    torch.cuda.synchronize(device)

    timings = []
    for _ in range(args.trials):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        _train_step(
            model,
            optimizer,
            inputs,
            targets,
            args.checkpoint,
            args.loss_chunk_tokens,
        )
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)

    record = {
        "hardware": torch.cuda.get_device_name(device),
        "rank": rank,
        "world_size": world_size,
        "config": vars(args) | {"output": str(args.output)},
        "first_step_seconds_including_triton_jit": first_step_seconds,
        "first_step_loss": first_loss,
        "memory_after_init_bytes": memory_after_init,
        "peak_first_step_bytes": peak_first_step,
        "steady_step_seconds": timings,
        "mean_step_seconds": statistics.mean(timings),
    }
    records = [None] * world_size
    dist.all_gather_object(records, record)
    if rank == 0:
        args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(records, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
