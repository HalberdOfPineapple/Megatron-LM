"""Profile dispatch / compute / combine latency of the MCore MoELayer.

Builds a perfectly balanced routing map (each expert gets exactly
num_tokens * topk / num_experts tokens) so that the measurement
reflects pure kernel + communication cost without load-imbalance noise.

Usage (via the companion shell script):
    torchrun --nproc_per_node=8 examples/moe_profile/profile_moe_latency.py <args>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE layer latency profiler")

    p.add_argument("--num-tokens", type=int, default=4096,
                    help="Total tokens per rank (seq_len * micro_batch)")
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--moe-ffn-hidden-size", type=int, default=960)
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--num-attention-heads", type=int, default=32)

    p.add_argument("--ep-size", type=int, default=None,
                    help="Expert-parallel size (default: world_size)")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--dispatcher", choices=["alltoall", "allgather"], default="alltoall")
    p.add_argument("--grouped-gemm", action="store_true", default=True)
    p.add_argument("--no-grouped-gemm", dest="grouped_gemm", action="store_false")

    p.add_argument("--warmup-iters", type=int, default=200,
                    help="Fixed warmup iterations (skipped before measurement). "
                         "Set to 0 to use adaptive warmup instead.")
    p.add_argument("--adaptive-warmup", action="store_true", default=False,
                    help="Auto-detect when latency stabilizes instead of "
                         "using a fixed warmup count.")
    p.add_argument("--measure-iters", type=int, default=100)
    p.add_argument("--output-dir", type=str, default=None,
                    help="Where to save the latency plot (default: script dir)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _build_config(args: argparse.Namespace) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=args.hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_moe_experts=args.num_experts,
        moe_router_topk=args.topk,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type=args.dispatcher,
        moe_grouped_gemm=args.grouped_gemm,
        use_cpu_initialization=True,
        add_bias_linear=False,
        sequence_parallel=args.tp_size > 1,
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep_size,
        expert_tensor_parallel_size=1,
        params_dtype=_resolve_dtype(args.dtype),
        bf16=(args.dtype == "bf16"),
    )


def _build_layer(config: TransformerConfig, args: argparse.Namespace) -> MoELayer:
    submodules = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.grouped_gemm,
    ).mlp.submodules
    layer = MoELayer(config=config, submodules=submodules)
    layer = layer.cuda().to(dtype=_resolve_dtype(args.dtype))
    layer.eval()
    return layer


def _build_balanced_routing(
    num_tokens: int, num_experts: int, topk: int, device: torch.device,
) -> torch.Tensor:
    """Build a perfectly balanced routing map.

    Returns a boolean tensor of shape [num_tokens, num_experts] where each
    token is assigned to exactly `topk` experts, and every expert receives
    exactly `num_tokens * topk // num_experts` tokens.
    """
    slots_per_expert = num_tokens * topk // num_experts
    assert slots_per_expert * num_experts == num_tokens * topk, (
        f"num_tokens({num_tokens}) * topk({topk}) must be divisible by "
        f"num_experts({num_experts})"
    )

    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device=device)
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device="cpu")
    token_counts = torch.zeros(num_tokens, dtype=torch.long, device="cpu")

    expert_ptr = 0
    for tok in range(num_tokens):
        assigned = 0
        while assigned < topk:
            eidx = expert_ptr % num_experts
            if expert_counts[eidx] < slots_per_expert:
                routing_map[tok, eidx] = True
                expert_counts[eidx] += 1
                token_counts[tok] += 1
                assigned += 1
            expert_ptr += 1
    return routing_map


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _run_one_iter(layer, hidden, routing_map, probs, events, idx):
    """Execute one MoE iteration, recording CUDA events at index *idx*."""
    iter_starts, dispatch_starts, dispatch_ends = events[:3]
    compute_starts, compute_ends, combine_starts, combine_ends, iter_ends = events[3:]

    torch.cuda.synchronize()

    h, p = layer.token_dispatcher.dispatch_preprocess(hidden, routing_map, probs)

    iter_starts[idx].record()

    dispatch_starts[idx].record()
    h_disp, p_disp = layer.dispatch(h, p)
    dispatch_ends[idx].record()

    compute_starts[idx].record()
    out, _ = layer.routed_experts_compute(h_disp, p_disp)
    compute_ends[idx].record()

    combine_starts[idx].record()
    out = layer.combine(out)
    combine_ends[idx].record()

    layer.token_dispatcher.combine_postprocess(out)

    iter_ends[idx].record()


def _adaptive_warmup(layer, hidden, routing_map, probs, *, window: int = 40, cv_threshold: float = 0.05, min_iters: int = 50, max_iters: int = 1000):
    """Run iterations until total-latency coefficient-of-variation (CV) over the
    last *window* iterations drops below *cv_threshold*, or *max_iters* is reached.

    Returns the number of warmup iterations actually executed.
    """
    import math
    rank = dist.get_rank() if dist.is_initialized() else 0
    recent: list[float] = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for n in range(1, max_iters + 1):
        torch.cuda.synchronize()
        h, p = layer.token_dispatcher.dispatch_preprocess(hidden, routing_map, probs)
        start_ev.record()
        h_disp, p_disp = layer.dispatch(h, p)
        out, _ = layer.routed_experts_compute(h_disp, p_disp)
        out = layer.combine(out)
        layer.token_dispatcher.combine_postprocess(out)
        end_ev.record()
        torch.cuda.synchronize()

        recent.append(start_ev.elapsed_time(end_ev))
        if len(recent) > window:
            recent.pop(0)

        if n >= min_iters and len(recent) == window:
            mean = sum(recent) / window
            std = math.sqrt(sum((x - mean) ** 2 for x in recent) / window)
            cv = std / mean if mean > 0 else 0.0
            if cv < cv_threshold:
                if rank == 0:
                    print(f"[adaptive warmup] converged after {n} iters "
                          f"(CV={cv:.4f} < {cv_threshold})")
                return n

    if rank == 0:
        print(f"[adaptive warmup] reached max {max_iters} iters without converging")
    return max_iters


def run_benchmark(layer: MoELayer, args: argparse.Namespace):
    device = torch.device("cuda")
    dtype = _resolve_dtype(args.dtype)
    num_tokens = args.num_tokens

    hidden = torch.randn(num_tokens, 1, args.hidden_size, device=device, dtype=dtype)

    routing_map = _build_balanced_routing(num_tokens, args.num_experts, args.topk, device)
    probs = torch.zeros(num_tokens, args.num_experts, device=device, dtype=torch.float32)
    probs[routing_map] = 1.0 / args.topk

    # ---- Warmup phase ----
    if args.adaptive_warmup:
        warmup_done = _adaptive_warmup(layer, hidden, routing_map, probs)
    else:
        warmup_done = args.warmup_iters
        for _ in range(warmup_done):
            torch.cuda.synchronize()
            h, p = layer.token_dispatcher.dispatch_preprocess(hidden, routing_map, probs)
            h_disp, p_disp = layer.dispatch(h, p)
            out, _ = layer.routed_experts_compute(h_disp, p_disp)
            out = layer.combine(out)
            layer.token_dispatcher.combine_postprocess(out)
        torch.cuda.synchronize()

    # ---- Measurement phase ----
    measure = args.measure_iters
    iter_starts      = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    dispatch_starts  = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    dispatch_ends    = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    compute_starts   = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    compute_ends     = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    combine_starts   = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    combine_ends     = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    iter_ends        = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]

    events = (iter_starts, dispatch_starts, dispatch_ends,
              compute_starts, compute_ends, combine_starts, combine_ends, iter_ends)

    for i in range(measure):
        _run_one_iter(layer, hidden, routing_map, probs, events, i)

    # ---- Final sync ----
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # ---- Collect timings ----
    dispatch_ms = [dispatch_starts[i].elapsed_time(dispatch_ends[i]) for i in range(measure)]
    compute_ms  = [compute_starts[i].elapsed_time(compute_ends[i])  for i in range(measure)]
    combine_ms  = [combine_starts[i].elapsed_time(combine_ends[i])  for i in range(measure)]
    total_ms    = [iter_starts[i].elapsed_time(iter_ends[i])        for i in range(measure)]

    return dispatch_ms, compute_ms, combine_ms, total_ms


# ---------------------------------------------------------------------------
# Plotting (rank 0 only)
# ---------------------------------------------------------------------------

def _percentile(vals: list[float], pct: float) -> float:
    s = sorted(vals)
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def save_plot(
    dispatch_ms: list[float],
    compute_ms: list[float],
    combine_ms: list[float],
    total_ms: list[float],
    output_dir: str,
    args: argparse.Namespace,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = list(range(len(dispatch_ms)))

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    phase_titles = [
        "Dispatch (All-to-All)",
        "Expert Compute (Grouped GEMM)",
        "Combine (All-to-All)",
    ]
    phase_data = [dispatch_ms, compute_ms, combine_ms]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for ax, title, vals, color in zip(axes[:3], phase_titles, phase_data, colors):
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        ax.plot(iters, vals, linewidth=0.8, color=color, alpha=0.7)
        ax.axhline(p50, linestyle="--", linewidth=1.2, color="black",
                    label=f"P50={p50:.3f} ms")
        ax.axhline(p95, linestyle=":", linewidth=1.0, color="red",
                    label=f"P95={p95:.3f} ms")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    t_p50 = _percentile(total_ms, 50)
    t_p95 = _percentile(total_ms, 95)
    axes[3].plot(iters, total_ms, linewidth=0.8, color="#E91E63", alpha=0.7)
    axes[3].axhline(t_p50, linestyle="--", linewidth=1.2, color="black",
                     label=f"P50={t_p50:.3f} ms")
    axes[3].axhline(t_p95, linestyle=":", linewidth=1.0, color="red",
                     label=f"P95={t_p95:.3f} ms")
    axes[3].set_ylabel("Latency (ms)")
    axes[3].set_title("Total (dispatch + compute + combine + postprocess)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("Iteration")

    fig.suptitle(
        f"MoE Layer Latency — {args.num_experts}E top-{args.topk}, "
        f"{args.num_tokens} tokens/rank, hidden={args.hidden_size}, "
        f"ffn={args.moe_ffn_hidden_size}, ep={args.ep_size}",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "moe_latency.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[rank 0] Latency plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.ep_size is None:
        args.ep_size = world_size

    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent)

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    ps.initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep_size,
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    config = _build_config(args)
    layer = _build_layer(config, args)

    dispatch_ms, compute_ms, combine_ms, total_ms = run_benchmark(layer, args)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        def _report(name: str, vals: list[float]):
            p50 = _percentile(vals, 50)
            p95 = _percentile(vals, 95)
            p99 = _percentile(vals, 99)
            mean = sum(vals) / len(vals)
            print(f"{name:<18} {p50:>10.3f} {p95:>10.3f} {p99:>10.3f} {mean:>10.3f}")

        print(f"\n{'Phase':<18} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'Mean (ms)':>10}")
        print("-" * 68)
        _report("Dispatch", dispatch_ms)
        _report("Compute", compute_ms)
        _report("Combine", combine_ms)
        _report("Total", total_ms)

        save_plot(dispatch_ms, compute_ms, combine_ms, total_ms, args.output_dir, args)

    if dist.is_initialized():
        ps.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
