"""
Benchmark: Expert-by-expert MoE decode (MOE_E2E) vs batched baseline,
with CUDA Graph enabled.

Usage:
    python bench_moe_decode_e2e.py [checkpoint_path]
"""
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp/.cache/modelscope"
os.environ["ATTN_FP8"] = "1"
os.environ["UNEMBED_FP8"] = "1"

import torch
from tritonllm.utils import get_model_with_checkpoint
from tritonllm.gpt_oss.triton.model import TokenGenerator


def set_e2e_flag(model, enabled: bool):
    """Toggle expert-by-expert scheduling on all MLP blocks."""
    for block in model.block:
        block.mlp._moe_e2e = enabled


def benchmark_e2e(generator, use_e2e, num_steps=200, warmup=50):
    """Benchmark decode TPS with/without expert-by-expert scheduling."""
    set_e2e_flag(generator.model, use_e2e)
    device = generator.device

    with torch.inference_mode():
        for cache in generator.caches:
            cache.reset()

        # Re-capture greedy graph with the new e2e setting
        generator._greedy_graph = torch.cuda.CUDAGraph()
        generator.input_token[0] = 0
        with torch.cuda.graph(generator._greedy_graph):
            generator._greedy_token = generator.model.forward_greedy(
                generator.input_token[None, :], caches=generator.caches
            )

        token = generator._greedy_token.item()
        for _ in range(warmup):
            generator.input_token[0] = token
            generator._greedy_graph.replay()
            token = generator._greedy_token.item()

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.inference_mode():
        for _ in range(num_steps):
            generator.input_token[0] = token
            generator._greedy_graph.replay()
            token = generator._greedy_token.item()
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    tps = num_steps / (elapsed_ms / 1000)
    return tps, elapsed_ms


def main():
    os.environ["CUDA_GRAPH"] = "1"
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "20b"
    checkpoint_path = get_model_with_checkpoint(checkpoint_path)
    device = torch.device("cuda:0")

    print(f"Loading model from {checkpoint_path}...")
    gen = TokenGenerator(checkpoint_path, context=4096, device=device)

    # Baseline: batched (current default)
    print("Benchmarking MOE_E2E=0 (baseline, batched)...")
    tps_off, ms_off = benchmark_e2e(gen, use_e2e=False)
    print(f"  E2E=off:  {tps_off:.1f} TPS  ({ms_off:.0f} ms for 200 steps)")

    # E2E on
    print("Benchmarking MOE_E2E=1 (expert-by-expert)...")
    tps_on, ms_on = benchmark_e2e(gen, use_e2e=True)
    print(f"  E2E=on:   {tps_on:.1f} TPS  ({ms_on:.0f} ms for 200 steps)")

    delta = (tps_on - tps_off) / tps_off * 100
    print(f"\n{'='*60}")
    print(f"  ATTN_FP8=1  UNEMBED_FP8=1  CUDA_GRAPH=1")
    print(f"  E2E=off: {tps_off:.1f} TPS  [baseline]")
    print(f"  E2E=on:  {tps_on:.1f} TPS  ({delta:+.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
