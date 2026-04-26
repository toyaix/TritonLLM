"""
Benchmark MoE decode block_k: 128 vs 256 (default) vs 512, with CUDA Graph enabled.
Re-captures the graph for each block_k value so we don't need to reload the model.
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


def recapture_and_benchmark(generator, block_k_val, num_steps=200, warmup=50):
    """Re-capture CUDA Graph with new block_k, warm up, then measure."""
    if block_k_val is not None:
        os.environ["MOE_BLOCK_K"] = str(block_k_val)
    elif "MOE_BLOCK_K" in os.environ:
        del os.environ["MOE_BLOCK_K"]

    device = generator.device

    with torch.inference_mode():
        for cache in generator.caches:
            cache.reset()

        # Re-capture the greedy graph with the new block_k constraint
        generator._greedy_graph = torch.cuda.CUDAGraph()
        generator.input_token[0] = 0
        with torch.cuda.graph(generator._greedy_graph):
            generator._greedy_token = generator.model.forward_greedy(
                generator.input_token[None, :], caches=generator.caches
            )

        # Run warmup through the graph
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
    os.environ["CUDA_GRAPH"] = "1"  # ensure graph is enabled

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "20b"
    checkpoint_path = get_model_with_checkpoint(checkpoint_path)
    device = torch.device("cuda:0")

    # Load model once with CUDA Graph
    print(f"Loading model from {checkpoint_path}...")
    gen = TokenGenerator(checkpoint_path, context=4096, device=device)

    results = {}
    for bk in [256, 512, 128]:
        label = f"block_k={bk}"
        print(f"Capturing graph + benchmarking {label} (warmup 50, measure 200)...")
        tps, ms = recapture_and_benchmark(gen, bk)
        results[bk] = (tps, ms)
        print(f"  {label}:  {tps:.1f} TPS  ({ms:.0f} ms for 200 steps)")

    # Also test without MOE_BLOCK_K at all to confirm baseline
    if "MOE_BLOCK_K" in os.environ:
        del os.environ["MOE_BLOCK_K"]
    print(f"Capturing graph + benchmarking (no constraint override)...")
    tps_none, ms_none = recapture_and_benchmark(gen, None)
    print(f"  no override:  {tps_none:.1f} TPS  ({ms_none:.0f} ms for 200 steps)")

    print(f"\n{'='*60}")
    baseline_tps = results[256][0]
    print(f"  CUDA Graph: ON  |  ATTN_FP8: ON  |  UNEMBED_FP8: ON")
    print(f"  {'-'*50}")
    for bk in [128, 256, 512]:
        tps, ms = results[bk]
        delta = (tps - baseline_tps) / baseline_tps * 100
        marker = " [baseline]" if bk == 256 else ""
        print(f"  block_k={bk:>3}:  {tps:.1f} TPS  ({ms:.0f} ms)  {delta:+.1f}%{marker}")
    print(f"  no override: {tps_none:.1f} TPS  (sanity check)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
