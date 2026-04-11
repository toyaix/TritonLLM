"""
MoE decode matmul micro-benchmark.
Tests W1 (gather+SwiGLU) and W2 (scatter) separately with different
split_k / fused_scatter combinations on RTX 5090.

Uses CUDA graph replay for timing to match the actual inference path
(model uses CUDA graphs → Python overhead is not included in GPU time).

Usage:
    python examples/bench_moe.py
"""
import torch
from tritonllm.utils import init_env

init_env()

from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.matmul_ogs_details.opt_flags import (
    update_opt_flags_constraints, reset_opt_flags_constraints,
)
from triton_kernels.routing import routing
from triton_kernels.numerics import InFlexData
from triton_kernels.swiglu import swiglu_fn
from tritonllm.gpt_oss.triton.moe import quantize_mx4, routing_decode_fast


# --------------------------------------------------------------------------
# Model config (matches gpt-oss-20b)
# --------------------------------------------------------------------------
N_EXPTS_TOT = 128
N_EXPTS_ACT = 4
HIDDEN      = 2880
INTER       = 2880          # per-expert; SwiGLU doubles to 2×INTER
SWIGLU_LIM  = 7.0
DEVICE      = torch.device("cuda")
WARMUP      = 30            # iterations before capture (includes JIT compile)
ITERS       = 500           # graph replay iterations for timing


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_weights(k_in, n_out, n_experts=N_EXPTS_TOT):
    raw = torch.randn(n_experts, k_in, n_out, device=DEVICE, dtype=torch.bfloat16) * 0.02
    return quantize_mx4(raw)


def make_bias(n_out, n_experts=N_EXPTS_TOT):
    return torch.randn(n_experts, n_out, device=DEVICE, dtype=torch.bfloat16) * 0.01


def bench_graph(fn, warmup=WARMUP, iters=ITERS):
    """
    Time fn() using CUDA graph replay.  fn() must be pure-CUDA (no
    Python-side allocation that changes shape between calls).
    """
    # Warmup — ensures Triton JIT compilation happens before capture
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    # Time replays
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3   # µs per call


def bench_nograph(fn, warmup=WARMUP, iters=ITERS):
    """Baseline timing without CUDA graph (includes Python overhead)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3   # µs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    print("Building weights…")

    # W1: hidden → 2×inter (SwiGLU interleaved)
    w1, w1_mx = make_weights(HIDDEN, 2 * INTER)
    b1 = make_bias(2 * INTER)

    # W2: inter → hidden  (W2 input is inter after SwiGLU)
    w2, w2_mx = make_weights(INTER, HIDDEN)
    b2 = make_bias(HIDDEN)

    # Single decode token input
    x = torch.randn(1, HIDDEN, device=DEVICE, dtype=torch.bfloat16)

    # Build routing data once (standard Triton path)
    logits_raw = torch.randn(1, N_EXPTS_TOT, device=DEVICE, dtype=torch.float32)
    expt_indx  = torch.topk(logits_raw, N_EXPTS_ACT, dim=1, sorted=False).indices.to(torch.int32)
    rdata_std, gi_std, si_std = routing(
        logits_raw, N_EXPTS_ACT, expt_indx=expt_indx, simulated_ep=1
    )

    # Build fast routing data
    rdata_fast, gi_fast, si_fast = routing_decode_fast(
        logits_raw, expt_indx, N_EXPTS_TOT, N_EXPTS_ACT
    )

    pc1 = PrecisionConfig(weight_scale=w1_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pc2 = PrecisionConfig(weight_scale=w2_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (1.702, SWIGLU_LIM), 2)

    # Pre-compute a fixed W1 output to use as W2 input
    x_w1 = matmul_ogs(x, w1, b1, rdata_std, gather_indx=gi_std,
                      precision_config=pc1, fused_activation=act)
    x_w1 = x_w1.detach()

    print(f"\n{'='*72}")
    print(f"Decode MoE benchmark (CUDA graph timing): 1 token, "
          f"{N_EXPTS_ACT}/{N_EXPTS_TOT} experts, hidden={HIDDEN}")
    print(f"{'='*72}")

    # ------------------------------------------------------------------
    # Routing overhead
    # ------------------------------------------------------------------
    print("\n--- Routing overhead (CUDA graph replay) ---")

    def _routing_std():
        rdata, gi, si = routing(logits_raw, N_EXPTS_ACT, expt_indx=expt_indx, simulated_ep=1)

    def _routing_fast():
        rdata, gi, si = routing_decode_fast(logits_raw, expt_indx, N_EXPTS_TOT, N_EXPTS_ACT)

    t_route_std  = bench_graph(_routing_std)
    t_route_fast = bench_graph(_routing_fast)
    print(f"  {'routing() [Triton]':<40}  {t_route_std:6.1f} µs")
    print(f"  {'routing_decode_fast() [PyTorch]':<40}  {t_route_fast:6.1f} µs  "
          f"({100*(t_route_std-t_route_fast)/t_route_std:+.0f}%)")

    # ------------------------------------------------------------------
    # W1 matmul configs
    # ------------------------------------------------------------------
    print("\n--- W1 (gather + fused SwiGLU) ---")
    def w1_default():
        return matmul_ogs(x, w1, b1, rdata_std, gather_indx=gi_std,
                          precision_config=pc1, fused_activation=act)
    def w1_sk1():
        update_opt_flags_constraints({"split_k": 1})
        out = matmul_ogs(x, w1, b1, rdata_std, gather_indx=gi_std,
                         precision_config=pc1, fused_activation=act)
        reset_opt_flags_constraints()
        return out
    def w1_fast_routing():
        return matmul_ogs(x, w1, b1, rdata_fast, gather_indx=gi_fast,
                          precision_config=pc1, fused_activation=act)

    t_w1_def   = bench_graph(w1_default)
    t_w1_sk1   = bench_graph(w1_sk1)
    t_w1_fast  = bench_graph(w1_fast_routing)

    print(f"  {'default (split_k=auto)':<40}  {t_w1_def:6.1f} µs  [baseline]")
    print(f"  {'split_k=1':<40}  {t_w1_sk1:6.1f} µs  "
          f"({100*(t_w1_def-t_w1_sk1)/t_w1_def:+.0f}%)")
    print(f"  {'fast_routing + default split_k':<40}  {t_w1_fast:6.1f} µs  "
          f"({100*(t_w1_def-t_w1_fast)/t_w1_def:+.0f}%)")

    # ------------------------------------------------------------------
    # W2 matmul configs
    # ------------------------------------------------------------------
    print("\n--- W2 (scatter, split_k variations) ---")
    def w2_default():
        return matmul_ogs(x_w1, w2, b2, rdata_std, scatter_indx=si_std,
                          precision_config=pc2, gammas=rdata_std.gate_scal)
    def w2_sk1():
        update_opt_flags_constraints({"split_k": 1})
        out = matmul_ogs(x_w1, w2, b2, rdata_std, scatter_indx=si_std,
                         precision_config=pc2, gammas=rdata_std.gate_scal)
        reset_opt_flags_constraints()
        return out
    def w2_sk2():
        update_opt_flags_constraints({"split_k": 2})
        out = matmul_ogs(x_w1, w2, b2, rdata_std, scatter_indx=si_std,
                         precision_config=pc2, gammas=rdata_std.gate_scal)
        reset_opt_flags_constraints()
        return out
    def w2_fast_sk1():
        update_opt_flags_constraints({"split_k": 1})
        out = matmul_ogs(x_w1, w2, b2, rdata_fast, scatter_indx=si_fast,
                         precision_config=pc2, gammas=rdata_fast.gate_scal)
        reset_opt_flags_constraints()
        return out

    t_w2_def     = bench_graph(w2_default)
    t_w2_sk1     = bench_graph(w2_sk1)
    t_w2_sk2     = bench_graph(w2_sk2)
    t_w2_fast_sk1 = bench_graph(w2_fast_sk1)

    print(f"  {'default (split_k=auto)':<40}  {t_w2_def:6.1f} µs  [baseline]")
    print(f"  {'split_k=1 (fused_scatter)':<40}  {t_w2_sk1:6.1f} µs  "
          f"({100*(t_w2_def-t_w2_sk1)/t_w2_def:+.0f}%)")
    print(f"  {'split_k=2':<40}  {t_w2_sk2:6.1f} µs  "
          f"({100*(t_w2_def-t_w2_sk2)/t_w2_def:+.0f}%)")
    print(f"  {'fast_routing + split_k=1':<40}  {t_w2_fast_sk1:6.1f} µs  "
          f"({100*(t_w2_def-t_w2_fast_sk1)/t_w2_def:+.0f}%)")

    # ------------------------------------------------------------------
    # Combined per-layer MoE time
    # ------------------------------------------------------------------
    print("\n--- Per-layer MoE total (routing + W1 + W2) ---")
    t_base  = t_route_std  + t_w1_def  + t_w2_def
    t_opt   = t_route_fast + t_w1_fast + t_w2_fast_sk1
    print(f"  {'Original path':<40}  {t_base:6.1f} µs")
    print(f"  {'Optimized (fast_routing + W2 sk=1)':<40}  {t_opt:6.1f} µs  "
          f"({100*(t_base-t_opt)/t_base:+.0f}%)")

    # ------------------------------------------------------------------
    # Theoretical bandwidth ceiling
    # ------------------------------------------------------------------
    print("\n--- Bandwidth ceiling (1.8 TB/s peak) ---")
    bw = 1.8e12  # bytes/s
    w1_bytes = N_EXPTS_ACT * HIDDEN * 2 * INTER * 0.5   # MXFP4
    w2_bytes = N_EXPTS_ACT * INTER  * HIDDEN      * 0.5
    print(f"  W1 weight: {w1_bytes/1e6:.1f} MB → floor {w1_bytes/bw*1e6:.1f} µs  "
          f"({100*w1_bytes/bw/t_w1_def*1e6:.0f}% of peak used)")
    print(f"  W2 weight: {w2_bytes/1e6:.1f} MB → floor {w2_bytes/bw*1e6:.1f} µs  "
          f"({100*w2_bytes/bw/t_w2_def*1e6:.0f}% of peak used)")


if __name__ == "__main__":
    main()
