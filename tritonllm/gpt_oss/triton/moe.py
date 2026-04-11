import torch
from torch.profiler import record_function

from triton_kernels.swiglu import swiglu_fn
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import routing, RoutingData, ExptData, GatherIndx, ScatterIndx
from triton_kernels.tensor import convert_layout
from triton_kernels.tensor_details.layout import StridedLayout, make_default_matmul_mxfp4_w_layout
from triton_kernels.tensor import wrap_torch_tensor, FP4


def quantize_mx4(w):
    w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
    value_layout, value_layout_opts = make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale


def swiglu(x, alpha: float = 1.702, limit: float = 7.0, interleaved: bool = True):
    if interleaved:
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
    else:
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


def moe(x, wg, w1, w1_mx, w2, w2_mx, bg, b1, b2, experts_per_token=4, num_experts=128, swiglu_limit=7.0, fused_act=True, interleaved=True):
    if x.numel() == 0:
        return x

    pc1 = PrecisionConfig(weight_scale=w1_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pc2 = PrecisionConfig(weight_scale=w2_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=InFlexData()))

    with record_function("wg"):
        logits = matmul_ogs(x, wg, bg, precision_config=pcg)
    with record_function("routing"):
        rdata, gather_indx, scatter_indx = routing(logits, experts_per_token, simulated_ep=1)

    if fused_act:
        assert interleaved, "Fused activation requires interleaved weights"
        with record_function("w1+swiglu"):
            act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (1.702, swiglu_limit), 2)
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
    else:
        with record_function("w1"):
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        with record_function("swiglu"):
            x = swiglu(x, limit=swiglu_limit, interleaved=interleaved)

    with record_function("w2"):
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2, gammas=rdata.gate_scal)
    return x


def routing_decode_fast(logits, expt_indx, n_expts_tot, n_expts_act):
    """
    Fast routing for single-token decode using pure PyTorch tensor ops.

    Replaces the Triton-kernel-based routing() for the 1-token decode case.
    All ops are CUDA-graph-compatible (fixed shapes, no Python branching on
    tensor values).

    Args:
        logits:      [1, n_expts_tot] float32  (gate logits for the single token)
        expt_indx:   [1, n_expts_act] int32    (pre-computed topk expert indices)
        n_expts_tot: total number of experts (128)
        n_expts_act: active experts per token (4)

    Returns:
        (RoutingData, GatherIndx, ScatterIndx)
    """
    device = logits.device
    expt_indx_flat = expt_indx.reshape(-1)  # [n_expts_act]

    # --- Sort expert indices by expert id for contiguous expert processing ---
    sort_order = torch.argsort(expt_indx_flat)          # [n_expts_act]
    expt_indx_sorted = expt_indx_flat[sort_order]        # [n_expts_act], sorted by id

    # --- Gate scores (softmax) in expert-id-sorted order ---
    gate_scal = torch.softmax(
        logits[0, expt_indx_sorted.long()], dim=0
    )  # [n_expts_act], float32

    # --- GatherIndx / ScatterIndx ---
    # topk_indx[i] = position in the unsorted expt_indx that maps to sorted slot i
    topk_indx = sort_order.to(torch.int32)               # [n_expts_act]
    gate_indx  = torch.argsort(topk_indx).to(torch.int32)  # inverse perm

    # --- Expert histogram: 1 at each active expert ---
    ones = torch.ones(n_expts_act, dtype=torch.int32, device=device)
    hist = torch.zeros(n_expts_tot, dtype=torch.int32, device=device)
    hist.scatter_add_(0, expt_indx_sorted.long(), ones)

    # --- token_offs_raw: cumsum(hist) prepended with 0 ---
    # token_offs_raw[e] = offset of expert e's first token in sorted layout
    token_offs_raw = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.cumsum(hist, dim=0).to(torch.int32),
    ])  # [n_expts_tot + 1]

    # --- token_offs_pad[16]: same as token_offs_raw for 1 token per expert ---
    # (ceil(1 / 16) = 1 tile = same as ceil(hist / 16) when hist ≤ 1)
    token_offs_pad_16 = token_offs_raw  # [n_expts_tot + 1]

    # --- block_pid_map[16]: one entry per active expert ---
    # Entry format: (block_id << 16) | expert_id.  block_id=0 always (1 tile).
    # Position of expert e in the map = token_offs_pad_16[e].
    positions = token_offs_pad_16[expt_indx_sorted.long()]  # [n_expts_act], values 0..3
    values    = expt_indx_sorted.to(torch.int32)             # (0 << 16) | expt_id
    block_pid_map_16 = torch.full((n_expts_act,), -1, dtype=torch.int32, device=device)
    block_pid_map_16.scatter_(0, positions.long(), values)

    expt_data = ExptData(
        hist=hist,
        token_offs_raw=token_offs_raw,
        token_offs_pad={16: token_offs_pad_16},
        block_pid_map={16: block_pid_map_16},
    )
    rdata = RoutingData(
        gate_scal=gate_scal,
        expt_hist=hist,
        n_expts_tot=n_expts_tot,
        n_expts_act=n_expts_act,
        expt_data=expt_data,
    )
    gather_indx  = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    return rdata, gather_indx, scatter_indx


def moe_decode(
    x,
    wg,
    w1,
    w1_mx,
    w2,
    w2_mx,
    bg,
    b1,
    b2,
    experts_per_token=4,
    num_experts=128,
    swiglu_limit=7.0,
    fused_act=True,
    interleaved=True,
):
    if x.numel() == 0:
        return x

    pc1 = PrecisionConfig(weight_scale=w1_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pc2 = PrecisionConfig(weight_scale=w2_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))

    with record_function("wg_decode"):
        logits = torch.matmul(x, wg).float()
        if bg is not None:
            logits = logits + bg
    with record_function("routing_decode"):
        expt_indx = torch.topk(logits, experts_per_token, dim=1, sorted=False).indices.to(torch.int32)
        rdata, gather_indx, scatter_indx = routing(
            logits, experts_per_token, expt_indx=expt_indx, simulated_ep=1,
        )

    if fused_act:
        assert interleaved, "Fused activation requires interleaved weights"
        with record_function("w1+swiglu_decode"):
            act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (1.702, swiglu_limit), 2)
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
    else:
        with record_function("w1_decode"):
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        with record_function("swiglu_decode"):
            x = swiglu(x, limit=swiglu_limit, interleaved=interleaved)

    with record_function("w2_decode"):
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2, gammas=rdata.gate_scal)
    return x

