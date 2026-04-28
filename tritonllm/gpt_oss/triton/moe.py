import os
import torch
from torch.profiler import record_function

from triton_kernels.swiglu import swiglu_fn
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import routing
from triton_kernels.tensor import convert_layout
from triton_kernels.tensor_details.layout import StridedLayout, make_default_matmul_mxfp4_w_layout
from triton_kernels.tensor import wrap_torch_tensor, FP4
from triton_kernels.matmul_ogs_details.opt_flags import update_opt_flags_constraints, reset_opt_flags_constraints


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


def moe_gate_routing(
    x: torch.Tensor,
    wg: torch.Tensor,
    bg: torch.Tensor | None,
    experts_per_token: int = 4,
    num_experts: int = 128,
):
    if x.numel() == 0:
        return None, None, None
    with record_function("wg"):
        logits = torch.matmul(x, wg).float()
        if bg is not None:
            logits = logits + bg
    with record_function("routing"):
        rdata, gather_indx, scatter_indx = routing(logits, experts_per_token, simulated_ep=1)
    return rdata, gather_indx, scatter_indx


def moe_experts(
    x: torch.Tensor,
    w1,
    w1_mx,
    w2,
    w2_mx,
    b1: torch.Tensor,
    b2: torch.Tensor,
    rdata,
    gather_indx,
    scatter_indx,
    swiglu_limit: float = 7.0,
    fused_act: bool = True,
    interleaved: bool = True,
):
    if x.numel() == 0:
        return x
    pc1 = PrecisionConfig(weight_scale=w1_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pc2 = PrecisionConfig(weight_scale=w2_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))

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


def moe(x, wg, w1, w1_mx, w2, w2_mx, bg, b1, b2, experts_per_token=4, num_experts=128, swiglu_limit=7.0, fused_act=True, interleaved=True):
    rdata, gather_indx, scatter_indx = moe_gate_routing(x, wg, bg, experts_per_token)
    if rdata is None:
        return x
    return moe_experts(x, w1, w1_mx, w2, w2_mx, b1, b2, rdata, gather_indx, scatter_indx,
                       swiglu_limit=swiglu_limit, fused_act=fused_act, interleaved=interleaved)


def moe_decode_gate_routing(
    x: torch.Tensor,
    wg: torch.Tensor,
    bg: torch.Tensor | None,
    experts_per_token: int = 4,
    num_experts: int = 128,
):
    if x.numel() == 0:
        return None, None, None
    with record_function("wg_decode"):
        logits = torch.matmul(x, wg).float()
        if bg is not None:
            logits = logits + bg
    with record_function("routing_decode"):
        rdata, gather_indx, scatter_indx = routing(
            logits, experts_per_token, simulated_ep=1
        )
    return rdata, gather_indx, scatter_indx


def moe_decode_experts(
    x: torch.Tensor,
    w1,
    w1_mx,
    w2,
    w2_mx,
    b1: torch.Tensor,
    b2: torch.Tensor,
    rdata,
    gather_indx,
    scatter_indx,
    swiglu_limit: float = 7.0,
    fused_act: bool = True,
    interleaved: bool = True,
):
    if x.numel() == 0:
        return x

    pc1 = PrecisionConfig(weight_scale=w1_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pc2 = PrecisionConfig(weight_scale=w2_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))

    _block_k = int(os.environ["MOE_BLOCK_K"]) if os.environ.get("MOE_BLOCK_K") else None

    if fused_act:
        assert interleaved, "Fused activation requires interleaved weights"
        if _block_k is not None:
            update_opt_flags_constraints({"block_k": _block_k})
        with record_function("w1+swiglu_decode"):
            act = FusedActivation(
                FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")),
                (1.702, swiglu_limit), 2,
            )
            x = matmul_ogs(
                x, w1, b1, rdata,
                gather_indx=gather_indx,
                precision_config=pc1,
                fused_activation=act,
            )
    else:
        if _block_k is not None:
            update_opt_flags_constraints({"block_k": _block_k})
        with record_function("w1_decode"):
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        with record_function("swiglu_decode"):
            x = swiglu(x, limit=swiglu_limit, interleaved=interleaved)

    if _block_k is not None:
        update_opt_flags_constraints({"block_k": _block_k})
    with record_function("w2_decode"):
        x = matmul_ogs(
            x, w2, b2, rdata,
            scatter_indx=scatter_indx,
            precision_config=pc2,
            gammas=rdata.gate_scal,
        )
    if _block_k is not None:
        reset_opt_flags_constraints()
    return x


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
    rdata, gather_indx, scatter_indx = moe_decode_gate_routing(
        x, wg, bg, experts_per_token, num_experts,
    )
    if rdata is None:
        return x
    return moe_decode_experts(
        x, w1, w1_mx, w2, w2_mx, b1, b2,
        rdata, gather_indx, scatter_indx,
        swiglu_limit=swiglu_limit,
        fused_act=fused_act,
        interleaved=interleaved,
    )
