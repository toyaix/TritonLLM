import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_decode_fwd(
    Q,
    K,
    V,
    Sinks,
    Out,
    sm_scale,
    Start_q,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vz,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    M,
    N_KV_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BANDWIDTH: tl.constexpr,
):
    pid_hm = tl.program_id(0)
    off_z = tl.program_id(1)
    off_h = pid_hm // M
    off_m = pid_hm % M
    start_q = tl.load(Start_q).to(tl.int32)

    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = (
        Q
        + off_z.to(tl.int64) * stride_qz
        + off_h.to(tl.int64) * stride_qh
        + off_m.to(tl.int64) * stride_qm
        + offs_d.to(tl.int64) * stride_qk
    )
    q = tl.load(q_ptrs, mask=offs_d < HEAD_DIM, other=0).to(tl.float32)

    if Sinks is not None:
        sink = tl.load(Sinks + off_h.to(tl.int64) * M + off_m).to(tl.float32)
    else:
        sink = 0

    if BANDWIDTH:
        lo = tl.maximum(0, start_q - BANDWIDTH + 1)
    else:
        lo = 0
    hi = tl.minimum(start_q + 1, N_KV_CTX)
    lo = tl.minimum(lo, hi)

    m_i = sink
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < hi

        k_ptrs = (
            K
            + off_z.to(tl.int64) * stride_kz
            + offs_n.to(tl.int64)[:, None] * stride_kn
            + off_h.to(tl.int64) * stride_kh
            + offs_d.to(tl.int64)[None, :] * stride_kk
        )
        v_ptrs = (
            V
            + off_z.to(tl.int64) * stride_vz
            + offs_n.to(tl.int64)[:, None] * stride_vn
            + off_h.to(tl.int64) * stride_vh
            + offs_d.to(tl.int64)[None, :] * stride_vk
        )

        k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < HEAD_DIM), other=0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < HEAD_DIM), other=0).to(tl.float32)

        qk = tl.sum(k * q[None, :], axis=1) * sm_scale
        qk = tl.where(mask_n, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.math.exp(qk - m_ij)
        alpha = tl.math.exp(m_i - m_ij)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_ij

    z = l_i + tl.math.exp(sink - m_i)
    out = acc / z

    o_ptrs = (
        Out
        + off_z.to(tl.int64) * stride_oz
        + off_h.to(tl.int64) * stride_oh
        + off_m.to(tl.int64) * stride_om
        + offs_d.to(tl.int64) * stride_ok
    )
    tl.store(o_ptrs, out.to(Out.type.element_ty), mask=offs_d < HEAD_DIM)

class _attention_decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q):
        assert len(start_q) == 1
        bs, n_ctx, n_kv_heads, repeat_kv, head_dim_q = q.shape
        bs, n_kv_ctx, n_kv_heads_k, head_dim_k = k.shape
        bs, n_kv_ctx, n_kv_heads_v, head_dim_v = v.shape
        assert n_ctx == 1
        assert n_kv_heads == n_kv_heads_k == n_kv_heads_v
        assert head_dim_q == head_dim_k == head_dim_v
        assert head_dim_q in {16, 32, 64, 128, 256}

        q = q[:, 0, :, :, :].contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = torch.empty_like(q)

        grid = (n_kv_heads * repeat_kv, bs)
        _attn_decode_fwd[grid](
            q,
            k,
            v,
            sinks,
            o,
            sm_scale,
            start_q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            repeat_kv,
            N_KV_CTX=n_kv_ctx,
            HEAD_DIM=head_dim_q,
            BANDWIDTH=bandwidth,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )
        return o.view(bs, 1, n_kv_heads * repeat_kv * head_dim_q)

attention_decode = _attention_decode.apply


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_keys", [128, 1024])
@pytest.mark.parametrize("num_key_value_heads", [8])
@pytest.mark.parametrize("num_key_value_groups", [8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("sm_scale", [0.125])
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("start_q", [0, 5, 63])
def test_decode_eq(batch_size, num_keys, num_key_value_heads, num_key_value_groups, head_dim, sm_scale, sliding_window, start_q):
    from gpt_oss.triton.attention_ref import attention_ref
    q = torch.randn(batch_size, 1, num_key_value_heads, num_key_value_groups, head_dim).bfloat16().cuda()
    k = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    v = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    sinks = torch.randn(num_key_value_heads * num_key_value_groups).bfloat16().cuda()

    start_q = torch.tensor([start_q], dtype=torch.int32).cuda()

    o1 = attention_decode(q, k, v, sinks, sm_scale, sliding_window, start_q)
    o2 = attention_ref(q, k, v, sinks, sm_scale, sliding_window, start_q)
    torch.testing.assert_close(o1, o2)
