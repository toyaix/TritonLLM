import torch
import triton
import triton.language as tl
import math


@triton.jit
def rmsnorm_kernel(x_ptr, t_ptr, scale_ptr, last_dim, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * last_dim
    t_ptr = t_ptr + row * last_dim
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, last_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < last_dim, other=0)
        _sum += x * x
    mean = tl.sum(_sum, axis=0) / last_dim + eps
    pid = tl.program_id(1)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset, mask=offset < last_dim, other=0)
    scale = tl.load(scale_ptr + offset, mask=offset < last_dim, other=0)
    y = x * tl.math.rsqrt(mean) * scale
    tl.store(t_ptr + offset, y, mask=offset < last_dim)


def rmsnorm_forward(x, scale, eps):
    t = torch.empty_like(x)
    *prefix_shape, last_dim = x.shape
    rows_total = math.prod(prefix_shape)
    grid = lambda META: (rows_total, triton.cdiv(last_dim, META['BLOCK_SIZE']))
    rmsnorm_kernel[grid](x, t, scale, last_dim, eps, BLOCK_SIZE=512)
    return t


@triton.jit
def rope_kernel(
    query_ptr,
    key_ptr,
    sin_ptr,
    cos_ptr,
    offset_ptr,
    max_context_length,
    num_tokens,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_key_value_heads: tl.constexpr,
    head_dim_div: tl.constexpr,
    TILE_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_t = tl.program_id(1)
    query_ptr += pid * num_tokens * num_heads * head_dim
    key_ptr += pid * num_tokens * num_key_value_heads * head_dim
    offs_token = pid_t * TILE_TOKENS + tl.arange(0, TILE_TOKENS)[:, None]
    offset = tl.load(offset_ptr)
    new_offs_token = (offs_token + offset) % max_context_length
    offs_dim_div = tl.arange(0, head_dim_div)[None, :]
    offs = new_offs_token * head_dim_div + offs_dim_div
    sin_cos_mask = offs_token < num_tokens
    sin = tl.load(sin_ptr + offs, mask=sin_cos_mask)
    sin = sin[:, None, :]
    cos = tl.load(cos_ptr + offs, mask=sin_cos_mask)
    cos = cos[:, None, :]

    offs_token = pid_t * TILE_TOKENS + tl.arange(0, TILE_TOKENS)[:, None, None]
    offs_head = tl.arange(0, num_heads)[None, :, None]
    offs_dim = tl.arange(0, head_dim_div)[None, None, :]
    offs = offs_token * (num_heads * head_dim) + offs_head * head_dim + offs_dim
    q_k_mask = offs_token < num_tokens
    q1 = tl.load(query_ptr + offs, mask=q_k_mask)
    q2 = tl.load(query_ptr + offs + head_dim_div, mask=q_k_mask)
    o1 = q1 * cos - q2 * sin
    o2 = q2 * cos + q1 * sin
    tl.store(query_ptr + offs, o1, mask=q_k_mask)
    tl.store(query_ptr + offs + head_dim_div, o2, mask=q_k_mask)
    stride_0 = num_key_value_heads * head_dim
    offs_num_key_value_heads = tl.arange(0, num_key_value_heads)[None, :, None]
    offs = offs_token * stride_0 + offs_num_key_value_heads * head_dim + offs_dim

    k1 = tl.load(key_ptr + offs, mask=q_k_mask)
    k2 = tl.load(key_ptr + offs + head_dim_div, mask=q_k_mask)
    o1 = k1 * cos - k2 * sin
    o2 = k2 * cos + k1 * sin
    tl.store(key_ptr + offs, o1, mask=q_k_mask)
    tl.store(key_ptr + offs + head_dim_div, o2, mask=q_k_mask)

def rope_forward(query, key, sin, cos, max_context_length, offset):
    batch_size, num_tokens, num_heads, head_dim = query.shape
    batch_size, num_tokens, num_key_value_heads, head_dim = key.shape

    grid = lambda META: (batch_size, triton.cdiv(num_tokens, META['TILE_TOKENS']))
    rope_kernel[grid](
        query,
        key,
        sin,
        cos,
        offset,
        max_context_length,
        num_tokens,
        num_heads,
        head_dim,
        num_key_value_heads,
        head_dim // 2,
        TILE_TOKENS = 1 if num_tokens < 60 else 32,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 128, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 256, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["hidden_size"],
)
@triton.jit
def unembedding_decode_kernel(
    hidden_ptr,
    weight_ptr,
    logits_ptr,
    vocab_size,
    hidden_size,
    stride_hidden_batch,
    stride_hidden_dim,
    stride_weight_vocab,
    stride_weight_dim,
    stride_logits_batch,
    stride_logits_vocab,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_vocab = tl.program_id(0)
    pid_batch = tl.program_id(1)

    offs_vocab = pid_vocab * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_k = tl.arange(0, BLOCK_K)

    hidden_ptr = hidden_ptr + pid_batch.to(tl.int64) * stride_hidden_batch
    logits_ptr = logits_ptr + pid_batch.to(tl.int64) * stride_logits_batch

    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        k_offsets = k_start + offs_k
        hidden = tl.load(
            hidden_ptr + k_offsets.to(tl.int64) * stride_hidden_dim,
            mask=k_offsets < hidden_size,
            other=0,
        )
        weight = tl.load(
            weight_ptr
            + offs_vocab[:, None].to(tl.int64) * stride_weight_vocab
            + k_offsets[None, :].to(tl.int64) * stride_weight_dim,
            mask=(offs_vocab[:, None] < vocab_size) & (k_offsets[None, :] < hidden_size),
            other=0,
        )
        acc += tl.sum(weight * hidden[None, :], axis=1)

    tl.store(
        logits_ptr + offs_vocab.to(tl.int64) * stride_logits_vocab,
        acc.to(logits_ptr.type.element_ty),
        mask=offs_vocab < vocab_size,
    )


def unembedding_decode_forward(hidden, weight):
    batch, num_tokens, hidden_size = hidden.shape
    vocab_size, hidden_size_weight = weight.shape
    assert num_tokens == 1
    assert hidden_size == hidden_size_weight

    hidden = hidden.reshape(batch, hidden_size)
    if hidden.stride(-1) != 1:
        hidden = hidden.contiguous()
    if weight.stride(-1) != 1:
        weight = weight.contiguous()

    logits = torch.empty((batch, vocab_size), device=hidden.device, dtype=hidden.dtype)
    grid = lambda META: (triton.cdiv(vocab_size, META["BLOCK_V"]), batch)
    unembedding_decode_kernel[grid](
        hidden,
        weight,
        logits,
        vocab_size,
        hidden_size,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        logits.stride(0),
        logits.stride(1),
    )
    return logits[:, None, :]


def unembedding_forward(hidden, weight):
    if (
        hidden.is_cuda
        and hidden.ndim == 3
        and hidden.shape[1] == 1
        and hidden.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
    ):
        return unembedding_decode_forward(hidden, weight)
    return torch.nn.functional.linear(hidden, weight, bias=None)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_OUT": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OUT": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OUT": 128, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_OUT": 256, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["hidden_size", "q_dim", "kv_dim"],
)
@triton.jit
def qkv_decode_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    hidden_size,
    q_dim,
    kv_dim,
    stride_hidden_batch,
    stride_hidden_dim,
    stride_weight_out,
    stride_weight_dim,
    stride_bias_out,
    stride_q_batch,
    stride_q_dim,
    stride_k_batch,
    stride_k_dim,
    stride_v_batch,
    stride_v_dim,
    BLOCK_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_out = tl.program_id(0)
    pid_batch = tl.program_id(1)

    total_out = q_dim + kv_dim + kv_dim
    offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    offs_k = tl.arange(0, BLOCK_K)

    hidden_ptr = hidden_ptr + pid_batch.to(tl.int64) * stride_hidden_batch
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        k_offsets = k_start + offs_k
        hidden = tl.load(
            hidden_ptr + k_offsets.to(tl.int64) * stride_hidden_dim,
            mask=k_offsets < hidden_size,
            other=0,
        )
        weight = tl.load(
            weight_ptr
            + offs_out[:, None].to(tl.int64) * stride_weight_out
            + k_offsets[None, :].to(tl.int64) * stride_weight_dim,
            mask=(offs_out[:, None] < total_out) & (k_offsets[None, :] < hidden_size),
            other=0,
        )
        acc += tl.sum(weight * hidden[None, :], axis=1)

    bias = tl.load(
        bias_ptr + offs_out.to(tl.int64) * stride_bias_out,
        mask=offs_out < total_out,
        other=0,
    )
    acc += bias.to(tl.float32)

    q_mask = offs_out < q_dim
    k_mask = (offs_out >= q_dim) & (offs_out < q_dim + kv_dim)
    v_mask = (offs_out >= q_dim + kv_dim) & (offs_out < total_out)

    q_offsets = tl.where(q_mask, offs_out, 0)
    k_offsets = tl.where(k_mask, offs_out - q_dim, 0)
    v_offsets = tl.where(v_mask, offs_out - q_dim - kv_dim, 0)

    tl.store(
        q_ptr + pid_batch.to(tl.int64) * stride_q_batch + q_offsets.to(tl.int64) * stride_q_dim,
        acc.to(q_ptr.type.element_ty),
        mask=q_mask,
    )
    tl.store(
        k_ptr + pid_batch.to(tl.int64) * stride_k_batch + k_offsets.to(tl.int64) * stride_k_dim,
        acc.to(k_ptr.type.element_ty),
        mask=k_mask,
    )
    tl.store(
        v_ptr + pid_batch.to(tl.int64) * stride_v_batch + v_offsets.to(tl.int64) * stride_v_dim,
        acc.to(v_ptr.type.element_ty),
        mask=v_mask,
    )


def qkv_decode_forward(hidden, weight, bias, q_dim, kv_dim):
    batch, num_tokens, hidden_size = hidden.shape
    total_out, hidden_size_weight = weight.shape

    assert num_tokens == 1
    assert hidden_size == hidden_size_weight
    assert bias.shape == (total_out,)
    assert total_out == q_dim + 2 * kv_dim

    hidden = hidden.reshape(batch, hidden_size)
    if hidden.stride(-1) != 1:
        hidden = hidden.contiguous()
    if weight.stride(-1) != 1:
        weight = weight.contiguous()
    if bias.stride(0) != 1:
        bias = bias.contiguous()

    q = torch.empty((batch, q_dim), device=hidden.device, dtype=hidden.dtype)
    k = torch.empty((batch, kv_dim), device=hidden.device, dtype=hidden.dtype)
    v = torch.empty((batch, kv_dim), device=hidden.device, dtype=hidden.dtype)

    grid = lambda META: (triton.cdiv(total_out, META["BLOCK_OUT"]), batch)
    qkv_decode_kernel[grid](
        hidden,
        weight,
        bias,
        q,
        k,
        v,
        hidden_size,
        q_dim,
        kv_dim,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        bias.stride(0),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
    )
    return q[:, None, :], k[:, None, :], v[:, None, :]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=2),
    ],
    key=["hidden_size", "num_q_heads", "num_kv_heads"],
)
@triton.jit
def qkv_rope_cache_decode_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    sin_ptr,
    cos_ptr,
    offset_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    hidden_size,
    num_q_heads,
    num_kv_heads,
    stride_hidden_batch,
    stride_hidden_dim,
    stride_weight_out,
    stride_weight_dim,
    stride_bias_out,
    stride_sin_ctx,
    stride_sin_dim,
    stride_cos_ctx,
    stride_cos_dim,
    stride_q_batch,
    stride_q_head,
    stride_q_dim,
    stride_k_batch,
    stride_k_ctx,
    stride_k_head,
    stride_k_dim,
    stride_v_batch,
    stride_v_ctx,
    stride_v_head,
    stride_v_dim,
    max_context_length,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_half = tl.arange(0, HEAD_DIM // 2)

    q_dim = num_q_heads * HEAD_DIM
    kv_dim = num_kv_heads * HEAD_DIM

    is_q = pid_head < num_q_heads
    is_k = (pid_head >= num_q_heads) & (pid_head < num_q_heads + num_kv_heads)
    is_v = pid_head >= num_q_heads + num_kv_heads

    q_head = tl.where(is_q, pid_head, 0)
    k_head = tl.where(is_k, pid_head - num_q_heads, 0)
    v_head = tl.where(is_v, pid_head - num_q_heads - num_kv_heads, 0)

    out_base = tl.where(
        is_q,
        q_head * HEAD_DIM,
        tl.where(
            is_k,
            q_dim + k_head * HEAD_DIM,
            q_dim + kv_dim + v_head * HEAD_DIM,
        ),
    )
    out_offsets = out_base + offs_d
    out_mask = out_offsets < total_heads * HEAD_DIM

    hidden_ptr = hidden_ptr + pid_batch.to(tl.int64) * stride_hidden_batch
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        k_offsets = k_start + offs_k
        hidden = tl.load(
            hidden_ptr + k_offsets.to(tl.int64) * stride_hidden_dim,
            mask=k_offsets < hidden_size,
            other=0,
        )
        weight = tl.load(
            weight_ptr
            + out_offsets[:, None].to(tl.int64) * stride_weight_out
            + k_offsets[None, :].to(tl.int64) * stride_weight_dim,
            mask=out_mask[:, None] & (k_offsets[None, :] < hidden_size),
            other=0,
        )
        acc += tl.sum(weight * hidden[None, :], axis=1)

    bias = tl.load(
        bias_ptr + out_offsets.to(tl.int64) * stride_bias_out,
        mask=out_mask,
        other=0,
    )
    acc += bias.to(tl.float32)

    raw_offset = tl.load(offset_ptr).to(tl.int64)
    rope_offset = raw_offset % max_context_length
    sin = tl.load(
        sin_ptr + rope_offset * stride_sin_ctx + offs_half.to(tl.int64) * stride_sin_dim,
        other=0,
    ).to(tl.float32)
    cos = tl.load(
        cos_ptr + rope_offset * stride_cos_ctx + offs_half.to(tl.int64) * stride_cos_dim,
        other=0,
    ).to(tl.float32)
    acc_lo = acc[: HEAD_DIM // 2]
    acc_hi = acc[HEAD_DIM // 2 :]
    rot_lo = acc_lo * cos - acc_hi * sin
    rot_hi = acc_hi * cos + acc_lo * sin

    q_mask = is_q & (offs_half < HEAD_DIM // 2)
    k_mask = is_k & (offs_half < HEAD_DIM // 2)
    v_mask = is_v & (offs_d < HEAD_DIM)

    q_ptr = q_ptr + pid_batch.to(tl.int64) * stride_q_batch + q_head.to(tl.int64) * stride_q_head
    tl.store(
        q_ptr + offs_half.to(tl.int64) * stride_q_dim,
        rot_lo.to(q_ptr.type.element_ty),
        mask=q_mask,
    )
    tl.store(
        q_ptr + (HEAD_DIM // 2 + offs_half).to(tl.int64) * stride_q_dim,
        rot_hi.to(q_ptr.type.element_ty),
        mask=q_mask,
    )

    k_cache_ptr = (
        k_cache_ptr
        + pid_batch.to(tl.int64) * stride_k_batch
        + raw_offset * stride_k_ctx
        + k_head.to(tl.int64) * stride_k_head
    )
    tl.store(
        k_cache_ptr + offs_half.to(tl.int64) * stride_k_dim,
        rot_lo.to(k_cache_ptr.type.element_ty),
        mask=k_mask,
    )
    tl.store(
        k_cache_ptr + (HEAD_DIM // 2 + offs_half).to(tl.int64) * stride_k_dim,
        rot_hi.to(k_cache_ptr.type.element_ty),
        mask=k_mask,
    )

    v_cache_ptr = (
        v_cache_ptr
        + pid_batch.to(tl.int64) * stride_v_batch
        + raw_offset * stride_v_ctx
        + v_head.to(tl.int64) * stride_v_head
    )
    tl.store(
        v_cache_ptr + offs_d.to(tl.int64) * stride_v_dim,
        acc.to(v_cache_ptr.type.element_ty),
        mask=v_mask,
    )


def qkv_rope_cache_decode_forward(
    hidden,
    weight,
    bias,
    sin,
    cos,
    max_context_length,
    cache_k,
    cache_v,
    offset,
    num_q_heads,
    num_kv_heads,
    head_dim,
):
    batch, num_tokens, hidden_size = hidden.shape
    total_out, hidden_size_weight = weight.shape

    assert num_tokens == 1
    assert hidden_size == hidden_size_weight
    assert total_out == (num_q_heads + 2 * num_kv_heads) * head_dim
    assert bias.shape == (total_out,)
    assert cache_k.shape == cache_v.shape
    assert cache_k.shape[0] == batch
    assert cache_k.shape[2] == num_kv_heads
    assert cache_k.shape[3] == head_dim
    assert offset.shape == (1,)

    hidden = hidden.reshape(batch, hidden_size)
    if hidden.stride(-1) != 1:
        hidden = hidden.contiguous()
    if weight.stride(-1) != 1:
        weight = weight.contiguous()
    if bias.stride(0) != 1:
        bias = bias.contiguous()
    if sin.stride(-1) != 1:
        sin = sin.contiguous()
    if cos.stride(-1) != 1:
        cos = cos.contiguous()

    q = torch.empty((batch, num_q_heads, head_dim), device=hidden.device, dtype=hidden.dtype)

    grid = (num_q_heads + 2 * num_kv_heads, batch)
    qkv_rope_cache_decode_kernel[grid](
        hidden,
        weight,
        bias,
        sin,
        cos,
        offset,
        q,
        cache_k,
        cache_v,
        hidden_size,
        num_q_heads,
        num_kv_heads,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        bias.stride(0),
        sin.stride(0),
        sin.stride(1),
        cos.stride(0),
        cos.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        cache_k.stride(0),
        cache_k.stride(1),
        cache_k.stride(2),
        cache_k.stride(3),
        cache_v.stride(0),
        cache_v.stride(1),
        cache_v.stride(2),
        cache_v.stride(3),
        max_context_length,
        HEAD_DIM=head_dim,
    )
    return q.reshape(batch, 1, num_q_heads * head_dim)
