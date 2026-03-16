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

    acc = tl.zeros((BLOCK_V, 1), dtype=tl.float32)

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
        acc += tl.dot(weight, hidden[:, None])

    tl.store(
        logits_ptr + offs_vocab.to(tl.int64) * stride_logits_vocab,
        acc[:, 0].to(logits_ptr.type.element_ty),
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
