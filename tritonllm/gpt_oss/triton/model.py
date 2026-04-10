import hashlib
import json
import math
import os
import struct
import termcolor
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch.profiler import record_function

from gpt_oss.triton.weights import Checkpoint
from gpt_oss.triton.attention_ref import attention_ref
from gpt_oss.triton.attention_tt_decode import attention_decode

from triton_kernels.target_info import cuda_capability_geq, cuda_capability_eq
if not cuda_capability_geq(9):
    from gpt_oss.triton.attention_tt import attention
else:
    from gpt_oss.triton.attention_tt_tma import attention

from gpt_oss.triton.moe import quantize_mx4, moe, moe_decode
from gpt_oss.triton.triton_kernels import (
    out_residual_decode_forward,
    qkv_rope_cache_decode_forward,
    qkv_decode_forward,
    rmsnorm_forward,
    rope_forward,
    unembedding_decode_forward,
)

ENABLE_RECORD_FUNCTION = os.getenv("ENABLE_RECORD_FUNCTION", "0") == "1"


def maybe_record_function(name: str):
    if ENABLE_RECORD_FUNCTION:
        return record_function(name)
    return nullcontext()


def maybe_record_decorator(name: str):
    if ENABLE_RECORD_FUNCTION:
        return record_function(name)

    def decorator(fn):
        return fn

    return decorator

@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    @maybe_record_decorator("rmsnorm_triton")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_forward(x, self.scale, self.eps)


class UnEmbedding(torch.nn.Module):
    def __init__(
        self, hidden_size: int, vocab_size: int, device: torch.device | None = None
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((vocab_size, hidden_size), device=device, dtype=torch.bfloat16)
        )

    @maybe_record_decorator("unembedding_linear")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.ndim == 3
            and x.shape[1] == 1
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            return unembedding_decode_forward(x, self.weight)
        return torch.nn.functional.linear(x, self.weight, bias=None)


class QKV(torch.nn.Module):
    def __init__(
        self, hidden_size: int, qkv_dim: int, device: torch.device | None = None
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((qkv_dim, hidden_size), device=device, dtype=torch.bfloat16)
        )
        self.bias = torch.nn.Parameter(
            torch.empty((qkv_dim), device=device, dtype=torch.bfloat16)
        )

    @maybe_record_decorator("qkv_linear")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

    @maybe_record_decorator("qkv_linear_decode")
    def decode(
        self,
        x: torch.Tensor,
        q_dim: int,
        kv_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            x.is_cuda
            and x.ndim == 3
            and x.shape[1] == 1
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
            and self.bias.dtype == torch.bfloat16
        ):
            return qkv_decode_forward(x, self.weight, self.bias, q_dim, kv_dim)

        qkv = self.forward(x)
        q, k, v = torch.split(qkv, (q_dim, kv_dim, kv_dim), dim=-1)
        return q.contiguous(), k.contiguous(), v.contiguous()


class OUT(torch.nn.Module):
    def __init__(
        self, out_dim: int, hidden_size: int, device: torch.device | None = None
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((hidden_size, out_dim), device=device, dtype=torch.bfloat16)
        )
        self.bias = torch.nn.Parameter(
            torch.empty((hidden_size), device=device, dtype=torch.bfloat16)
        )

    @maybe_record_decorator("out_linear")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

    @maybe_record_decorator("out_linear_decode")
    def decode_residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.ndim == 3
            and x.shape[1] == 1
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
            and self.bias.dtype == torch.bfloat16
            and residual.shape == (x.shape[0], 1, self.weight.shape[0])
            and residual.dtype == torch.bfloat16
        ):
            return out_residual_decode_forward(x, self.weight, self.bias, residual)
        return self.forward(x) + residual


class RotaryEmbedding(torch.nn.Module):
    _cos_sin_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        max_context_length: int = 131072,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.max_context_length = max_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device
        self.cos, self.sin = self._get_or_create_cos_sin()

    def _cache_key(self) -> tuple:
        return (
            self.device.type if self.device is not None else None,
            self.device.index if self.device is not None else None,
            self.head_dim,
            self.base,
            self.dtype,
            self.initial_context_length,
            self.max_context_length,
            self.scaling_factor,
            self.ntk_alpha,
            self.ntk_beta,
        )

    def _get_or_create_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._cache_key()
        cached = self._cos_sin_cache.get(key)
        if cached is None:
            cached = self._compute_cos_sin(0, self.max_context_length)
            self._cos_sin_cache[key] = cached
        return cached

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, start: int, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(start, start + num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    @maybe_record_decorator("rotate")
    def _rotate(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos[None, :, None, :].to(x.dtype)
        sin = sin[None, :, None, :].to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    @maybe_record_decorator("rope_triton")
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        offset: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_forward(query, key, self.sin, self.cos, self.max_context_length, offset)
        return query, key


# ---------------------------------------------------------------------------
# Paged KV-cache (nano-vllm approach)
# ---------------------------------------------------------------------------

@triton.jit
def _store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Scatter one token's K/V into a paged cache.

    Each program handles one token. slot_mapping[idx] gives the flat slot
    index into k_cache / v_cache viewed as [total_slots, D].
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, D)
    key   = tl.load(key_ptr   + idx * key_stride   + offsets)
    value = tl.load(value_ptr + idx * value_stride + offsets)
    base  = slot.to(tl.int64) * D
    tl.store(k_cache_ptr + base + offsets, key)
    tl.store(v_cache_ptr + base + offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter key/value pairs into a paged KV cache.

    Args:
        key:          [N, n_kv_heads, head_dim]
        value:        [N, n_kv_heads, head_dim]
        k_cache:      [num_blocks, block_size, n_kv_heads, head_dim]
        v_cache:      [num_blocks, block_size, n_kv_heads, head_dim]
        slot_mapping: [N] int32 — flat slot index for each token
                      (slot = block_id * block_size + offset_in_block)
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    _store_kvcache_kernel[(N,)](
        key,   key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping, D,
    )


class Block:
    """A single physical KV-cache block with ref-counting and hash-based identity."""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, hash: int, token_ids: list[int]) -> None:
        self.hash = hash
        self.token_ids = token_ids

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Paged KV-cache block allocator (per-layer, single sequence).

    Physical layout of each layer's cache:
        [num_blocks, block_size, n_kv_heads, head_dim]

    A block_table maps logical block indices → physical block ids.
    slot = block_table[pos // block_size] * block_size + pos % block_size

    Supports hash-based prefix caching and ref-counted
    block ownership for shared blocks between sequences.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """Compute a deterministic 64-bit SHA-256 hash over block tokens."""
        h = hashlib.sha256()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little", signed=False))
        if token_ids:
            h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
        return int.from_bytes(h.digest()[:8], "little", signed=False)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def capacity(self) -> int:
        return self.num_blocks * self.block_size

    def can_fit(self, num_tokens: int) -> bool:
        needed = math.ceil(num_tokens / self.block_size)
        return self.num_free_blocks >= needed

    def _allocate_first_free_block(self) -> Block:
        """Pop the first free block in O(1) and mark it as in-use."""
        assert self.free_block_ids, "KV cache exhausted: no free blocks available"
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def deallocate(self, block_table: list[int]) -> None:
        """Release blocks in block_table with ref-count semantics."""
        for bid in reversed(block_table):
            block = self.blocks[bid]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(bid)
        block_table.clear()

    def append_block(self, block_table: list[int]) -> None:
        """Append a single new block to block_table (used during generation)."""
        block = self._allocate_first_free_block()
        block_table.append(block.block_id)



class DenseCache:
    """Contiguous KV cache matching main branch behavior."""

    def __init__(
        self,
        batch_size: int,
        n_ctx: int,
        n_kv_heads: int,
        d_head: int = 64,
        device: torch.device | None = None,
    ):
        self.k = torch.zeros(
            (batch_size, n_ctx, n_kv_heads, d_head),
            dtype=torch.bfloat16,
            device=device,
        )
        self.v = torch.zeros(
            (batch_size, n_ctx, n_kv_heads, d_head),
            dtype=torch.bfloat16,
            device=device,
        )
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)

    @property
    def k_flat(self) -> torch.Tensor:
        return self.k

    @property
    def v_flat(self) -> torch.Tensor:
        return self.v

    def append_block(self) -> None:
        return None

    def reset(self) -> None:
        self.k.zero_()
        self.v.zero_()
        self.offset.zero_()

    def repeat_interleave(self, n: int) -> None:
        self.k = self.k.repeat_interleave(n, dim=0)
        self.v = self.v.repeat_interleave(n, dim=0)

    def truncate(self, n_ctx: int):
        batch_size, _, n_kv_heads, d_head = self.k.shape
        assert batch_size == self.v.shape[0]
        assert n_ctx <= self.k.shape[1]
        self.k[:, n_ctx:, :, :].zero_()
        self.v[:, n_ctx:, :, :].zero_()
        self.offset.fill_(n_ctx)
        return self.k, self.v

    def extend(self, k: torch.Tensor, v: torch.Tensor):
        batch_size, n_ctx, *_rest = k.shape
        assert batch_size == self.k.shape[0]
        indices = torch.arange(0, n_ctx, device=k.device, dtype=torch.long) + self.offset
        self.k.index_copy_(1, indices, k)
        self.v.index_copy_(1, indices, v)
        self.offset.add_(n_ctx)
        return self.k, self.v


class PagedCache:
    """Per-layer KV cache with paged block management (nano-vllm approach).

    Physical storage
    ----------------
    k / v : [num_blocks, block_size, n_kv_heads, d_head]   (stride(1) = D)

    Kernel-compatible flat view
    ---------------------------
    k_flat / v_flat : [1, num_blocks * block_size, n_kv_heads, d_head]

    The fused decode kernel and attention kernels receive k_flat / v_flat
    unchanged — the existing strides are compatible because the underlying
    storage is contiguous and the flat view has the same element order.
    Unused slots contain zeros and are naturally masked out by causal
    attention (via start_q / cu_seqlens).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        d_head: int,
        device: torch.device | None = None,
    ):
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.device = device
        self.block_manager = BlockManager(num_blocks, block_size)
        self.k = torch.zeros(
            (num_blocks, block_size, n_kv_heads, d_head),
            dtype=torch.bfloat16, device=device,
        )
        self.v = torch.zeros(
            (num_blocks, block_size, n_kv_heads, d_head),
            dtype=torch.bfloat16, device=device,
        )
        self._k_flat = self.k.view(1, num_blocks * block_size, n_kv_heads, d_head)
        self._v_flat = self.v.view(1, num_blocks * block_size, n_kv_heads, d_head)
        self.block_table: list[int] = []
        self.window_start = 0
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)
        self._offset_int = 0
        # Pre-allocated slot tensor for decode steps (nano-vllm graph_vars pattern).
        # Updated by prepare_decode_slot() *before* each graph.replay() so that
        # the CUDA graph only sees pure GPU ops (no Python/D2H inside the graph).
        self._decode_slot = torch.zeros((1,), dtype=torch.int32, device=device)
        self._block_table_i32 = torch.empty((num_blocks,), dtype=torch.int32, device=device)
        self._block_table_long = torch.empty((num_blocks,), dtype=torch.long, device=device)
        self._block_table_dirty = False
        self._position_indices = torch.empty((0,), dtype=torch.int32, device=device)
        self._prefill_slot_mapping = torch.empty((0,), dtype=torch.int32, device=device)

    @property
    def block_size(self) -> int:
        return self.block_manager.block_size

    @property
    def k_flat(self) -> torch.Tensor:
        """[1, total_slots, n_kv_heads, d_head] — zero-copy view."""
        return self._k_flat

    @property
    def v_flat(self) -> torch.Tensor:
        return self._v_flat

    def reset(self) -> None:
        if self.block_table:
            self.block_manager.deallocate(self.block_table)
            # deallocate() clears block_table in place
        self.window_start = 0
        self.offset.zero_()
        self._offset_int = 0
        self._block_table_dirty = False

    def _slot(self, pos: int) -> int:
        bs = self.block_size
        local_pos = pos - self.window_start
        if local_pos < 0:
            raise ValueError(f"position {pos} is older than the retained window starting at {self.window_start}")
        return self.block_table[local_pos // bs] * bs + local_pos % bs

    def _visible_tokens(self, length: int | None = None) -> int:
        if length is None:
            length = self._offset_int
        return max(0, length - self.window_start)

    def _ensure_position_indices(self, length: int) -> None:
        if self._position_indices.numel() >= length:
            return
        new_size = max(length, max(1024, self._position_indices.numel() * 2))
        self._position_indices = torch.arange(
            new_size,
            dtype=torch.int32,
            device=self.k.device,
        )

    def _ensure_prefill_slot_mapping(self, length: int) -> None:
        if self._prefill_slot_mapping.numel() >= length:
            return
        new_size = max(length, max(1024, self._prefill_slot_mapping.numel() * 2))
        self._prefill_slot_mapping = torch.empty(
            (new_size,),
            dtype=torch.int32,
            device=self.k.device,
        )

    def _sync_block_table_device(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_blocks = len(self.block_table)
        if num_blocks == 0:
            return self._block_table_i32[:0], self._block_table_long[:0]
        if self._block_table_dirty:
            host_i32 = torch.tensor(self.block_table, dtype=torch.int32, device=self.k.device)
            host_long = host_i32.to(torch.long)
            self._block_table_i32[:num_blocks].copy_(host_i32)
            self._block_table_long[:num_blocks].copy_(host_long)
            self._block_table_dirty = False
        return self._block_table_i32[:num_blocks], self._block_table_long[:num_blocks]

    def _logical_kv_view(self, length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        visible_tokens = self._visible_tokens(length)
        if not self.block_table or visible_tokens == 0:
            empty_shape = (1, 0, self.n_kv_heads, self.d_head)
            return self.k.new_empty(empty_shape), self.v.new_empty(empty_shape)

        _, block_order = self._sync_block_table_device()
        k_view = self.k.index_select(0, block_order).view(1, -1, self.n_kv_heads, self.d_head)
        v_view = self.v.index_select(0, block_order).view(1, -1, self.n_kv_heads, self.d_head)
        return k_view[:, :visible_tokens], v_view[:, :visible_tokens]

    def _evict_oldest_block(self) -> None:
        if not self.block_table:
            raise RuntimeError("Cannot evict from an empty KV cache")
        oldest_bid = self.block_table.pop(0)
        block = self.block_manager.blocks[oldest_bid]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.block_manager._deallocate_block(oldest_bid)
        self.window_start += self.block_size
        self._block_table_dirty = True

    def _ensure_blocks(self, length: int) -> None:
        capacity = self.block_manager.capacity
        min_window_start = max(0, length - capacity)
        while self.window_start < min_window_start:
            self._evict_oldest_block()
        needed = math.ceil(max(0, length - self.window_start) / self.block_size) if length > 0 else 0
        while len(self.block_table) < needed:
            self.append_block()

    def append_block(self) -> None:
        if not self.block_manager.free_block_ids:
            self._evict_oldest_block()
        self.block_manager.append_block(self.block_table)
        self._block_table_dirty = True

    def set_decode_slot(self, offset: int) -> None:
        self._decode_slot[0] = self._slot(offset)

    def prepare_decode_slot(self, offset: int | None = None) -> None:
        """Compute and cache the slot for the next decode step.

        Must be called on the CPU *before* every graph.replay() (and before
        graph capture itself).  This mirrors nano-vllm's pattern of updating
        graph_vars outside the CUDA graph so that the graph body contains only
        pure GPU ops — no Python computation or D2H syncs.

        offset: pass the already-known token position to avoid a D2H sync.
                When None, falls back to self.offset.item() (only for
                standalone callers such as warmup / graph-capture setup).
        """
        if offset is None:
            offset = int(self.offset.item())   # D2H sync — only OK outside the graph
        self._ensure_blocks(offset + 1)
        self.set_decode_slot(offset)

    def extend(self, k: torch.Tensor, v: torch.Tensor):
        """Scatter k, v [1, n_ctx, n_kv_heads, d_head] into the paged cache.

        Decode path (n_ctx == 1)
        ------------------------
        Uses the pre-allocated self._decode_slot tensor which was filled by
        prepare_decode_slot() *outside* the CUDA graph.  No Python computation
        or D2H syncs happen here — the body is fully graph-capturable.

        Prefill path (n_ctx > 1)
        ------------------------
        Builds slot_mapping dynamically (fine; prefill is never graph-captured).
        Returns a slice covering only the filled region so attention doesn't pay
        memory-bandwidth for empty trailing slots.
        """
        _, n_ctx, *_ = k.shape
        if n_ctx == 1:
            # Decode: slot was pre-computed outside the graph
            store_kvcache(
                k.view(1, self.n_kv_heads, self.d_head),
                v.view(1, self.n_kv_heads, self.d_head),
                self.k, self.v, self._decode_slot,
            )
            self._offset_int += 1
            self.offset.add_(1)
            return self._logical_kv_view()
        else:
            # Prefill: compute slot_mapping entirely on GPU (no Python loop)
            offset = self._offset_int
            self._ensure_blocks(offset + n_ctx)
            bs = self.block_size
            bt, _ = self._sync_block_table_device()
            local_offset = offset - self.window_start
            self._ensure_position_indices(local_offset + n_ctx)
            self._ensure_prefill_slot_mapping(n_ctx)
            local_positions = self._position_indices[local_offset:local_offset + n_ctx]
            slot_mapping = self._prefill_slot_mapping[:n_ctx]
            slot_mapping.copy_(bt[local_positions // bs])
            slot_mapping.mul_(bs)
            slot_mapping.add_(local_positions.remainder(bs))
            store_kvcache(
                k.view(n_ctx, self.n_kv_heads, self.d_head),
                v.view(n_ctx, self.n_kv_heads, self.d_head),
                self.k, self.v, slot_mapping,
            )
            self._offset_int = offset + n_ctx
            self.offset.fill_(self._offset_int)
            new_len = offset + n_ctx
            return self._logical_kv_view(new_len)

    def truncate(self, n_ctx: int):
        """Truncate cache to n_ctx tokens, freeing excess blocks."""
        bs = self.block_size
        keep = math.ceil(n_ctx / bs) if n_ctx > 0 else 0
        while len(self.block_table) > keep:
            bid = self.block_table.pop()
            bm = self.block_manager
            bm.blocks[bid].ref_count -= 1
            if bm.blocks[bid].ref_count == 0:
                bm.used_block_ids.remove(bid)
                bm.free_block_ids.append(bid)
        self.window_start = min(self.window_start, n_ctx)
        self._block_table_dirty = True
        self._offset_int = n_ctx
        self.offset.fill_(n_ctx)
        return self._logical_kv_view(n_ctx)

    def repeat_interleave(self, n: int) -> None:
        raise NotImplementedError("repeat_interleave is not supported with PagedCache")


KVCache = DenseCache | PagedCache


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.layer_idx = layer_idx
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = QKV(config.hidden_size, qkv_dim, device=device)
        self.out = OUT(config.head_dim * config.num_attention_heads, config.hidden_size, device=device)
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    @maybe_record_decorator("attn")
    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        batch_size, n_ctx, dim = x.shape
        fused_decode = False

        t = self.norm(x)
        with maybe_record_function("qkv"):
            q_dim = self.num_attention_heads * self.head_dim
            kv_dim = self.num_key_value_heads * self.head_dim
            if (
                cache is not None
                and n_ctx == 1
                and t.is_cuda
                and t.dtype == torch.bfloat16
            ):
                fused_decode = True
                offset = cache.offset
                q = qkv_rope_cache_decode_forward(
                    t,
                    self.qkv.weight,
                    self.qkv.bias,
                    self.rope.sin,
                    self.rope.cos,
                    self.rope.max_context_length,
                    cache.k_flat,
                    cache.v_flat,
                    offset,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                )
            else:
                qkv = self.qkv(t)
                q, k, v = torch.split(qkv, (q_dim, kv_dim, kv_dim), dim=-1)
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        q = q.view(batch_size, n_ctx, self.num_attention_heads, self.head_dim)
        if fused_decode:
            k = cache.k_flat
            v = cache.v_flat
        else:
            k = k.view(batch_size, n_ctx, self.num_key_value_heads, self.head_dim)
            v = v.view(batch_size, n_ctx, self.num_key_value_heads, self.head_dim)

        if fused_decode:
            pass
        elif cache is not None:
            offset = cache.offset.clone()
            q, k = self.rope(q, k, offset=offset)
            k, v = cache.extend(k, v)
        else:
            offset = torch.zeros((1,), dtype=torch.long, device=x.device)
            q, k = self.rope(q, k, offset=offset)

        q = q.view(
            batch_size,
            n_ctx,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        with maybe_record_function("attn_kernel"):
            if cache is not None and n_ctx == 1:
                t = attention_decode(
                    q,
                    k,
                    v,
                    self.sinks,
                    self.sm_scale,
                    self.sliding_window,
                    offset,
                )
            elif n_ctx <= 8:
                t = attention_ref(
                    q,
                    k,
                    v,
                    self.sinks,
                    self.sm_scale,
                    self.sliding_window,
                    offset,
                )
            else:
                t = attention(
                    q,
                    k,
                    v,
                    self.sinks,
                    self.sm_scale,
                    self.sliding_window,
                    offset,
                )

        with maybe_record_function("c_proj"):
            if fused_decode:
                t = self.out.decode_residual(t, x)
                cache.offset.add_(1)
            else:
                t = self.out(t)
                t = x + t
        return t


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.ParameterDict({
            "weight": torch.nn.Parameter(
                torch.empty(
                    (config.hidden_size, config.num_experts),
                    device=device,
                    dtype=torch.bfloat16,
                )
            ),
            "bias": torch.nn.Parameter(
                torch.empty(
                    (config.num_experts,),
                    device=device,
                    dtype=torch.bfloat16,
                )
            ),
        })
        self.register_buffer(
            "gate_bias_fp32_cache",
            self.gate["bias"].detach().float().clone(),
            persistent=False,
        )
        self._gate_bias_fp32_version = self._maybe_tensor_version(self.gate["bias"])
        self.mlp1_weight_tensor, self.mlp1_weight_mx = quantize_mx4(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size * 2,
                ),
                device=device,
                dtype=torch.bfloat16,
            ),
        )
        self.mlp1_weight = torch.nn.Parameter(self.mlp1_weight_tensor.storage.data, requires_grad=False)
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.register_buffer(
            "mlp1_bias_fp32_cache",
            self.mlp1_bias.detach().float().clone(),
            persistent=False,
        )
        self._mlp1_bias_fp32_version = self._maybe_tensor_version(self.mlp1_bias)
        self.mlp2_weight_tensor, self.mlp2_weight_mx = quantize_mx4(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            ),
        )
        self.mlp2_weight = torch.nn.Parameter(self.mlp2_weight_tensor.storage.data, requires_grad=False)
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.register_buffer(
            "mlp2_bias_fp32_cache",
            self.mlp2_bias.detach().float().clone(),
            persistent=False,
        )
        self._mlp2_bias_fp32_version = self._maybe_tensor_version(self.mlp2_bias)

    @staticmethod
    def _maybe_tensor_version(param: torch.Tensor) -> int | None:
        try:
            return param._version
        except RuntimeError:
            return None

    def _refresh_fp32_bias_cache(
        self,
        param: torch.Tensor,
        cache_name: str,
        version_name: str,
    ) -> torch.Tensor:
        cache = getattr(self, cache_name)
        if cache.device != param.device or cache.shape != param.shape:
            cache = torch.empty_like(param, dtype=torch.float32, device=param.device)
            setattr(self, cache_name, cache)
        cache.copy_(param.detach())
        setattr(self, version_name, self._maybe_tensor_version(param))
        return cache

    def refresh_fp32_bias_caches(self) -> None:
        self._refresh_fp32_bias_cache(
            self.gate["bias"],
            "gate_bias_fp32_cache",
            "_gate_bias_fp32_version",
        )
        self._refresh_fp32_bias_cache(
            self.mlp1_bias,
            "mlp1_bias_fp32_cache",
            "_mlp1_bias_fp32_version",
        )
        self._refresh_fp32_bias_cache(
            self.mlp2_bias,
            "mlp2_bias_fp32_cache",
            "_mlp2_bias_fp32_version",
        )

    def _get_fp32_bias_cache(
        self,
        param: torch.Tensor,
        cache_name: str,
        version_name: str,
    ) -> torch.Tensor:
        cache = getattr(self, cache_name)
        if cache.device != param.device or cache.shape != param.shape:
            return self._refresh_fp32_bias_cache(param, cache_name, version_name)

        version = self._maybe_tensor_version(param)
        if version is None:
            return cache
        if getattr(self, version_name) != version:
            return self._refresh_fp32_bias_cache(param, cache_name, version_name)
        return cache

    def _get_gate_bias_fp32(self) -> torch.Tensor:
        return self._get_fp32_bias_cache(
            self.gate["bias"],
            "gate_bias_fp32_cache",
            "_gate_bias_fp32_version",
        )

    def _get_mlp1_bias_fp32(self) -> torch.Tensor:
        return self._get_fp32_bias_cache(
            self.mlp1_bias,
            "mlp1_bias_fp32_cache",
            "_mlp1_bias_fp32_version",
        )

    def _get_mlp2_bias_fp32(self) -> torch.Tensor:
        return self._get_fp32_bias_cache(
            self.mlp2_bias,
            "mlp2_bias_fp32_cache",
            "_mlp2_bias_fp32_version",
        )

    @maybe_record_decorator("mlp")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_ctx, dim = x.shape
        t = self.norm(x)
        gate_bias = self._get_gate_bias_fp32()
        mlp1_bias = self._get_mlp1_bias_fp32()
        mlp2_bias = self._get_mlp2_bias_fp32()

        t = t.view(batch_size * n_ctx, dim)
        if (
            x.is_cuda
            and n_ctx == 1
            and x.dtype == torch.bfloat16
            and self.gate["weight"].dtype == torch.bfloat16
        ):
            t = moe_decode(
                t,
                self.gate["weight"],
                self.mlp1_weight_tensor, self.mlp1_weight_mx,
                self.mlp2_weight_tensor, self.mlp2_weight_mx,
                gate_bias,
                mlp1_bias,
                mlp2_bias,
                experts_per_token=self.experts_per_token,
                num_experts=self.num_experts,
                swiglu_limit=self.swiglu_limit,
            )
        else:
            t = moe(
                t,
                self.gate["weight"],
                self.mlp1_weight_tensor, self.mlp1_weight_mx,
                self.mlp2_weight_tensor, self.mlp2_weight_mx,
                gate_bias,
                mlp1_bias,
                mlp2_bias,
                experts_per_token=self.experts_per_token,
                num_experts=self.num_experts,
                swiglu_limit=self.swiglu_limit,
            )
        t = t.view(batch_size, n_ctx, dim)

        return x + t


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, layer_idx, device)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = self.attn(x, cache=cache)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = UnEmbedding(config.hidden_size, config.vocab_size, device=device)

    def _forward_hidden(
        self,
        x: torch.Tensor,
        caches: list[KVCache] | None = None,
    ) -> torch.Tensor:
        caches = caches or [None] * len(self.block)
        x = self.embedding(x)
        for block, cache in zip(self.block, caches):
            x = block(x, cache=cache)
        return self.norm(x)

    def forward(self, x: torch.Tensor, caches: list[KVCache] | None = None) -> torch.Tensor:
        x = self._forward_hidden(x, caches)
        x = self.unembedding(x)
        return x.float()

    def prefill(self, x: torch.Tensor, caches: list[KVCache] | None = None) -> torch.Tensor:
        """Populate KV cache for the prompt without paying vocab projection cost."""
        return self._forward_hidden(x, caches)

    @staticmethod
    def from_checkpoint(
        path: str, config: ModelConfig | None = None, device: str | torch.device = "cuda",
    ) -> "Transformer":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        if config is None:
            config_path = os.path.join(path, "config.json")
            with open(config_path, "r") as f:
                json_config = json.load(f)
                config = ModelConfig(**json_config)

        model = Transformer(config=config, device=device)
        model.eval()

        checkpoint = Checkpoint(path, device)

        for name, param in model.named_parameters():
            torch.cuda.empty_cache()
            loaded_tensor = checkpoint.get(name)

            if "mlp1" in name:
                if "weight" in name:
                    loaded_tensor, scales = quantize_mx4(loaded_tensor.mT.contiguous())
                    _, block_index, _, _ = name.split(".")
                    model.block[int(block_index)].mlp.mlp1_weight_mx = scales
                    with torch.no_grad():
                        param.copy_(loaded_tensor.storage.data)
                else:
                    with torch.no_grad():
                        param.copy_(loaded_tensor)

            elif "mlp2_weight" in name:
                loaded_tensor, scales = quantize_mx4(loaded_tensor.mT.contiguous())
                _, block_index, _, _ = name.split(".")
                model.block[int(block_index)].mlp.mlp2_weight_mx = scales
                with torch.no_grad():
                    param.copy_(loaded_tensor.storage.data)

            elif "gate" in name and loaded_tensor.ndim == 2:
                loaded_tensor = loaded_tensor.mT.contiguous()
                with torch.no_grad():
                    param.copy_(loaded_tensor)

            else:
                with torch.no_grad():
                    param.copy_(loaded_tensor)

        for block in model.block:
            block.mlp.refresh_fp32_bias_caches()

        # NOTE: Required to avoid OOM errors
        torch.cuda.empty_cache()
        return model


class TokenGenerator:
    @torch.inference_mode()
    def __init__(
        self,
        checkpoint: str,
        context: int,
        device: torch.device,
        gpu_memory_utilization: float | None = None,
    ):
        self.device = device
        print(termcolor.colored("Loading model checkpoint...", "yellow"), flush=True)
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)
        cache_env = os.getenv("USE_PAGED_CACHE", "1").strip().lower()
        self.use_paged_cache = cache_env not in {"0", "false", "no", "off", ""}
        cache_mode = "paged" if self.use_paged_cache else "dense"
        print(termcolor.colored(f"KV cache implementation: {cache_mode}", "yellow"), flush=True)
        # By default allocate exactly enough KV cache for the requested context.
        # Passing gpu_memory_utilization opts into the more aggressive auto-sizing mode.
        self.caches = self._allocate_kv_cache(
            min_context=context,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.block_size = self.caches[0].block_size if self.use_paged_cache else 0
        self.input_token = torch.zeros(1, dtype=torch.int32, device=self.device)
        # Warm up and capture the single-token decode graph with the first block allocated.
        for cache in self.caches:
            cache.append_block()
        self.model(self.input_token[None, :], caches=self.caches)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self.model(self.input_token[None, :], caches=self.caches)[0]
        self._sampling_probs = torch.empty_like(self.logits[0])

    @property
    def max_model_len(self) -> int:
        """Maximum total sequence length (prompt + generated tokens).

        Bounded by two independent limits that must both hold:
          • GPU KV-cache capacity  — how many token slots we allocated
          • RoPE max_context_length — positions beyond this have no valid
            positional encoding and were not seen during training
        One decode slot is reserved, so the usable prompt budget is
        max_model_len - 1.
        """
        rope_max = self.model.block[0].attn.rope.max_context_length
        if self.use_paged_cache:
            kv_capacity = self.caches[0].block_manager.capacity
        else:
            kv_capacity = self.caches[0].k.shape[1]
        return min(rope_max, kv_capacity)

    @torch.inference_mode()
    def _allocate_kv_cache(
        self,
        min_context: int = 8192,
        block_size: int = 256,
        gpu_memory_utilization: float | None = None,
    ) -> list[KVCache]:
        """Allocate paged KV cache.

        By default, allocate exactly enough blocks to satisfy min_context.
        If gpu_memory_utilization is provided, use the previous auto-sizing mode
        that grows the cache to consume a fraction of free GPU memory.
        """
        config = self.model.config
        n_kv_heads = config.num_key_value_heads
        d_head = config.head_dim
        num_layers = len(self.model.block)
        rope_max = self.model.block[0].attn.rope.max_context_length
        if not self.use_paged_cache:
            context = min(min_context, rope_max)
            bytes_per_layer = 2 * context * n_kv_heads * d_head * 2
            print(termcolor.colored(
                f"KV cache: dense mode, {context:,} max context tokens "
                f"(rope limit: {rope_max:,}, {bytes_per_layer / 1024 / 1024:.2f} MiB/layer)",
                "yellow",
            ), flush=True)
            return [
                DenseCache(1, context, n_kv_heads, d_head, device=self.device)
                for _ in range(num_layers)
            ]

        min_blocks = max(1, math.ceil(min_context / block_size))

        # Bytes for one block in one layer (K + V, bfloat16 = 2 bytes)
        bytes_per_layer_block = 2 * block_size * n_kv_heads * d_head * 2

        if gpu_memory_utilization is None:
            num_blocks = min(min_blocks, math.ceil(rope_max / block_size))
            mode_label = "exact context mode"
        else:
            free, total = torch.cuda.mem_get_info()
            stats = torch.cuda.memory_stats()
            peak = stats.get("allocated_bytes.all.peak", 0)
            current = stats.get("allocated_bytes.all.current", 0)
            used = total - free
            available = int(total * gpu_memory_utilization - used - peak + current)
            num_blocks = max(min_blocks, available // bytes_per_layer_block)
            mode_label = f"auto mode ({gpu_memory_utilization:.2f} GPU mem)"

        # Never allocate more blocks than the model's RoPE can address
        num_blocks = min(num_blocks, math.ceil(rope_max / block_size))

        print(termcolor.colored(
            f"KV cache: {mode_label}, {num_blocks} blocks/layer × {block_size} tokens/block "
            f"= {num_blocks * block_size:,} max context tokens (rope limit: {rope_max:,}, "
            f"{bytes_per_layer_block / 1024:.1f} KiB/block/layer)",
            "yellow",
        ), flush=True)

        return [
            PagedCache(num_blocks, block_size, n_kv_heads, d_head, device=self.device)
            for _ in range(num_layers)
        ]

    @torch.inference_mode()
    def sample_next_token(self, logits: torch.Tensor, temperature: float) -> int:
        """Executed only on rank 0."""
        logits = logits[-1]
        if temperature == 0.0:
            return logits.argmax().item()
        self._sampling_probs.copy_(logits)
        self._sampling_probs.div_(temperature)
        torch.softmax(self._sampling_probs, dim=-1, out=self._sampling_probs)
        return torch.multinomial(self._sampling_probs, num_samples=1).item()

    def _prepare_decode_slots(self, decode_offset: int) -> None:
        if not self.use_paged_cache:
            return
        if decode_offset % self.block_size == 0:
            for cache in self.caches:
                cache.append_block()

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int] | None = None,
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        stop_tokens = stop_tokens or []
        for cache in self.caches:
            cache.reset()
        # max_model_len - 1: leave one slot for the first decode step
        max_prompt = self.max_model_len - 1
        if len(prompt_tokens) > max_prompt:
            raise ValueError(
                f"Prompt is too long: {len(prompt_tokens)} tokens "
                f"exceeds max_model_len={self.max_model_len}. "
                "Truncate the conversation history before calling generate()."
            )
        prompt_tokens = torch.as_tensor(prompt_tokens, dtype=torch.int32, device=self.device)
        predicted_token = prompt_tokens[-1]
        decode_offset = prompt_tokens.numel() - 1
        if decode_offset > 0:
            self.model.prefill(prompt_tokens[None, :-1], self.caches)
        num_generated_tokens = 0

        while max_tokens == 0 or num_generated_tokens < max_tokens:
            if decode_offset + 1 >= self.max_model_len:
                print(
                    termcolor.colored(
                        f"Decode stopped at max_model_len={self.max_model_len}",
                        "yellow",
                    ),
                    flush=True,
                )
                break
            self.input_token[0] = predicted_token
            self._prepare_decode_slots(decode_offset)
            self.graph.replay()
            predicted_token = self.sample_next_token(self.logits, temperature)
            decode_offset += 1
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(self.logits[-1, :], dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break
