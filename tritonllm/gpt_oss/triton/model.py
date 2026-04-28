import json
import math
import os
import termcolor
import time
from dataclasses import dataclass

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from gpt_oss.triton.weights import Checkpoint
from gpt_oss.triton.attention_ref import attention_ref
from gpt_oss.triton.attention_tt_decode import attention_decode

from triton_kernels.target_info import cuda_capability_geq, cuda_capability_eq
if not cuda_capability_geq(9):
    from gpt_oss.triton.attention_tt import attention
else:
    from gpt_oss.triton.attention_tt_tma import attention

from gpt_oss.triton.moe import quantize_mx4, moe, moe_decode, moe_decode_gate_routing, moe_decode_experts, moe_gate_routing, moe_experts
from gpt_oss.triton.expert_cache import ExpertCache
from gpt_oss.triton.triton_kernels import (
    out_residual_decode_forward,
    qkv_rope_cache_decode_forward,
    qkv_decode_forward,
    rmsnorm_forward,
    rope_forward,
    unembedding_decode_forward,
    unembedding_decode_fp8_forward,
    unembedding_decode_argmax_forward,
    unembedding_decode_fp8_argmax_forward,
    _quantize_unembed_fp8,
)

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

    @record_function("rmsnorm_triton")
    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return rmsnorm_forward(x, self.scale, self.eps, out=out)


class UnEmbedding(torch.nn.Module):
    def __init__(
        self, hidden_size: int, vocab_size: int, device: torch.device | None = None
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((vocab_size, hidden_size), device=device, dtype=torch.bfloat16)
        )
        self._use_fp8 = os.getenv("UNEMBED_FP8", "0").strip().lower() not in {
            "0", "false", "no", "off",
        }
        self._weight_fp8: torch.Tensor | None = None
        self._weight_scale: torch.Tensor | None = None

    @record_function("unembedding_linear")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.ndim == 3
            and x.shape[1] == 1
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            if self._use_fp8:
                if self._weight_fp8 is None:
                    self._weight_fp8, self._weight_scale = _quantize_unembed_fp8(self.weight)
                return unembedding_decode_fp8_forward(x, self._weight_fp8, self._weight_scale)
            return unembedding_decode_forward(x, self.weight)
        return torch.nn.functional.linear(x, self.weight, bias=None)

    @record_function("unembedding_argmax")
    def forward_argmax(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.ndim == 3
            and x.shape[1] == 1
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            if self._use_fp8:
                if self._weight_fp8 is None:
                    self._weight_fp8, self._weight_scale = _quantize_unembed_fp8(self.weight)
                return unembedding_decode_fp8_argmax_forward(x, self._weight_fp8, self._weight_scale)
            return unembedding_decode_argmax_forward(x, self.weight)
        logits = self.forward(x)
        return logits[-1].argmax(dim=-1)


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
        self._use_fp8 = os.getenv("ATTN_FP8", "0").strip().lower() not in {
            "0", "false", "no", "off",
        }
        self._weight_fp8: torch.Tensor | None = None
        self._weight_scale: torch.Tensor | None = None

    @record_function("qkv_linear")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

    @record_function("qkv_linear_decode")
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

    @property
    def _fp8_weight(self):
        if not self._use_fp8:
            return None, None
        if self._weight_fp8 is None:
            self._weight_fp8, self._weight_scale = _quantize_unembed_fp8(self.weight)
        return self._weight_fp8, self._weight_scale


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
        self._use_fp8 = os.getenv("ATTN_FP8", "0").strip().lower() not in {
            "0", "false", "no", "off",
        }
        self._weight_fp8: torch.Tensor | None = None
        self._weight_scale: torch.Tensor | None = None

    @property
    def _fp8_weight(self):
        if not self._use_fp8:
            return None, None
        if self._weight_fp8 is None:
            self._weight_fp8, self._weight_scale = _quantize_unembed_fp8(self.weight)
        return self._weight_fp8, self._weight_scale

    @record_function("out_linear")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

    @record_function("out_linear_decode")
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
            if self._use_fp8:
                w_fp8, w_scale = self._fp8_weight
                return out_residual_decode_forward(x, w_fp8, self.bias, residual, weight_scale=w_scale)
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

    @record_function("rotate")
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

    @record_function("rope_triton")
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        offset: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_forward(query, key, self.sin, self.cos, self.max_context_length, offset)
        return query, key


class Cache:
    def __init__(self, batch_size, n_ctx, n_kv_heads, d_head=64, device: torch.device | None = None):
        self.k = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.v = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)
        sliding_env = os.getenv("DENSE_CACHE_SLIDING", "1").strip().lower()
        self.enable_sliding_window = sliding_env not in {"0", "false", "no", "off", ""}
        default_slide_chunk = max(256, min(n_ctx // 8, 4096))
        slide_chunk_env = os.getenv("DENSE_CACHE_SLIDE_CHUNK")
        slide_chunk = int(slide_chunk_env) if slide_chunk_env is not None else default_slide_chunk
        self.slide_chunk = max(1, min(slide_chunk, n_ctx - 1)) if n_ctx > 1 else 0
        self._slide_k = self._slide_v = None
        self._ensure_slide_buffers()

    def reset(self):
        self.k.zero_()
        self.v.zero_()
        self.offset.zero_()

    def repeat_interleave(self, n):
        """Repeat each cache entry n times along the batch dimension."""
        self.k = self.k.repeat_interleave(n, dim=0)
        self.v = self.v.repeat_interleave(n, dim=0)
        self._ensure_slide_buffers()

    def truncate(self, n_ctx):
        """Truncate the cache to the first n_ctx tokens."""
        batch_size, _, n_kv_heads, d_head = self.k.shape
        assert batch_size == self.v.shape[0]
        assert n_ctx <= self.k.shape[1]
        self.k[:, n_ctx:, :, :].zero_()
        self.v[:, n_ctx:, :, :].zero_()
        self.offset.fill_(n_ctx)
        return self.k, self.v

    def can_slide(self) -> bool:
        return self.enable_sliding_window and self.slide_chunk > 0

    def _ensure_slide_buffers(self) -> None:
        """Keep the sliding-window staging buffers in sync with the KV cache."""
        if not self.can_slide():
            self._slide_k = self._slide_v = None
            return

        shape = (self.k.shape[0], self.slide_chunk, self.k.shape[2], self.k.shape[3])
        if self._slide_k is not None and self._slide_k.shape == shape:
            return

        self._slide_k = self.k.new_empty(shape)
        self._slide_v = self.v.new_empty(shape)

    def slide_window(self, keep_last_n: int | None = None, current: int | None = None) -> int:
        """Drop the oldest KV entries and keep only the most recent suffix.

        ``current`` may be supplied by the caller (e.g. from a CPU-side
        decode_offset counter) to avoid the GPU→CPU sync of offset.item().
        """
        if current is None:
            current = int(self.offset.item())
        capacity = self.k.shape[1]
        if current < capacity:
            return current
        if not self.can_slide():
            return current
        if keep_last_n is None:
            keep_last_n = capacity - self.slide_chunk
        keep_last_n = max(1, min(keep_last_n, capacity - 1, current))
        self._ensure_slide_buffers()
        start = current - keep_last_n  # first token to keep
        chunk = self.slide_chunk
        # Move the retained suffix in chunk-sized blocks via a reusable staging
        # buffer, avoiding a per-slide clone() allocation on the hot path.
        pos = 0
        while pos < keep_last_n:
            n = min(chunk, keep_last_n - pos)
            src = start + pos
            self._slide_k[:, :n].copy_(self.k[:, src:src + n])
            self._slide_v[:, :n].copy_(self.v[:, src:src + n])
            self.k[:, pos:pos + n].copy_(self._slide_k[:, :n])
            self.v[:, pos:pos + n].copy_(self._slide_v[:, :n])
            pos += n
        self.k[:, keep_last_n:, :, :].zero_()
        self.v[:, keep_last_n:, :, :].zero_()
        self.offset.fill_(keep_last_n)
        return keep_last_n

    def extend(self, k, v):
        batch_size, n_ctx, *_rest = k.shape
        assert batch_size == self.k.shape[0]
        indices = torch.arange(0, n_ctx, device=k.device, dtype=torch.long) + self.offset
        self.k.index_copy_(1, indices, k)
        self.v.index_copy_(1, indices, v)
        self.offset.add_(n_ctx)
        return self.k, self.v


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

    @record_function("attn")
    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        batch_size, n_ctx, dim = x.shape
        fused_decode = False

        t = self.norm(x)
        with record_function("qkv"):
            q_dim = self.num_attention_heads * self.head_dim
            kv_dim = self.num_key_value_heads * self.head_dim
            if (
                cache is not None
                and n_ctx == 1
                and t.is_cuda
                and t.dtype == torch.bfloat16
            ):
                fused_decode = True
                offset = cache.offset.clone()
                qkv_w, qkv_ws = self.qkv._fp8_weight
                q = qkv_rope_cache_decode_forward(
                    t,
                    qkv_w if qkv_ws is not None else self.qkv.weight,
                    self.qkv.bias,
                    self.rope.sin,
                    self.rope.cos,
                    self.rope.max_context_length,
                    cache.k,
                    cache.v,
                    offset,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    weight_scale=qkv_ws,
                )
                cache.offset.add_(1)
            else:
                qkv = self.qkv(t)
                q, k, v = torch.split(qkv, (q_dim, kv_dim, kv_dim), dim=-1)
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        q = q.view(batch_size, n_ctx, self.num_attention_heads, self.head_dim)
        if fused_decode:
            k = cache.k
            v = cache.v
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
        with record_function("attn_kernel"):
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

        with record_function("c_proj"):
            if fused_decode:
                t = self.out.decode_residual(t, x)
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
        cpu_offload: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self._cpu_offload = cpu_offload
        self.expert_cache: ExpertCache | None = None
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
        if cpu_offload:
            self.mlp1_weight_tensor = torch.empty(1, device=device)
            self.mlp1_weight_mx = torch.empty(1, device=device)
            self.mlp1_weight = torch.nn.Parameter(torch.empty(1, device=device), requires_grad=False)
            self.mlp1_bias = torch.nn.Parameter(torch.empty(1, device=device), requires_grad=False)
            self.mlp2_weight_tensor = torch.empty(1, device=device)
            self.mlp2_weight_mx = torch.empty(1, device=device)
            self.mlp2_weight = torch.nn.Parameter(torch.empty(1, device=device), requires_grad=False)
            self.mlp2_bias = torch.nn.Parameter(torch.empty(1, device=device), requires_grad=False)
        else:
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
            "mlp1_bias_fp32_cache",
            self.mlp1_bias.detach().float().clone(),
            persistent=False,
        )
        self._mlp1_bias_fp32_version = self._maybe_tensor_version(self.mlp1_bias)
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

    @record_function("mlp")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_ctx, dim = x.shape
        t = self.norm(x)
        gate_bias = self._get_gate_bias_fp32()

        t = t.view(batch_size * n_ctx, dim)
        if (
            x.is_cuda
            and n_ctx == 1
            and x.dtype == torch.bfloat16
            and self.gate["weight"].dtype == torch.bfloat16
        ):
            if self._cpu_offload and self.expert_cache is not None:
                t = self._forward_decode_offload(t, gate_bias)
            else:
                mlp1_bias = self._get_mlp1_bias_fp32()
                mlp2_bias = self._get_mlp2_bias_fp32()
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
            if self._cpu_offload and self.expert_cache is not None:
                rdata, gather_indx, scatter_indx = moe_gate_routing(
                    t, self.gate["weight"], gate_bias,
                    experts_per_token=self.experts_per_token,
                )
                if rdata is not None:
                    expert_ids = torch.where(rdata.expt_hist > 0)[0]
                    self.expert_cache.ensure_experts(self.layer_idx, expert_ids.tolist())
                    t = moe_experts(
                        t,
                        self.expert_cache.gpu_w1, self.expert_cache.gpu_w1_mx,
                        self.expert_cache.gpu_w2, self.expert_cache.gpu_w2_mx,
                        self.expert_cache.gpu_b1,
                        self.expert_cache.gpu_b2,
                        rdata, gather_indx, scatter_indx,
                        swiglu_limit=self.swiglu_limit,
                    )
            else:
                mlp1_bias = self._get_mlp1_bias_fp32()
                mlp2_bias = self._get_mlp2_bias_fp32()
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

        x.add_(t)
        return x

    def _forward_decode_offload(self, t: torch.Tensor, gate_bias: torch.Tensor) -> torch.Tensor:
        cache = self.expert_cache

        if cache.route_graph is None:
            # First call: run routing + matmul eagerly, then capture both graphs
            rdata, gather_indx, scatter_indx = moe_decode_gate_routing(
                t, self.gate["weight"], gate_bias,
                experts_per_token=self.experts_per_token,
                num_experts=self.num_experts,
            )
            if rdata is None:
                return t
            expert_ids = torch.where(rdata.expt_hist > 0)[0]
            cache.ensure_experts(self.layer_idx, expert_ids.tolist())

            t_out = moe_decode_experts(
                t,
                cache.gpu_w1, cache.gpu_w1_mx,
                cache.gpu_w2, cache.gpu_w2_mx,
                cache.gpu_b1, cache.gpu_b2,
                rdata, gather_indx, scatter_indx,
                swiglu_limit=self.swiglu_limit,
            )

            cache.init_route_graph(
                t, self.gate["weight"], gate_bias,
                experts_per_token=self.experts_per_token,
                num_experts=self.num_experts,
            )
            cache.init_decode_graph(rdata, gather_indx, scatter_indx, swiglu_limit=self.swiglu_limit)
            cache._g_x.copy_(t)
            cache._g_out.copy_(t_out)
            return t_out

        # Replay path: route graph → copy experts → matmul graph
        cache._gr_t.copy_(t)
        cache._gr_wg.copy_(self.gate["weight"])
        if cache._gr_bg is not None and gate_bias is not None:
            cache._gr_bg.copy_(gate_bias)

        cache.route_graph.replay()
        expert_ids = torch.where(cache._gr_rdata.expt_hist > 0)[0]
        cache.ensure_experts(self.layer_idx, expert_ids.tolist())

        cache.route_to_matmul_proxies()
        cache._g_x.copy_(t)
        cache.decode_graph.replay()

        if self.layer_idx == 0 and cache._miss_count + cache._hit_count > 0:
            if os.getenv("PRINT_HIT_RATE", "0").strip().lower() not in {"0", "false", "no", "off"}:
                if getattr(cache, '_print_ctr', 0) % 10 == 0:
                    hr = cache.hit_rate()
                    print(f"[gpu_layer reuse] hit={cache._hit_count} miss={cache._miss_count} "
                          f"rate={hr:.1%}")
            cache._print_ctr = getattr(cache, '_print_ctr', 0) + 1
            cache.reset_hit_rate()

        return cache._g_out

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
        cpu_offload: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, layer_idx, device, cpu_offload=cpu_offload)

    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        x = self.attn(x, cache=cache)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
        cpu_offload: bool = False,
    ):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device, cpu_offload=cpu_offload)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = UnEmbedding(config.hidden_size, config.vocab_size, device=device)

    def _forward_hidden(self, x: torch.Tensor, caches: list[Cache] | None = None) -> torch.Tensor:
        caches = caches or [None] * len(self.block)
        x = self.embedding(x)
        for block, cache in zip(self.block, caches):
            x = block(x, cache=cache)
        return self.norm(x, out=x)

    def forward(self, x: torch.Tensor, caches: list[Cache] | None = None) -> torch.Tensor:
        x = self._forward_hidden(x, caches)
        x = self.unembedding(x)
        return x.float()

    @record_function("forward_greedy")
    def forward_greedy(self, x: torch.Tensor, caches: list[Cache] | None = None) -> torch.Tensor:
        """Fused forward that returns token IDs instead of logits (for greedy T=0)."""
        x = self._forward_hidden(x, caches)
        return self.unembedding.forward_argmax(x)

    def prefill(self, x: torch.Tensor, caches):
        """Populate KV cache for the prompt without paying vocab projection cost."""
        return self._forward_hidden(x, caches)

    @staticmethod
    def from_checkpoint(
        path: str,
        config: ModelConfig | None = None,
        device: str | torch.device = "cuda",
        cpu_offload: bool = False,
    ) -> "Transformer":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        if config is None:
            config_path = os.path.join(path, "config.json")
            with open(config_path, "r") as f:
                json_config = json.load(f)
                config = ModelConfig(**json_config)

        model = Transformer(config=config, device=device, cpu_offload=cpu_offload)
        model.eval()

        checkpoint = Checkpoint(path, device)

        if cpu_offload:
            model._load_with_cpu_offload(checkpoint, config)
        else:
            model._load_gpu_only(checkpoint)

        return model

    def _load_gpu_only(self, checkpoint: Checkpoint) -> None:
        for name, param in self.named_parameters():
            torch.cuda.empty_cache()
            loaded_tensor = checkpoint.get(name)

            if "mlp1" in name:
                if "weight" in name:
                    loaded_tensor, scales = quantize_mx4(loaded_tensor.mT.contiguous())
                    _, block_index, _, _ = name.split(".")
                    self.block[int(block_index)].mlp.mlp1_weight_mx = scales
                    with torch.no_grad():
                        param.copy_(loaded_tensor.storage.data)
                else:
                    with torch.no_grad():
                        param.copy_(loaded_tensor)

            elif "mlp2_weight" in name:
                loaded_tensor, scales = quantize_mx4(loaded_tensor.mT.contiguous())
                _, block_index, _, _ = name.split(".")
                self.block[int(block_index)].mlp.mlp2_weight_mx = scales
                with torch.no_grad():
                    param.copy_(loaded_tensor.storage.data)

            elif "gate" in name and loaded_tensor.ndim == 2:
                loaded_tensor = loaded_tensor.mT.contiguous()
                with torch.no_grad():
                    param.copy_(loaded_tensor)

            else:
                with torch.no_grad():
                    param.copy_(loaded_tensor)

        for block in self.block:
            block.mlp.refresh_fp32_bias_caches()

        torch.cuda.empty_cache()

    def _load_with_cpu_offload(self, checkpoint: Checkpoint, config: ModelConfig) -> None:
        device = torch.device("cuda")
        cache = ExpertCache(
            num_experts=config.num_experts,
            experts_per_token=config.experts_per_token,
            device=device,
        )

        # Determine how many hot layers can be cached on GPU
        _n_cache = cache.init_layer_cache(max_layers=10)

        # Phase 1: load non-MLP params normally (small, stay on GPU)
        for name, param in self.named_parameters():
            if "mlp1" in name or "mlp2" in name:
                continue
            torch.cuda.empty_cache()
            loaded_tensor = checkpoint.get(name)
            if "gate" in name and loaded_tensor.ndim == 2:
                loaded_tensor = loaded_tensor.mT.contiguous()
            with torch.no_grad():
                param.copy_(loaded_tensor)
            del loaded_tensor

        _after_p1 = torch.cuda.memory_stats().get('allocated_bytes.all.current', 0)
        print(f"GPU after Phase 1: {_after_p1/1e9:.2f} GB")

        # Offload large non-MLP tensors to CPU to free GPU memory for quantization
        # Embedding + UnEmbedding (~2.32 GB) + 36 layers attention QKV+OUT (~1.9 GB)
        _cpu_offload_tensors: dict[str, torch.Tensor] = {}

        def _move_to_cpu(model, attr_path):
            obj = model
            for part in attr_path.split(".")[:-1]:
                obj = getattr(obj, part)
            attr = attr_path.split(".")[-1]
            param = getattr(obj, attr)
            cpu_copy = param.data.cpu()
            _cpu_offload_tensors[attr_path] = cpu_copy
            param.data = torch.empty(1, device=device, dtype=param.dtype)

        _move_to_cpu(self, "embedding.weight")
        _move_to_cpu(self, "unembedding.weight")
        for i in range(config.num_hidden_layers):
            _move_to_cpu(self, f"block.{i}.attn.qkv.weight")
            _move_to_cpu(self, f"block.{i}.attn.qkv.bias")
            _move_to_cpu(self, f"block.{i}.attn.out.weight")
            _move_to_cpu(self, f"block.{i}.attn.out.bias")
        torch.cuda.empty_cache()

        _after_offload = torch.cuda.memory_stats().get('allocated_bytes.all.current', 0)
        print(f"GPU after offload: {_after_offload/1e9:.2f} GB")

        # Phase 2: load MLP expert weights layer-by-layer
        for block_idx, block in enumerate(self.block):
            mlp = block.mlp

            # mlp1_weight
            loaded = checkpoint.get(f"block.{block_idx}.mlp.mlp1_weight")
            torch.cuda.empty_cache()
            loaded_t = loaded.mT.contiguous()
            del loaded
            w1, w1_mx = quantize_mx4(loaded_t)
            del loaded_t
            mlp.mlp1_weight_tensor = w1
            mlp.mlp1_weight_mx = w1_mx
            mlp.mlp1_weight = torch.nn.Parameter(w1.storage.data, requires_grad=False)

            # mlp1_bias
            loaded = checkpoint.get(f"block.{block_idx}.mlp.mlp1_bias")
            mlp.mlp1_bias = torch.nn.Parameter(loaded, requires_grad=False)
            del loaded

            # mlp2_weight
            loaded = checkpoint.get(f"block.{block_idx}.mlp.mlp2_weight")
            torch.cuda.empty_cache()
            loaded_t = loaded.mT.contiguous()
            del loaded
            w2, w2_mx = quantize_mx4(loaded_t)
            del loaded_t
            mlp.mlp2_weight_tensor = w2
            mlp.mlp2_weight_mx = w2_mx
            mlp.mlp2_weight = torch.nn.Parameter(w2.storage.data, requires_grad=False)

            # mlp2_bias
            loaded = checkpoint.get(f"block.{block_idx}.mlp.mlp2_bias")
            mlp.mlp2_bias = torch.nn.Parameter(loaded, requires_grad=False)
            del loaded

            mlp.refresh_fp32_bias_caches()

            _hot = block_idx < _n_cache
            cache.register_layer(
                block_idx,
                w1, w1_mx, mlp.mlp1_bias,
                w2, w2_mx, mlp.mlp2_bias,
                cache_on_gpu=_hot,
            )
            mlp._cpu_offload = True
            mlp.expert_cache = cache

            if not _hot:
                # Free GPU expert tensors for this layer (cold layers)
                mlp.mlp1_weight_tensor = torch.empty(1, device=device)
                mlp.mlp1_weight_mx = torch.empty(1, device=device)
                mlp.mlp2_weight_tensor = torch.empty(1, device=device)
                mlp.mlp2_weight_mx = torch.empty(1, device=device)
                mlp.mlp1_weight = torch.nn.Parameter(torch.empty(1, device=device), requires_grad=False)
                mlp.mlp1_bias = torch.nn.Parameter(torch.empty(1, device=device), requires_grad=False)
                mlp.mlp2_bias = torch.nn.Parameter(torch.empty(1, device=device), requires_grad=False)

            torch.cuda.empty_cache()

        # Restore offloaded tensors to GPU
        def _restore_to_gpu(model, attr_path):
            obj = model
            for part in attr_path.split(".")[:-1]:
                obj = getattr(obj, part)
            attr = attr_path.split(".")[-1]
            cpu_copy = _cpu_offload_tensors[attr_path]
            getattr(obj, attr).data = cpu_copy.to(device)

        for attr_path in _cpu_offload_tensors:
            _restore_to_gpu(self, attr_path)
        _cpu_offload_tensors.clear()
        torch.cuda.empty_cache()

        _final = torch.cuda.memory_stats().get('allocated_bytes.all.current', 0)
        print(f"GPU after Phase 2 + restore: {_final/1e9:.2f} GB")

        # Allocate victim cache from remaining free VRAM
        _vc_env = os.getenv("VICTIM_CACHE", "1").strip().lower()
        if _vc_env not in {"0", "false", "no", "off"}:
            cache.init_victim_cache(max_slots=120)


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, context: int, device: torch.device, cpu_offload: bool | None = None):
        self.device = device
        if cpu_offload is None:
            cpu_offload_env = os.getenv("CPU_OFFLOAD", "0").strip().lower()
            cpu_offload = cpu_offload_env not in {"0", "false", "no", "off", ""}
        self.cpu_offload = cpu_offload
        self.use_global_cuda_graph = (
            os.getenv("CUDA_GRAPH", "1").strip().lower() not in {"0", "false", "no", "off"}
        )
        if self.use_global_cuda_graph and self.cpu_offload:
            self.use_global_cuda_graph = False  # global graph incompatible; per-layer graphs used instead
        print(termcolor.colored("Loading model checkpoint...", "yellow"), flush=True)
        if self.cpu_offload:
            print(termcolor.colored("CPU_OFFLOAD enabled: MoE expert weights on CPU, attention on GPU", "yellow"), flush=True)
        self.model = Transformer.from_checkpoint(
            checkpoint, device=self.device, cpu_offload=self.cpu_offload,
        )
        self.caches = [Cache(1, context, self.model.config.num_key_value_heads, device=self.device) for _ in range(len(self.model.block))]
        self.enable_dense_cache_sliding = all(cache.can_slide() for cache in self.caches)
        repeat_stop_env = os.getenv("DENSE_REPEAT_PATTERN_STOP", "0").strip().lower()
        self.enable_repeat_pattern_stop = repeat_stop_env not in {"0", "false", "no", "off", ""}
        self._dense_slide_warned = False
        self.input_token = torch.zeros(1, dtype=torch.int32, device=self.device)
        # warmup: trigger Triton JIT. With CPU offload only need one layer
        # since all layers share the same kernel shapes.
        if self.cpu_offload:
            x = self.model.embedding(self.input_token[None, :])
            x = self.model.block[0](x, cache=self.caches[0])
        else:
            self.model(self.input_token[None, :], caches=self.caches)
        self._sampling_graph = None
        self._greedy_graph = None
        self._greedy_token: torch.Tensor | None = None
        if self.use_global_cuda_graph:
            self._sampling_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._sampling_graph):
                self.logits = self.model(self.input_token[None, :], caches=self.caches)[0]
            self._greedy_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._greedy_graph):
                self._greedy_token = self.model.forward_greedy(self.input_token[None, :], caches=self.caches)
        else:
            self.logits = None
            if not self.cpu_offload:
                print(termcolor.colored("CUDA Graph disabled, using eager mode", "yellow"), flush=True)
        self._sampling_probs = torch.empty(self.model.config.vocab_size, device=self.device, dtype=torch.bfloat16)
        self.last_generation_stats = None
        if self.enable_dense_cache_sliding:
            print(
                termcolor.colored(
                    f"Dense KV sliding enabled: evict {self.caches[0].slide_chunk} oldest tokens when full",
                    "yellow",
                ),
                flush=True,
            )

    @property
    def max_model_len(self) -> int:
        """Maximum prompt length accepted by the dense KV cache path."""
        rope_max = self.model.block[0].attn.rope.max_context_length
        kv_capacity = self.caches[0].k.shape[1]
        return min(rope_max, kv_capacity)

    @torch.inference_mode()
    def sample_next_token(self, logits: torch.Tensor, temperature: float) -> int:
        """Executed only on rank 0."""
        logits = logits[-1]
        if temperature == 0.0:
            return logits.argmax().item()
        # Gumbel-max trick: argmax(logits + T*Gumbel(0,1)) is distributed as
        # categorical(softmax(logits/T)).  Avoids softmax's exp+normalize and
        # multinomial's serial cumsum; argmax is a fast parallel tree-reduction.
        # _sampling_probs is pre-allocated (vocab,) float32 on GPU.
        self._sampling_probs.exponential_().log_().neg_().mul_(temperature).add_(logits)
        return self._sampling_probs.argmax().item()

    def _handle_kv_capacity(self, decode_offset: int) -> tuple[bool, int]:
        if decode_offset < self.max_model_len:
            return True, decode_offset
        if not self.enable_dense_cache_sliding:
            print(
                termcolor.colored(
                    f"Decode stopped at max_model_len={self.max_model_len}",
                    "yellow",
                ),
                flush=True,
            )
            return False, decode_offset

        new_decode_offset = None
        for cache in self.caches:
            # Pass decode_offset (CPU-tracked) as current to avoid offset.item() sync.
            # decode_offset == cache.offset here because both are incremented in lockstep.
            kept = cache.slide_window(current=decode_offset)
            if new_decode_offset is None:
                new_decode_offset = kept
            else:
                assert new_decode_offset == kept
        if not self._dense_slide_warned:
            self._dense_slide_warned = True
            print(
                termcolor.colored(
                    f"Dense KV cache full at {self.max_model_len} tokens; sliding window reuse activated",
                    "yellow",
                ),
                flush=True,
            )
        return True, new_decode_offset

    @staticmethod
    def _has_repeating_token_pattern(
        recent_tokens: list[int],
        min_pattern_len: int = 2,
        max_pattern_len: int = 8,
        min_repeats: int = 8,
    ) -> bool:
        for pattern_len in range(min_pattern_len, max_pattern_len + 1):
            required_tokens = pattern_len * min_repeats
            if len(recent_tokens) < required_tokens:
                continue
            pattern = recent_tokens[-pattern_len:]
            if all(
                recent_tokens[-repeat_idx * pattern_len: -(repeat_idx - 1) * pattern_len or None] == pattern
                for repeat_idx in range(1, min_repeats + 1)
            ):
                return True
        return False

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int] | None = None,
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False,
                 enable_repeat_pattern_stop: bool | None = None):
        stop_tokens = stop_tokens or []
        if enable_repeat_pattern_stop is None:
            enable_repeat_pattern_stop = self.enable_repeat_pattern_stop
        self.last_generation_stats = None
        for cache in self.caches:
            cache.reset()
        if len(prompt_tokens) > self.max_model_len:
            raise ValueError(
                f"Prompt is too long: {len(prompt_tokens)} tokens "
                f"exceeds max_model_len={self.max_model_len}. "
                "Truncate the conversation history before calling generate()."
            )
        prompt_tokens = torch.as_tensor(prompt_tokens, dtype=torch.int32, device=self.device)
        predicted_token = prompt_tokens[-1]
        decode_offset = prompt_tokens.numel() - 1
        prefill_elapsed = 0.0
        if decode_offset > 0:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            prefill_start = time.perf_counter()
            self.model.prefill(prompt_tokens[None, :-1], self.caches)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            prefill_elapsed = time.perf_counter() - prefill_start
        num_generated_tokens = 0
        recent_generated_tokens: list[int] = []
        loop_detected = False
        if os.getenv("profile", "0") == "1":
            print("DEBUG: You are currently in profiling mode. To disable, run `export profile=0`", flush=True)
            with profile(
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_dir"),
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    can_continue, decode_offset = self._handle_kv_capacity(decode_offset)
                    if not can_continue:
                        return
                    self.input_token[0] = predicted_token
                    if self.use_global_cuda_graph and self._greedy_graph is not None and temperature == 0.0:
                        self._greedy_graph.replay()
                    elif self.use_global_cuda_graph:
                        self._sampling_graph.replay()
                    else:
                        self.logits = self.model(self.input_token[None, :], caches=self.caches)[0]
                    if temperature == 0.0:
                        self.logits.argmax()
                    else:
                        self.sample_next_token(self.logits, temperature)
            self.last_generation_stats = {
                "generated_tokens": 0,
                "prefill_time_s": prefill_elapsed,
                "decode_time_s": 0.0,
            }
            return

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decode_start = time.perf_counter()
        use_fused_greedy = temperature == 0.0 and not return_logprobs
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            can_continue, decode_offset = self._handle_kv_capacity(decode_offset)
            if not can_continue:
                break
            self.input_token[0] = predicted_token
            if use_fused_greedy:
                if self.use_global_cuda_graph:
                    self._greedy_graph.replay()
                    predicted_token = self._greedy_token.item()
                else:
                    predicted_token = self.model.forward_greedy(self.input_token[None, :], caches=self.caches).item()
            else:
                if self.use_global_cuda_graph:
                    self._sampling_graph.replay()
                else:
                    self.logits = self.model(self.input_token[None, :], caches=self.caches)[0]
                predicted_token = self.sample_next_token(self.logits, temperature)
            decode_offset += 1
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(self.logits[-1, :], dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if enable_repeat_pattern_stop:
                recent_generated_tokens.append(predicted_token)
                if len(recent_generated_tokens) > 64:
                    recent_generated_tokens.pop(0)
                if self._has_repeating_token_pattern(recent_generated_tokens):
                    loop_detected = True
                    print(
                        termcolor.colored(
                            "Stopping generation after detecting 8 consecutive repeats of a 2-8 token pattern",
                            "yellow",
                        ),
                        flush=True,
                    )
                    break
            if predicted_token in stop_tokens:
                break
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decode_elapsed = time.perf_counter() - decode_start
        self.last_generation_stats = {
            "generated_tokens": num_generated_tokens,
            "prefill_time_s": prefill_elapsed,
            "decode_time_s": decode_elapsed,
            "loop_detected": loop_detected,
        }
