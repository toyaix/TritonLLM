"""Shape-keyed scratch buffer pool for decode kernels.

All decode operations run sequentially on a single CUDA stream, so a single
buffer per (shape, dtype) suffices for the entire model — each layer reuses
the same slots, overwriting data that has already been consumed.
"""
import torch
from typing import Any

_pool: dict[tuple, torch.Tensor] = {}


def _key(shape: tuple, dtype: torch.dtype, device: torch.device) -> tuple:
    return (shape, dtype, device)


def get(shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get or create a reusable scratch buffer."""
    k = _key(shape, dtype, device)
    if k not in _pool:
        _pool[k] = torch.empty(shape, dtype=dtype, device=device)
    return _pool[k]


def clear() -> None:
    """Release all pooled buffers."""
    _pool.clear()


def stats() -> dict[str, Any]:
    """Return pool size info (for debugging)."""
    total = sum(t.numel() * t.element_size() for t in _pool.values())
    return {"num_buffers": len(_pool), "total_bytes": total}
