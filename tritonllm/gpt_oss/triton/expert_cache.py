"""CPU/GPU hybrid expert weight cache for MoE models.

Stores MoE expert weights on CPU and maintains a shared set of GPU buffers
that are filled on-demand as each layer's experts are activated. The routing
step tells us exactly which experts are needed before the heavy matmul, so
we only transfer the activated experts (4/128) per layer.

The custom Tensor types from ``quantize_mx4`` do not expose ``cpu()`` or
``clone()`` directly, so we work with the underlying ``torch.Tensor`` via
``.storage.data``.  The swizzled layouts preserve the leading (expert)
dimension, so ``storage.data[eid]`` correctly isolates a single expert.
"""

from typing import Optional

import torch


class ExpertCache:
    """Global GPU buffers for expert weights, shared across all layers.

    Each layer's expert weights are stored on CPU (per-expert slices of the
    raw storage).  A single set of ``(num_experts, ...)`` shaped GPU buffers
    is shared across layers — before each MLP forward the activated experts
    for that layer are copied from CPU into the corresponding GPU slots.
    """

    def __init__(
        self,
        num_experts: int,
        experts_per_token: int,
        device: torch.device,
    ):
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.device = device

        # Per-layer, per-expert CPU storage (raw torch.Tensor slices)
        # cpu[layer_idx][expert_id] = {"w1": Tensor, "w1_mx": Tensor, ...}
        self.cpu: dict[int, dict[int, dict[str, torch.Tensor]]] = {}

        # Global GPU buffers — the custom Tensor objects themselves
        self.gpu_w1 = None           # custom Tensor from quantize_mx4
        self.gpu_w1_mx = None        # custom Tensor (or None)
        self.gpu_b1: Optional[torch.Tensor] = None  # fp32 (E, intermed*2)
        self.gpu_w2 = None           # custom Tensor
        self.gpu_w2_mx = None        # custom Tensor (or None)
        self.gpu_b2: Optional[torch.Tensor] = None  # fp32 (E, hidden)

        # Track which layer's data is currently in each GPU expert slot (CPU-side)
        self.gpu_layer: list[int] = []  # [layer_idx] * num_experts, -1 = invalid

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_layer(
        self,
        layer_idx: int,
        w1,           # custom Tensor from quantize_mx4, shape (E, ...)
        w1_mx,        # MXFP4 scales, shape (E, ...) — custom Tensor or None
        b1: torch.Tensor,   # (E, intermediate_size*2) bf16
        w2,           # custom Tensor, shape (E, ...)
        w2_mx,        # MXFP4 scales
        b2: torch.Tensor,   # (E, hidden_size) bf16
    ) -> None:
        """Store per-expert CPU copies and initialise global GPU buffers.

        Called once per MLP layer after weight loading + quantization.
        """
        # Allocate global GPU buffers on first call (keep first layer's tensors)
        if self.gpu_w1 is None:
            self.gpu_w1 = w1
            self.gpu_w1_mx = w1_mx if w1_mx is not None else None
            self.gpu_b1 = torch.empty_like(
                b1, dtype=torch.float32, device=self.device,
            )
            self.gpu_w2 = w2
            self.gpu_w2_mx = w2_mx if w2_mx is not None else None
            self.gpu_b2 = torch.empty_like(
                b2, dtype=torch.float32, device=self.device,
            )
            self.gpu_layer = [-1] * self.num_experts

        # Slice by expert and move to CPU (via .storage.data for custom Tensors)
        cpu_layer: dict[int, dict[str, torch.Tensor]] = {}
        for eid in range(self.num_experts):
            cpu_layer[eid] = {
                "w1": self._raw_slice_cpu(w1, eid),
                "w1_mx": self._raw_slice_cpu(w1_mx, eid)
                if w1_mx is not None else None,
                "b1": b1[eid].cpu(),
                "w2": self._raw_slice_cpu(w2, eid),
                "w2_mx": self._raw_slice_cpu(w2_mx, eid)
                if w2_mx is not None else None,
                "b2": b2[eid].cpu(),
            }
        self.cpu[layer_idx] = cpu_layer

    @staticmethod
    def _raw_slice_cpu(tensor, index: int) -> torch.Tensor:
        """Slice *tensor* at *index* and move the slice to CPU.

        Works with both regular ``torch.Tensor`` and the custom ``Tensor``
        from ``triton_kernels.tensor`` (by accessing ``.storage.data``).
        """
        if hasattr(tensor, "storage") and hasattr(tensor.storage, "data"):
            raw = tensor.storage.data  # underlying torch.Tensor
        else:
            raw = tensor
        return raw[index].cpu()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Synchronously copy expert weights from CPU → GPU if not already present."""
        gpu_w1_raw = self.gpu_w1.storage.data
        gpu_w2_raw = self.gpu_w2.storage.data
        gpu_w1_mx_raw = (
            self.gpu_w1_mx.storage.data
            if self.gpu_w1_mx is not None and hasattr(self.gpu_w1_mx, "storage")
            else self.gpu_w1_mx
        )
        gpu_w2_mx_raw = (
            self.gpu_w2_mx.storage.data
            if self.gpu_w2_mx is not None and hasattr(self.gpu_w2_mx, "storage")
            else self.gpu_w2_mx
        )
        cpu_layer = self.cpu[layer_idx]

        for eid in expert_ids:
            eid_i = int(eid)
            if self.gpu_layer[eid_i] == layer_idx:
                continue
            cw = cpu_layer[eid_i]
            gpu_w1_raw[eid_i].copy_(cw["w1"])
            if gpu_w1_mx_raw is not None and cw["w1_mx"] is not None:
                gpu_w1_mx_raw[eid_i].copy_(cw["w1_mx"])
            self.gpu_b1[eid_i].copy_(cw["b1"])
            gpu_w2_raw[eid_i].copy_(cw["w2"])
            if gpu_w2_mx_raw is not None and cw["w2_mx"] is not None:
                gpu_w2_mx_raw[eid_i].copy_(cw["w2_mx"])
            self.gpu_b2[eid_i].copy_(cw["b2"])
            self.gpu_layer[eid_i] = layer_idx

    def load_all_experts(self, layer_idx: int) -> None:
        """Load every expert for *layer_idx* into the GPU buffers (for prefill)."""
        self.ensure_experts(layer_idx, list(range(self.num_experts)))

    def free_gpu_buffers(self) -> None:
        """Release all global GPU buffers to reclaim memory."""
        self.gpu_w1 = None
        self.gpu_w1_mx = None
        self.gpu_b1 = None
        self.gpu_w2 = None
        self.gpu_w2_mx = None
        self.gpu_b2 = None
        self.gpu_layer = []
