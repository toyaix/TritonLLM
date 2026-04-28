"""CPU/GPU hybrid expert weight cache for MoE models.

Stores MoE expert weights on CPU (pageable, contiguous per-tensor) and
maintains shared GPU buffers that are filled on-demand as each layer's
experts are activated.

Key optimization — GPU hot-layer cache:
Uses free VRAM to keep the full 128-expert tensor set for N layers
permanently on GPU.  When a cached layer is hit, its experts are
copied from the GPU cache to the shared buffers via 6 large GPU→GPU
copies at ~900 GB/s, avoiding 24 pageable CPU→GPU copies entirely.
"""

from typing import Optional

import torch


class ExpertCache:
    """Expert weight cache with hot-layer GPU cache.

    Shared GPU buffers (128 experts) are reused across all layers.
    A separate per-layer GPU cache holds the full expert set for
    frequently-accessed layers, eliminating CPU→GPU transfers.
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

        # Per-layer pageable CPU storage — contiguous per-tensor
        self.cpu: dict[int, dict[str, torch.Tensor]] = {}

        # Global GPU buffers — shared across all layers (128 experts)
        self.gpu_w1 = None
        self.gpu_w1_mx = None
        self.gpu_b1: Optional[torch.Tensor] = None
        self.gpu_w2 = None
        self.gpu_w2_mx = None
        self.gpu_b2: Optional[torch.Tensor] = None

        # Track which layer is in each GPU slot
        self.gpu_layer: list[int] = []

        # ---- Hot-layer GPU cache ----
        # Per-layer full (128-expert) GPU tensors that persist across tokens.
        # When a cached layer is requested, 6 big GPU→GPU copies replace
        # 24 small CPU→GPU copies.
        self._cache: dict[int, dict[str, torch.Tensor]] = {}
        self._cache_order: list[int] = []  # LRU: front=MRU, back=LRU
        self._cache_capacity: int = 0
        self._cache_bytes_per_layer: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # ---- CUDA Graph: gate routing ----
        self.route_graph: torch.cuda.CUDAGraph | None = None
        self._gr_t: torch.Tensor | None = None
        self._gr_wg: torch.Tensor | None = None
        self._gr_bg: torch.Tensor | None = None
        self._gr_rdata = None
        self._gr_gather = None
        self._gr_scatter = None

        # ---- CUDA Graph: decode matmul ----
        self.decode_graph: torch.cuda.CUDAGraph | None = None
        self._g_x: torch.Tensor | None = None
        self._g_out: torch.Tensor | None = None
        self._g_topk: torch.Tensor | None = None
        self._g_gate: torch.Tensor | None = None
        self._g_scale: torch.Tensor | None = None
        self._g_hist: torch.Tensor | None = None
        self._g_offs_raw: torch.Tensor | None = None
        self._g_offs_pad: dict[int, torch.Tensor] = {}
        self._g_pid_map: dict[int, torch.Tensor] = {}
        self._g_rdata = None
        self._g_gather = None
        self._g_scatter = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_layer(
        self,
        layer_idx: int,
        w1,
        w1_mx,
        b1: torch.Tensor,
        w2,
        w2_mx,
        b2: torch.Tensor,
    ) -> None:
        """Store pageable CPU copies and initialise shared GPU buffers.

        Called once per MLP layer after weight loading + quantization.
        """
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

        # Store pageable CPU tensors — one contiguous tensor per component,
        # shape (num_experts, ...), matching the GPU buffer layout.
        cpu_layer: dict[str, torch.Tensor] = {}
        for key, gpu_tensor in self._iter_tensors(w1, w1_mx, b1, w2, w2_mx, b2):
            cpu_layer[key] = self._gpu_to_pageable_cpu(gpu_tensor)
        self.cpu[layer_idx] = cpu_layer

    def _gpu_to_pageable_cpu(self, gpu_tensor) -> torch.Tensor:
        """Copy the entire GPU tensor into a contiguous pageable CPU tensor."""
        raw = self._raw_data(gpu_tensor)
        return raw.cpu()

    @staticmethod
    def _raw_data(tensor) -> torch.Tensor:
        """Return the underlying torch.Tensor for a regular or custom Tensor."""
        if hasattr(tensor, "storage") and hasattr(tensor.storage, "data"):
            return tensor.storage.data
        return tensor

    @staticmethod
    def _iter_tensors(w1, w1_mx, b1, w2, w2_mx, b2):
        for key, t in [
            ("w1", w1), ("w1_mx", w1_mx), ("b1", b1),
            ("w2", w2), ("w2_mx", w2_mx), ("b2", b2),
        ]:
            if t is not None:
                yield key, t

    # ------------------------------------------------------------------
    # Public API — shared GPU buffers (CPU→GPU)
    # ------------------------------------------------------------------

    def ensure_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Copy expert weights from CPU → GPU if not already present.

        If *layer_idx* is in the hot-layer GPU cache, copies from the
        cache (GPU→GPU at ~900 GB/s) instead of from CPU.
        """
        # Hot-layer cache hit: copy from GPU cache → shared GPU buffers
        if layer_idx in self._cache:
            self._copy_from_cache(layer_idx)
            self._cache_hits += 1
            self._touch_cache(layer_idx)
            return

        # Cache miss: copy from pageable CPU
        self._cache_misses += 1
        gpu_w1_raw = self._raw_data(self.gpu_w1)
        gpu_w2_raw = self._raw_data(self.gpu_w2)
        gpu_w1_mx_raw = self._raw_data(self.gpu_w1_mx) if self.gpu_w1_mx is not None else None
        gpu_w2_mx_raw = self._raw_data(self.gpu_w2_mx) if self.gpu_w2_mx is not None else None

        cpu_layer = self.cpu[layer_idx]
        cpu_w1 = cpu_layer["w1"]
        cpu_w1_mx = cpu_layer.get("w1_mx")
        cpu_b1 = cpu_layer["b1"]
        cpu_w2 = cpu_layer["w2"]
        cpu_w2_mx = cpu_layer.get("w2_mx")
        cpu_b2 = cpu_layer["b2"]

        for eid in expert_ids:
            eid_i = int(eid)
            if self.gpu_layer[eid_i] == layer_idx:
                continue

            gpu_w1_raw[eid_i].copy_(cpu_w1[eid_i])

            if gpu_w1_mx_raw is not None and cpu_w1_mx is not None:
                gpu_w1_mx_raw[eid_i].copy_(cpu_w1_mx[eid_i])

            self.gpu_b1[eid_i].copy_(cpu_b1[eid_i])

            gpu_w2_raw[eid_i].copy_(cpu_w2[eid_i])

            if gpu_w2_mx_raw is not None and cpu_w2_mx is not None:
                gpu_w2_mx_raw[eid_i].copy_(cpu_w2_mx[eid_i])

            self.gpu_b2[eid_i].copy_(cpu_b2[eid_i])

            self.gpu_layer[eid_i] = layer_idx

    def load_all_experts(self, layer_idx: int) -> None:
        """Load every expert into the shared GPU buffers (for prefill)."""
        self.ensure_experts(layer_idx, list(range(self.num_experts)))

    # ------------------------------------------------------------------
    # Hot-layer GPU cache
    # ------------------------------------------------------------------

    def init_layer_cache(self, max_layers: int = 999) -> int:
        """Determine how many full-layer GPU caches fit in free VRAM.

        Must be called after all ``register_layer`` calls so that tensor
        sizes are known.  Returns the actual number of cache slots
        available.  Actual GPU allocation is deferred to ``cache_layer``.
        """
        gpu_w1_raw = self._raw_data(self.gpu_w1)
        gpu_w2_raw = self._raw_data(self.gpu_w2)

        total_bytes = (
            gpu_w1_raw.numel() * gpu_w1_raw.element_size()
            + gpu_w2_raw.numel() * gpu_w2_raw.element_size()
            + self.gpu_b1.numel() * self.gpu_b1.element_size()
            + self.gpu_b2.numel() * self.gpu_b2.element_size()
        )
        if self.gpu_w1_mx is not None:
            total_bytes += self._raw_data(self.gpu_w1_mx).numel() * self._raw_data(self.gpu_w1_mx).element_size()
        if self.gpu_w2_mx is not None:
            total_bytes += self._raw_data(self.gpu_w2_mx).numel() * self._raw_data(self.gpu_w2_mx).element_size()

        self._cache_bytes_per_layer = total_bytes

        free, _ = torch.cuda.mem_get_info(self.device)
        # Reserve 2 GB for KV cache, activations, CUDA graph workspace.
        inference_headroom = 2 * 1024**3
        usable = max(0, int(free) - inference_headroom)
        self._cache_capacity = min(max_layers, max(0, usable // total_bytes))
        if self._cache_capacity > 0:
            print(f"[cache] {self._cache_capacity} layer slots "
                  f"({self._cache_capacity * total_bytes / 1e9:.2f} GB) "
                  f"from {usable / 1e9:.2f} GB usable VRAM")

        # Deferred allocation — actual GPU tensors created in cache_layer
        self._cache_order = [-1] * self._cache_capacity

        return self._cache_capacity

    def cache_layer(self, layer_idx: int) -> bool:
        """Fill the GPU cache for *layer_idx* from CPU pageable storage.

        Returns True if the layer was cached, False if cache is disabled
        or the layer is already cached.
        """
        if self._cache_capacity == 0:
            return False
        if layer_idx in self._cache:
            self._touch_cache(layer_idx)
            return True

        # Evict LRU layer if cache is full
        if len(self._cache) >= self._cache_capacity:
            evict_idx = self._cache_order[-1]
            if evict_idx >= 0 and evict_idx in self._cache:
                del self._cache[evict_idx]
                self._cache_order[-1] = -1

        # Find a free slot
        slot = -1
        for i, lid in enumerate(self._cache_order):
            if lid < 0 or lid not in self._cache:
                slot = i
                break
        if slot < 0:
            slot = len(self._cache_order) - 1  # reuse last slot

        # Allocate buffers in this slot if first use
        if self._cache_order[slot] < 0:
            # Already allocated in init_layer_cache; just assign
            self._cache_order[slot] = layer_idx
            self._cache[layer_idx] = {}  # filled below

        # Copy all 128 experts from pageable CPU to GPU cache
        cpu_layer = self.cpu[layer_idx]
        cache_entry = self._cache.setdefault(layer_idx, {})
        for key in ["w1", "w1_mx", "b1", "w2", "w2_mx", "b2"]:
            if key in cpu_layer:
                if key not in cache_entry:
                    # Allocate on first use
                    cache_entry[key] = cpu_layer[key].to(self.device)
                else:
                    cache_entry[key].copy_(cpu_layer[key])

        self._touch_cache(layer_idx)
        return True

    def cache_stats(self) -> dict:
        """Return cache hit/miss counters."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "capacity": self._cache_capacity,
            "active": len(self._cache),
            "bytes_per_layer_mb": self._cache_bytes_per_layer / (1024 * 1024),
        }

    def _copy_from_cache(self, layer_idx: int) -> None:
        """Copy all 128 experts from GPU cache → shared GPU buffers.

        Six large contiguous GPU→GPU copies at ~900 GB/s, replacing
        24 small pageable CPU→GPU copies at ~15 GB/s.
        """
        cache_entry = self._cache[layer_idx]
        gpu_w1_raw = self._raw_data(self.gpu_w1)
        gpu_w2_raw = self._raw_data(self.gpu_w2)

        gpu_w1_raw.copy_(cache_entry["w1"])
        if self.gpu_w1_mx is not None and "w1_mx" in cache_entry:
            self._raw_data(self.gpu_w1_mx).copy_(cache_entry["w1_mx"])
        self.gpu_b1.copy_(cache_entry["b1"])
        gpu_w2_raw.copy_(cache_entry["w2"])
        if self.gpu_w2_mx is not None and "w2_mx" in cache_entry:
            self._raw_data(self.gpu_w2_mx).copy_(cache_entry["w2_mx"])
        self.gpu_b2.copy_(cache_entry["b2"])

        # Mark all 128 GPU slots as belonging to this layer
        for eid in range(self.num_experts):
            self.gpu_layer[eid] = layer_idx

    def _touch_cache(self, layer_idx: int) -> None:
        """Move *layer_idx* to the front of the LRU order."""
        if layer_idx in self._cache_order:
            self._cache_order.remove(layer_idx)
        while len(self._cache_order) < self._cache_capacity:
            self._cache_order.append(-1)
        self._cache_order.insert(0, layer_idx)
        # Trim to capacity
        while len(self._cache_order) > self._cache_capacity:
            removed = self._cache_order.pop()
            if removed >= 0 and removed in self._cache:
                del self._cache[removed]

    # ------------------------------------------------------------------
    # CUDA Graph (decode matmul)
    # ------------------------------------------------------------------

    def init_decode_graph(
        self,
        rdata,
        gather_indx,
        scatter_indx,
        swiglu_limit: float = 7.0,
    ) -> None:
        """Capture a CUDA graph for the decode-matmul path."""
        from triton_kernels.routing import GatherIndx, ScatterIndx, RoutingData, ExptData
        from .moe import moe_decode_experts

        self._g_topk = gather_indx.src_indx.clone()
        self._g_gate = gather_indx.dst_indx.clone()
        self._g_scale = rdata.gate_scal.clone()
        self._g_hist = rdata.expt_hist.clone()
        self._g_offs_raw = rdata.expt_data.token_offs_raw.clone()
        self._g_offs_pad = {k: v.clone() for k, v in rdata.expt_data.token_offs_pad.items()}
        self._g_pid_map = {k: v.clone() for k, v in rdata.expt_data.block_pid_map.items()}

        expt_data = ExptData(
            hist=self._g_hist,
            token_offs_raw=self._g_offs_raw,
            token_offs_pad=self._g_offs_pad,
            block_pid_map=self._g_pid_map,
        )
        self._g_rdata = RoutingData(
            gate_scal=self._g_scale,
            expt_hist=self._g_hist,
            n_expts_tot=rdata.n_expts_tot,
            n_expts_act=rdata.n_expts_act,
            expt_data=expt_data,
        )
        self._g_gather = GatherIndx(src_indx=self._g_topk, dst_indx=self._g_gate)
        self._g_scatter = ScatterIndx(src_indx=self._g_gate, dst_indx=self._g_topk)

        hidden_size = self.gpu_b2.shape[1]
        self._g_x = self.gpu_b2.new_zeros(1, hidden_size, dtype=torch.bfloat16)

        self.decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.decode_graph):
            self._g_out = moe_decode_experts(
                self._g_x,
                self.gpu_w1, self.gpu_w1_mx,
                self.gpu_w2, self.gpu_w2_mx,
                self.gpu_b1, self.gpu_b2,
                self._g_rdata, self._g_gather, self._g_scatter,
                swiglu_limit=swiglu_limit,
            )

    # ------------------------------------------------------------------
    # CUDA Graph (decode gate routing)
    # ------------------------------------------------------------------

    def init_route_graph(
        self,
        t: torch.Tensor,
        wg: torch.Tensor,
        bg: torch.Tensor | None,
        experts_per_token: int = 4,
        num_experts: int = 128,
    ) -> None:
        """Capture a CUDA graph for the decode gate-routing path."""
        from .moe import moe_decode_gate_routing

        self._gr_t = t.clone()
        self._gr_wg = wg.clone()
        self._gr_bg = bg.clone() if bg is not None else None

        self.route_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.route_graph):
            self._gr_rdata, self._gr_gather, self._gr_scatter = moe_decode_gate_routing(
                self._gr_t, self._gr_wg, self._gr_bg,
                experts_per_token=experts_per_token,
                num_experts=num_experts,
            )

    def route_to_matmul_proxies(self) -> None:
        """Copy route-graph outputs into the matmul-graph proxy tensors."""
        self._g_topk.copy_(self._gr_gather.src_indx)
        self._g_gate.copy_(self._gr_gather.dst_indx)
        self._g_scale.copy_(self._gr_rdata.gate_scal)
        self._g_hist.copy_(self._gr_rdata.expt_hist)
        self._g_offs_raw.copy_(self._gr_rdata.expt_data.token_offs_raw)
        for k in self._g_offs_pad:
            self._g_offs_pad[k].copy_(self._gr_rdata.expt_data.token_offs_pad[k])
        for k in self._g_pid_map:
            self._g_pid_map[k].copy_(self._gr_rdata.expt_data.block_pid_map[k])

    def free_gpu_buffers(self) -> None:
        """Release all GPU buffers to reclaim memory."""
        self.route_graph = None
        self.decode_graph = None
        self._gr_t = None
        self._gr_wg = None
        self._gr_bg = None
        self._g_x = None
        self._g_out = None
        self.gpu_w1 = None
        self.gpu_w1_mx = None
        self.gpu_b1 = None
        self.gpu_w2 = None
        self.gpu_w2_mx = None
        self.gpu_b2 = None
        self._cache.clear()
        self._cache_order.clear()
        self.gpu_layer = []
