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

        # GPU hot-layer cache: permanent GPU storage for selected layers
        # gpu_cache[layer_idx] = {"w1": raw_tensor, "w2": ..., "b1": ..., "b2": ...}
        self.gpu_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._cache_bytes_per_layer: int = 0

        # Hit/miss counters for gpu_layer reuse across tokens
        self._hit_count: int = 0
        self._miss_count: int = 0

        # Victim cache: manually-managed GPU buffer for evicted expert data.
        # When a GPU slot is overwritten by a different layer, the old data is
        # saved here.  On a later reuse, a D2D copy (~900 GB/s) restores it.
        self._vc_slots: int = 0
        self._vc_w1: torch.Tensor | None = None
        self._vc_w1_mx: torch.Tensor | None = None
        self._vc_b1: torch.Tensor | None = None
        self._vc_w2: torch.Tensor | None = None
        self._vc_w2_mx: torch.Tensor | None = None
        self._vc_b2: torch.Tensor | None = None
        self._vc_expert_shapes: dict | None = None  # set on first register_layer
        self._vc_tag: dict[tuple[int, int], int] = {}  # (layer, expert) → slot idx
        self._vc_free: list[int] = []   # free slot indices
        self._vc_lru: list[int] = []    # front=LRU, back=MRU

        # Per-layer CUDA graph: gate routing (captured once, shared across layers)
        self.route_graph: torch.cuda.CUDAGraph | None = None
        self._gr_t: torch.Tensor | None = None        # input activation buffer
        self._gr_wg: torch.Tensor | None = None        # gate weight buffer
        self._gr_bg: torch.Tensor | None = None        # gate bias buffer
        self._gr_rdata = None   # RoutingData output (set during capture)
        self._gr_gather = None  # GatherIndx output
        self._gr_scatter = None # ScatterIndx output

        # Per-layer CUDA graph for decode matmul (captured once, shared across layers)
        self.decode_graph: torch.cuda.CUDAGraph | None = None
        self._g_x: torch.Tensor | None = None       # input activation buffer
        self._g_out: torch.Tensor | None = None      # output activation (set during capture)
        self._g_topk: torch.Tensor | None = None     # gather.src_indx / scatter.dst_indx
        self._g_gate: torch.Tensor | None = None     # gather.dst_indx / scatter.src_indx
        self._g_scale: torch.Tensor | None = None    # gate_scal
        self._g_hist: torch.Tensor | None = None     # expt_hist
        self._g_offs_raw: torch.Tensor | None = None # token_offs_raw
        self._g_offs_pad: dict[int, torch.Tensor] = {}
        self._g_pid_map: dict[int, torch.Tensor] = {}
        self._g_rdata = None   # proxy RoutingData
        self._g_gather = None  # proxy GatherIndx
        self._g_scatter = None # proxy ScatterIndx

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
        cache_on_gpu: bool = False,
    ) -> None:
        """Store per-expert CPU copies and initialise global GPU buffers.

        Called once per MLP layer after weight loading + quantization.
        When ``cache_on_gpu=True``, weights stay on GPU for fast D2D access.
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

            # Measure per-layer bytes for cache capacity planning
            w1_raw = self._raw_data(w1)
            w2_raw = self._raw_data(w2)
            b1_f32 = b1.to(torch.float32)
            b2_f32 = b2.to(torch.float32)
            w1_mx_raw = self._raw_data(w1_mx) if w1_mx is not None else None
            w2_mx_raw = self._raw_data(w2_mx) if w2_mx is not None else None
            self._cache_bytes_per_layer = (
                w1_raw.numel() * w1_raw.element_size() +
                w2_raw.numel() * w2_raw.element_size() +
                b1_f32.numel() * 4 +
                b2_f32.numel() * 4 +
                (w1_mx_raw.numel() * w1_mx_raw.element_size() if w1_mx_raw is not None else 0) +
                (w2_mx_raw.numel() * w2_mx_raw.element_size() if w2_mx_raw is not None else 0)
            )

            # Save per-expert shapes for deferred victim cache allocation
            self._vc_expert_shapes = {
                "w1": w1_raw.shape[1:], "w1_dtype": w1_raw.dtype,
                "w1_mx": w1_mx_raw.shape[1:] if w1_mx_raw is not None else None,
                "w1_mx_dtype": w1_mx_raw.dtype if w1_mx_raw is not None else None,
                "b1": b1_f32.shape[1:],
                "w2": w2_raw.shape[1:], "w2_dtype": w2_raw.dtype,
                "w2_mx": w2_mx_raw.shape[1:] if w2_mx_raw is not None else None,
                "w2_mx_dtype": w2_mx_raw.dtype if w2_mx_raw is not None else None,
                "b2": b2_f32.shape[1:],
                "bytes_per_expert": self._cache_bytes_per_layer // self.num_experts,
            }

        if cache_on_gpu:
            # Keep raw tensors on GPU for fast D2D copies
            self.gpu_cache[layer_idx] = {
                "w1": self._raw_data(w1),
                "w1_mx": self._raw_data(w1_mx) if w1_mx is not None else None,
                "b1": b1.to(torch.float32),
                "w2": self._raw_data(w2),
                "w2_mx": self._raw_data(w2_mx) if w2_mx is not None else None,
                "b2": b2.to(torch.float32),
            }
            # Also store CPU copies for the non-cache path if needed
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
        else:
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
    def _raw_data(tensor) -> torch.Tensor:
        """Return the underlying torch.Tensor for a regular or custom Tensor."""
        if hasattr(tensor, "storage") and hasattr(tensor.storage, "data"):
            return tensor.storage.data
        return tensor

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
    # Victim cache — manually managed GPU memory
    # ------------------------------------------------------------------

    def init_victim_cache(self, max_slots: int = 200) -> int:
        """Pre-allocate a contiguous GPU buffer for evicted expert data.

        Must be called AFTER model loading is complete (sufficient free VRAM).
        Returns the number of allocated cache slots.
        """
        if self._vc_expert_shapes is None:
            return 0
        s = self._vc_expert_shapes
        expert_bytes = s["bytes_per_expert"]
        free, _ = torch.cuda.mem_get_info()
        usable = max(0, free - 2.5 * 1024**3)  # reserve 2.5 GB for kernels, KV cache, etc.
        slots = min(int(usable // expert_bytes), max_slots)
        if slots < self.experts_per_token:
            print(f"Victim cache: insufficient VRAM "
                  f"(need {expert_bytes * self.experts_per_token / 1e9:.2f} GB, "
                  f"have {usable / 1e9:.2f} GB usable). Victim cache disabled.")
            return 0

        self._vc_slots = slots
        self._vc_w1 = torch.empty(slots, *s["w1"], dtype=s["w1_dtype"], device=self.device)
        self._vc_w1_mx = (
            torch.empty(slots, *s["w1_mx"], dtype=s["w1_mx_dtype"], device=self.device)
            if s["w1_mx"] is not None else None
        )
        self._vc_b1 = torch.empty(slots, *s["b1"], dtype=torch.float32, device=self.device)
        self._vc_w2 = torch.empty(slots, *s["w2"], dtype=s["w2_dtype"], device=self.device)
        self._vc_w2_mx = (
            torch.empty(slots, *s["w2_mx"], dtype=s["w2_mx_dtype"], device=self.device)
            if s["w2_mx"] is not None else None
        )
        self._vc_b2 = torch.empty(slots, *s["b2"], dtype=torch.float32, device=self.device)
        self._vc_free = list(range(slots))
        self._vc_tag = {}
        self._vc_lru = []
        vc_bytes = slots * expert_bytes
        print(f"Victim cache: {slots} slots ({vc_bytes / 1e9:.2f} GB, "
              f"{vc_bytes / 1024**3:.2f} GiB)")
        return slots

    def _vc_evict_one(self) -> int:
        """Evict the LRU entry, returning its slot index."""
        slot = self._vc_lru.pop(0)
        # Find and remove the tag pointing to this slot
        for key, s in list(self._vc_tag.items()):
            if s == slot:
                del self._vc_tag[key]
                break
        return slot

    def _vc_put(self, layer_idx: int, eid_i: int,
                w1_src, w1_mx_src, b1_src, w2_src, w2_mx_src, b2_src):
        """Save expert data into the victim cache before it gets overwritten."""
        key = (layer_idx, eid_i)
        if key in self._vc_tag:
            # Already cached — update LRU position
            slot = self._vc_tag[key]
            if slot in self._vc_lru:
                self._vc_lru.remove(slot)
            self._vc_lru.append(slot)
            # Still need to refresh the data since shared buffer may have changed
        else:
            slot = self._vc_free.pop() if self._vc_free else self._vc_evict_one()
            self._vc_tag[key] = slot
            if slot in self._vc_lru:
                self._vc_lru.remove(slot)
            self._vc_lru.append(slot)

        self._vc_w1[slot].copy_(w1_src)
        if self._vc_w1_mx is not None and w1_mx_src is not None:
            self._vc_w1_mx[slot].copy_(w1_mx_src)
        self._vc_b1[slot].copy_(b1_src)
        self._vc_w2[slot].copy_(w2_src)
        if self._vc_w2_mx is not None and w2_mx_src is not None:
            self._vc_w2_mx[slot].copy_(w2_mx_src)
        self._vc_b2[slot].copy_(b2_src)

    def _vc_get(self, layer_idx: int, eid_i: int,
                w1_dst, w1_mx_dst, b1_dst, w2_dst, w2_mx_dst, b2_dst) -> bool:
        """Restore expert data from victim cache. Returns True on hit."""
        key = (layer_idx, eid_i)
        slot = self._vc_tag.get(key)
        if slot is None:
            return False
        w1_dst.copy_(self._vc_w1[slot])
        if w1_mx_dst is not None and self._vc_w1_mx is not None:
            w1_mx_dst.copy_(self._vc_w1_mx[slot])
        b1_dst.copy_(self._vc_b1[slot])
        w2_dst.copy_(self._vc_w2[slot])
        if w2_mx_dst is not None and self._vc_w2_mx is not None:
            w2_mx_dst.copy_(self._vc_w2_mx[slot])
        b2_dst.copy_(self._vc_b2[slot])
        # Update LRU
        if slot in self._vc_lru:
            self._vc_lru.remove(slot)
        self._vc_lru.append(slot)
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_layer_cache(self, max_layers: int = 0) -> int:
        """Determine how many hot layers can be cached in free GPU memory.

        Returns the number of layers that can be cached (0 if insufficient VRAM).
        The caller should then call ``cache_on_gpu=True`` during
        ``register_layer`` for the first *n* layers.
        """
        if self._cache_bytes_per_layer == 0 or max_layers <= 0:
            return 0
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # Reserve 2 GB for inference overhead (activations, CUDA graphs, etc.)
        usable = max(0, free - 2 * 1024**3)
        n = min(max_layers, int(usable // self._cache_bytes_per_layer))
        if n > 0:
            print(f"GPU layer cache: {n} slots "
                  f"({self._cache_bytes_per_layer * n / 1e9:.2f} GB of "
                  f"{free / 1e9:.2f} GB free VRAM)")
        else:
            print(f"GPU layer cache: insufficient free VRAM "
                  f"(need {self._cache_bytes_per_layer / 1e9:.2f} GB/layer, "
                  f"{free / 1e9:.2f} GB free)")
        return n

    def is_cached(self, layer_idx: int) -> bool:
        return layer_idx in self.gpu_cache

    def ensure_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Copy expert weights to GPU shared buffers if not already present.

        Three-path lookup:
        1. Hot-layer cache: entire layer permanently on GPU → D2D copy
        2. gpu_layer match: slot still holds this layer's data from prev token
        3. Victim cache: data evicted by another layer → D2D restore
        4. CPU→GPU copy (with victim-cache save of overwritten data)
        """
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

        # Path 1: hot-layer cache
        cached = self.gpu_cache.get(layer_idx)
        if cached is not None:
            cw1 = cached["w1"]
            cw1_mx = cached["w1_mx"]
            cb1 = cached["b1"]
            cw2 = cached["w2"]
            cw2_mx = cached["w2_mx"]
            cb2 = cached["b2"]
            for eid in expert_ids:
                eid_i = int(eid)
                if self.gpu_layer[eid_i] == layer_idx:
                    continue
                gpu_w1_raw[eid_i].copy_(cw1[eid_i])
                if gpu_w1_mx_raw is not None and cw1_mx is not None:
                    gpu_w1_mx_raw[eid_i].copy_(cw1_mx[eid_i])
                self.gpu_b1[eid_i].copy_(cb1[eid_i])
                gpu_w2_raw[eid_i].copy_(cw2[eid_i])
                if gpu_w2_mx_raw is not None and cw2_mx is not None:
                    gpu_w2_mx_raw[eid_i].copy_(cw2_mx[eid_i])
                self.gpu_b2[eid_i].copy_(cb2[eid_i])
                self.gpu_layer[eid_i] = layer_idx
            return

        # Path 2-4: CPU-offloaded layers
        cpu_layer = self.cpu[layer_idx]

        for eid in expert_ids:
            eid_i = int(eid)
            if self.gpu_layer[eid_i] == layer_idx:
                self._hit_count += 1
                continue

            # Path 3: victim cache hit → D2D restore
            if self._vc_slots > 0 and self._vc_get(
                layer_idx, eid_i,
                gpu_w1_raw[eid_i], gpu_w1_mx_raw[eid_i] if gpu_w1_mx_raw is not None else None,
                self.gpu_b1[eid_i],
                gpu_w2_raw[eid_i], gpu_w2_mx_raw[eid_i] if gpu_w2_mx_raw is not None else None,
                self.gpu_b2[eid_i],
            ):
                self._hit_count += 1
                self.gpu_layer[eid_i] = layer_idx
                continue

            self._miss_count += 1
            # Path 4: CPU→GPU copy. Save overwritten data to victim cache first.
            prev_layer = self.gpu_layer[eid_i]
            if self._vc_slots > 0 and prev_layer >= 0 and prev_layer != layer_idx:
                self._vc_put(
                    prev_layer, eid_i,
                    gpu_w1_raw[eid_i],
                    gpu_w1_mx_raw[eid_i] if gpu_w1_mx_raw is not None else None,
                    self.gpu_b1[eid_i],
                    gpu_w2_raw[eid_i],
                    gpu_w2_mx_raw[eid_i] if gpu_w2_mx_raw is not None else None,
                    self.gpu_b2[eid_i],
                )

            cw = cpu_layer[eid_i]
            gpu_w1_raw[eid_i].copy_(cw["w1"])
            if gpu_w1_mx_raw is not None and cw.get("w1_mx") is not None:
                gpu_w1_mx_raw[eid_i].copy_(cw["w1_mx"])
            self.gpu_b1[eid_i].copy_(cw["b1"])
            gpu_w2_raw[eid_i].copy_(cw["w2"])
            if gpu_w2_mx_raw is not None and cw.get("w2_mx") is not None:
                gpu_w2_mx_raw[eid_i].copy_(cw["w2_mx"])
            self.gpu_b2[eid_i].copy_(cw["b2"])
            self.gpu_layer[eid_i] = layer_idx

    def reset_hit_rate(self) -> None:
        self._hit_count = 0
        self._miss_count = 0

    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def load_all_experts(self, layer_idx: int) -> None:
        """Load every expert for *layer_idx* into the GPU buffers (for prefill)."""
        self.ensure_experts(layer_idx, list(range(self.num_experts)))

    # ------------------------------------------------------------------
    # CUDA Graph (decode matmul)
    # ------------------------------------------------------------------

    def init_decode_graph(
        self,
        rdata,           # RoutingData from triton_kernels.routing
        gather_indx,     # GatherIndx
        scatter_indx,    # ScatterIndx
        swiglu_limit: float = 7.0,
    ) -> None:
        """Capture a CUDA graph for the decode-matmul path.

        Called once after the first decode routing step.  The graph
        captures ``moe_decode_experts`` reading from the shared GPU
        buffers, so it can be replayed for every layer / every token
        as long as the buffers are refreshed beforehand.
        """
        from triton_kernels.routing import GatherIndx, ScatterIndx, RoutingData, ExptData
        from .moe import moe_decode_experts

        # Snapshot the routing tensors (these become the fixed-address proxies)
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

        # Input buffer — same shape as the decode activation
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

    def update_graph_routing(
        self,
        rdata,
        gather_indx,
        scatter_indx,
    ) -> None:
        """Copy fresh routing results into the graph's proxy tensors."""
        self._g_topk.copy_(gather_indx.src_indx)
        self._g_gate.copy_(gather_indx.dst_indx)
        self._g_scale.copy_(rdata.gate_scal)
        self._g_hist.copy_(rdata.expt_hist)
        self._g_offs_raw.copy_(rdata.expt_data.token_offs_raw)
        for k in self._g_offs_pad:
            self._g_offs_pad[k].copy_(rdata.expt_data.token_offs_pad[k])
        for k in self._g_pid_map:
            self._g_pid_map[k].copy_(rdata.expt_data.block_pid_map[k])

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
        """Capture a CUDA graph for the decode gate-routing path.

        The graph replays ``moe_decode_gate_routing`` writing into
        fixed-address proxy tensors.  Before each replay the caller
        must copy the current layer's *t*, *wg*, *bg* into
        ``_gr_t``, ``_gr_wg``, ``_gr_bg``.
        """
        from .moe import moe_decode_gate_routing

        # Pre-allocate input buffers
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
        """Release all global GPU buffers to reclaim memory."""
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
        self.gpu_layer = []
