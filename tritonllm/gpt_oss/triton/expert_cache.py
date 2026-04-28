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
