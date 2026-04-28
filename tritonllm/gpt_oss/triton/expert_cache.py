"""CPU/GPU hybrid expert weight cache for MoE models.

Stores MoE expert weights on CPU (pageable, contiguous per-tensor) and
maintains shared GPU buffers that are filled on-demand as each layer's
experts are activated.  Only the activated experts (4/128) are transferred
per layer.
"""

from typing import Optional

import torch


class ExpertCache:
    """Global GPU buffers for expert weights, shared across all layers.

    Each layer's expert weights are stored on CPU as contiguous pageable
    tensors whose layout matches the GPU buffers (``(num_experts, ...)``).
    Before each MLP forward the activated experts for that layer are copied
    from CPU into the corresponding GPU slots.
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

        # Global GPU buffers — the custom Tensor objects themselves
        self.gpu_w1 = None
        self.gpu_w1_mx = None
        self.gpu_b1: Optional[torch.Tensor] = None
        self.gpu_w2 = None
        self.gpu_w2_mx = None
        self.gpu_b2: Optional[torch.Tensor] = None

        # Track which layer is in each GPU slot
        self.gpu_layer: list[int] = []

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
    # Public API
    # ------------------------------------------------------------------

    def ensure_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Copy expert weights from CPU → GPU if not already present.

        CPU storage is contiguous per-tensor (shape ``(num_experts, ...)``),
        so each ``cpu_tensor[eid]`` is a contiguous slice that maps directly
        to the corresponding GPU slot.
        """
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
        self.gpu_layer = []
