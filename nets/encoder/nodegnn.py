# nodegnn_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data


class NodeGNNLayer(nn.Module):
    """
    Node-level GNN layer with learned edge attention and multiplicative gating.

    Expects:
      - h: [N, dim] node embeddings
      - edge_index: [2, E] with (src, dst)
      - edge_attr: [E, K] where:
            edge_attr[:,0] = bandwidth (bw)
            edge_attr[:,1] = delay/latency (optional)

    Key weighting:
      alpha_e : normalized attention over incoming edges to each dst
      bw_factor_e    = bw / (1 + bw)
      delay_factor_e = 1 / (1 + delay)
      weight_e = alpha_e * bw_factor_e * delay_factor_e
    """

    def __init__(self, dim: int, eps: float = 1e-6, negative_slope: float = 0.05, use_log1p: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.negative_slope = float(negative_slope)
        self.use_log1p = bool(use_log1p)

        # linear transforms
        self.linear_src = nn.Linear(dim, dim, bias=False)
        self.linear_dst = nn.Linear(dim, dim, bias=False)

        # message transform
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Linear(dim, dim)
        )

        # edge attention MLP
        # input = [h_dst_e, h_src_e, bw, delay]  -> scalar score
        self.edge_score = nn.Sequential(
            nn.Linear(dim * 2 + 2, dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Linear(dim, 1)
        )

        self.norm = nn.LayerNorm(dim)

        # init
        nn.init.xavier_uniform_(self.linear_src.weight)
        nn.init.xavier_uniform_(self.linear_dst.weight)

        for m in self.msg_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.edge_score:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, edge_index: torch.LongTensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        device = h.device
        N = int(h.size(0))

        if edge_index is None or edge_index.numel() == 0:
            out = self.norm(h)
            return F.leaky_relu(out, negative_slope=self.negative_slope)

        src = edge_index[0].long()
        dst = edge_index[1].long()
        E = int(src.numel())

        # edge features
        if edge_attr is None:
            bw = torch.ones(E, device=device, dtype=h.dtype)
            delay = torch.zeros(E, device=device, dtype=h.dtype)
        else:
            ea = edge_attr.to(device=device, dtype=h.dtype)
            bw = torch.clamp(ea[:, 0], min=0.0)
            if ea.size(1) > 1:
                delay = torch.clamp(ea[:, 1], min=0.0)
            else:
                delay = torch.zeros(E, device=device, dtype=h.dtype)

        # optional log scaling (sometimes helps if bw/delay have large ranges)
        if self.use_log1p:
            bw_in = torch.log1p(bw)
            delay_in = torch.log1p(delay)
        else:
            bw_in = bw
            delay_in = delay

        # transforms
        h_src_lin = self.linear_src(h)   # [N, D]
        h_dst_lin = self.linear_dst(h)   # [N, D]

        h_src_e = h_src_lin[src]         # [E, D]
        h_dst_e = h_dst_lin[dst]         # [E, D]

        # message content uses raw source embedding (common choice)
        msg = self.msg_mlp(h[src])       # [E, D]

        # attention score uses dst/src + bw + delay
        score_in = torch.cat([h_dst_e, h_src_e, bw_in.view(-1, 1), delay_in.view(-1, 1)], dim=1)  # [E, 2D+2]
        raw_score = self.edge_score(score_in).view(-1)  # [E]

        # ----- stable normalization per-dst -----
        # compute max score per dst for stability
        # scatter_add can't max, so we do a safe trick:
        # we approximate with global max shift if needed, but better is scatter_reduce.
        # torch_scatter has scatter_max in newer versions; if not available, fall back.
        try:
            from torch_scatter import scatter_max  # type: ignore
            max_per_dst, _ = scatter_max(raw_score, dst, dim=0, dim_size=N)
            max_shift = max_per_dst[dst]
        except Exception:
            # fallback: global max (less sharp but still stable)
            max_shift = raw_score.max().detach()

        exp_score = torch.exp(raw_score - max_shift)
        denom = scatter_add(exp_score, dst, dim=0, dim_size=N).clamp(min=self.eps)  # [N]
        alpha = exp_score / (denom[dst] + self.eps)  # [E]

        # ----- multiplicative gating for bw and delay -----
        # bw_factor in (0,1)
        bw_factor = bw / (1.0 + bw + self.eps)

        # delay_factor in (0,1]  (higher delay => smaller factor)
        delay_factor = 1.0 / (1.0 + delay + self.eps)

        weight = alpha * bw_factor * delay_factor  # [E]

        msg_weighted = msg * weight.view(-1, 1)
        agg = scatter_add(msg_weighted, dst, dim=0, dim_size=N)  # [N, D]

        # mild normalization by in-degree
        deg = torch.bincount(dst, minlength=N).to(h.dtype).view(N, 1)
        agg = agg / (deg + 1.0)

        # residual + norm + activation
        h_new = h + agg
        h_new = self.norm(h_new)
        h_new = F.leaky_relu(h_new, negative_slope=self.negative_slope)
        return h_new


class NodeGNN(nn.Module):
    """
    Stacked NodeGNNLayer.
    - in_node_feat: how many node features to use from data.x
    - dim: embedding dim
    - n_layers: number of GNN layers
    """

    def __init__(self, in_node_feat: int = 5, in_edge_feat: int = 2, dim: int = 8, n_layers: int = 3):
        super().__init__()
        self.dim = int(dim)
        self.n_layers = int(n_layers)

        self.node_in = nn.Linear(in_node_feat, dim)
        nn.init.xavier_uniform_(self.node_in.weight)
        nn.init.zeros_(self.node_in.bias)

        self.layers = nn.ModuleList([NodeGNNLayer(dim) for _ in range(n_layers)])
        self.out_proj = nn.Identity()

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        if x is None:
            raise ValueError("NodeGNN.forward: data.x is required")

        # take first in_node_feat columns (e.g., cpu, memory)
        h = self.node_in(x[:, : self.node_in.in_features].to(dtype=torch.float32, device=x.device))

        edge_index = getattr(data, "edge_index", None)
        edge_attr = getattr(data, "edge_attr", None)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        return self.out_proj(h)
