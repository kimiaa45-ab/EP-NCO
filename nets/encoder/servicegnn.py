# nets/encoder/servicegnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_softmax, scatter_add


class PerServiceModule(nn.Module):
    """
    Per-service message passing module.

    IMPORTANT (dataset compatibility):
      In your Dataset, service edge_attr[:,0] is dataSize (NOT bandwidth).
      Therefore edge weighting must DECREASE as dataSize increases.

    Weight per edge e:
      weight_e = alpha_e * ds_factor_e * lat_factor_e

    alpha_e      : attention softmax over incoming edges to each dst
    ds_factor_e  : smaller when dataSize larger
    lat_factor_e : smaller when latency larger (if provided)
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        n_layers: int,
        *,
        use_log1p: bool = True,   # stabilize ds/lat scale
        ds_mode: str = "inv",     # "inv" => 1/(1+ds) ; "exp" => exp(-ds)
        eps: float = 1e-9,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, model_dim)
        self.out_proj = nn.Identity()

        # attention-like parameters
        self.UHS = nn.Parameter(torch.randn(model_dim, model_dim) * (1.0 / (model_dim ** 0.5)))
        self.VHS = nn.Parameter(torch.randn(model_dim, model_dim) * (1.0 / (model_dim ** 0.5)))
        self.WS = nn.Parameter(torch.randn(2 * model_dim) * 0.1)

        self.n_layers = max(1, int(n_layers))

        self.msg_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )

        self.use_log1p = bool(use_log1p)
        self.ds_mode = str(ds_mode).lower()
        if self.ds_mode not in ("inv", "exp"):
            raise ValueError("ds_mode must be 'inv' or 'exp'")
        self.eps = float(eps)

        # init
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        for m in self.msg_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_edge_factors(self, ds: torch.Tensor, lat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ds: dataSize (>=0)
        lat: latency/delay (>=0) if provided else zeros
        """
        if self.use_log1p:
            ds_in = torch.log1p(ds)
            lat_in = torch.log1p(lat)
        else:
            ds_in = ds
            lat_in = lat

        if self.ds_mode == "inv":
            ds_factor = 1.0 / (1.0 + ds_in + self.eps)
        else:
            ds_factor = torch.exp(-ds_in)

        lat_factor = 1.0 / (1.0 + lat_in + self.eps)
        return ds_factor, lat_factor

    def forward(
        self,
        x_sub: torch.Tensor,
        local_edge_index: torch.LongTensor,
        edge_attr_sub: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_sub: [n_sub, in_dim]
        local_edge_index: [2, E_sub] local indices 0..n_sub-1
        edge_attr_sub:
          - [E_sub, 1] => col0=dataSize
          - [E_sub, 2] => col0=dataSize, col1=latency (optional)
        output: h_sub [n_sub, model_dim]
        """
        device = x_sub.device
        dtype = x_sub.dtype
        D = self.in_proj.out_features

        h = self.in_proj(x_sub)  # [n_sub, D]

        if local_edge_index is None or local_edge_index.numel() == 0:
            for _ in range(self.n_layers):
                h = F.relu(h)
            return h

        src = local_edge_index[0].long()
        dst = local_edge_index[1].long()
        E = int(src.numel())

        # edge attrs
        if edge_attr_sub is None or edge_attr_sub.numel() == 0:
            ds = torch.zeros(E, device=device, dtype=dtype)
            lat = torch.zeros(E, device=device, dtype=dtype)
        else:
            ea = edge_attr_sub.to(device=device, dtype=dtype)
            ds = torch.clamp(ea[:, 0], min=0.0)  # dataSize
            if ea.size(1) > 1:
                lat = torch.clamp(ea[:, 1], min=0.0)
            else:
                lat = torch.zeros(E, device=device, dtype=dtype)

        ds_factor, lat_factor = self._compute_edge_factors(ds, lat)

        for _ in range(self.n_layers):
            h_src = h[src]  # [E, D]
            h_dst = h[dst]  # [E, D]

            concat = torch.cat([h_dst, h_src], dim=-1)     # [E, 2D]
            scaled = concat * self.WS.unsqueeze(0)         # [E, 2D]
            left = scaled[:, :D]
            right = scaled[:, D:]

            a_left = torch.matmul(left, self.UHS)          # [E, D]
            a_right = torch.matmul(right, self.VHS)        # [E, D]

            score = (a_left * a_right).sum(dim=-1)         # [E]
            alpha = scatter_softmax(score, dst)            # [E] over incoming edges to dst

            msg = torch.matmul(h_src, self.VHS)            # [E, D]
            msg = self.msg_mlp(msg)                        # [E, D]

            # IMPORTANT: dataSize reduces weight
            weight = alpha * ds_factor * lat_factor        # [E]
            msg = msg * weight.view(-1, 1)

            agg = scatter_add(msg, dst, dim=0, dim_size=h.size(0))  # [n_sub, D]

            deg = torch.bincount(dst, minlength=h.size(0)).to(h.dtype).view(-1, 1)
            agg = agg / (deg + 1.0)

            h = h + agg
            h = F.layer_norm(h, h.shape[1:])
            h = F.relu(h)

        return h


class ServiceGNN(nn.Module):
    """
    ServiceGNN:
      - service_config: {service_id: {'in_dim','model_dim','n_layers', ...}}
      - forward accepts: Batch, Data, or list[Data]
      - returns: (node_embeddings [total_nodes, max_model_dim], batch)
    """

    def __init__(self, service_config: Dict[int, Dict], device=None):
        super().__init__()
        self.service_config = {int(k): v for k, v in service_config.items()}
        self.device = device if device is not None else torch.device("cpu")

        self.service_modules = nn.ModuleDict()
        self._max_model_dim = 0

        for sid, cfg in self.service_config.items():
            sid_s = str(sid)
            in_dim = int(cfg.get("in_dim", 3))
            model_dim = int(cfg.get("model_dim", 8))
            n_layers = int(cfg.get("n_layers", 2))

            use_log1p = bool(cfg.get("use_log1p", True))
            ds_mode = str(cfg.get("ds_mode", "inv"))

            self.service_modules[sid_s] = PerServiceModule(
                in_dim=in_dim,
                model_dim=model_dim,
                n_layers=n_layers,
                use_log1p=use_log1p,
                ds_mode=ds_mode,
            )
            self._max_model_dim = max(self._max_model_dim, model_dim)

    def _ensure_batch(self, batch_or_list: Union[Batch, Data, List[Data]]) -> Batch:
        if isinstance(batch_or_list, list):
            return Batch.from_data_list(batch_or_list)
        if isinstance(batch_or_list, Batch):
            return batch_or_list
        if isinstance(batch_or_list, Data):
            return Batch.from_data_list([batch_or_list])
        raise ValueError("Unsupported input type to ServiceGNN.forward")

    def forward(self, batch_or_list: Union[Batch, Data, List[Data]]):
        batch = self._ensure_batch(batch_or_list)
        x = batch.x
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "edge_attr", None)

        # service_id per node
        if hasattr(batch, "service_id") and batch.service_id is not None:
            # dataset sets d.service_id = tensor([sid]) per-graph
            if batch.service_id.numel() == x.size(0):
                service_ids_per_node = batch.service_id.to(torch.long)
            else:
                per_graph_ids = batch.service_id.view(-1).to(torch.long)  # [num_graphs]
                ptr = getattr(batch, "ptr", None)
                if ptr is None:
                    counts = torch.bincount(batch.batch)
                    ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
                service_ids_per_node = x.new_zeros(x.size(0), dtype=torch.long)
                for g in range(len(per_graph_ids)):
                    start = int(ptr[g].item())
                    end = int(ptr[g + 1].item())
                    service_ids_per_node[start:end] = int(per_graph_ids[g].item())
        else:
            raise RuntimeError(
                "ServiceGNN: batch must carry `service_id` (per-graph tensor([sid])). "
                "Your Dataset already sets this on each service Data."
            )

        device = x.device
        out = x.new_zeros((x.size(0), self._max_model_dim), dtype=x.dtype, device=device)

        unique_service_ids = torch.unique(service_ids_per_node)
        inv_map = -torch.ones(x.size(0), dtype=torch.long, device=device)

        for sid_t in unique_service_ids.tolist():
            sid = int(sid_t)
            sid_s = str(sid)
            if sid_s not in self.service_modules:
                continue

            mask = (service_ids_per_node == sid)
            if int(mask.sum().item()) == 0:
                continue

            node_idx = mask.nonzero(as_tuple=False).view(-1)
            x_sub = x[node_idx]

            inv_map.fill_(-1)
            inv_map[node_idx] = torch.arange(node_idx.size(0), device=device)

            if edge_index is None or edge_index.numel() == 0:
                local_ei = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_attr_sub = None
            else:
                src = edge_index[0].long()
                dst = edge_index[1].long()
                sel = (inv_map[src] >= 0) & (inv_map[dst] >= 0)
                if int(sel.sum().item()) == 0:
                    local_ei = torch.empty((2, 0), dtype=torch.long, device=device)
                    edge_attr_sub = None
                else:
                    local_src = inv_map[src[sel]]
                    local_dst = inv_map[dst[sel]]
                    local_ei = torch.stack([local_src, local_dst], dim=0)
                    edge_attr_sub = edge_attr[sel] if edge_attr is not None else None

            h_sub = self.service_modules[sid_s](x_sub, local_ei, edge_attr_sub)

            md = h_sub.size(1)
            out[node_idx, :md] = h_sub

        return out, batch


def make_service_data(x, edge_index, edge_attr=None, service_id: int = 0) -> Data:
    """
    helper to build per-service Data with service_id
    """
    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    d.service_id = torch.tensor([service_id], dtype=torch.long)
    return d
