from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_log1p_feature(
    x: torch.Tensor,
    *,
    posinf: float = 1e6,
    clamp_max: float = 20.0,
) -> torch.Tensor:
    """
    Keep decoder side-features finite and on a sane scale.

    Raw capacities in this project are often O(1e3-1e4), which can dominate the
    policy MLP and make the learned logits effectively static. Missing-link
    delays can also arrive as +inf before masking. We compress both cases here.
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=0.0)
    x = torch.clamp(x, min=0.0)
    x = torch.log1p(x)
    return torch.clamp(x, max=clamp_max)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, nl=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nl(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
# def _build_dense_bw_delay_from_node_graph(
#     node_graph,
#     device: torch.device,
#     dtype: torch.dtype = torch.float32,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Build dense bw_mat, delay_mat from PyG node_graph (edge_index, edge_attr).
#     edge_attr is expected shape [E,2] with columns [bw, delay].
#     Missing edges => bw=0, delay=0.
#     """
#     N = int(node_graph.x.size(0))
#     bw_mat = torch.zeros((N, N), dtype=dtype, device=device)
#     delay_mat = torch.zeros((N, N), dtype=dtype, device=device)
#     ei = getattr(node_graph, "edge_index", None)
#     ea = getattr(node_graph, "edge_attr", None)
#     if ei is None or ea is None or ei.numel() == 0 or ea.numel() == 0:
#         return bw_mat, delay_mat
#     src = ei[0].long().to(device)
#     dst = ei[1].long().to(device)
#     ea = ea.to(device=device, dtype=dtype)
#     bw = ea[:, 0]
#     delay = ea[:, 1]
#     bw_mat[src, dst] = bw
#     delay_mat[src, dst] = delay
#     return bw_mat, delay_mat

def _build_dense_bw_delay_from_node_graph(
    node_graph,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build dense bw_mat, delay_mat from PyG node_graph (edge_index, edge_attr).
    edge_attr expected [E,2] columns [bw, delay].
    Missing edges => bw=0, delay=+inf (safer than 0).
    """
    N = int(node_graph.x.size(0))
    bw_mat = torch.zeros((N, N), dtype=dtype, device=device)
    # delay for missing links should be large, not 0
    delay_mat = torch.full((N, N), float("inf"), dtype=dtype, device=device)
    # self-link delay is 0
    delay_mat.fill_diagonal_(0.0)

    ei = getattr(node_graph, "edge_index", None)
    ea = getattr(node_graph, "edge_attr", None)
    if ei is None or ea is None or ei.numel() == 0 or ea.numel() == 0:
        return bw_mat, delay_mat

    src = ei[0].long().to(device)
    dst = ei[1].long().to(device)
    ea = ea.to(device=device, dtype=dtype)
    bw = ea[:, 0]
    delay = ea[:, 1]

    # bw_mat[src, dst] = bw
    # delay_mat[src, dst] = delay
    bw_mat[src, dst] = bw
    delay_mat[src, dst] = delay
    bw_mat[dst, src] = bw
    delay_mat[dst, src] = delay

    return bw_mat, delay_mat

class ARPlacementPipeline(nn.Module):
    """
    Autoregressive placement policy (component -> node).
    Inputs:
        comp_emb: [C, E]
        node_emb: [N, E]
        node_caps: [N, cap_dim]        available resources per node
        comp_reqs: [C, cap_dim]        requirement per component
        bw_mat:    [N, N]              bandwidth matrix (0 => no link)
        comp_predecessors: List[List[int]] length C
    Optional (recommended for network-aware scoring):
        delay_mat: [N, N]              delay matrix (0 if missing)
        node_graph: PyG Data           used to build delay_mat if delay_mat not provided
        service_edge_attr: Tensor      [E,1] or [E,k]; col0=dataSize for edges in components graph
        service_edge_index: Tensor     [2,E] edge_index for components graph
    Network feasibility:
        For each predecessor p already assigned to node i:
           candidate node j is feasible iff (j==i) OR (bw_mat[i,j] > 0)
    New (important):
        Adds per-candidate network features into policy input:
           - bw_feat(j): aggregated bandwidth from assigned predecessors to j (min or mean)
           - delay_feat(j): aggregated delay from assigned predecessors to j (min or mean)
           - ds_feat: aggregated dataSize from predecessors (scalar per component, broadcast)
    This helps the decoder "see" transmission cost components.
    """
    def __init__(
        self,
        embed_dim: int,
        proj_dim: int,
        policy_hidden: int = 128,
        device: Optional[torch.device] = None,
        cap_dim: int = 2,
        # network feature controls
        use_net_features: bool = True,
        net_agg: str = "min",  # "min" or "mean"
        use_datasize_feature: bool = True,
        net_feat_log: bool = True,  # use log1p on bw,delay,ds
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.proj_comp = nn.Linear(embed_dim, proj_dim)
        self.proj_node = nn.Linear(embed_dim, proj_dim)
        self.cap_dim = int(cap_dim)
        self.use_net_features = bool(use_net_features)
        self.net_agg = str(net_agg).lower()
        # if self.net_agg not in ("min", "mean"):
        #     raise ValueError("net_agg must be 'min' or 'mean'")
        if self.net_agg not in ("min", "mean", "max"):
            raise ValueError("net_agg must be 'min', 'mean', or 'max'")


        self.use_datasize_feature = bool(use_datasize_feature)
        self.net_feat_log = bool(net_feat_log)
        # input dims:
        # base: [qt_rep (P) , k (P) , node_caps (D)]  => 2P + D
        # + net feats: bw_feat, delay_feat (2 scalars)
        # + ds_feat (1 scalar) if enabled
        extra = 0
        if self.use_net_features:
            extra += 2
            if self.use_datasize_feature:
                extra += 1
        self.policy = MLP(
            in_dim=proj_dim * 2 + self.cap_dim + extra,
            hidden=policy_hidden,
            out_dim=1,
        )
        self.register_buffer("_eps", torch.tensor(1e-9))
    def _prepare_embeddings(
        self, comp_emb: torch.Tensor, node_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.proj_comp(comp_emb)  # [C, P]
        k = self.proj_node(node_emb)  # [N, P]
        return q, k
    @staticmethod
    def _ensure_2d(x: torch.Tensor, N_or_C: int, name: str) -> torch.Tensor:
        if x.dim() == 1:
            if x.numel() != N_or_C:
                raise ValueError(f"{name} expected length {N_or_C}, got {x.numel()}")
            return x.view(N_or_C, 1)
        if x.dim() == 2:
            if x.size(0) != N_or_C:
                raise ValueError(f"{name} expected first dim {N_or_C}, got {x.size(0)}")
            return x
        raise ValueError(f"{name} must be 1D or 2D, got shape {tuple(x.shape)}")
    @staticmethod
    def _network_mask_for_component(
        t, N, assignments, bw_mat, comp_predecessors, device
    ):
        preds = comp_predecessors[t]
        if not preds:
            return torch.ones(N, dtype=torch.bool, device=device)
        cand = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        for p in preds:
            p_node = assignments[p]
            if p_node < 0:
                continue
            same = cand == p_node
            # اگر directed می‌خوای فقط bw_mat[p_node, cand]
            bw_ok = (bw_mat[p_node, cand] > 0) | (bw_mat[cand, p_node] > 0)
            mask &= same | bw_ok
            if not mask.any():
                break
        return mask
    @staticmethod
    def _datasize_pred_feature(
        t: int,
        comp_predecessors: List[List[int]],
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns scalar ds_sum for component t = sum dataSize over edges (p -> t) for p in preds.
        If not available, returns 0.
        """
        if (
            edge_index is None
            or edge_attr is None
            or edge_index.numel() == 0
            or edge_attr.numel() == 0
        ):
            return torch.zeros((), device=device, dtype=dtype)
        preds = comp_predecessors[t]
        if not preds:
            return torch.zeros((), device=device, dtype=dtype)
        # Find edges where dst == t and src in preds
        # edge_index: [2,E]
        src = edge_index[0].long().to(device)
        dst = edge_index[1].long().to(device)
        ds = edge_attr[:, 0].to(device=device, dtype=dtype)
        dst_mask = dst == int(t)
        if not torch.any(dst_mask):
            return torch.zeros((), device=device, dtype=dtype)
        src2 = src[dst_mask]
        ds2 = ds[dst_mask]
        preds_t = torch.tensor(preds, device=device, dtype=torch.long)
        # membership test: for each src2, is in preds?
        # Use broadcasting (small preds) - fine for typical sizes
        in_preds = (src2.view(-1, 1) == preds_t.view(1, -1)).any(dim=1)
        if not torch.any(in_preds):
            return torch.zeros((), device=device, dtype=dtype)
        return ds2[in_preds].sum()
    def _net_features_for_component(
        self,
        t: int,
        N: int,
        assignments: torch.Tensor,  # [C]
        bw_mat: torch.Tensor,  # [N,N]
        delay_mat: torch.Tensor,  # [N,N]
        comp_predecessors: List[List[int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build candidate-wise bw_feat, delay_feat for component t.
        Defaults when no assigned predecessors:
            bw_feat = ones(N)
            delay_feat = zeros(N)
        """
        bw_feat = torch.ones((N,), device=device, dtype=dtype)
        delay_feat = torch.zeros((N,), device=device, dtype=dtype)
        preds = comp_predecessors[t]
        if not preds:
            return bw_feat, delay_feat
        cand = torch.arange(N, device=device)
        bw_vals = []
        d_vals = []
        for p in preds:
            p_node = assignments[p]
            if p_node >= 0:
                bw_vals.append(bw_mat[p_node, cand].to(dtype))
                d_vals.append(delay_mat[p_node, cand].to(dtype))
        if len(bw_vals) == 0:
            return bw_feat, delay_feat
        bw_stack = torch.stack(bw_vals, dim=0)  # [P, N]
        d_stack = torch.stack(d_vals, dim=0)  # [P, N]
        if self.net_agg == "min":
            bw_feat = bw_stack.min(dim=0).values
            delay_feat = d_stack.min(dim=0).values
        elif self.net_agg == "max":
            # bw still bottleneck-ish: min; delay worst-case: max
            bw_feat = bw_stack.min(dim=0).values
            delay_feat = d_stack.max(dim=0).values
        else:
            bw_feat = bw_stack.mean(dim=0)
            delay_feat = d_stack.mean(dim=0)

        # if self.net_agg == "min":
        #     bw_feat = bw_stack.min(dim=0).values
        #     delay_feat = d_stack.min(dim=0).values
        # else:
        #     bw_feat = bw_stack.mean(dim=0)
        #     delay_feat = d_stack.mean(dim=0)
        bw_feat = torch.nan_to_num(bw_feat, nan=0.0, posinf=1e6, neginf=0.0)
        delay_feat = torch.nan_to_num(delay_feat, nan=0.0, posinf=1e6, neginf=0.0)
        return bw_feat, delay_feat
    

    def _datasize_map_for_preds(
        self,
        t: int,
        preds: List[int],
        service_edge_index: Optional[torch.Tensor],
        service_edge_attr: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        """
        Returns dict: p -> dataSize for edges (p -> t). If missing, returns empty dict.
        """
        if (
            service_edge_index is None
            or service_edge_attr is None
            or service_edge_index.numel() == 0
            or service_edge_attr.numel() == 0
            or len(preds) == 0
        ):
            return {}

        src = service_edge_index[0].long().to(device)
        dst = service_edge_index[1].long().to(device)
        ds = service_edge_attr[:, 0].to(device=device, dtype=dtype)

        mask = (dst == int(t))
        if not torch.any(mask):
            return {}

        src2 = src[mask]
        ds2 = ds[mask]

        pred_set = torch.tensor(preds, device=device, dtype=torch.long)
        in_preds = (src2.view(-1, 1) == pred_set.view(1, -1)).any(dim=1)
        if not torch.any(in_preds):
            return {}

        out: Dict[int, torch.Tensor] = {}
        for p, val in zip(src2[in_preds].tolist(), ds2[in_preds]):
            # اگر چند edge تکراری بود، جمع کن
            if p in out:
                out[p] = out[p] + val
            else:
                out[p] = val
        return out


    def _net_cost_for_component(
        self,
        t: int,
        N: int,
        assignments: torch.Tensor,   # [C]
        bw_mat: torch.Tensor,        # [N,N]
        delay_mat: torch.Tensor,     # [N,N]
        comp_predecessors: List[List[int]],
        service_edge_index: Optional[torch.Tensor],
        service_edge_attr: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        *,
        lambda_delay: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns net_cost per candidate node j: shape [N]
        net_cost(j) = sum_{p in preds(t)} [ ds(p,t)/(bw_eff+eps) + lambda_delay * delay_eff ]
        where bw_eff/delay_eff are chosen consistent with feasibility mask (either direction).
        For same-node (j == p_node): cost = 0 for that predecessor.
        """
        preds = comp_predecessors[t]
        if not preds:
            return torch.zeros((N,), device=device, dtype=dtype)

        cand = torch.arange(N, device=device)
        eps = self._eps.to(device=device, dtype=dtype)

        # map p -> ds(p,t) if available
        ds_map = self._datasize_map_for_preds(
            t=t,
            preds=preds,
            service_edge_index=service_edge_index,
            service_edge_attr=service_edge_attr,
            device=device,
            dtype=dtype,
        )

        total = torch.zeros((N,), device=device, dtype=dtype)

        for p in preds:
            p_node = int(assignments[p].item())
            if p_node < 0:
                continue

            # choose direction consistent with your feasibility rule (either direction ok):
            bw_fwd = bw_mat[p_node, cand]
            bw_bwd = bw_mat[cand, p_node]
            use_fwd = bw_fwd >= bw_bwd
            bw_eff = torch.where(use_fwd, bw_fwd, bw_bwd)

            d_fwd = delay_mat[p_node, cand]
            d_bwd = delay_mat[cand, p_node]
            d_eff = torch.where(use_fwd, d_fwd, d_bwd)

            # same node => cost 0 for this predecessor
            same = cand == p_node
            bw_eff = torch.where(same, torch.full_like(bw_eff, 1e9), bw_eff)
            d_eff = torch.where(same, torch.zeros_like(d_eff), d_eff)

            ds_val = ds_map.get(p, None)
            if ds_val is None:
                # اگر dataSize نداریم، هنوز delay مهمه؛ ds رو 1 بگیر که transfer جزء کوچیک باشه
                ds_val = torch.ones((), device=device, dtype=dtype)

            transfer = ds_val / (bw_eff + eps)
            total = total + transfer + (float(lambda_delay) * d_eff)

        return torch.nan_to_num(total, nan=1e6, posinf=1e6, neginf=0.0)


    def assign_greedy_or_stochastic(
        self,
        comp_emb: torch.Tensor,
        node_emb: torch.Tensor,
        node_caps: torch.Tensor,  # [N,cap_dim] or [N]
        comp_reqs: torch.Tensor,  # [C,cap_dim] or [C]
        bw_mat: torch.Tensor,  # [N,N]
        comp_predecessors: List[List[int]],
        comp_fixed_node_rows: Optional[torch.Tensor] = None,  # [C] or None
        *,
        # optional for network-aware scoring
        delay_mat: Optional[torch.Tensor] = None,
        node_graph: Optional[object] = None,  # PyG Data
        service_edge_index: Optional[torch.Tensor] = None,
        service_edge_attr: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        greedy: bool = True,
        allow_fallback: bool = False,
        alpha_net: float = 1.0,
        lambda_delay: float = 1.0,
        sim_weight: float = 1.0,
        beta_crowd: float = 0.2,         # وزن ضدتمرکز
        crowd_use_util: bool = True,     # True: utilization-based, False: count-based
        crowd_gamma: float = 2.0,        # شدت نابرابری

        return_probs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = comp_emb.device
        dtype = torch.float32
        C = int(comp_emb.size(0))
        N = int(node_emb.size(0))
        if len(comp_predecessors) != C:
            raise ValueError(
                f"comp_predecessors length must be C={C}, got {len(comp_predecessors)}"
            )
        if comp_fixed_node_rows is not None:
            comp_fixed_node_rows = comp_fixed_node_rows.to(
                device=device, dtype=torch.long
            ).view(-1)
            if int(comp_fixed_node_rows.numel()) != C:
                raise ValueError(
                    f"comp_fixed_node_rows length must be C={C}, got {int(comp_fixed_node_rows.numel())}"
                )
        # move mats to device (environment, no grad needed)
        bw_mat = bw_mat.to(device=device, dtype=dtype)
        # delay_mat: use provided, else build from node_graph if possible, else zeros
        if delay_mat is not None:
            delay_mat = delay_mat.to(device=device, dtype=dtype)
        else:
            if (
                node_graph is not None
                and hasattr(node_graph, "edge_index")
                and hasattr(node_graph, "edge_attr")
            ):
                delay_mat = _build_dense_bw_delay_from_node_graph(
                    node_graph, device=device, dtype=dtype
                )[1]
            else:
                delay_mat = torch.full((N, N), float("inf"), device=device, dtype=dtype)
                delay_mat.fill_diagonal_(0.0)


            # else:
            #     delay_mat = torch.zeros((N, N), device=device, dtype=dtype)
        q, k = self._prepare_embeddings(comp_emb, node_emb)  # [C,P], [N,P]
        node_caps = node_caps.to(device=device, dtype=dtype)
        comp_reqs = comp_reqs.to(device=device, dtype=dtype)
        node_caps_2d = self._ensure_2d(node_caps, N, "node_caps").clone()
        node_caps_init = node_caps_2d.clone()  # برای utilization
        node_count = torch.zeros((N,), device=device, dtype=dtype)  # برای count-based

        comp_reqs_2d = self._ensure_2d(comp_reqs, C, "comp_reqs")
        cap_dim_runtime = int(node_caps_2d.size(1))
        if int(comp_reqs_2d.size(1)) != cap_dim_runtime:
            raise ValueError(
                f"cap_dim mismatch: node_caps D={cap_dim_runtime}, comp_reqs D={int(comp_reqs_2d.size(1))}"
            )
        if cap_dim_runtime != self.cap_dim:
            raise ValueError(
                f"ARPlacementPipeline initialized with cap_dim={self.cap_dim}, "
                f"but got node_caps D={cap_dim_runtime}. Re-init with cap_dim={cap_dim_runtime}."
            )
        assignments = torch.full((C,), -1, dtype=torch.long, device=device)
        log_probs_list: List[torch.Tensor] = []
        entropies_list: List[torch.Tensor] = []
        feasible_counts_list: List[torch.Tensor] = []
        soft_list: List[torch.Tensor] = []
        onehot_list: List[torch.Tensor] = []
        eps = self._eps.to(device=device, dtype=dtype)
        zero_scalar = node_caps_2d.new_zeros(())
        temp = max(float(temperature), 1e-6)
        # cache candidate indices once
        cand_idx = torch.arange(N, device=device)
        for t in range(C):
            req_vec = comp_reqs_2d[t]  # [D]
            qt = q[t : t + 1]  # [1,P]
            # base similarity logits: [N]
            sim = torch.matmul(qt, k.t()).view(-1)
            # capacity feasibility
            cap_mask = (node_caps_2d >= req_vec.view(1, -1)).all(dim=1)
            # network feasibility mask
            net_mask = self._network_mask_for_component(
                t=t,
                N=N,
                assignments=assignments,
                bw_mat=bw_mat,
                comp_predecessors=comp_predecessors,
                device=device,
            )
            feasible = cap_mask & net_mask
            # Optional hard placement constraints (e.g., first comp on user node, last on helper node)
            if comp_fixed_node_rows is not None:
                fixed_row = int(comp_fixed_node_rows[t].item())
                if 0 <= fixed_row < N:
                    fixed_mask = torch.zeros((N,), dtype=torch.bool, device=device)
                    fixed_mask[fixed_row] = True
                    feasible = feasible & fixed_mask
            feasible_counts_list.append(feasible.sum().to(dtype))
            if not feasible.any():
                if allow_fallback:
                    # fallback: فقط بین نودهایی که از نظر ظرفیت OK هستند انتخاب کن (شبکه را نادیده می‌گیریم، ولی ظرفیت نه)
                    cap_only = cap_mask
                    if cap_only.any():
                        scores_fb = node_caps_2d[:, 0].masked_fill(
                            ~cap_only, float("-inf")
                        )
                        idx = torch.argmax(scores_fb)
                        assignments[t] = idx
                        node_caps_2d[idx] = node_caps_2d[idx] - req_vec
                        node_count[idx] = node_count[idx] + 1.0

                    else:
                        assignments[t] = -1
                else:
                    assignments[t] = -1
                log_probs_list.append(zero_scalar)
                entropies_list.append(zero_scalar)
                if return_probs:
                    soft_list.append(torch.zeros((N,), device=device, dtype=dtype))
                    onehot_list.append(torch.zeros((N,), device=device, dtype=dtype))
                continue
            # -------- build policy inputs --------
            qt_rep = qt.repeat(N, 1)  # [N,P]
            # Use compressed residual-capacity features. Raw capacities are in
            # the thousands, which can swamp the learned embeddings/logits.
            cap_feat = _safe_log1p_feature(node_caps_2d)
            feats = [qt_rep, k, cap_feat]  # base
            if self.use_net_features:
                bw_feat, d_feat = self._net_features_for_component(
                    t=t,
                    N=N,
                    assignments=assignments,
                    bw_mat=bw_mat,
                    delay_mat=delay_mat,
                    comp_predecessors=comp_predecessors,
                    device=device,
                    dtype=dtype,
                )
                # Optional log transform for stability / scale
                if self.net_feat_log:
                    bw_feat_in = _safe_log1p_feature(bw_feat)
                    d_feat_in = _safe_log1p_feature(d_feat)
                else:
                    bw_feat_in = torch.nan_to_num(
                        bw_feat, nan=0.0, posinf=1e6, neginf=0.0
                    )
                    d_feat_in = torch.nan_to_num(
                        d_feat, nan=0.0, posinf=1e6, neginf=0.0
                    )
                feats.append(bw_feat_in.view(N, 1))
                feats.append(d_feat_in.view(N, 1))
                if self.use_datasize_feature:
                    ds_sum = self._datasize_pred_feature(
                        t=t,
                        comp_predecessors=comp_predecessors,
                        edge_index=service_edge_index,
                        edge_attr=service_edge_attr,
                        device=device,
                        dtype=dtype,
                    )
                    ################################
                    # if t == 1:  # فقط یکبار
                    #     print("DEBUG delay_mat nonzero:", (delay_mat > 0).float().mean().item())
                    #     print("DEBUG bw_feat:", bw_feat.min().item(), bw_feat.mean().item(), bw_feat.max().item())
                    #     print("DEBUG d_feat:", d_feat.min().item(), d_feat.mean().item(), d_feat.max().item())
                    #     if self.use_datasize_feature:
                    #         print("DEBUG ds_sum:", ds_sum.item())
                    if self.net_feat_log:
                        ds_in = _safe_log1p_feature(ds_sum)
                    else:
                        ds_in = torch.nan_to_num(
                            ds_sum, nan=0.0, posinf=1e6, neginf=0.0
                        )
                    feats.append(ds_in.view(1, 1).repeat(N, 1))
            x = torch.cat(feats, dim=1)  # [N, 2P + D (+ extras)]
            x = torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
            x = torch.clamp(x, min=-20.0, max=20.0)
            logits = self.policy(x).view(-1) + float(sim_weight) * sim
            # ---- anti-collapse / anti-crowding penalty ----
            if float(beta_crowd) != 0.0:
                if crowd_use_util:
                    used = (node_caps_init - node_caps_2d).clamp(min=0.0)
                    denom = node_caps_init.clamp(min=1e-6)
                    util = (used / denom).max(dim=1).values  # [N]
                    crowd_pen = util.pow(float(crowd_gamma))
                else:
                    crowd_pen = torch.log1p(node_count)
                logits = logits - float(beta_crowd) * crowd_pen

            # ---- net-cost shaping: force attention to network delay/transfer ----
            if float(alpha_net) != 0.0:
                net_cost = self._net_cost_for_component(
                    t=t,
                    N=N,
                    assignments=assignments,
                    bw_mat=bw_mat,
                    delay_mat=delay_mat,
                    comp_predecessors=comp_predecessors,
                    service_edge_index=service_edge_index,
                    service_edge_attr=service_edge_attr,
                    device=device,
                    dtype=dtype,
                    lambda_delay=float(lambda_delay),
                )
                logits = logits - float(alpha_net) * net_cost

            # numerical guard
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)
            # mask infeasible
            scores = logits.masked_fill(~feasible, float("-inf"))
            if greedy:
                # ✅ greedy واقعی: argmax مستقیم روی scores (بدون temp و بدون softmax)
                idx = torch.argmax(scores)
                logp = zero_scalar  # در eval اهمیتی نداره
                entropy_t = zero_scalar  # در eval اهمیتی نداره
            else:
                # stochastic فقط برای train
                finite_scores = scores[feasible]
                score_shift = finite_scores.max().detach() if finite_scores.numel() > 0 else zero_scalar
                scores_t = (scores - score_shift) / temp
                scores_t = torch.nan_to_num(
                    scores_t, nan=-1e3, posinf=0.0, neginf=-1e3
                )
                scores_t = torch.clamp(scores_t, min=-60.0, max=0.0)
                m = torch.distributions.Categorical(logits=scores_t)  # بهتر از probs
                idx = m.sample()
                logp = m.log_prob(idx)
                entropy_t = m.entropy()
            # اگر return_probs می‌خوای:
            if return_probs:
                probs = F.softmax(scores, dim=0)  # بدون temp (برای نمایش)
            assignments[t] = idx
            node_caps_2d[idx] = node_caps_2d[idx] - req_vec
            node_count[idx] = node_count[idx] + 1.0

            log_probs_list.append(logp)
            entropies_list.append(entropy_t)
            if return_probs:
                soft_list.append(probs)
                one_hot = torch.zeros_like(probs)
                one_hot[idx] = 1.0
                onehot_list.append(one_hot)
        out: Dict[str, torch.Tensor] = {
            "assignments": assignments,
            "log_probs": torch.stack(log_probs_list),
            "entropies": torch.stack(entropies_list),
            "feasible_counts": torch.stack(feasible_counts_list),
            "final_node_caps": node_caps_2d.detach().clone(),
        }
        if return_probs:
            out["soft"] = torch.stack(soft_list, dim=0)
            out["one_hot"] = torch.stack(onehot_list, dim=0)
        return out
    @staticmethod
    def group_by_service(
        assignments: torch.Tensor, comp_service_map: List[int]
    ) -> Dict[int, List[Tuple[int, int]]]:
        out: Dict[int, List[Tuple[int, int]]] = {}
        for i, s in enumerate(comp_service_map):
            out.setdefault(s, []).append((i, int(assignments[i].detach().cpu().item())))
        return out
