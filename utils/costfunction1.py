# cost.py
from __future__ import annotations
from typing import Dict, List, Tuple, Union, Optional
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
DataBatch = Batch
def _to_tensor(x, device: torch.device) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)
def _ensure_bw_delay_mats(nodes_data: Data) -> Tuple[Tensor, Tensor]:
    """
    Build dense bandwidth/delay matrices on the same device as nodes_data.x.
    bw_mat[u,v]    = bandwidth (0 if missing)
    delay_mat[u,v] = delay     (0 if missing)
    Self-edges are ignored here (caller may treat u==v as bw=inf, delay=0).
    """
    device = nodes_data.x.device
    N = int(nodes_data.x.size(0))
    bw_mat = getattr(nodes_data, "_bw_mat", None)
    delay_mat = getattr(nodes_data, "_delay_mat", None)
    if (
        isinstance(bw_mat, torch.Tensor)
        and isinstance(delay_mat, torch.Tensor)
        and bw_mat.device == device
        and delay_mat.device == device
        and bw_mat.shape == (N, N)
        and delay_mat.shape == (N, N)
    ):
        return bw_mat, delay_mat
    bw_mat = torch.zeros((N, N), dtype=torch.float32, device=device)
    delay_mat = torch.zeros((N, N), dtype=torch.float32, device=device)
    ei = getattr(nodes_data, "edge_index", None)
    ea = getattr(nodes_data, "edge_attr", None)
    if (
        isinstance(ei, torch.Tensor)
        and isinstance(ea, torch.Tensor)
        and ei.numel() > 0
        and ea.numel() > 0
    ):
        src = ei[0].long()
        dst = ei[1].long()
        bw = ea[:, 0].float()
        delay = ea[:, 1].float()
        bw_mat[src, dst] = bw
        delay_mat[src, dst] = delay
    # Optional caching (enable if nodes_data lives long enough and you want speed)
    # nodes_data._bw_mat = bw_mat
    # nodes_data._delay_mat = delay_mat
    return bw_mat, delay_mat
def _ensure_nodeid_mapping(nodes_data: Data) -> Tuple[Tensor, Tensor]:
    """
    Prepare mapping from nodeID -> row index via sorting + searchsorted.
    Assumes nodes_data.x[:,2] = nodeID.
    """
    device = nodes_data.x.device
    node_ids = nodes_data.x[:, 2].long()
    sids = getattr(nodes_data, "_sorted_nodeids", None)
    srows = getattr(nodes_data, "_sorted_rowidx", None)
    if (
        isinstance(sids, torch.Tensor)
        and isinstance(srows, torch.Tensor)
        and sids.device == device
        and srows.device == device
        and sids.numel() == node_ids.numel()
    ):
        return sids, srows
    sids, perm = torch.sort(node_ids)
    srows = perm.long()
    nodes_data._sorted_nodeids = sids
    nodes_data._sorted_rowidx = srows
    return sids, srows
def _map_nodeids_to_rows(node_ids_query: Tensor, nodes_data: Data) -> Tensor:
    """
    Vectorized mapping from nodeID -> row index.
    If not found exactly, maps to 0.
    """
    device = nodes_data.x.device
    q = node_ids_query.long().to(device)
    sids, srows = _ensure_nodeid_mapping(nodes_data)
    pos = torch.searchsorted(sids, q)
    pos = torch.clamp(pos, 0, sids.numel() - 1)
    matched = sids[pos] == q
    out = torch.where(matched, srows[pos], torch.zeros_like(pos))
    return out.long()
def _get_cc_for_service(
    components_batch: DataBatch,
    s: int,
    device: torch.device,
    starts: Tensor,
    ends: Tensor,
) -> Tensor:
    """
    Returns componentConnections matrix for service s as float32 on device.
    Supports:
      - list/tuple: cc[s] is [m,m]
      - Tensor [S,m,m]: cc[s]
      - Tensor concatenated [sum(m), m]: slice rows for service
      - (common in your json) single [m,m] shared by all services: Tensor dim==2 and rows==m
    """
    cc = getattr(components_batch, "component_connections", None)
    if cc is None:
        raise ValueError("components_batch must have attribute `component_connections`")
    s_start = int(starts[s].item())
    s_end = int(ends[s].item())
    m = s_end - s_start
    if isinstance(cc, (list, tuple)):
        return torch.as_tensor(cc[s], device=device, dtype=torch.float32)
    if isinstance(cc, torch.Tensor):
        if cc.dim() == 3:
            return cc[s].to(device).float()
        if cc.dim() == 2:
            # Case A: shared [m,m]
            if cc.size(0) == m and cc.size(1) == m:
                return cc.to(device).float()
            # Case B: concatenated
            return cc[s_start:s_end, :m].to(device).float()
    raise ValueError(
        f"Unsupported component_connections type/shape: type={type(cc)}, shape={getattr(cc, 'shape', None)}"
    )
def _get_comp_datasize(components_batch: DataBatch) -> Tensor:
    """
    Returns per-component datasize vector [C] on the same device.
    Priority:
      1) components_batch.comp_datasize
      2) components_batch.x[:,2] if exists
      3) fallback zeros
    """
    if hasattr(components_batch, "comp_datasize"):
        ds = components_batch.comp_datasize
        return ds if isinstance(ds, torch.Tensor) else torch.as_tensor(ds)
    x = getattr(components_batch, "x", None)
    if isinstance(x, torch.Tensor) and x.size(-1) >= 3:
        return x[:, 2]
    return torch.zeros(
        (int(components_batch.x.size(0)),),
        device=components_batch.x.device,
        dtype=torch.float32,
    )
def _get_node_reliability(nodes_data: Data) -> Tensor:
    """
    Node reliability vector [N].
    Priority:
      1) nodes_data.x[:,4] if exists ([cpu, mem, nodeID, tier, reliability])
      2) fallback ones
    """
    x = getattr(nodes_data, "x", None)
    if isinstance(x, torch.Tensor) and x.size(-1) >= 5:
        return x[:, 4].float()
    return torch.ones((int(nodes_data.x.size(0)),), device=nodes_data.x.device, dtype=torch.float32)
def _get_comp_reliability(components_batch: DataBatch) -> Tensor:
    """
    Component reliability vector [C].
    Priority:
      1) components_batch.x[:,2] if exists ([cpu, mem, reliability])
      2) fallback ones
    """
    x = getattr(components_batch, "x", None)
    if isinstance(x, torch.Tensor) and x.size(-1) >= 3:
        return x[:, 2].float()
    return torch.ones((int(components_batch.x.size(0)),), device=components_batch.x.device, dtype=torch.float32)
def compute_cost_ga_like(
    assignments: Union[Tensor, List[int]],
    nodes_data: Data,
    components_batch: DataBatch,
    *,
    heal: bool = True,
    ga_compat: bool = True,
    normalize_by=None,
    bandwidth_sharing: bool = True,
    count_edges_once: bool = True,
    log: bool = False,
    return_breakdown: bool = True,
    eps: float = 1e-9,
    **kwargs,
) -> Dict[str, Union[Tensor, float, List[float]]]:
    device = nodes_data.x.device
    violation_weight = float(kwargs.get("violation_weight", 1))
    # NEW: utilization regularizer strength (default 0 => backward compatible)
    lambda_u = float(kwargs.get("lambda_u", 0.0))
    spread_guard = str(kwargs.get("spread_guard", "ga")).lower()  # "ga" | "always"
    # ---- multi-objective controls ----
    objective_mode = str(kwargs.get("objective_mode", "weighted_sum")).lower()
    w_rt = float(kwargs.get("w_rt", kwargs.get("response_time_weight", 1.0)))
    w_pr = float(kwargs.get("w_pr", kwargs.get("platform_reliability_weight", 0.0)))
    w_sr = float(kwargs.get("w_sr", kwargs.get("service_reliability_weight", 0.0)))
    normalize_rt = bool(kwargs.get("normalize_rt", False))
    rt_ref_cfg = kwargs.get("rt_ref", None)

    # ---- GA guard (anti "too-good-to-be-true") ----
    ga_ref = kwargs.get("ga_ref", None)  # scalar tensor/float
    ga_tol = float(kwargs.get("ga_tol", 0.10))  # 10%
    ga_penalty_weight = float(kwargs.get("ga_penalty_weight", 1))
    # ---- anti-collapse spread regularizer (only when GA guard triggers) ----
    lambda_spread = float(kwargs.get("lambda_spread", 0.0))  # strength
    min_unique_nodes = int(kwargs.get("min_unique_nodes", 2))  # per service
    assignments = _to_tensor(assignments, device=device).long()
    comp_fixed_node_rows = kwargs.get("comp_fixed_node_rows", None)
    # ---- invalid assignment check ----
    invalid = assignments < 0
    if torch.any(invalid):
        num_invalid = invalid.sum().to(torch.float32)
        huge_penalty = violation_weight * num_invalid
        z = torch.zeros((), device=device, dtype=torch.float32)
        out = {
            "objective": huge_penalty,
            "total_response_time": huge_penalty,
            "legacy_total_response_time": huge_penalty,
            "response_time": z,
            "platform_reliability": z,
            "service_reliability": z,
            "normalized_response_time": z,
            "exec_time": z,
            "transmission": z,
            "provider_delay": 0.0,
            "codec_delay": 0.0,
            "violations": num_invalid,
            "bw0_violations": num_invalid,
            # NEW: keep key for consistency
            "util_penalty": z,
        }
        if return_breakdown:
            out["per_service_exec"] = []
            out["per_service_trans"] = []
            out["per_service_total"] = []
            out["per_service_bw0_viol"] = []
        return out
    N = int(nodes_data.x.size(0))
    C = int(components_batch.x.size(0))
    # ---- map assignment -> node row (either already row index or nodeID) ----
    min_a = int(assignments.min().item())
    max_a = int(assignments.max().item())
    if 0 <= min_a and max_a < N:
        node_rows = assignments.clone()
    else:
        node_rows = _map_nodeids_to_rows(assignments, nodes_data)
    fixed_viol = torch.zeros((), device=device, dtype=torch.float32)
    if comp_fixed_node_rows is not None:
        fixed_rows = _to_tensor(comp_fixed_node_rows, device=device).long().view(-1)
        if int(fixed_rows.numel()) == C:
            fixed_mask = fixed_rows >= 0
            if torch.any(fixed_mask):
                fixed_viol = (
                    node_rows[fixed_mask] != fixed_rows[fixed_mask]
                ).sum().to(torch.float32)
    # ---- dense infra mats ----
    bw_mat, delay_mat = _ensure_bw_delay_mats(nodes_data)
    # ---- capacity violations (CPU+MEM) ----
    cpu_req = components_batch.x[:, 0].float()
    mem_req = components_batch.x[:, 1].float()
    cpu_cap_node = nodes_data.x[:, 0].float()
    mem_cap_node = nodes_data.x[:, 1].float()
    cpu_used = torch.zeros((N,), device=device, dtype=torch.float32)
    mem_used = torch.zeros((N,), device=device, dtype=torch.float32)
    cpu_used.index_add_(0, node_rows, cpu_req)
    mem_used.index_add_(0, node_rows, mem_req)
    # cpu_excess = torch.clamp(cpu_used - cpu_cap_node, min=0.0)
    # mem_excess = torch.clamp(mem_used - mem_cap_node, min=0.0)
    # cap_viol = cpu_excess.sum() + mem_excess.sum()
    cpu_excess = torch.clamp(cpu_used - cpu_cap_node, min=0.0)
    mem_excess = torch.clamp(mem_used - mem_cap_node, min=0.0)

    # ✅ count-based violation: هر نودی که CPU یا MEM رو نقض کرد => 1
    cap_viol = ((cpu_excess > 0) | (mem_excess > 0)).sum().to(torch.float32)

    # NEW: utilization penalty to discourage collapse (soft, no hard K)
    # Normalize by the number of active nodes so this term stays on the same scale
    # as response_time instead of dominating the objective.
    cpu_cap_safe_node = torch.clamp(cpu_cap_node, min=eps)
    mem_cap_safe_node = torch.clamp(mem_cap_node, min=eps)
    util_cpu = cpu_used / cpu_cap_safe_node
    util_mem = mem_used / mem_cap_safe_node
    active_nodes = ((cpu_used > eps) | (mem_used > eps)).sum().to(torch.float32)
    active_nodes = torch.clamp(active_nodes, min=1.0)
    # Sum of squares penalizes concentrated load more than spread load.
    util_penalty = (
        lambda_u * (util_cpu.pow(2) + util_mem.pow(2)).sum() / active_nodes
    )
    # ---- execution time (global) ----
    cpu_cap_assigned = nodes_data.x[node_rows, 0].float()
    cpu_cap_safe = torch.clamp(cpu_cap_assigned, min=eps)
    comp_exec = cpu_req / cpu_cap_safe  # [C]
    exec_time = comp_exec.sum()
    # ---- determine per-service ranges ----
    ptr = getattr(components_batch, "ptr", None)
    B = getattr(components_batch, "batch", None)
    if ptr is not None:
        num_services = int(ptr.numel() - 1)
        starts = ptr[:-1].long()
        ends = ptr[1:].long()
    else:
        if B is None:
            raise ValueError("components_batch must have either ptr or batch.")
        num_services = int(B.max().long().detach().cpu().item()) + 1
        counts = torch.bincount(B.long(), minlength=num_services)
        starts = torch.cumsum(
            torch.cat([torch.zeros(1, device=device, dtype=torch.long), counts[:-1]]),
            dim=0,
        )
        ends = starts + counts
    # ---- reliability objectives ----
    node_rel = _get_node_reliability(nodes_data).to(device=device, dtype=torch.float32)
    comp_rel = _get_comp_reliability(components_batch).to(device=device, dtype=torch.float32)
    pr_sum = torch.zeros((), device=device, dtype=torch.float32)
    sr_sum = torch.zeros((), device=device, dtype=torch.float32)
    counted_services = 0
    for s in range(num_services):
        s_start = int(starts[s].item())
        s_end = int(ends[s].item())
        m = s_end - s_start
        if m <= 0:
            continue
        counted_services += 1
        rows_s = node_rows[s_start:s_end]
        uniq_nodes = torch.unique(rows_s)
        pr_s = node_rel[uniq_nodes].prod() if uniq_nodes.numel() > 0 else torch.ones((), device=device, dtype=torch.float32)
        sr_s = comp_rel[s_start:s_end].prod()
        pr_sum = pr_sum + pr_s
        sr_sum = sr_sum + sr_s
    denom_s = float(max(1, counted_services))
    platform_reliability = pr_sum / denom_s
    service_reliability = sr_sum / denom_s
    # ---- prefetch datasize ----
    comp_datasize = _get_comp_datasize(components_batch).to(device).float()
    if log:
        print(
            "comp_datasize stats:",
            comp_datasize.min().item(),
            comp_datasize.mean().item(),
            comp_datasize.max().item(),
            "x_dim:",
            components_batch.x.size(1),
        )
    # ---- build conn[u,v] for bandwidth sharing ----
    conn = None
    if bandwidth_sharing:
        conn = torch.ones((N, N), dtype=torch.float32, device=device)
        for s in range(num_services):
            s_start = int(starts[s].item())
            s_end = int(ends[s].item())
            m = s_end - s_start
            if m <= 0:
                continue
            cc_s = _get_cc_for_service(
                components_batch, s, device, starts, ends
            )  # [m,m]
            ii, jj = torch.nonzero(
                cc_s != 0, as_tuple=True
            )  # همه‌ی edgeها (directed/undirected)
            if ii.numel() == 0:
                continue
            # اگر self-edge نمی‌خوای:
            not_self_ij = ii != jj
            ii, jj = ii[not_self_ij], jj[not_self_ij]
            if ii.numel() == 0:
                continue
            es = (ii + s_start).long()
            ed = (jj + s_start).long()
            src_rows = node_rows[es]
            dst_rows = node_rows[ed]
            not_self = src_rows != dst_rows
            if not torch.any(not_self):
                continue
            src_rows = src_rows[not_self]
            dst_rows = dst_rows[not_self]
            raw_bw = bw_mat[src_rows, dst_rows]
            # has_link = raw_bw > 0
            # if not torch.any(has_link):
            #     continue
            conn[src_rows, dst_rows] += 1.0

            #conn[src_rows[has_link], dst_rows[has_link]] += 1.0
    # ---- transmission + bw0 violations + per-service breakdown ----
    transmission = torch.zeros((), device=device, dtype=torch.float32)
    bw0_viol = torch.zeros((), device=device, dtype=torch.float32)
    per_service_exec: List[float] = []
    per_service_trans: List[float] = []
    per_service_total: List[float] = []
    per_service_bw0: List[float] = []
    edge_index = getattr(components_batch, "edge_index", None)
    edge_attr = getattr(components_batch, "edge_attr", None)
    for s in range(num_services):
        s_start = int(starts[s].item())
        s_end = int(ends[s].item())
        m = s_end - s_start
        s_exec_t = (
            comp_exec[s_start:s_end].sum() if m > 0 else torch.zeros((), device=device)
        )
        s_trans_t = torch.zeros((), device=device)
        s_bw0_t = torch.zeros((), device=device)
        if m > 0:
            if edge_index is None or edge_index.numel() == 0:
                if return_breakdown:
                    per_service_exec.append(float(s_exec_t.detach().cpu().item()))
                    per_service_trans.append(0.0)
                    per_service_total.append(float(s_exec_t.detach().cpu().item()))
                    per_service_bw0.append(0.0)
                continue
            src_comp = edge_index[0]
            dst_comp = edge_index[1]
            mask_s = (
                (src_comp >= s_start)
                & (src_comp < s_end)
                & (dst_comp >= s_start)
                & (dst_comp < s_end)
            )
            if not torch.any(mask_s):
                if return_breakdown:
                    per_service_exec.append(float(s_exec_t.detach().cpu().item()))
                    per_service_trans.append(0.0)
                    per_service_total.append(float(s_exec_t.detach().cpu().item()))
                    per_service_bw0.append(0.0)
                continue
            src_comp_s = src_comp[mask_s]
            dst_comp_s = dst_comp[mask_s]
            src_rows = node_rows[src_comp_s]
            dst_rows = node_rows[dst_comp_s]
            not_self = src_rows != dst_rows
            src_comp_s = src_comp_s[not_self]
            dst_comp_s = dst_comp_s[not_self]
            src_rows = src_rows[not_self]
            dst_rows = dst_rows[not_self]
            raw_bw = bw_mat[src_rows, dst_rows]
            link_delay = delay_mat[src_rows, dst_rows]
            s_bw0_t = (raw_bw == 0).sum().to(torch.float32)
            bw0_viol = bw0_viol + s_bw0_t
            # raw_bw = bw_mat[src_rows, dst_rows]
            # link_delay = delay_mat[src_rows, dst_rows]

            # has_link = raw_bw > 0
            # src_comp_s = src_comp_s[has_link]
            # dst_comp_s = dst_comp_s[has_link]
            # src_rows = src_rows[has_link]
            # dst_rows = dst_rows[has_link]
            # raw_bw = raw_bw[has_link]
            # link_delay = link_delay[has_link]
            # ✅ datasize edge (یا fallback)
            if edge_attr is not None and edge_attr.numel() > 0:
                edge_attr_s = edge_attr[mask_s]  # [E_s,1]
                edge_attr_s = edge_attr_s[not_self]  # [E_s2,1]
                #edge_attr_s = edge_attr_s[has_link]  # [E_s3,1]
                data_size = edge_attr_s[:, 0].float()
            else:
                data_size = comp_datasize[src_comp_s].float()
            if log and s == 0:
                print(
                    "edge datasize stats:",
                    data_size.min().item(),
                    data_size.mean().item(),
                    data_size.max().item(),
                    "E=",
                    data_size.numel(),
                )
            if bandwidth_sharing:
                k = conn[src_rows, dst_rows]
                eff_bw = raw_bw / torch.clamp(k, min=1.0)
            else:
                eff_bw = raw_bw
                
            is_bw0 = (eff_bw <= 0)

            bw_floor = float(kwargs.get("bw_floor", 1.0))  # مثلا 1.0
            eff_bw_safe = torch.where(is_bw0, torch.full_like(eff_bw, bw_floor), eff_bw)

            td = (data_size / torch.clamp(eff_bw_safe, min=eps)) + link_delay

            bw0_penalty_weight = float(kwargs.get("bw0_penalty_weight", violation_weight))
            td = td + is_bw0.to(td.dtype) * bw0_penalty_weight

           # td = (data_size / torch.clamp(eff_bw, min=eps)) + link_delay
            s_trans_t = td.sum()
            transmission = transmission + s_trans_t
        if return_breakdown:
            per_service_exec.append(float(s_exec_t.detach().cpu().item()))
            per_service_trans.append(float(s_trans_t.detach().cpu().item()))
            per_service_total.append(
                float((s_exec_t + s_trans_t).detach().cpu().item())
            )
            per_service_bw0.append(float(s_bw0_t.detach().cpu().item()))
    # violations = bw0_viol + cap_viol
    # if log: print(f"cap_viol={cap_viol:.4f}, bw0_viol={bw0_viol}, penalty= {violations:.1f}")
    # # NEW: include util_penalty
    # total_response_time = exec_time + transmission + violation_weight * violations + util_penalty
    violations = cap_viol + fixed_viol
    if log:
        print(
            f"cap_viol={cap_viol:.4f}, bw0_viol={bw0_viol}, penalty= {violations:.1f}"
        )
    response_time = exec_time + transmission
    penalties_base = violation_weight * violations + util_penalty
    base_cost = response_time + penalties_base
    # =========================
    # GA guard: اگر بیشتر از ga_tol بهتر از GA شد => جریمه
    # =========================
    ga_penalty = torch.zeros((), device=device, dtype=torch.float32)
    ga_trigger = torch.zeros((), device=device, dtype=torch.float32)
    if ga_ref is not None:
        ga_ref_t = _to_tensor(ga_ref, device=device).to(torch.float32)
        # target = (1 - tol) * GA  => اگر base_cost از این کمتر شد یعنی > tol بهتر شده
        target = (1.0 - ga_tol) * ga_ref_t
        improve_too_much = torch.clamp(target - base_cost, min=0.0)
        if improve_too_much > 0:
            ga_trigger = torch.ones((), device=device, dtype=torch.float32)
        ga_penalty = ga_penalty_weight * improve_too_much
    # =========================
    # Anti-collapse: پخش شدن روی node های بیشتر (فقط وقتی GA guard فعال شد)
    # =========================
    spread_penalty = torch.zeros((), device=device, dtype=torch.float32)
    #if (lambda_spread > 0.0) and (ga_trigger > 0):
    if lambda_spread > 0.0:
        if spread_guard == "ga":
            do_spread = (ga_trigger > 0)
        else:
            # "always"
            do_spread = True

    if (lambda_spread > 0.0) and do_spread:
        # per-service ranges
        ptr = getattr(components_batch, "ptr", None)
        B = getattr(components_batch, "batch", None)
        if ptr is not None:
            num_services = int(ptr.numel() - 1)
            starts = ptr[:-1].long()
            ends = ptr[1:].long()
        else:
            if B is None:
                raise ValueError("components_batch must have either ptr or batch.")
            num_services = int(B.max().item()) + 1
            starts = torch.zeros((num_services,), device=device, dtype=torch.long)
            ends = torch.zeros((num_services,), device=device, dtype=torch.long)
            for s in range(num_services):
                idxs = (B == s).nonzero(as_tuple=False).view(-1)
                starts[s] = idxs.min()
                ends[s] = idxs.max() + 1
        # node_rows: [C] (already computed earlier)
        for s in range(num_services):
            s_start = int(starts[s].item())
            s_end = int(ends[s].item())
            rows_s = node_rows[s_start:s_end]
            # unique nodes used by this service
            uniq = torch.unique(rows_s).numel()
            deficit = max(0, min_unique_nodes - int(uniq))
            if deficit > 0:
                # quadratic penalty (smooth-ish)
                spread_penalty = spread_penalty + (deficit * deficit)
        spread_penalty = lambda_spread * spread_penalty
    penalties_total = penalties_base + ga_penalty + spread_penalty
    legacy_total_response_time = response_time + penalties_total
    # RT normalization reference
    if isinstance(rt_ref_cfg, str):
        key = rt_ref_cfg.strip().lower()
        if key in ("ga", "ga_baseline", "gabaseline"):
            rt_ref_t = _to_tensor(ga_ref, device=device).to(torch.float32) if ga_ref is not None else response_time.detach()
        elif key in ("self", "response_time", "rt"):
            rt_ref_t = response_time.detach()
        else:
            try:
                rt_ref_t = torch.tensor(float(rt_ref_cfg), device=device, dtype=torch.float32)
            except Exception:
                rt_ref_t = response_time.detach()
    elif rt_ref_cfg is not None:
        rt_ref_t = _to_tensor(rt_ref_cfg, device=device).to(torch.float32)
    elif normalize_rt and (ga_ref is not None):
        rt_ref_t = _to_tensor(ga_ref, device=device).to(torch.float32)
    else:
        rt_ref_t = torch.ones((), device=device, dtype=torch.float32)
    rt_ref_t = torch.clamp(rt_ref_t, min=eps)
    normalized_response_time = response_time / rt_ref_t if normalize_rt else response_time
    # Weighted-sum objective:
    # minimize RT, maximize PR/SR (negative sign), keep penalties positive
    if objective_mode == "weighted_sum":
        objective = (w_rt * normalized_response_time) - (w_pr * platform_reliability) - (w_sr * service_reliability) + penalties_total
    else:
        objective = legacy_total_response_time
    out: Dict[str, Union[Tensor, float, List[float]]] = {
        "objective": objective,
        "total_response_time": legacy_total_response_time,
        "legacy_total_response_time": legacy_total_response_time,
        "response_time": response_time,
        "platform_reliability": platform_reliability,
        "service_reliability": service_reliability,
        "normalized_response_time": normalized_response_time,
        "exec_time": exec_time,
        "transmission": transmission,
        "provider_delay": 0.0,
        "codec_delay": 0.0,
        "violations": violations,
        "fixed_violations": fixed_viol,
        "bw0_violations": bw0_viol,
        # NEW: expose utilization penalty & optionally some monitors
        "util_penalty": util_penalty,
        "max_util_cpu": util_cpu.max(),
        "max_util_mem": util_mem.max(),
    }
    if return_breakdown:
        out["per_service_exec"] = per_service_exec
        out["per_service_trans"] = per_service_trans
        out["per_service_total"] = per_service_total
        out["per_service_bw0_viol"] = per_service_bw0
    if log:
        print(
            f"[compute_cost_ga_like] exec={exec_time.detach().cpu().item():.6f} "
            f"trans={transmission.detach().cpu().item():.6f} "
            f"pr={platform_reliability.detach().cpu().item():.6f} "
            f"sr={service_reliability.detach().cpu().item():.6f} "
            f"util={util_penalty.detach().cpu().item():.6f} "
            f"bw0={bw0_viol.detach().cpu().item():.0f} "
            f"cap_viol={cap_viol.detach().cpu().item():.6f} "
            f"objective={objective.detach().cpu().item():.6f}"
        )
    return out
