# state.py
import torch
from typing import Optional, List


class WarningTracker:
    def __init__(self, warn_once: bool = True):
        self.warn_once = warn_once
        self._counts = {}

    def warned_before(self, key: str) -> bool:
        return self._counts.get(key, 0) > 0

    def record(self, key: str):
        self._counts[key] = self._counts.get(key, 0) + 1

    def maybe_warn(self, key: str, msg: str):
        if (not self.warn_once) or (not self.warned_before(key)):
            print("Warning:", msg)
        self.record(key)

    def reset(self):
        self._counts.clear()


class NodeState:
    """
    Holds current node capacities and utilities to check/apply assignments.
    All tensors are kept on self.device.

    Important GPU note:
      - apply_assignments_batch is fully tensorized (no .item(), no Python loop).
      - apply_assignment remains scalar-oriented (kept for compatibility/debug).
    """

    def __init__(
        self,
        node_caps: torch.Tensor,
        device: Optional[torch.device] = None,
        warn_once: bool = True,
    ):
        if device is None:
            device = (
                node_caps.device
                if isinstance(node_caps, torch.Tensor)
                else torch.device("cpu")
            )
        self.device = device

        # caps = node_caps.detach().clone().to(self.device)
        # if caps.dim() > 1:
        #     caps = caps.view(-1)

        # self.initial_caps = caps.clone()
        # self.final_caps = caps.clone()
        # self.N = int(self.final_caps.shape[0])
        caps = node_caps.detach().clone().to(self.device)

        # Expect [N] (legacy) or [N,2] (cpu, mem)
        if caps.dim() == 1:
            # legacy single-resource mode -> treat as [N,1]
            caps = caps.view(-1, 1)
        elif caps.dim() == 2:
            # keep as [N, R]
            pass
        else:
            raise ValueError(
                f"[NodeState] node_caps must be 1D or 2D, got shape={tuple(caps.shape)}"
            )

        self.initial_caps = caps.clone()
        self.final_caps = caps.clone()
        self.N = int(self.final_caps.shape[0])
        self.R = int(self.final_caps.shape[1])

        self.warning_tracker = WarningTracker(warn_once=warn_once)

    def reset(self, node_caps: Optional[torch.Tensor] = None):
        if node_caps is None:
            self.final_caps = self.initial_caps.clone()
        else:
            caps = node_caps.detach().clone().to(self.device)

            if caps.dim() == 1:
                caps = caps.view(-1, 1)
            elif caps.dim() == 2:
                pass
            else:
                raise ValueError(
                    f"[NodeState] node_caps must be 1D or 2D, got shape={tuple(caps.shape)}"
                )

            self.initial_caps = caps.clone()
            self.final_caps = caps.clone()
            self.N = int(self.final_caps.shape[0])
            self.R = int(self.final_caps.shape[1])

        self.warning_tracker.reset()

    # def feasible_mask(self, comp_reqs: torch.Tensor) -> torch.BoolTensor:
    #     cr = comp_reqs.view(-1).to(self.device)  # [C]
    #     caps = self.final_caps.view(1, -1)  # [1, N]
    #     reqs = cr.view(-1, 1)  # [C, 1]
    #     return caps >= reqs
    def feasible_mask(self, comp_reqs: torch.Tensor) -> torch.BoolTensor:
        # comp_reqs: [C, R] or [C] (legacy)
        cr = comp_reqs.to(self.device)

        if cr.dim() == 1:
            cr = cr.view(-1, 1)  # [C,1]
        elif cr.dim() == 2:
            pass
        else:
            raise ValueError(
                f"[NodeState] comp_reqs must be 1D or 2D, got shape={tuple(cr.shape)}"
            )

        # final_caps: [N,R] -> [1,N,R]
        caps = self.final_caps.view(1, self.N, self.R)
        # reqs: [C,1,R]
        reqs = cr.view(-1, 1, self.R)

        # [C,N,R] then reduce over R -> [C,N]
        return (caps >= reqs).all(dim=2)

    def apply_assignment(
        self,
        comp_idx: int,
        node_idx: int,
        comp_req: torch.Tensor,
        warn_on_fail: bool = True,
    ) -> bool:
        """
        Scalar/debug path (kept). Not used in GPU-optimized training.
        """
        if node_idx is None or int(node_idx) < 0:
            if warn_on_fail:
                key = f"no_node_for_comp_{comp_idx}"
                self.warning_tracker.maybe_warn(
                    key, f"No valid computing node for component {comp_idx}"
                )
            return False

        node_idx = int(node_idx)

        req = comp_req.to(self.device)
        if req.dim() == 0:
            req = req.view(1)
        elif req.dim() == 1:
            pass
        else:
            req = req.view(-1)

        # req should be [R]
        if req.numel() != self.R:
            # legacy single-resource
            if self.R == 1:
                req = req.view(1)
            else:
                raise ValueError(
                    f"[NodeState] comp_req has {req.numel()} elems but NodeState expects R={self.R}"
                )

        if torch.any(self.final_caps[node_idx] < req):
            if warn_on_fail:
                key = f"insufficient_cap_comp_{comp_idx}_node_{node_idx}"
                self.warning_tracker.maybe_warn(
                    key,
                    f"Insufficient capacity for comp {comp_idx} on node {node_idx}: "
                    f"need={req.detach().cpu().tolist()} have={self.final_caps[node_idx].detach().cpu().tolist()}",
                )
            return False

        self.final_caps[node_idx] = self.final_caps[node_idx] - req
        return True
        # req = comp_req.view(-1)[0].to(self.device)

        # if node_idx < 0 or node_idx >= self.N:
        #     if warn_on_fail:
        #         key = f"invalid_node_index_{node_idx}"
        #         self.warning_tracker.maybe_warn(
        #             key, f"Invalid node index {node_idx} for component {comp_idx}"
        #         )
        #     return False

        # if self.final_caps[node_idx] < req:
        #     if warn_on_fail:
        #         key = f"insufficient_cap_comp_{comp_idx}_node_{node_idx}"
        #         self.warning_tracker.maybe_warn(
        #             key,
        #             f"Insufficient capacity for component {comp_idx} on node {node_idx}",
        #         )
        #     return False

        # self.final_caps[node_idx] = self.final_caps[node_idx] - req
        # return True

    def apply_assignments_batch(
        self,
        assignments: torch.Tensor,
        comp_reqs: torch.Tensor,
        warn_on_fail: bool = True,
        return_list: bool = True,
    ):
        """
        GPU-friendly batch apply:
          - assignments: [C] (node row index per component, -1 if unassigned)
          - comp_reqs:   [C] (resource requirements)

        Behavior:
          - Only assignments >=0 and <N are considered candidates.
          - For candidates, we compute total demand per node and check if total demand <= capacity.
          - If a node is over-subscribed, we mark ALL components assigned to that node as failed
            (consistent with "batch apply" semantics; if you need sequential feasibility, enforce
             feasibility in the policy step).
          - For feasible nodes, we subtract aggregated demand from capacities using scatter_add.

        Returns:
          - list[bool] by default (for backward compatibility),
            or a boolean tensor if return_list=False.
        """
        device = self.device
        a = assignments.to(device=device, dtype=torch.long).view(-1)  # [C]
        r = comp_reqs.to(device=device, dtype=torch.float32)
        if r.dim() == 1:
            r = r.view(-1, 1)  # [C,1]
        elif r.dim() == 2:
            pass  # [C,R]
        else:
            raise ValueError(
                f"[NodeState] comp_reqs must be 1D or 2D, got {tuple(r.shape)}"
            )

        C = int(a.numel())
        R = int(r.size(1))

        valid = (a >= 0) & (a < self.N)
        ok = torch.zeros((C,), device=device, dtype=torch.bool)

        if not torch.any(valid):
            if warn_on_fail and C > 0:
                self.warning_tracker.maybe_warn(
                    "all_unassigned", "All components are unassigned (-1)."
                )
            return ok.tolist() if return_list else ok

        a_v = a[valid]  # [Cv]
        r_v = r[valid, :]  # [Cv,R]

        # demand: [N,R]
        demand = torch.zeros((self.N, R), device=device, dtype=torch.float32)
        for k in range(R):
            demand[:, k].scatter_add_(0, a_v, r_v[:, k])

        cap = self.final_caps.to(device=device, dtype=torch.float32)  # [N,R]
        node_feasible = (demand <= cap).all(dim=1)  # [N]

        ok_valid = node_feasible[a_v]  # [Cv]
        ok[valid] = ok_valid

        # subtract only feasible nodes
        feasible_demand = torch.where(
            node_feasible.view(-1, 1), demand, torch.zeros_like(demand)
        )
        self.final_caps = (cap - feasible_demand).to(self.final_caps.dtype)

        # Optional warnings (summarized, not per component)
        if warn_on_fail:
            num_unassigned = int((a < 0).sum().detach().cpu().item())
            num_invalid = int(((a >= self.N) & (a >= 0)).sum().detach().cpu().item())
            # num_over = int((~node_feasible & (demand > 0)).sum().detach().cpu().item())
            num_over = int(
                (~node_feasible & (demand.sum(dim=1) > 0)).sum().detach().cpu().item()
            )

            if num_unassigned > 0:
                self.warning_tracker.maybe_warn(
                    "some_unassigned",
                    f"{num_unassigned} components are unassigned (-1).",
                )
            if num_invalid > 0:
                self.warning_tracker.maybe_warn(
                    "some_invalid_nodes",
                    f"{num_invalid} components assigned to invalid node indices (>= N).",
                )
            if num_over > 0:
                self.warning_tracker.maybe_warn(
                    "some_oversubscribed_nodes",
                    f"{num_over} nodes are oversubscribed (total assigned req > capacity).",
                )

        return ok.tolist() if return_list else ok

    def get_final_caps(self) -> torch.Tensor:
        return self.final_caps.clone()

    def get_initial_caps(self) -> torch.Tensor:
        return self.initial_caps.clone()
