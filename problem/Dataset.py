# Dataset.py

import os

import glob

import json

from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple



import torch

import yaml

from torch_geometric.data import Data, Dataset, Batch





def _load_config_default_root() -> str:

    with open("configs/config.yaml", "r") as f:

        config = yaml.safe_load(f)

    data_type = config["data"]["type"]      # small / medium / large

    base_dir = config["data"]["base_dir"]   # data/generated

    return os.path.join(base_dir, data_type)





def _numeric_sort_key(path: str) -> Any:

    name = os.path.basename(path)

    stem, _ = os.path.splitext(name)

    try:

        return int(stem)

    except ValueError:

        return stem





def service_node_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    # return list as-is (batch_size=1 in your training)

    return batch





@dataclass

class CacheOptions:

    use_cache: bool = True

    cache_dir: Optional[str] = None

    cache_in_memory: bool = False





class ServiceNodeDataset(Dataset):

    """

    Each index corresponds to one JSON file.



    Returns a dict with:

      - node_graph: PyG Data (node-level graph)

      - bw_mat: [N,N] float32 bandwidth matrix (0 => no link)

      - service_graphs: List[PyG Data] (each service is a component-graph)

      - service_batch: PyG Batch (batch of all service graphs)

      - comp_reqs: [C] float32 CPU req per component (global order)

      - comp_predecessors: List[List[int]] length C (global predecessor indices)

      - comp_service_map: List[int] length C (service id per component)

      - node_caps_cpu: [N] float32 CPU capacity per node (from characteristics.cpu)

      - sample_idx: int (stable dataset index)

    """



    def __init__(

        self,

        root: Optional[str] = None,

        transform=None,

        pre_transform=None,

        use_cache: bool = True,

        cache_dir: Optional[str] = None,

        cache_in_memory: bool = False,

    ):

        if root is None:

            root = _load_config_default_root()

        super().__init__(root, transform, pre_transform)



        self.files = sorted(glob.glob(os.path.join(root, "*.json")), key=_numeric_sort_key)



        self.cache_opts = CacheOptions(

            use_cache=use_cache,

            cache_dir=cache_dir,

            cache_in_memory=cache_in_memory,

        )



        if self.cache_opts.cache_dir is None:

            self.cache_opts.cache_dir = os.path.join(root, "_cache_pt")



        if self.cache_opts.use_cache:

            os.makedirs(self.cache_opts.cache_dir, exist_ok=True)



        self._mem_cache: Dict[int, Dict[str, Any]] = {}



    def len(self) -> int:

        return len(self.files)



    def _cache_path(self, json_path: str) -> str:

        stem = os.path.splitext(os.path.basename(json_path))[0]

        return os.path.join(self.cache_opts.cache_dir, f"{stem}.pt")



    def get(self, idx: int) -> Dict[str, Any]:

        # 1) memory cache

        if self.cache_opts.cache_in_memory and idx in self._mem_cache:

            sample = self._mem_cache[idx]

            sample["sample_idx"] = int(idx)

            return sample



        file_path = self.files[idx]

        cache_path = self._cache_path(file_path)



        # 2) disk cache

        if self.cache_opts.use_cache and os.path.exists(cache_path):

            sample = torch.load(cache_path, map_location="cpu")
            # Refresh stale cache when feature widths changed.
            if isinstance(sample, dict):
                ng = sample.get("node_graph", None)
                sb = sample.get("service_batch", None)
                node_stale = bool(ng is not None and hasattr(ng, "x") and ng.x is not None and ng.x.dim() == 2 and ng.x.size(1) < 5)
                comp_stale = bool(sb is not None and hasattr(sb, "x") and sb.x is not None and sb.x.dim() == 2 and sb.x.size(1) < 3)
                if node_stale or comp_stale:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    sample = self._build_sample(data)
                    torch.save(sample, cache_path)

            sample["sample_idx"] = int(idx)

            if self.cache_opts.cache_in_memory:

                self._mem_cache[idx] = sample

            return sample



        # 3) build from json

        with open(file_path, "r") as f:

            data = json.load(f)



        sample = self._build_sample(data)

        sample["sample_idx"] = int(idx)



        if self.cache_opts.use_cache:

            torch.save(sample, cache_path)



        if self.cache_opts.cache_in_memory:

            self._mem_cache[idx] = sample



        return sample



    # -----------------------------

    # Helpers for building graphs

    # -----------------------------

    @staticmethod

    def _build_bw_mat_and_node_graph(data: Dict[str, Any]) -> Tuple[Data, torch.Tensor, torch.Tensor]:

        """

        Build:

          - node_graph: Data(x, edge_index, edge_attr)   edge_attr=[bw, delay]

          - bw_mat: [N,N] bandwidth matrix

          - node_caps_cpu: [N] cpu capacity extracted from characteristics.cpu

        """

        #all_nodes = data["computingNodes"] + data["helperNodes"] + data["usersNodes"]

        all_nodes = data["computingNodes"] + data["helperNodes"] + data["usersNodes"]

        all_nodes = sorted(all_nodes, key=lambda n: int(n["nodeID"]))



        net = data.get("networkConnections", [])

        n = len(net)



        if len(all_nodes) != n:

            raise ValueError(f"[Dataset] all_nodes={len(all_nodes)} != networkConnections={n}")



        node_ids = [int(node["nodeID"]) for node in all_nodes]

        if min(node_ids) != 1 or max(node_ids) != n or len(set(node_ids)) != n:

            raise ValueError(

                f"[Dataset] nodeID sanity failed: min={min(node_ids)}, max={max(node_ids)}, uniq={len(set(node_ids))}, N={n}"

            )



        # Node features: [cpu, memory, nodeID, tier, reliability]

        nodes_feat: List[List[float]] = []

        node_caps_cpu_list: List[float] = []

        for node in all_nodes:

            cpu = float(node["characteristics"]["cpu"])

            mem = float(node["characteristics"]["memory"])

            nid = float(node["nodeID"])

            tier = float(node["nodeTier"])
            rel = ServiceNodeDataset._safe_get_node_reliability(node)

            nodes_feat.append([cpu, mem, nid, tier, rel])

            node_caps_cpu_list.append(cpu)



        x_nodes = torch.tensor(nodes_feat, dtype=torch.float32)

        node_caps_cpu = torch.tensor(node_caps_cpu_list, dtype=torch.float32)



        edge_index_nodes: List[List[int]] = []

        edge_attr_nodes: List[List[float]] = []



        bw_mat = torch.zeros((n, n), dtype=torch.float32)



        for i in range(n):

            row = net[i]

            for j in range(n):

                bw, delay = row[j]

                bw = float(bw)

                delay = float(delay)

                bw_mat[i, j] = bw

                if i != j and bw > 0:

                    edge_index_nodes.append([i, j])

                    edge_attr_nodes.append([bw, delay])



        if len(edge_index_nodes) == 0:

            edge_index_nodes_t = torch.empty((2, 0), dtype=torch.long)

            edge_attr_nodes_t = torch.empty((0, 2), dtype=torch.float32)

        else:

            edge_index_nodes_t = torch.tensor(edge_index_nodes, dtype=torch.long).t().contiguous()

            edge_attr_nodes_t = torch.tensor(edge_attr_nodes, dtype=torch.float32)



        node_graph = Data(x=x_nodes, edge_index=edge_index_nodes_t, edge_attr=edge_attr_nodes_t)

        return node_graph, bw_mat, node_caps_cpu



    @staticmethod

    def _safe_get_datasize(comp: Dict[str, Any]) -> float:

        """

        Robustly read component characteristics dataSize.

        """

        ch = comp.get("characteristics", {})

        # common fallbacks in case generator used different key

        for k in ("dataSize", "datasize", "data_size", "data"):

            if k in ch:

                try:

                    return float(ch[k])

                except Exception:

                    pass

        return 0.0

    @staticmethod
    def _safe_get_node_reliability(node: Dict[str, Any]) -> float:
        ch = node.get("characteristics", {})
        if "reliabilityScore" in ch:
            return float(ch["reliabilityScore"])
        if "reliability" in ch:
            return float(ch["reliability"])
        return 1.0

    @staticmethod
    def _safe_get_component_reliability(comp: Dict[str, Any]) -> float:
        ch = comp.get("characteristics", {})
        if "reliabilityScore" in ch:
            return float(ch["reliabilityScore"])
        if "reliability" in ch:
            return float(ch["reliability"])
        return 1.0



    @staticmethod

    def _build_service_graphs_and_predecessors(data: Dict[str, Any]) -> Tuple[

        List[Data],

        Batch,

        torch.Tensor,

        List[List[int]],

        List[int],
        torch.Tensor,

    ]:

        """

        Builds:

          - service_graphs (list of Data)

          - service_batch (Batch)

          - comp_reqs [C] (CPU only; you later use x[:, :2] for CPU/MEM)

          - comp_predecessors: List[List[int]] length C (global indices)

          - comp_service_map: List[int] length C
          - comp_fixed_node_rows: [C] int64, fixed node row per component (-1 means free)

        """

        services = data["services"]

        conn = data.get("componentConnections", [])  # can be 0/1 OR can be dataSize weights



        service_graphs: List[Data] = []

        comp_reqs_list: List[float] = []

        comp_service_map: List[int] = []

        comp_predecessors: List[List[int]] = []
        comp_fixed_node_rows: List[int] = []



        global_offset = 0



        for sid, service in enumerate(services):

            comps = service["components"]

            m = len(comps)
            # GA convention in this dataset:
            # component 1 of each service is pinned to userID,
            # last component is pinned to helperID.
            user_row = int(service.get("userID", -1)) - 1
            helper_row = int(service.get("helperID", -1)) - 1

            conn_is_local = (

                isinstance(conn, list)

                and len(conn) == m

                and all(isinstance(r, list) and len(r) == m for r in conn)

            )



            # --- component node features ---

            x_list: List[List[float]] = []

            data_sizes: List[float] = []

            comp_cpu_local: List[float] = []



            for comp in comps:

                cpu = float(comp["characteristics"]["cpu"])

                mem = float(comp["characteristics"]["memory"])

                ds = ServiceNodeDataset._safe_get_datasize(comp)

                rel = ServiceNodeDataset._safe_get_component_reliability(comp)
                x_list.append([cpu, mem, rel])

                #x_list.append([cpu, mem, ds])   # ds همان dataSize کامپوننت



                data_sizes.append(float(ds))

                comp_cpu_local.append(float(cpu))



            x = torch.tensor(x_list, dtype=torch.float32)



            # --- edges (local indices) ---

            edge_index: List[List[int]] = []

            edge_attr: List[List[float]] = []

            local_preds: List[List[int]] = [[] for _ in range(m)]



            # --- edges (local indices) ---

            edge_index: List[List[int]] = []

            edge_attr: List[List[float]] = []

            local_preds: List[List[int]] = [[] for _ in range(m)]



            if isinstance(conn, list):

                if conn_is_local:

                    # ✅ LOCAL (m×m): same matrix reused for every service

                    for i in range(m):

                        row = conn[i]

                        if not isinstance(row, list):

                            continue



                        for j in range(m):

                            if i == j:

                                continue



                            val = row[j]



                            is_edge = False

                            ds_edge = 0.0



                            if isinstance(val, (int, float)):

                                v = float(val)

                                if v > 0:

                                    is_edge = True

                                    ds_edge = v if abs(v - 1.0) > 1e-9 else float(data_sizes[i])

                            else:

                                if str(val).strip() in ("1", "true", "True"):

                                    is_edge = True

                                    ds_edge = float(data_sizes[i])



                            if is_edge:

                                edge_index.append([i, j])

                                edge_attr.append([ds_edge])  # edge_attr[:,0] = dataSize

                                local_preds[j].append(i)



                else:

                    # ✅ GLOBAL (C_total×C_total): use global_offset slicing

                    for i in range(m):

                        gi = global_offset + i

                        if gi >= len(conn):

                            continue



                        row = conn[gi]

                        if not isinstance(row, list):

                            continue



                        for j in range(m):

                            if i == j:

                                continue



                            gj = global_offset + j

                            if gj >= len(row):

                                continue



                            val = row[gj]



                            is_edge = False

                            ds_edge = 0.0



                            if isinstance(val, (int, float)):

                                v = float(val)

                                if v > 0:

                                    is_edge = True

                                    ds_edge = v if abs(v - 1.0) > 1e-9 else float(data_sizes[i])

                            else:

                                if str(val).strip() in ("1", "true", "True"):

                                    is_edge = True

                                    ds_edge = float(data_sizes[i])



                            if is_edge:

                                edge_index.append([i, j])

                                edge_attr.append([ds_edge])

                                local_preds[j].append(i)



            if len(edge_index) == 0:

                edge_index_t = torch.empty((2, 0), dtype=torch.long)

                edge_attr_t = torch.empty((0, 1), dtype=torch.float32)

            else:

                edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)



            d = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)

            d.comp_datasize = torch.tensor(data_sizes, dtype=torch.float32)  # shape [m]

            # و اگر طبق پیام قبلی componentConnections رو هم اضافه کرده‌ای:

            # d.component_connections = cc_local  # shape [m,m]

       

            # ---- attach componentConnections for GA-style transmission (m x m) ----

            if isinstance(conn, list) and len(conn) > 0:

                if conn_is_local:

                    # local matrix already m x m

                    cc_local = torch.tensor(conn, dtype=torch.float32)

                else:

                    # global matrix: slice out this service block [global_offset:global_offset+m]

                    cc_block = []

                    for ii in range(m):

                        gi = global_offset + ii

                        row = conn[gi]

                        cc_block.append(row[global_offset : global_offset + m])

                    cc_local = torch.tensor(cc_block, dtype=torch.float32)



                d.component_connections = cc_local  # shape [m, m]



            d.service_id = torch.tensor([sid], dtype=torch.long)

            service_graphs.append(d)



            # --- global component info ---

            for local_idx in range(m):

                comp_reqs_list.append(comp_cpu_local[local_idx])

                comp_service_map.append(sid)
                if local_idx == 0 and user_row >= 0:
                    comp_fixed_node_rows.append(user_row)
                elif local_idx == (m - 1) and helper_row >= 0:
                    comp_fixed_node_rows.append(helper_row)
                else:
                    comp_fixed_node_rows.append(-1)

                preds_global = [global_offset + p for p in local_preds[local_idx]]

                comp_predecessors.append(preds_global)



            global_offset += m





        service_batch = Batch.from_data_list(service_graphs)

        comp_reqs = torch.tensor(comp_reqs_list, dtype=torch.float32)

        comp_fixed_node_rows_t = torch.tensor(comp_fixed_node_rows, dtype=torch.long)
        return (
            service_graphs,
            service_batch,
            comp_reqs,
            comp_predecessors,
            comp_service_map,
            comp_fixed_node_rows_t,
        )



    @staticmethod

    def _build_sample(data: Dict[str, Any]) -> Dict[str, Any]:

        node_graph, bw_mat, node_caps_cpu = ServiceNodeDataset._build_bw_mat_and_node_graph(data)



        (
            service_graphs,
            service_batch,
            comp_reqs,
            comp_predecessors,
            comp_service_map,
            comp_fixed_node_rows,
        ) = (

            ServiceNodeDataset._build_service_graphs_and_predecessors(data)

        )



        return {

            "node_graph": node_graph,

            "bw_mat": bw_mat,

            "node_caps_cpu": node_caps_cpu,

            "service_graphs": service_graphs,

            "service_batch": service_batch,

            "comp_reqs": comp_reqs,

            "comp_predecessors": comp_predecessors,

            "comp_service_map": comp_service_map,
            "comp_fixed_node_rows": comp_fixed_node_rows,

        }

