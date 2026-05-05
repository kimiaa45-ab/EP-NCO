# =========================
# Test_GNN.py (CLEAN)
# =========================
import os
import json
import time
import yaml
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ---- Project modules ----
from problem.Dataset import ServiceNodeDataset, service_node_collate_fn
from utils.costfunction1 import compute_cost_ga_like as compute_cost

from nets.encoder.servicegnn import ServiceGNN
from nets.encoder.nodegnn import NodeGNN
from nets.hard_decoder import ARPlacementPipeline


# =========================
# Helpers
# =========================
def to_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_float(v):
    if v is None:
        return 0.0
    if torch.is_tensor(v):
        return float(v.detach().cpu().item())
    try:
        return float(v)
    except Exception:
        return 0.0


def to_device(obj, device: torch.device):
    # keep python lists as-is
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=(device.type == "cuda"))
    if hasattr(obj, "to"):
        try:
            return obj.to(device, non_blocking=(device.type == "cuda"))
        except TypeError:
            return obj.to(device)
    return obj


def manual_bw0_violations(assignments: torch.Tensor, bw_mat: torch.Tensor, comp_predecessors):
    """
    Count number of predecessor edges (p -> t) where:
      assigned(p)=i, assigned(t)=j, i!=j and bw_mat[i,j]==0
    We ignore unassigned (-1).
    """
    if not torch.is_tensor(assignments):
        assignments = torch.tensor(assignments, dtype=torch.long)
    a = assignments.detach().cpu().long().tolist()

    if torch.is_tensor(bw_mat):
        bw = bw_mat.detach().cpu().numpy()
    else:
        bw = np.asarray(bw_mat)

    viol = 0
    C = len(a)
    for t in range(C):
        jt = a[t]
        if jt is None or jt < 0:
            continue
        preds = comp_predecessors[t] if t < len(comp_predecessors) else []
        for p in preds:
            if p < 0 or p >= C:
                continue
            ip = a[p]
            if ip is None or ip < 0:
                continue
            if ip != jt and bw[ip, jt] == 0:
                viol += 1
    return viol


def build_component_records(sample, in_json, assignments):
    """
    Export placements in JSON-like format.
    Attempts to map assignment indices to nodeID via node_graph.x[:,2] if present.
    """
    node_graph = sample.get("node_graph", None)

    # row -> nodeID map (node_graph.x[:,2] is nodeID)
    if (
        node_graph is not None
        and hasattr(node_graph, "x")
        and node_graph.x is not None
        and node_graph.x.dim() == 2
        and node_graph.x.size(1) >= 3
    ):
        row_to_nodeid = node_graph.x[:, 2].long().detach().cpu().tolist()
    else:
        row_to_nodeid = None

    node_ids_raw = to_list(assignments) or []
    if row_to_nodeid is not None:
        Nmap = len(row_to_nodeid)
        node_ids = []
        for a in node_ids_raw:
            a_int = int(a)
            if 0 <= a_int < Nmap:
                node_ids.append(int(row_to_nodeid[a_int]))
            else:
                node_ids.append(a_int)
    else:
        node_ids = [int(a) for a in node_ids_raw]

    n_comp = len(node_ids)

    # fallback: read from input JSON (services->components)
    comps_flat = None
    if isinstance(in_json, dict) and "services" in in_json and isinstance(in_json["services"], list):
        tmp = []
        for s in in_json["services"]:
            sid = s.get("serviceID", s.get("id", None))
            clist = s.get("components", [])
            for c in clist:
                tmp.append(
                    {
                        "serviceID": c.get("serviceID", sid),
                        "componentID": c.get("componentID", c.get("id", None)),
                        "versionID": c.get("versionID", c.get("version", 1)),
                    }
                )
        comps_flat = tmp

    if comps_flat is None:
        return [{"componentIndex": i, "nodeID": int(node_ids[i])} for i in range(n_comp)]

    m = min(len(comps_flat), n_comp)
    recs = []
    for i in range(m):
        recs.append(
            {
                "serviceID": int(comps_flat[i].get("serviceID", -1)),
                "componentID": int(comps_flat[i].get("componentID", i)),
                "versionID": int(comps_flat[i].get("versionID", 1)),
                "nodeID": int(node_ids[i]),
            }
        )
    for i in range(m, n_comp):
        recs.append({"componentIndex": i, "nodeID": int(node_ids[i])})
    return recs


def percent_diff(nco, ga):
    """Percent change vs GA. Negative => better (lower cost), Positive => worse."""
    if ga is None:
        return None
    try:
        ga = float(ga)
        nco = float(nco)
    except Exception:
        return None
    if ga == 0:
        return None
    return 100.0 * (nco - ga) / ga


def safe_get(arr, i):
    """Safe index getter for numpy arrays containing NaN."""
    if i is None:
        return None
    v = arr[i]
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return float(v)


def numeric_sort_key(name: str):
    stem, _ = os.path.splitext(os.path.basename(name))
    try:
        return int(stem)
    except ValueError:
        return stem


# =========================
# Main
# =========================
def main():
    # =========================
    # Load config
    # =========================
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    objective_cfg = config.get("objective", {})
    base_dir = data_cfg.get("base_dir", "data/generated")
    train_type = data_cfg["type"]          # small / medium / large
    test_type = data_cfg["test_type"]      # e.g. test_small

    train_root = os.path.join(base_dir, train_type)
    test_root = os.path.join(base_dir, test_type)

    print("Train root:", train_root)
    print("Test root :", test_root)

    # =========================
    # Model config
    # =========================
    model_cfg = config["model"][train_type]
    n_layers = int(model_cfg.get("n_encode_layers", model_cfg.get("num_layers", 3)))
    num_services = int(model_cfg.get("charnum_service", 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = int(model_cfg["gpu_hidden_dim"] if device.type == "cuda" else model_cfg["cpu_hidden_dim"])

    print("Device:", device)
    print("Hidden dim:", hidden_dim)

    COST_VIOLATION_WEIGHT = 1

    w_rt_cfg = float(objective_cfg.get("response_time_weight", 1.0))
    w_pr_cfg = float(objective_cfg.get("platform_reliability_weight", 0.0))
    w_sr_cfg = float(objective_cfg.get("service_reliability_weight", 0.0))
    COST_OBJECTIVE_KWARGS = {
        "objective_mode": objective_cfg.get("mode", "weighted_sum"),
        "w_rt": w_rt_cfg,
        "w_pr": w_pr_cfg,
        "w_sr": w_sr_cfg,
        # Match nco_rl_hard_dec.py: keep objective on raw RT scale.
        "normalize_rt": False,
        "rt_ref": objective_cfg.get("rt_ref", "ga_baseline"),
    }

    decoder_shaping_cfg = config.get("decoder_shaping", {})
    decoder_shaping = dict(
        alpha_net=float(decoder_shaping_cfg.get("alpha_net", 0.5)),
        lambda_delay=float(decoder_shaping_cfg.get("lambda_delay", 1.0)),
        sim_weight=float(decoder_shaping_cfg.get("sim_weight", 0.1)),
        beta_crowd=float(decoder_shaping_cfg.get("beta_crowd", 0.05)),
        crowd_use_util=bool(decoder_shaping_cfg.get("crowd_use_util", True)),
        crowd_gamma=float(decoder_shaping_cfg.get("crowd_gamma", 1.5)),
    )
    if bool(config.get("separate_eval_decoder_shaping", False)):
        eval_decoder_shaping_cfg = config.get("eval_decoder_shaping", {})
        EVAL_DECODER_SHAPING = dict(
            alpha_net=float(
                eval_decoder_shaping_cfg.get("alpha_net", decoder_shaping["alpha_net"])
            ),
            lambda_delay=float(
                eval_decoder_shaping_cfg.get(
                    "lambda_delay", decoder_shaping["lambda_delay"]
                )
            ),
            sim_weight=float(
                eval_decoder_shaping_cfg.get("sim_weight", decoder_shaping["sim_weight"])
            ),
            beta_crowd=float(
                eval_decoder_shaping_cfg.get("beta_crowd", decoder_shaping["beta_crowd"])
            ),
            crowd_use_util=bool(
                eval_decoder_shaping_cfg.get(
                    "crowd_use_util", decoder_shaping["crowd_use_util"]
                )
            ),
            crowd_gamma=float(
                eval_decoder_shaping_cfg.get(
                    "crowd_gamma", decoder_shaping["crowd_gamma"]
                )
            ),
        )
    else:
        EVAL_DECODER_SHAPING = dict(decoder_shaping)

    anneal_epochs = int(config.get("anneal_epochs", int(model_cfg.get("num_epochs", 1))))

    def get_lambda_u(epoch: int) -> float:
        progress = min(1.0, epoch / float(max(1, anneal_epochs)))
        return float(0.3 + (0.15 - 0.3) * progress)

    eval_lambda_u = float(
        model_cfg.get(
            "eval_lambda_u",
            config.get("eval_lambda_u", get_lambda_u(anneal_epochs)),
        )
    )
    eval_lambda_spread = float(
        model_cfg.get("eval_lambda_spread", config.get("eval_lambda_spread", 0.2))
    )
    allow_fallback_eval = bool(config.get("allow_fallback_eval", True))

    # =========================
    # Dataset + DataLoader
    # =========================
    assert os.path.isdir(test_root), f"test_root not found: {test_root}"

    test_dataset = ServiceNodeDataset(
        root=test_root,
        use_cache=True,
        cache_in_memory=False,
    )

    num_test_samples = len(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=service_node_collate_fn,
    )

    # =========================
    # Build models + load checkpoint
    # =========================
    service_gnn_cfg = {
        int(sid): {
            "in_dim": 3,
            "model_dim": hidden_dim,
            "n_layers": n_layers,
            "use_log1p": True,
            "ds_mode": "inv",
        }
        for sid in range(int(num_services))
    }

    comp_encoder = ServiceGNN(service_config=service_gnn_cfg, device=device).to(device)
    node_encoder = NodeGNN(in_node_feat=5, in_edge_feat=2, dim=hidden_dim, n_layers=n_layers).to(device)
    use_datasize_feature = bool(config.get("use_datasize_feature", False))
    pipeline = ARPlacementPipeline(
        embed_dim=hidden_dim,
        proj_dim=8,
        cap_dim=2,
        net_agg="max",
        use_datasize_feature=use_datasize_feature,
    ).to(device)

    checkpoint_path = "best_rl_model.pt"
    assert os.path.exists(checkpoint_path), f"{checkpoint_path} not found."

    ckpt = torch.load(checkpoint_path, map_location=device)
    comp_encoder.load_state_dict(ckpt["comp_encoder"], strict=True)
    node_encoder.load_state_dict(ckpt["node_encoder"], strict=True)
    pipeline.load_state_dict(ckpt["pipeline"], strict=True)

    comp_encoder.eval()
    node_encoder.eval()
    pipeline.eval()

    print(
        f"[OK] Loaded from {checkpoint_path} | epoch={ckpt.get('epoch', None)} "
        f"| use_datasize_feature={use_datasize_feature}"
    )

    # =========================
    # File order for JSON readback
    # =========================
    test_files = sorted(
        [f for f in os.listdir(test_root) if f.endswith(".json")],
        key=numeric_sort_key,
    )
    assert len(test_files) == num_test_samples, (
        f"Mismatch: dataset len={num_test_samples} but json files={len(test_files)} in {test_root}"
    )

    algo = config.get("eval", {}).get("algorithm", "NCOSPP")
    out_root = config.get("eval", {}).get("out_dir", "data")
    processed_dir = os.path.join(out_root, algo, f"train_{train_type}", f"test_{test_type}")
    os.makedirs(processed_dir, exist_ok=True)

    plot_dir = "img"
    os.makedirs(plot_dir, exist_ok=True)

    # =========================
    # Evaluation loop
    # =========================
    rl_costs = []
    rl_objectives = []
    rl_legacy_totals = []
    ga_costs = []
    sample_times = []

    rl_bw0_costfn = []
    rl_bw0_manual = []
    exec_times = []
    trans_times = []
    viol_list = []
    runtime_ms = []

    total_start = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            start_time = time.time()
            sample = batch[0]

            # move tensors; keep lists
            sample = {
                k: (to_device(v, device) if torch.is_tensor(v) or hasattr(v, "to") else v)
                for k, v in sample.items()
            }

            # bw_mat shape fix (sometimes [1,N,N])
            bw_mat = sample["bw_mat"]
            if isinstance(bw_mat, torch.Tensor) and bw_mat.dim() == 3:
                bw_mat = bw_mat[0]

            comp_predecessors = sample["comp_predecessors"]

            # ---- embeddings ----
            sb_batch = to_device(sample["service_batch"], device)
            comp_emb, _ = comp_encoder(sb_batch)
            node_emb = node_encoder(sample["node_graph"])

            # 2-resource req/caps
            comp_reqs2 = sb_batch.x[:, :2].to(torch.float32)
            node_caps2 = sample["node_graph"].x[:, :2].to(torch.float32)
            # ---- greedy placement ----
            out = pipeline.assign_greedy_or_stochastic(
                comp_emb=comp_emb,
                node_emb=node_emb,
                node_caps=node_caps2,
                comp_reqs=comp_reqs2,
                bw_mat=bw_mat,
                comp_predecessors=comp_predecessors,
                comp_fixed_node_rows=sample.get("comp_fixed_node_rows", None),
                node_graph=sample["node_graph"],
                service_edge_index=sb_batch.edge_index,
                service_edge_attr=getattr(sb_batch, "edge_attr", None),
                greedy=True,
                temperature=1.0,
                **EVAL_DECODER_SHAPING,
                allow_fallback=allow_fallback_eval,
                return_probs=False,
            )
            assignments = out["assignments"]

            # ---- read original JSON for GA reference / export ----
            in_path = os.path.join(test_root, test_files[idx])
            with open(in_path, "r") as f:
                in_data = json.load(f)

            ga_cost = None
            results_arr = in_data.get("results", [])
            if isinstance(results_arr, list) and len(results_arr) > 0 and isinstance(results_arr[0], dict):
                ga_cost = results_arr[0].get("totalResponseTime", None)

            # ---- cost ----
            results = compute_cost(
                assignments,
                sample["node_graph"],
                sample["service_batch"],
                heal=False,
                log=False,
                return_breakdown=True,
                violation_weight=COST_VIOLATION_WEIGHT,
                count_edges_once=True,
                lambda_u=eval_lambda_u,
                lambda_spread=eval_lambda_spread,
                min_unique_nodes=2,
                spread_guard="always",
                ga_ref=ga_cost,
                comp_fixed_node_rows=sample.get("comp_fixed_node_rows", None),
                **COST_OBJECTIVE_KWARGS,
            )

            response_time = results.get("response_time", 0.0)
            objective = results.get("objective", response_time)
            legacy_total = results.get(
                "legacy_total_response_time",
                results.get("total_response_time", response_time),
            )
            exec_time = results.get("exec_time", 0.0)
            trans = results.get("transmission", 0.0)

            viol = results.get("violations", results.get("bw0_violations", 0.0))
            bw0 = results.get("bw0_violations", viol)

            bw0_costfn = to_float(bw0)
            bw0_man = manual_bw0_violations(assignments, bw_mat, comp_predecessors)

            rt_val = to_float(response_time)
            objective_val = to_float(objective)
            legacy_total_val = to_float(legacy_total)

            rl_costs.append(rt_val)
            rl_objectives.append(objective_val)
            rl_legacy_totals.append(legacy_total_val)
            exec_times.append(to_float(exec_time))
            trans_times.append(to_float(trans))
            viol_list.append(to_float(viol))
            rl_bw0_costfn.append(bw0_costfn)
            rl_bw0_manual.append(float(bw0_man))

            ga_costs.append(ga_cost)

            # ---- export placement ----
            placement_records = build_component_records(sample, in_data, assignments)

            sample_time_sec = time.time() - start_time
            sample_times.append(sample_time_sec)
            runtime_ms.append(sample_time_sec * 1000.0)

            sample_pct_vs_ga = percent_diff(rt_val, ga_cost)

            ncospp_out = {
                "algorithm": "NCOSPP",
                "mode": "greedy",
                "totalResponseTime": float(rt_val),
                "objective": float(objective_val),
                "legacyTotalResponseTime": float(legacy_total_val),
                "pct_vs_ga_totalResponseTime": (None if sample_pct_vs_ga is None else float(sample_pct_vs_ga)),
                "algorithmRuntime": float(sample_time_sec * 1000.0),
                "checkpoint": checkpoint_path,
                "epoch": ckpt.get("epoch", None),
                "finalSolution": placement_records,
                "ga_totalResponseTime": (None if ga_cost is None else float(ga_cost)),
                "input_file": test_files[idx],
            }

            out_path = os.path.join(processed_dir, f"test{idx}.json")
            with open(out_path, "w") as f:
                json.dump(ncospp_out, f, indent=2)

            ga_cost_str = "None" if ga_cost is None else f"{float(ga_cost):.4f}"
            print(
                f"[{idx+1}/{num_test_samples}] "
                f"NCOSPP_RT={rt_val:.4f} | Obj={objective_val:.4f} | GA_RT={ga_cost_str} | "
                f"BW0(costfn)={bw0_costfn:.0f} | BW0(man)={bw0_man:d} | "
                f"Time={sample_time_sec:.3f}s | Saved={out_path}"
            )

    total_time = time.time() - total_start

    # =========================
    # Summary
    # =========================
    rl_costs_arr = np.array(rl_costs, dtype=float)
    rl_objectives_arr = np.array(rl_objectives, dtype=float)
    rl_legacy_totals_arr = np.array(rl_legacy_totals, dtype=float)
    times_arr = np.array(sample_times, dtype=float)

    avg_rl_cost = float(rl_costs_arr.mean()) if len(rl_costs_arr) else 0.0
    min_rl_cost = float(rl_costs_arr.min()) if len(rl_costs_arr) else 0.0
    max_rl_cost = float(rl_costs_arr.max()) if len(rl_costs_arr) else 0.0
    avg_rl_objective = float(rl_objectives_arr.mean()) if len(rl_objectives_arr) else 0.0
    min_rl_objective = float(rl_objectives_arr.min()) if len(rl_objectives_arr) else 0.0
    max_rl_objective = float(rl_objectives_arr.max()) if len(rl_objectives_arr) else 0.0

    ga_arr = np.array([np.nan if x is None else float(x) for x in ga_costs], dtype=float)
    ga_has_any = np.isfinite(ga_arr).any()
    ga_numeric = ga_arr[np.isfinite(ga_arr)]

    if ga_has_any:
        avg_ga_cost = float(np.nanmean(ga_arr))
        min_ga_cost = float(np.nanmin(ga_arr))
        max_ga_cost = float(np.nanmax(ga_arr))
    else:
        avg_ga_cost = min_ga_cost = max_ga_cost = 0.0

    # ---- extremes with indices ----
    nco_min_idx = int(np.argmin(rl_costs_arr)) if len(rl_costs_arr) else None
    nco_max_idx = int(np.argmax(rl_costs_arr)) if len(rl_costs_arr) else None
    ga_min_idx = int(np.nanargmin(ga_arr)) if ga_has_any else None
    ga_max_idx = int(np.nanargmax(ga_arr)) if ga_has_any else None

    nco_min = safe_get(rl_costs_arr, nco_min_idx)
    nco_max = safe_get(rl_costs_arr, nco_max_idx)
    ga_min = safe_get(ga_arr, ga_min_idx)
    ga_max = safe_get(ga_arr, ga_max_idx)

    ga_at_nco_min = safe_get(ga_arr, nco_min_idx)
    ga_at_nco_max = safe_get(ga_arr, nco_max_idx)
    nco_at_ga_min = safe_get(rl_costs_arr, ga_min_idx) if ga_min_idx is not None else None
    nco_at_ga_max = safe_get(rl_costs_arr, ga_max_idx) if ga_max_idx is not None else None

    summary = {
        "algorithm": "NCOSPP",
        "train_scale": train_type,
        "test_scale": test_type,
        "n_samples": int(num_test_samples),
        "mean_response_time": avg_rl_cost,
        "min_response_time": min_rl_cost,
        "max_response_time": max_rl_cost,
        "mean_objective": avg_rl_objective,
        "min_objective": min_rl_objective,
        "max_objective": max_rl_objective,
        "mean_legacy_total_response_time": float(rl_legacy_totals_arr.mean()) if len(rl_legacy_totals_arr) else 0.0,
        "mean_exec_time": float(np.mean(exec_times)) if exec_times else 0.0,
        "mean_transmission": float(np.mean(trans_times)) if trans_times else 0.0,
        "mean_violations": float(np.mean(viol_list)) if viol_list else 0.0,
        "mean_runtime_ms": float(np.mean(runtime_ms)) if runtime_ms else 0.0,
        "std_response_time": float(np.std(rl_costs_arr)) if len(rl_costs_arr) else 0.0,
        "mean_bw0_manual": float(np.mean(rl_bw0_manual)) if rl_bw0_manual else 0.0,
        "mean_bw0_costfn": float(np.mean(rl_bw0_costfn)) if rl_bw0_costfn else 0.0,
        "ga_avg": avg_ga_cost,
        "ga_min": min_ga_cost,
        "ga_max": max_ga_cost,
        "total_test_time_s": float(total_time),
        "nco_min_idx": nco_min_idx,
        "nco_max_idx": nco_max_idx,
        "ga_min_idx": ga_min_idx,
        "ga_max_idx": ga_max_idx,
    }

    with open(os.path.join(processed_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Test Summary =====")
    print(f"NCOSPP RT   -> avg={avg_rl_cost:.4f} | min={min_rl_cost:.4f} | max={max_rl_cost:.4f}")
    print(f"NCOSPP Obj  -> avg={avg_rl_objective:.4f} | min={min_rl_objective:.4f} | max={max_rl_objective:.4f}")
    print(f"GA RT       -> avg={avg_ga_cost:.4f} | min={min_ga_cost:.4f} | max={max_ga_cost:.4f}")

    bw0_costfn_arr = np.array(rl_bw0_costfn, dtype=float)
    bw0_manual_arr = np.array(rl_bw0_manual, dtype=float)

    if len(bw0_costfn_arr):
        print(
            f"BW0(costfn)-> avg={float(bw0_costfn_arr.mean()):.2f} | "
            f"min={float(bw0_costfn_arr.min()):.0f} | max={float(bw0_costfn_arr.max()):.0f}"
        )
    else:
        print("BW0(costfn)-> n/a")

    if len(bw0_manual_arr):
        print(
            f"BW0(manual)-> avg={float(bw0_manual_arr.mean()):.2f} | "
            f"min={float(bw0_manual_arr.min()):.0f} | max={float(bw0_manual_arr.max()):.0f}"
        )
    else:
        print("BW0(manual)-> n/a")

    if len(times_arr):
        print(
            f"Time/sample -> avg={float(times_arr.mean()):.3f}s | "
            f"min={float(times_arr.min()):.3f}s | max={float(times_arr.max()):.3f}s"
        )
    else:
        print("Time/sample -> n/a")

    print(f"Total Test Time: {total_time:.3f}s")
    print(f"Processed outputs: {processed_dir}")

    print("\n===== Extremes (with cross-values) =====")
    if nco_min_idx is not None:
        print(
            f"NCOSPP MIN: {nco_min:.4f} at sample {nco_min_idx} | "
            f"GA at same sample: {('None' if ga_at_nco_min is None else f'{ga_at_nco_min:.4f}')}"
        )
    if nco_max_idx is not None:
        print(
            f"NCOSPP MAX: {nco_max:.4f} at sample {nco_max_idx} | "
            f"GA at same sample: {('None' if ga_at_nco_max is None else f'{ga_at_nco_max:.4f}')}"
        )
    if ga_min_idx is not None:
        print(
            f"GA MIN    : {ga_min:.4f} at sample {ga_min_idx} | "
            f"NCOSPP at same sample: {('None' if nco_at_ga_min is None else f'{nco_at_ga_min:.4f}')}"
        )
    if ga_max_idx is not None:
        print(
            f"GA MAX    : {ga_max:.4f} at sample {ga_max_idx} | "
            f"NCOSPP at same sample: {('None' if nco_at_ga_max is None else f'{nco_at_ga_max:.4f}')}"
        )

    # =========================
    # Plot: NCOSPP vs GA + min/max lines
    # =========================
    plt.figure(figsize=(9, 5))
    plt.plot(rl_costs_arr, "o-", color="C0", label=f"NCOSPP RT(avg={avg_rl_cost:.2f})")
    plt.plot(ga_arr, "s--", color="C3", label=f"GA RT(avg={avg_ga_cost:.2f})" if ga_has_any else "GA RT(available)")

    # Horizontal min/max guide lines (min = lighter, max = stronger)
    if nco_min is not None:
        plt.axhline(nco_min, color="C0", linestyle="--", linewidth=1.5, alpha=0.35)
        plt.text(0, nco_min, " min NCOSPP RT", va="bottom", ha="left")
    if nco_max is not None:
        plt.axhline(nco_max, color="C0", linestyle="--", linewidth=2.5, alpha=0.9)
        plt.text(0, nco_max, " max NCOSPP RT", va="bottom", ha="left")

    if ga_min is not None:
        plt.axhline(ga_min, color="C3", linestyle="--", linewidth=1.5, alpha=0.35)
        plt.text(0, ga_min, " min GA RT", va="bottom", ha="left")
    if ga_max is not None:
        plt.axhline(ga_max, color="C3", linestyle="--", linewidth=2.5, alpha=0.9)
        plt.text(0, ga_max, " max GA RT", va="bottom", ha="left")

    plt.xlabel("Test Sample Index")
    plt.ylabel("Response Time")
    plt.title("NCOSPP RT vs GA RT per Test Sample (with min/max lines)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "test_ncospp_vs_ga.png"), dpi=200)
    plt.close()

    # =========================
    # Plot: BW0 violations
    # =========================
    if len(rl_bw0_manual) > 0:
        plt.figure(figsize=(9, 5))
        plt.plot(rl_bw0_manual, "o-", label="BW0(manual)")
        plt.plot(rl_bw0_costfn, "s--", label="BW0(costfn)")
        plt.xlabel("Test Sample Index")
        plt.ylabel("BW0 Violations")
        plt.title("BW0 violations per Test Sample")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "test_bw0_violations.png"), dpi=200)
        plt.close()

    # Relative performance vs GA (using summary values)
    avg_diff = percent_diff(avg_rl_cost, avg_ga_cost)
    min_diff = percent_diff(min_rl_cost, min_ga_cost)
    max_diff = percent_diff(max_rl_cost, max_ga_cost)

    print("\n📈 Relative Performance vs GA:")
    print(f"Avg : {('None' if avg_diff is None else f'{avg_diff:+.2f}%')}")
    print(f"Min : {('None' if min_diff is None else f'{min_diff:+.2f}%')}")
    print(f"Max : {('None' if max_diff is None else f'{max_diff:+.2f}%')}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
