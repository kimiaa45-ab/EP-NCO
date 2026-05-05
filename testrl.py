

def main():

    import os

    import json

    import yaml

    import time

    import numpy as np

    import torch

    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader



    # ---- Project modules ----


    from nets.ardecoder import ARPlacementPipeline

    from problem.Dataset import ServiceNodeDataset, service_node_collate_fn

    from utils.costfunction1 import compute_cost_ga_like as compute_cost

    from src.state import NodeState



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



    # independent bw=0 violation check (based only on assignments + bw_mat + predecessors)

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



    # --- Placement export (kept; minor safety) ---

    def build_component_records(sample, in_json, assignments):

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

        if (

            isinstance(in_json, dict)

            and "services" in in_json

            and isinstance(in_json["services"], list)

        ):

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



    # =========================

    # Load config

    # =========================

    with open("configs/config.yaml", "r") as f:

        config = yaml.safe_load(f)



    data_cfg = config.get("data", {})

    base_dir = data_cfg.get("base_dir", "data/generated")



    train_type = data_cfg.get("type")  # small / medium / large

    test_type = data_cfg.get("test_type", f"test_{train_type}")



    train_root = os.path.join(base_dir, train_type)

    test_root = os.path.join(base_dir, test_type)



    print("Train root:", train_root)

    print("Test root :", test_root)



    # =========================

    # Model config

    # =========================

    model_cfg = config["model"][train_type]

    num_services = int(model_cfg["charnum_service"])

    n_layers = int(model_cfg.get("n_encode_layers", model_cfg.get("num_layers", model_cfg["num_layers"])))



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = int(model_cfg["gpu_hidden_dim"] if device.type == "cuda" else model_cfg["cpu_hidden_dim"])

    use_amp = device.type == "cuda"



    print("Device:", device)

    print("Hidden dim:", hidden_dim)



    COST_VIOLATION_WEIGHT = 1e6



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

    # Models

    # =========================

    # -------- Simple encoders (must match training) --------
    comp_encoder = torch.nn.Sequential(
        torch.nn.Linear(2, hidden_dim),
        torch.nn.ReLU(),
    ).to(device)

    node_encoder = torch.nn.Sequential(
        torch.nn.Linear(4, hidden_dim),
        torch.nn.ReLU(),
    ).to(device)

    pipeline = ARPlacementPipeline(embed_dim=hidden_dim, proj_dim=8, cap_dim=2).to(device)



    # =========================

    # Load checkpoint

    # =========================

    checkpoint_path = "best_rl_model.pt"

    assert os.path.exists(checkpoint_path), "best_rl_model.pt not found (train first)."



    ckpt = torch.load(checkpoint_path, map_location=device)

       # ckpt = torch.load(checkpoint_path, map_location=device)



    # common accidental wrap: ckpt = {torch.load(...)}  -> set

    if isinstance(ckpt, set):

        # if it contains exactly one element, unwrap it

        if len(ckpt) == 1:

            ckpt = next(iter(ckpt))

        else:

            raise TypeError(f"Checkpoint is a set of size {len(ckpt)}; expected dict.")



    if not isinstance(ckpt, dict):

        raise TypeError(f"Checkpoint type is {type(ckpt)}; expected dict with state_dicts.")



    for k in ["comp_encoder", "node_encoder", "pipeline"]:
        if k not in ckpt:
            raise KeyError(f"Checkpoint missing key '{k}'. موجودها: {list(ckpt.keys())}")

    comp_encoder.load_state_dict(ckpt["comp_encoder"], strict=True)
    node_encoder.load_state_dict(ckpt["node_encoder"], strict=True)
    pipeline.load_state_dict(ckpt["pipeline"], strict=True)

    comp_encoder.eval()
    node_encoder.eval()
    pipeline.eval()





    # =========================

    # File order for JSON readback

    # =========================
    def numeric_key(name):
        stem = os.path.splitext(name)[0]
        try:
            return int(stem)
        except:
            return stem

    test_files = sorted([f for f in os.listdir(test_root) if f.endswith(".json")], key=numeric_key)

   
    assert len(test_files) == num_test_samples, (

        f"Mismatch: dataset len={num_test_samples} but json files={len(test_files)} in {test_root}"

    )



    processed_dir = os.path.join("data", "processed", test_type)

    os.makedirs(processed_dir, exist_ok=True)



    plot_dir = "img"

    os.makedirs(plot_dir, exist_ok=True)



    # =========================

    # Evaluation loop

    # =========================

    rl_costs = []

    rl_bw0_costfn = []

    rl_bw0_manual = []

    ga_costs = []

    sample_times = []



    total_start = time.time()



    with torch.no_grad():

        for idx, batch in enumerate(test_loader):

            start_time = time.time()



            sample = batch[0]

            # move tensors; keep lists

            sample = {k: (to_device(v, device) if torch.is_tensor(v) or hasattr(v, "to") else v) for k, v in sample.items()}



            sb_batch = to_device(sample["service_batch"], device)



            with torch.cuda.amp.autocast(enabled=use_amp):
                comp_emb = comp_encoder(sb_batch.x[:, :2])        # [C,2] -> [C,H]
                node_emb = node_encoder(sample["node_graph"].x)   # [N,4] -> [N,H]


            # -------- 2-resource req/caps from graphs (authoritative) --------

            comp_reqs2 = sb_batch.x[:, :2].to(torch.float32)                # [C,2] cpu,mem

            node_caps2 = sample["node_graph"].x[:, :2].to(torch.float32)    # [N,2] cpu,mem

            node_state = NodeState(node_caps=node_caps2, device=device, warn_once=False)



            bw_mat = sample["bw_mat"]

            if isinstance(bw_mat, torch.Tensor) and bw_mat.dim() == 3:

                bw_mat = bw_mat[0]



            comp_predecessors = sample["comp_predecessors"]  # List[List[int]]



            out = pipeline.assign_greedy_or_stochastic(

                comp_emb=comp_emb,

                node_emb=node_emb,

                node_caps=node_state.get_final_caps(),  # [N,2]

                comp_reqs=comp_reqs2,                   # [C,2]

                bw_mat=bw_mat,

                comp_predecessors=comp_predecessors,

                # NEW (برای network-aware logits)

                node_graph=sample["node_graph"],

                service_edge_index=sb_batch.edge_index,

                service_edge_attr=getattr(sb_batch, "edge_attr", None),



                greedy=True,

                temperature=1.0,

                return_probs=False,

            )



            assignments = out["assignments"]



            # node_state.apply_assignments_batch(

            #     assignments, comp_reqs2, warn_on_fail=False, return_list=False

            # )



            results = compute_cost(

                assignments,

                sample["node_graph"],

                sample["service_batch"],

                heal=False,

                log=False,

                violation_weight=COST_VIOLATION_WEIGHT,

                count_edges_once=False,

            )



            total_cost = results.get("total_response_time", 0.0)

            exec_time = results.get("exec_time", 0.0)

            trans = results.get("transmission", 0.0)



            # costfunction-reported bw0

            bw0 = results.get("bw0_violations", results.get("violations", 0.0))

            bw0_costfn = to_float(bw0)



            # independent manual bw0

            bw0_man = manual_bw0_violations(assignments, bw_mat, comp_predecessors)



            cost_val = to_float(total_cost)  # already penalized



            rl_costs.append(cost_val)

            rl_bw0_costfn.append(bw0_costfn)

            rl_bw0_manual.append(float(bw0_man))



            # read original JSON
            # sidx = sample.get("sample_idx", idx)
            # in_path = os.path.join(test_root, f"{int(sidx)+1}.json")   # اگر فایل‌ها 1.json.. هستند

            in_path = os.path.join(test_root, test_files[idx])

            with open(in_path, "r") as f:

                in_data = json.load(f)



            # GA baseline

            ga_result = (in_data.get("results", []) or [None])[0]

            ga_cost = None

            if isinstance(ga_result, dict):

                ga_cost = ga_result.get("totalResponseTime", None)

            ga_costs.append(ga_cost)



            # placement records

            placement_records = build_component_records(sample, in_data, assignments)



            sample_time_sec = time.time() - start_time

            sample_times.append(sample_time_sec)

    # ===================== SAVE ONLY rl RESULT =====================

            viol = results.get("violations")

            bw0 = results.get("bw0_violations")

            rl_out = {

                "algorithm": "rl",

                "mode": "greedy",

                "totalResponseTime": float(cost_val),

                "execTime": float(to_float(exec_time)),

                "transmission": float(to_float(trans)),

                #"violations": int(bw0_costfn),

                #"bw0_violations": float(bw0_costfn),

                "violations": float(to_float(viol)),

                "bw0_violations": float(to_float(bw0)),

                "bw0_manual": int(bw0_man),

                "algorithmRuntime": float(sample_time_sec * 1000.0),

                "checkpoint": checkpoint_path,

                "epoch": ckpt.get("epoch", None),

                "finalSolution": placement_records,



                # (اختیاری) اگر دوست داری GA رو هم تو فایل خروجی داشته باشی:

                "ga_totalResponseTime": (None if ga_cost is None else float(ga_cost)),

                "input_file": test_files[idx],

            }



            out_path = os.path.join(processed_dir, f"test{idx}.json")

            with open(out_path, "w") as f:

                json.dump(rl_out, f, indent=2)



            ga_cost_str = "None" if ga_cost is None else f"{float(ga_cost):.4f}"

            print(

                f"[{idx+1}/{num_test_samples}] "

                f"rl_Cost={cost_val:.4f} | GA_Cost={ga_cost_str} | "

                f"BW0(costfn)={bw0_costfn:.0f} | BW0(man)={bw0_man:d} | "

                f"Time={sample_time_sec:.3f}s | Saved={out_path}"

            )

            # in_data["NCOSPP_result"] = {

            #     "algorithm": "NCOSPP",

            #     "mode": "greedy",

            #     "totalResponseTime": float(cost_val),

            #     "execTime": float(to_float(exec_time)),

            #     "transmission": float(to_float(trans)),

            #     "violations": int(bw0_costfn),

            #     "bw0_violations": float(bw0_costfn),

            #     "bw0_manual": int(bw0_man),

            #     "algorithmRuntime": float(sample_time_sec * 1000.0),

            #     "checkpoint": checkpoint_path,

            #     "epoch": ckpt.get("epoch", None),

            #     "finalSolution": placement_records,

            # }



            # out_path = os.path.join(processed_dir, f"test{idx}.json")

            # with open(out_path, "w") as f:

            #     json.dump(in_data, f, indent=2)



            # ga_cost_str = "None" if ga_cost is None else f"{float(ga_cost):.4f}"

            # print(

            #     f"[{idx+1}/{num_test_samples}] "

            #     f"NCOSPP_Cost={cost_val:.4f} | GA_Cost={ga_cost_str} | "

            #     f"BW0(costfn)={bw0_costfn:.0f} | BW0(man)={bw0_man:d} | "

            #     f"Time={sample_time_sec:.3f}s | Saved={out_path}"

            # )



    total_time = time.time() - total_start



    # =========================

    # Summary + plot

    # =========================

    rl_costs_arr = np.array(rl_costs, dtype=float)

    bw0_costfn_arr = np.array(rl_bw0_costfn, dtype=float)

    bw0_manual_arr = np.array(rl_bw0_manual, dtype=float)

    times_arr = np.array(sample_times, dtype=float)



    avg_rl_cost = float(rl_costs_arr.mean()) if len(rl_costs_arr) else 0.0

    min_rl_cost = float(rl_costs_arr.min()) if len(rl_costs_arr) else 0.0

    max_rl_cost = float(rl_costs_arr.max()) if len(rl_costs_arr) else 0.0



    ga_numeric = [float(x) for x in ga_costs if x is not None]

    if len(ga_numeric) > 0:

        avg_ga_cost = float(np.mean(ga_numeric))

        min_ga_cost = float(np.min(ga_numeric))

        max_ga_cost = float(np.max(ga_numeric))

    else:

        avg_ga_cost = min_ga_cost = max_ga_cost = 0.0



    print("\n===== Test Summary =====")

    print(f"rl Cost -> avg={avg_rl_cost:.4f} | min={min_rl_cost:.4f} | max={max_rl_cost:.4f}")

    print(f"GA Cost     -> avg={avg_ga_cost:.4f} | min={min_ga_cost:.4f} | max={max_ga_cost:.4f}")

    print(

        f"BW0(costfn)-> avg={float(bw0_costfn_arr.mean()):.2f} | min={float(bw0_costfn_arr.min()):.0f} | max={float(bw0_costfn_arr.max()):.0f}"

        if len(bw0_costfn_arr) else "BW0(costfn)-> n/a"

    )

    print(

        f"BW0(manual)-> avg={float(bw0_manual_arr.mean()):.2f} | min={float(bw0_manual_arr.min()):.0f} | max={float(bw0_manual_arr.max()):.0f}"

        if len(bw0_manual_arr) else "BW0(manual)-> n/a"

    )

    print(

        f"Time/sample -> avg={float(times_arr.mean()):.3f}s | min={float(times_arr.min()):.3f}s | max={float(times_arr.max()):.3f}s"

        if len(times_arr) else "Time/sample -> n/a"

    )

    print(f"Total Test Time: {total_time:.3f}s")

    print(f"Processed outputs: {processed_dir}")



    # plot costs

    plt.figure(figsize=(8, 5))

    plt.plot(rl_costs, "o-", label=f"rl(avg={avg_rl_cost:.2f})")



    ga_plot = [np.nan if x is None else float(x) for x in ga_costs]

    plt.plot(ga_plot, "s--", label=f"GA(avg={avg_ga_cost:.2f})" if len(ga_numeric) else "GA(available)")



    plt.xlabel("Test Sample Index")

    plt.ylabel("Cost")

    plt.title("rl vs GA per Test Sample")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, "test_rl_vs_ga.png"), dpi=200)

    plt.close()



    # plot bw0 violations (manual vs costfn)

    if len(rl_bw0_manual) > 0:

        plt.figure(figsize=(8, 5))

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





if __name__ == "__main__":

    import multiprocessing as mp

    mp.freeze_support()

    main()

