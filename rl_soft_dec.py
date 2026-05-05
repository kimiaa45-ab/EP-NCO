def main():
    import os
    import time
    import re
    import json
    import yaml
    import random
    import numpy as np
    import torch
    import torch.optim as optim
    from torch.utils.data import Subset, DataLoader
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone

    # Optional: make deterministic behavior tighter on CUDA (slower but stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from nets.soft_decoder import ARPlacementPipeline
    from problem.Dataset import ServiceNodeDataset, service_node_collate_fn
    from utils.costfunction1 import compute_cost_ga_like as compute_cost
    from src.state import NodeState

    # ------------------ Val split config ------------------
    VAL_FRAC_DEFAULT = 0.2

    # ================== GLOBAL MODES ==================
    FAST_MODE = True

    if FAST_MODE:
        EVAL_EVERY = 5
        SAVE_PLACEMENTS = False
        MAX_EVAL_SAMPLES = 50
        VERBOSE = True
    else:
        EVAL_EVERY = 1
        SAVE_PLACEMENTS = True
        MAX_EVAL_SAMPLES = None
        VERBOSE = True

    # ================== Plot config ==================
    PLOT_DIR = "img"
    os.makedirs(PLOT_DIR, exist_ok=True)
    BEST_AVG_MULT = 1.00

    # ------------------ Time parsing ------------------
    def parse_time_to_seconds(t):
        if t is None:
            return None
        if isinstance(t, (int, float)):
            return int(t)

        t = str(t).strip().lower()
        if t.isdigit():
            return int(t)

        hours = re.search(r"(\d+)\s*h", t)
        mins = re.search(r"(\d+)\s*m", t)
        secs = re.search(r"(\d+)\s*s", t)

        total = 0
        if hours:
            total += int(hours.group(1)) * 3600
        if mins:
            total += int(mins.group(1)) * 60
        if secs:
            total += int(secs.group(1))

        if total == 0:
            try:
                return int(float(t))
            except Exception:
                return None
        return total

    # ------------------ Load config ------------------
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ------------------ Seeding ------------------
    SEED = int(config.get("seed", 0))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data_cfg = config.get("data", {})
    objective_cfg = config.get("objective", {})
    data_type = data_cfg.get("type")

    all_models_cfg = config.get("model", {})
    model_cfg = all_models_cfg.get(data_type, {})

    num_epochs = int(model_cfg.get("num_epochs"))
    num_services = int(model_cfg.get("charnum_service"))

    gpu_hidden_dim = int(model_cfg.get("gpu_hidden_dim"))
    cpu_hidden_dim = int(model_cfg.get("cpu_hidden_dim"))
    device_str = model_cfg.get("device", "auto")

    num_train_samples = int(model_cfg.get("num_samples"))
    n_layers = int(model_cfg.get("n_encode_layers", model_cfg.get("num_layers")))
    w_rt_cfg = float(objective_cfg.get("response_time_weight", 1.0))
    w_pr_cfg = float(objective_cfg.get("platform_reliability_weight", 0.0))
    w_sr_cfg = float(objective_cfg.get("service_reliability_weight", 0.0))
    COST_OBJECTIVE_KWARGS = {
        "objective_mode": objective_cfg.get("mode", "weighted_sum"),
        "w_rt": w_rt_cfg,
        "w_pr": w_pr_cfg,
        "w_sr": w_sr_cfg,
        "normalize_rt": False,
        "rt_ref": objective_cfg.get("rt_ref", "ga_baseline"),
    }

    base_dir = data_cfg.get("base_dir", "data/generated")
    dataset_root = os.path.join(base_dir, data_type)

    lr = float(model_cfg.get("lr", 2e-4))
    max_train_time = parse_time_to_seconds(config.get("max_train_time", None))

    # ------------------ Device ------------------
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    hidden_dim = gpu_hidden_dim if device.type == "cuda" else cpu_hidden_dim
    use_amp = device.type == "cuda"

    print("Device:", device)
    print("Using dataset_root:", dataset_root)

    # ------------------ Performance flags ------------------
    if device.type == "cuda" and not FAST_MODE:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def to_device(obj, dev: torch.device):
        if hasattr(obj, "to"):
            try:
                return obj.to(dev, non_blocking=True)
            except TypeError:
                return obj.to(dev)
        return obj

    def load_state_dict_compatible(module, loaded_state, module_name: str):
        current_state = module.state_dict()
        compatible = {}
        skipped = []
        for k, v in loaded_state.items():
            if k in current_state and current_state[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(k)
        missing, unexpected = module.load_state_dict(compatible, strict=False)
        if VERBOSE:
            if skipped:
                print(
                    f"[ckpt] {module_name}: skipped {len(skipped)} incompatible keys "
                    f"(shape/name mismatch)."
                )
            if missing:
                print(f"[ckpt] {module_name}: missing keys after load: {len(missing)}")
            if unexpected:
                print(f"[ckpt] {module_name}: unexpected keys: {len(unexpected)}")

    use_datasize_feature = bool(config.get("use_datasize_feature", False))
    decoder_shaping_cfg = config.get("rl_decoder_shaping", {})
    DECODER_SHAPING = dict(
        alpha_net=float(decoder_shaping_cfg.get("alpha_net", 0.0)),
        lambda_delay=float(decoder_shaping_cfg.get("lambda_delay", 1.0)),
        sim_weight=float(decoder_shaping_cfg.get("sim_weight", 0.0)),
        beta_crowd=float(decoder_shaping_cfg.get("beta_crowd", 0.0)),
        crowd_use_util=bool(decoder_shaping_cfg.get("crowd_use_util", True)),
        crowd_gamma=float(decoder_shaping_cfg.get("crowd_gamma", 1.5)),
    )
    if bool(config.get("rl_separate_eval_decoder_shaping", False)):
        eval_decoder_shaping_cfg = config.get("rl_eval_decoder_shaping", {})
        EVAL_DECODER_SHAPING = dict(
            alpha_net=float(
                eval_decoder_shaping_cfg.get("alpha_net", DECODER_SHAPING["alpha_net"])
            ),
            lambda_delay=float(
                eval_decoder_shaping_cfg.get(
                    "lambda_delay", DECODER_SHAPING["lambda_delay"]
                )
            ),
            sim_weight=float(
                eval_decoder_shaping_cfg.get("sim_weight", DECODER_SHAPING["sim_weight"])
            ),
            beta_crowd=float(
                eval_decoder_shaping_cfg.get("beta_crowd", DECODER_SHAPING["beta_crowd"])
            ),
            crowd_use_util=bool(
                eval_decoder_shaping_cfg.get(
                    "crowd_use_util", DECODER_SHAPING["crowd_use_util"]
                )
            ),
            crowd_gamma=float(
                eval_decoder_shaping_cfg.get(
                    "crowd_gamma", DECODER_SHAPING["crowd_gamma"]
                )
            ),
        )
    else:
        EVAL_DECODER_SHAPING = dict(DECODER_SHAPING)
    allow_fallback_train = bool(
        config.get("rl_allow_fallback_train", config.get("allow_fallback_train", True))
    )
    allow_fallback_eval = bool(
        config.get("rl_allow_fallback_eval", config.get("allow_fallback_eval", True))
    )
    use_ga_baseline_for_policy = bool(
        config.get(
            "rl_use_ga_baseline_for_policy",
            config.get("use_ga_baseline_for_policy", True),
        )
    )
    normalize_advantage = bool(config.get("normalize_advantage", True))


    # ------------------ Dataset & DataLoader ------------------
    dataset = ServiceNodeDataset(
        root=dataset_root,
        use_cache=False,
        cache_in_memory=False,
    )

    train_subset = Subset(dataset, list(range(min(num_train_samples, len(dataset)))))
    num_samples = len(train_subset)
    print("Num train samples:", num_samples)

    # -------- split train/val deterministically --------
    n_total = min(num_train_samples, len(dataset))
    all_indices = list(range(n_total))
    rnd = random.Random(SEED)
    rnd.shuffle(all_indices)

    # val_frac = float(config.get("val_frac", VAL_FRAC_DEFAULT))
    val_frac = float(model_cfg.get("val_frac", VAL_FRAC_DEFAULT))

    val_size = max(1, int(round(n_total * val_frac)))
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    num_train = len(train_subset)
    num_val = len(val_subset)
    print(
        f"Num total samples: {n_total} | train: {num_train} | val: {num_val} (val_frac={val_frac})"
    )

    loader_workers = int(
        model_cfg.get("dataloader_num_workers", config.get("dataloader_num_workers", 0))
    )
    prefetch_factor = int(
        model_cfg.get("dataloader_prefetch_factor", config.get("dataloader_prefetch_factor", 2))
    )

    # dl_kwargs = dict(
    dl_train_kwargs = dict(
        batch_size=1,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=service_node_collate_fn,
    )

    if loader_workers > 0:
        # dl_kwargs["persistent_workers"] = (device.type == "cuda")
        # dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_train_kwargs["persistent_workers"] = (device.type == "cuda")
        dl_train_kwargs["prefetch_factor"] = prefetch_factor

    # deterministic eval loader (no shuffle)
    dl_val_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=service_node_collate_fn,
    )

    if loader_workers > 0:
        dl_val_kwargs["persistent_workers"] = (device.type == "cuda")
        dl_val_kwargs["prefetch_factor"] = prefetch_factor

    # train_loader = DataLoader(train_subset, **dl_kwargs)
    train_loader = DataLoader(train_subset, **dl_train_kwargs)
    val_loader = DataLoader(val_subset, **dl_val_kwargs)

    # Eval loader (deterministic order) - still on train data (your request)
    dl_eval_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=service_node_collate_fn,
    )

    if loader_workers > 0:
        dl_eval_kwargs["persistent_workers"] = (device.type == "cuda")
        dl_eval_kwargs["prefetch_factor"] = prefetch_factor

    eval_loader = DataLoader(train_subset, **dl_eval_kwargs)

    # ------------------ Load GA baselines ------------------
    # def load_ga_baselines(dataset_root: str, device: torch.device):
    #     """
    #     Returns:
    #     ga_tensor_full: [M] float32  (M = max_file_stem)  index0-based => stem-1
    #     ga_mask_full:   [M] bool
    #     index_offset:   int (always 1 here because stem->index0 is stem-1)
    #     """
    #     files = [f for f in os.listdir(dataset_root) if f.endswith(".json")]
    #     stems = []
    #     for f in files:
    #         try:
    #             stems.append(int(os.path.splitext(f)[0]))
    #         except Exception:
    #             pass
    #
    #     if len(stems) == 0:
    #         raise RuntimeError(f"No numeric json files found in {dataset_root}")
    #
    #     max_stem = max(stems)
    #     # allocate [max_stem] so that stem s maps to idx = s-1
    #     baselines = [0.0] * max_stem
    #     mask = [False] * max_stem
    #
    #     for f in files:
    #         try:
    #             stem = int(os.path.splitext(f)[0])
    #         except Exception:
    #             continue
    #
    #         path = os.path.join(dataset_root, f)
    #         try:
    #             with open(path, "r") as fp:
    #                 data = json.load(fp)
    #         except Exception:
    #             continue
    #
    #         # Extract GA baseline
    #         ga_val = None
    #         if (
    #             isinstance(data, dict)
    #             and "results" in data
    #             and isinstance(data["results"], list)
    #             and len(data["results"]) > 0
    #         ):
    #             r0 = data["results"][0]
    #             if isinstance(r0, dict):
    #                 ga_val = r0.get("totalResponseTime", None)
    #
    #         if ga_val is None:
    #             continue
    #
    #         idx0 = stem - 1
    #         if 0 <= idx0 < max_stem:
    #             baselines[idx0] = float(ga_val)
    #             mask[idx0] = True
    #
    #     ga_tensor_full = torch.tensor(baselines, dtype=torch.float32, device=device)
    #     ga_mask_full = torch.tensor(mask, dtype=torch.bool, device=device)
    #
    #     print(f"[GA] loaded={sum(mask)}/{len(mask)} from {dataset_root} (index0 = stem-1)")
    #     return ga_tensor_full, ga_mask_full, 1  # offset=1 means stem -> idx0 = stem-1

    def load_ga_baselines(folder, n):
        baselines = [None] * n
        for i in range(n):
            file_idx = i + 1
            path = os.path.join(folder, f"{file_idx}.json")
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                data = json.load(f)
            if "results" in data and len(data["results"]) > 0:
                baselines[i] = float(data["results"][0].get("totalResponseTime", 0.0))

        print(f"Loaded {sum(b is not None for b in baselines)} GA baselines (aligned 0..{n-1})")
        return baselines

    ga_baselines = load_ga_baselines(dataset_root, num_train_samples)
    ga_array = [b if (b is not None) else 0.0 for b in ga_baselines]
    ga_tensor_full = torch.tensor(ga_array, dtype=torch.float32, device=device)
    ga_mask_full = torch.tensor([b is not None for b in ga_baselines], dtype=torch.bool, device=device)

    # ------------------ Build Models ------------------

    # -------- Simple encoders --------
    # service_batch.x is [C,3] => (cpu, mem, reliability)
    # node_graph.x is [N,5]   => (cpu, mem, nodeID, tier, reliability)

    comp_encoder = torch.nn.Sequential(
        torch.nn.Linear(3, hidden_dim),
        torch.nn.ReLU(),
    ).to(device)

    node_encoder = torch.nn.Sequential(
        torch.nn.Linear(5, hidden_dim),
        torch.nn.ReLU(),
    ).to(device)

    # IMPORTANT: ARDecoder must be the new 2-resource version (cap_dim=2)
    pipeline = ARPlacementPipeline(
        embed_dim=hidden_dim,
        proj_dim=8,
        cap_dim=2,
        net_agg="max",
        use_datasize_feature=use_datasize_feature,
    ).to(device)
    print(f"Decoder features: use_datasize_feature={use_datasize_feature}")

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)

    # -------- Optimizer / Scheduler --------
    # Warm restarts can cause sudden LR jumps -> policy drift in REINFORCE.
    # base_lr = float(config.get("lr", LR))
    # min_lr = float(config.get("min_lr", base_lr * 0.1))
    # use_warm_restarts = bool(config.get("use_warm_restarts", False))

    base_lr = float(model_cfg.get("lr", lr))
    min_lr = float(model_cfg.get("min_lr", base_lr * 0.1))
    use_warm_restarts = bool(model_cfg.get("use_warm_restarts", False))
    decoder_lr_mult = float(config.get("decoder_lr_mult", 10.0))
    decoder_lr = float(config.get("decoder_lr", base_lr * decoder_lr_mult))
    freeze_encoders_epochs = int(config.get("freeze_encoders_epochs", 5))

    optimizer = optim.Adam(
        [
            {
                "params": list(comp_encoder.parameters()) + list(node_encoder.parameters()),
                "lr": base_lr,
            },
            {"params": list(pipeline.parameters()), "lr": decoder_lr},
        ]
    )

    min_delta = float(config.get("min_delta", 0.02))

    if use_warm_restarts:
        # keep old behavior only if explicitly enabled
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=int(config.get("T_0", 10)),
        #     T_mult=int(config.get("T_mult", 2))
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, threshold=min_delta, min_lr=1e-4
        )
    else:
        # stable decay (no restart)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("num_epochs", num_epochs)),
            eta_min=min_lr,
        )

    # ------------------ RL Hyperparams ------------------
    init_temp = float(config.get("init_temp", 0.8))
    final_temp = float(config.get("final_temp", 0.1))
    init_entropy = float(config.get("init_entropy", 0.02))
    final_entropy = float(config.get("final_entropy", 0.0))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    anneal_epochs = max(1, int(0.6 * num_epochs))
    eval_lambda_u = float(
        model_cfg.get("eval_lambda_u", config.get("eval_lambda_u", 0.25))
    )
    eval_lambda_spread = float(
        model_cfg.get("eval_lambda_spread", config.get("eval_lambda_spread", 0.2))
    )
    print(
        f"Eval regularization fixed: eval_lambda_u={eval_lambda_u:.4f}, "
        f"eval_lambda_spread={eval_lambda_spread:.4f}"
    )
    print(
        f"RL-only settings: use_ga_baseline_for_policy={use_ga_baseline_for_policy}, "
        f"allow_fallback_train={allow_fallback_train}, "
        f"allow_fallback_eval={allow_fallback_eval}"
    )
    print(
        f"Optimizer LR groups: encoders_lr={base_lr:.6g}, decoder_lr={decoder_lr:.6g}, "
        f"freeze_encoders_epochs={freeze_encoders_epochs}"
    )

    # IMPORTANT: penalty for bw=0 / missing links is applied INSIDE costfunction
    COST_VIOLATION_WEIGHT = 1

    # ------------------ Early stopping ------------------
    min_epochs = int(config.get("min_epochs", 20))
    patience = int(config.get("patience", 8))
    max_consecutive_increases = int(config.get("max_consecutive_increases", 5))

    # best_eval_cost = float("inf")
    # best_eval_epoch = None
    # last_eval_cost = float("inf")
    best_val_cost = float("inf")
    best_val_epoch = None
    last_val_cost = float("inf")

    epochs_no_improve = 0
    consecutive_increases = 0

    print(
        f"Early stopping: min_epochs={min_epochs}, patience={patience}, "
        f"min_delta={min_delta}, max_consecutive_increases={max_consecutive_increases}"
    )

    # ------------------ History ------------------
    loss_history = []
    reward_history = []
    cost_history = []
    min_cost_history = []
    max_cost_history = []
    train_eval_history = []
    greedy_avg_history = []
    greedy_min_history = []
    greedy_max_history = []
    improved_epochs = []
    improved_costs = []

    # ------------------ Training time tracking ------------------
    training_start_time = time.time()
    training_start_dt = datetime.now(timezone.utc).isoformat()
    print(f"Training started at {training_start_dt} (UTC)")
    if max_train_time is not None:
        print(f"Max training time: {max_train_time} seconds")

    if device.type == "cpu":
        policy_batch_size = int(
            config.get("cpu_policy_batch_size", config.get("policy_batch_size", 8))
        )
    else:
        policy_batch_size = int(config.get("policy_batch_size", 8))
    print(f"Policy batch size: {policy_batch_size} (device={device.type})")

    def _ensure_scalar_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    policy_cost_mode = str(config.get("policy_cost_mode", "response_time")).strip().lower()
    print(f"Policy cost mode: {policy_cost_mode}")

    def get_policy_cost_from_result(res):
        if policy_cost_mode in ("objective", "obj"):
            return _ensure_scalar_tensor(res["objective"])
        if policy_cost_mode in ("legacy", "legacy_total_response_time", "total"):
            return _ensure_scalar_tensor(
                res.get("legacy_total_response_time", res["objective"])
            )
        if policy_cost_mode in ("response_time", "rt"):
            violations_t = _ensure_scalar_tensor(res.get("violations", 0.0))
            rt_t = _ensure_scalar_tensor(res["response_time"])
            return rt_t + (COST_VIOLATION_WEIGHT * violations_t)
        return _ensure_scalar_tensor(res["objective"])

    print("Sanity check sample_idx:")
    seen = []
    for i, batch in enumerate(train_loader):
        s = batch[0]
        seen.append(s.get("sample_idx", None))
        if i == 20:
            break
    print("first 20 sample_idx:", seen)
    print("unique:", len(set(seen)))

    # Optional sanity: val order should be deterministic
    seen_val = []
    for i, batch in enumerate(val_loader):
        s = batch[0]
        seen_val.append(s.get("sample_idx", None))
        if i == 20:
            break
    print("val first 20 sample_idx (deterministic):", seen_val)

    # ------------------ Checkpoint load ------------------
    ckpt_path = "best_rl_soft_model.pt"
    resume_from_checkpoint = bool(config.get("resume_from_checkpoint", False))
    if resume_from_checkpoint and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)

        if "comp_encoder" in checkpoint:
            load_state_dict_compatible(comp_encoder, checkpoint["comp_encoder"], "comp_encoder")
        if "node_encoder" in checkpoint:
            load_state_dict_compatible(node_encoder, checkpoint["node_encoder"], "node_encoder")
        if "pipeline" in checkpoint:
            load_state_dict_compatible(pipeline, checkpoint["pipeline"], "pipeline")

        if VERBOSE:
            print("Loaded checkpoint:", ckpt_path)
    else:
        print("Starting from scratch (resume_from_checkpoint=False)")


    ema_baseline = None
    ema_beta = 0.98


    def get_lambda_u(epoch: int) -> float:
        progress = min(1.0, epoch / float(anneal_epochs))
        return float(0.5 + (0.25 - 0.5) * progress)  # linear decay

    def get_temperature(epoch: int) -> float:
        progress = min(1.0, epoch / float(anneal_epochs))
        return float(init_temp + (final_temp - init_temp) * progress)  # linear

    # ======================== TRAINING LOOP ==========================
    for epoch in range(1, num_epochs + 1):
        elapsed = time.time() - training_start_time
        if max_train_time is not None and elapsed >= max_train_time:
            print(
                f"Time budget exceeded at start of epoch {epoch}: "
                f"elapsed={int(elapsed)}s >= max={max_train_time}s"
            )
            break

        if VERBOSE:
            print(f"\nEpoch {epoch}/{num_epochs} | elapsed {int(elapsed)}s")

        progress = min(1.0, epoch / anneal_epochs)
        temperature = get_temperature(epoch)
        lambda_u = get_lambda_u(epoch)
        # temperature = final_temp + (init_temp - final_temp) * np.exp(-progress * 1.0)
        entropy_coeff = final_entropy + (init_entropy - final_entropy) * np.exp(-progress * 3)

        if best_val_epoch is not None and epoch > best_val_epoch + 20:
            temperature = final_temp
            entropy_coeff = 0.0

        comp_encoder.train()
        node_encoder.train()
        pipeline.train()
        freeze_enc = epoch <= freeze_encoders_epochs
        for p in comp_encoder.parameters():
            p.requires_grad = not freeze_enc
        for p in node_encoder.parameters():
            p.requires_grad = not freeze_enc

        optimizer.zero_grad(set_to_none=True)

        epoch_loss_sum = 0.0
        epoch_updates = 0

        epoch_cost_t = torch.zeros((), device=device)
        epoch_reward_t = torch.zeros((), device=device)

        batch_costs_cpu = []

        mb_logps = []
        mb_entropies = []
        mb_costs = []
        mb_sample_idx = []
        last_baseline_mean = None

        # for batch in train_loader:
        # for it, batch in enumerate(train_loader):
        #     is_last = (it == num_samples - 1)

        for it, batch in enumerate(train_loader):
            is_last = (it == num_train - 1)

            sample = batch[0]
            sample = {
                k: (to_device(v, device) if torch.is_tensor(v) or hasattr(v, "to") else v)
                for k, v in sample.items()
            }

            sb_batch = to_device(sample["service_batch"], device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                comp_emb = comp_encoder(sb_batch.x[:, :3])        # [C,3] -> [C,H]
                node_emb = node_encoder(sample["node_graph"].x)   # [N,5] -> [N,H]

            # -------- decoder inputs (2-resource) --------
            comp_reqs2 = sb_batch.x[:, :2].to(torch.float32)              # [C,2]
            node_caps2 = sample["node_graph"].x[:, :2].to(torch.float32)  # [N,2]
            node_state = NodeState(node_caps=node_caps2, device=device, warn_once=True)

            bw_mat = sample["bw_mat"]
            if isinstance(bw_mat, torch.Tensor) and bw_mat.dim() == 3:
                bw_mat = bw_mat[0]

            comp_predecessors = sample["comp_predecessors"]  # List[List[int]]

            
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = pipeline.assign_greedy_or_stochastic(
                    comp_emb=comp_emb,
                    node_emb=node_emb,
                    node_caps=sample["node_graph"].x[:, :2],
                    comp_reqs=sb_batch.x[:, :2],
                    bw_mat=bw_mat,
                    comp_predecessors=comp_predecessors,
                    comp_fixed_node_rows=sample.get("comp_fixed_node_rows", None),
                    node_graph=sample["node_graph"],
                    service_edge_index=sb_batch.edge_index,
                    service_edge_attr=getattr(sb_batch, "edge_attr", None),
                    greedy=False,
                    **DECODER_SHAPING,
                    temperature=temperature,
                    allow_fallback=allow_fallback_train,
                    return_probs=False,
                )


            assignments = out["assignments"]
            if (assignments < 0).any():
                logp = out["log_probs"].sum()
                entropy = out["entropies"].mean()
                sidx_bad = int(sample.get("sample_idx", 0))
                ga_ref_bad = None
                if 0 <= sidx_bad < ga_tensor_full.numel() and bool(
                    ga_mask_full[sidx_bad].item()
                ):
                    ga_ref_bad = ga_tensor_full[sidx_bad]
                bad_res = compute_cost(
                    assignments,
                    sample["node_graph"],
                    sample["service_batch"],
                    heal=False,
                    log=False,
                    violation_weight=COST_VIOLATION_WEIGHT,
                    count_edges_once=True,
                    lambda_u=lambda_u,
                    lambda_spread=0.2,
                    min_unique_nodes=2,
                    spread_guard="always",
                    ga_ref=ga_ref_bad,
                    comp_fixed_node_rows=sample.get("comp_fixed_node_rows", None),
                    **COST_OBJECTIVE_KWARGS,
                )
                cost_tensor = bad_res["objective"]
                if not isinstance(cost_tensor, torch.Tensor):
                    cost_tensor = torch.as_tensor(
                        cost_tensor, device=device, dtype=torch.float32
                    )
                policy_cost_tensor = get_policy_cost_from_result(bad_res)
                cost_detached = policy_cost_tensor.detach()
                reward = -cost_detached

                mb_logps.append(logp)
                mb_entropies.append(entropy)
                mb_costs.append(cost_detached)
                mb_sample_idx.append(int(sample.get("sample_idx", 0)))

                epoch_cost_t = epoch_cost_t + cost_tensor.detach()
                epoch_reward_t = epoch_reward_t + reward
                batch_costs_cpu.append(float(cost_tensor.detach().cpu().item()))
                if (len(mb_sample_idx) == policy_batch_size) or is_last:
                    logps_tensor = torch.stack(mb_logps)           # [B]
                    entropies_tensor = torch.stack(mb_entropies)   # [B]
                    costs_tensor = torch.stack(mb_costs)           # [B]

                    mb_idx_tensor = torch.tensor(mb_sample_idx, dtype=torch.long, device=device)
                    mb_idx_tensor = torch.clamp(mb_idx_tensor, 0, ga_tensor_full.numel() - 1)

                    ga_batch = ga_tensor_full[mb_idx_tensor]       # [B]
                    mask_batch = ga_mask_full[mb_idx_tensor]       # [B] bool

                    prev_ema = ema_baseline
                    if prev_ema is None:
                        prev_ema = costs_tensor.mean().detach()

                    if use_ga_baseline_for_policy:
                        baseline = torch.where(
                            mask_batch, ga_batch, prev_ema.expand_as(costs_tensor)
                        )
                    else:
                        baseline = prev_ema.expand_as(costs_tensor)

                    last_baseline_mean = float(baseline.mean().detach().cpu().item())

                    adv = (baseline - costs_tensor)
                    if normalize_advantage and adv.numel() > 1:
                        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)

                    adv = torch.clamp(adv, -10.0, 10.0)

                    policy_loss = -(logps_tensor * adv.detach()).mean()
                    entropy_loss = -entropy_coeff * entropies_tensor.mean()
                    loss = policy_loss + entropy_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        list(comp_encoder.parameters())
                        + list(node_encoder.parameters())
                        + list(pipeline.parameters()),
                        max_norm=grad_clip_norm,
                    )

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    epoch_loss_sum += float(loss.detach().cpu().item())
                    epoch_updates += 1

                    batch_mean = costs_tensor.mean().detach()
                    if ema_baseline is None:
                        ema_baseline = batch_mean
                    else:
                        ema_baseline = (
                            ema_beta * ema_baseline + (1.0 - ema_beta) * batch_mean
                        )

                    mb_logps.clear()
                    mb_entropies.clear()
                    mb_costs.clear()
                    mb_sample_idx.clear()
                continue

            # Update state (capacity) using 2-resource reqs
            node_state.apply_assignments_batch(
                assignments, comp_reqs2, warn_on_fail=False, return_list=False
            )

            sidx_for_ref = int(sample.get("sample_idx", 0))
            ga_ref_val = None
            if 0 <= sidx_for_ref < ga_tensor_full.numel() and bool(
                ga_mask_full[sidx_for_ref].item()
            ):
                ga_ref_val = ga_tensor_full[sidx_for_ref]

            # Cost (includes violation penalty internally)
            res = compute_cost(
                assignments,
                sample["node_graph"],
                sample["service_batch"],
                heal=False,
                log=False,
                violation_weight=COST_VIOLATION_WEIGHT,
                count_edges_once=True,
                lambda_u= lambda_u,              # شروع پیشنهادی: 0.01 تا 0.05
                lambda_spread=0.2,          # شروع پیشنهادی: 0.2 تا 2.0
                min_unique_nodes=2,         # حداقل 2 نود برای هر سرویس
                spread_guard="always",
                ga_ref=ga_ref_val,
                comp_fixed_node_rows=sample.get("comp_fixed_node_rows", None),
                **COST_OBJECTIVE_KWARGS,
            )

            cost_tensor = res["objective"]
            if not isinstance(cost_tensor, torch.Tensor):
                cost_tensor = torch.as_tensor(cost_tensor, device=device, dtype=torch.float32)
            policy_cost_tensor = get_policy_cost_from_result(res)

            # IMPORTANT: reward/cost are environment signals → detach to avoid storing huge graphs
            cost_detached = policy_cost_tensor.detach()
            reward = -cost_detached

            logp = out["log_probs"].sum()         # keep graph (policy gradient)
            entropy = out["entropies"].mean()     # normalized entropy reg

            mb_logps.append(logp)
            mb_entropies.append(entropy)
            mb_costs.append(cost_detached)        # <-- detach here

            # IMPORTANT: use stable dataset index (works under shuffle)
            sidx = sample.get("sample_idx", None)
            if sidx is None:
                # fallback (not ideal): but keep training running
                sidx = 0
            mb_sample_idx.append(int(sidx))

            epoch_cost_t = epoch_cost_t + cost_tensor.detach()
            epoch_reward_t = epoch_reward_t + reward.detach()
            batch_costs_cpu.append(float(cost_tensor.detach().cpu().item()))

            # ---------- mini-batch update ----------
            # (قبلش باید is_last را ساخته باشی: is_last = (it == num_samples - 1))
            if (len(mb_sample_idx) == policy_batch_size) or is_last:
                logps_tensor = torch.stack(mb_logps)           # [B]
                entropies_tensor = torch.stack(mb_entropies)   # [B]
                costs_tensor = torch.stack(mb_costs)           # [B]

                mb_idx_tensor = torch.tensor(mb_sample_idx, dtype=torch.long, device=device)
                mb_idx_tensor = torch.clamp(mb_idx_tensor, 0, ga_tensor_full.numel() - 1)

                ga_batch = ga_tensor_full[mb_idx_tensor]       # [B]
                mask_batch = ga_mask_full[mb_idx_tensor]       # [B] bool

                # ---------- EMA baseline for samples without GA ----------
                batch_mean = costs_tensor.mean().detach()
                if ema_baseline is None:
                    ema_baseline = batch_mean
                else:
                    ema_baseline = ema_beta * ema_baseline + (1.0 - ema_beta) * batch_mean

                if use_ga_baseline_for_policy:
                    baseline = torch.where(
                        mask_batch, ga_batch, ema_baseline.expand_as(costs_tensor)
                    )
                else:
                    baseline = ema_baseline.expand_as(costs_tensor)

                last_baseline_mean = float(baseline.mean().detach().cpu().item())

                # ---------- Advantage (REINFORCE) ----------
                adv = (baseline - costs_tensor)  # اگر cost بهتر از baseline باشد => مثبت
                if normalize_advantage and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)

                # clip for stability
                adv = torch.clamp(adv, -10.0, 10.0)

                # ---------- Loss ----------
                policy_loss = -(logps_tensor * adv.detach()).mean()
                entropy_loss = -entropy_coeff * entropies_tensor.mean()
                loss = policy_loss + entropy_loss

                # ---------- Optim step ----------
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    list(comp_encoder.parameters())
                    + list(node_encoder.parameters())
                    + list(pipeline.parameters()),
                    max_norm=grad_clip_norm,
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                epoch_loss_sum += float(loss.detach().cpu().item())
                epoch_updates += 1

                mb_logps.clear()
                mb_entropies.clear()
                mb_costs.clear()
                mb_sample_idx.clear()

        # ---------- epoch aggregates ----------
        avg_loss = epoch_loss_sum / max(1, epoch_updates)

        # avg_cost = float((epoch_cost_t / max(1, num_samples)).detach().cpu().item())
        # avg_reward = float((epoch_reward_t / max(1, num_samples)).detach().cpu().item())
        avg_cost = float((epoch_cost_t / max(1, num_train)).detach().cpu().item())
        avg_reward = float((epoch_reward_t / max(1, num_train)).detach().cpu().item())

        min_cost = min(batch_costs_cpu) if batch_costs_cpu else avg_cost
        max_cost = max(batch_costs_cpu) if batch_costs_cpu else avg_cost

        loss_history.append(avg_loss)
        reward_history.append(avg_reward)
        cost_history.append(avg_cost)
        min_cost_history.append(min_cost)
        max_cost_history.append(max_cost)

        if VERBOSE:
            if last_baseline_mean is None:
                last_baseline_mean = float("nan")
            print(
                f"Epoch {epoch} done | AvgLoss={avg_loss:.6f} "
                f"AvgCost={avg_cost:.6f} AvgReward={avg_reward:.6f} "
                f"BaselineMean={last_baseline_mean:.6f} Temp={temperature:.4f}"
            )

        # ================== EVALUATION (greedy) ==================
        if epoch % EVAL_EVERY == 0:
            comp_encoder.eval()
            node_encoder.eval()
            pipeline.eval()

            eval_costs = []
            seen = 0

            with torch.no_grad():
                # for batch in train_loader:
                # Evaluate on VAL set (deterministic: shuffle=False)
                for batch in val_loader:
                    sample = batch[0]
                    sample = {
                        k: (to_device(v, device) if torch.is_tensor(v) or hasattr(v, "to") else v)
                        for k, v in sample.items()
                    }

                    sb_batch = to_device(sample["service_batch"], device)

                    comp_emb = comp_encoder(sb_batch.x[:, :3])
                    node_emb = node_encoder(sample["node_graph"].x)

                    comp_reqs2 = sb_batch.x[:, :2].to(torch.float32)
                    node_caps2 = sample["node_graph"].x[:, :2].to(torch.float32)
                    node_state = NodeState(node_caps=node_caps2, device=device, warn_once=False)

                    bw_mat = sample["bw_mat"]
                    if isinstance(bw_mat, torch.Tensor) and bw_mat.dim() == 3:
                        bw_mat = bw_mat[0]

                    comp_predecessors = sample["comp_predecessors"]


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
                    if (assignments < 0).any():
                        sidx_for_ref = int(sample.get("sample_idx", 0))
                        ga_ref_val = None
                        if 0 <= sidx_for_ref < ga_tensor_full.numel() and bool(
                            ga_mask_full[sidx_for_ref].item()
                        ):
                            ga_ref_val = ga_tensor_full[sidx_for_ref]
                        bad_eval = compute_cost(
                            assignments,
                            sample["node_graph"],
                            sample["service_batch"],
                            heal=False,
                            log=False,
                            violation_weight=COST_VIOLATION_WEIGHT,
                            count_edges_once=True,
                            lambda_u=eval_lambda_u,
                            lambda_spread=eval_lambda_spread,
                            min_unique_nodes=2,
                            spread_guard="always",
                            ga_ref=ga_ref_val,
                            comp_fixed_node_rows=sample.get("comp_fixed_node_rows", None),
                            **COST_OBJECTIVE_KWARGS,
                        )
                        total = bad_eval["objective"]
                        if not isinstance(total, torch.Tensor):
                            total = torch.as_tensor(total, device=device, dtype=torch.float32)
                        eval_costs.append(float(total.detach().cpu().item()))
                        continue

                    sidx_for_ref = int(sample.get("sample_idx", 0))
                    ga_ref_val = None
                    if 0 <= sidx_for_ref < ga_tensor_full.numel() and bool(
                        ga_mask_full[sidx_for_ref].item()
                    ):
                        ga_ref_val = ga_tensor_full[sidx_for_ref]

                    cost_eval = compute_cost(
                        assignments,
                        sample["node_graph"],
                        sample["service_batch"],
                        heal=False,
                        log=False,
                        violation_weight=COST_VIOLATION_WEIGHT,
                        count_edges_once=True,
                        lambda_u=eval_lambda_u,
                        lambda_spread=eval_lambda_spread,
                        min_unique_nodes=2,         # حداقل 2 نود برای هر سرویس
                        spread_guard="always",
                        ga_ref=ga_ref_val,
                        comp_fixed_node_rows=sample.get("comp_fixed_node_rows", None),
                        **COST_OBJECTIVE_KWARGS,
                    )

                    total = cost_eval["objective"]
                    if not isinstance(total, torch.Tensor):
                        total = torch.as_tensor(total, device=device, dtype=torch.float32)

                    eval_costs.append(float(total.detach().cpu().item()))

                    seen += 1
                    if MAX_EVAL_SAMPLES is not None and seen >= MAX_EVAL_SAMPLES:
                        break

            if len(eval_costs) > 0:
                # avg_train_eval = sum(eval_costs) / len(eval_costs)
                # min_train_eval = min(eval_costs)
                # max_train_eval = max(eval_costs)

                avg_val_eval = sum(eval_costs) / len(eval_costs)
                min_val_eval = min(eval_costs)
                max_val_eval = max(eval_costs)

                # greedy_avg_history.append(avg_train_eval)
                # greedy_min_history.append(min_train_eval)
                # greedy_max_history.append(max_train_eval)
                # train_eval_history.append(avg_train_eval)

                greedy_avg_history.append(avg_val_eval)
                greedy_min_history.append(min_val_eval)
                greedy_max_history.append(max_val_eval)
                train_eval_history.append(avg_val_eval)

                if epoch_updates > 0:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_eval)
                    else:
                        scheduler.step()


                if VERBOSE:
                    print(
                        f"Greedy VAL eval: avg={avg_val_eval:.6f}, "
                        f"min={min_val_eval:.6f}, max={max_val_eval:.6f}"
                    )

                # improved = avg_train_eval < best_eval_cost - min_delta
                improved = avg_val_eval < best_val_cost - min_delta

                if improved:
                    # after improvement: reduce exploration + reduce lr
                    for g in optimizer.param_groups:
                        g["lr"] = max(g["lr"] * 0.5, 1e-6)

                    # best_eval_cost = avg_train_eval
                    # best_eval_epoch = epoch
                    best_val_cost = avg_val_eval
                    best_val_epoch = epoch
                    epochs_no_improve = 0
                    consecutive_increases = 0

                    ckpt = {
                        "comp_encoder": comp_encoder.state_dict(),
                        "node_encoder": node_encoder.state_dict(),
                        "pipeline": pipeline.state_dict(),
                        "epoch": int(epoch),
                        "best_val_cost": float(best_val_cost),
                        "cap_dim": 2,
                    }

                    torch.save(ckpt, ckpt_path)

                    improved_epochs.append(epoch)
                    # improved_costs.append(best_eval_cost)
                    # print(f"Saved new best checkpoint at epoch {epoch}: {best_eval_cost:.6f}")

                    improved_costs.append(best_val_cost)
                    print(f"Saved new best checkpoint at epoch {epoch}: {best_val_cost:.6f}")

                else:
                    epochs_no_improve += 1

                    # if avg_train_eval > last_eval_cost:
                    if avg_val_eval > last_val_cost:
                        consecutive_increases += 1
                    else:
                        consecutive_increases = 0

                    if epoch >= min_epochs:
                        if consecutive_increases >= max_consecutive_increases:
                            print(f"Stopping: {max_consecutive_increases} consecutive increases")
                            print("Restoring best checkpoint before exit:", ckpt_path)

                            checkpoint = torch.load(ckpt_path, map_location=device)
                            load_state_dict_compatible(comp_encoder, checkpoint["comp_encoder"], "comp_encoder")
                            load_state_dict_compatible(node_encoder, checkpoint["node_encoder"], "node_encoder")
                            load_state_dict_compatible(pipeline, checkpoint["pipeline"], "pipeline")
                            break

                        if epochs_no_improve >= patience:
                            print("Stopping: no improvement")
                            print("Restoring best checkpoint before exit:", ckpt_path)

                            checkpoint = torch.load(ckpt_path, map_location=device)
                            load_state_dict_compatible(comp_encoder, checkpoint["comp_encoder"], "comp_encoder")
                            load_state_dict_compatible(node_encoder, checkpoint["node_encoder"], "node_encoder")
                            load_state_dict_compatible(pipeline, checkpoint["pipeline"], "pipeline")
                            break

                # last_eval_cost = avg_train_eval
                last_val_cost = avg_val_eval

    # ------------------ Training end & logging ------------------
    training_end_time = time.time()
    training_end_dt = datetime.now(timezone.utc).isoformat()
    total_training_seconds = int(training_end_time - training_start_time)

    print(f"Training finished at {training_end_dt} (UTC). Total duration: {total_training_seconds}s")

    log_data = {
        "loss_history": loss_history,
        "reward_history": reward_history,
        "cost_history": cost_history,
        "min_cost_history": min_cost_history,
        "max_cost_history": max_cost_history,
        "train_eval_history": train_eval_history,
        "greedy_avg_history": greedy_avg_history,
        "greedy_min_history": greedy_min_history,
        "greedy_max_history": greedy_max_history,
        "final_epoch": epoch,
        "training_start": training_start_dt,
        "training_end": training_end_dt,
        "training_time_seconds": total_training_seconds,
        "max_train_time": max_train_time,
        # "best_eval_cost": best_eval_cost,
        # "best_eval_epoch": best_eval_epoch,
        "best_val_cost": best_val_cost,
        "best_val_epoch": best_val_epoch,
        "improved_epochs": improved_epochs,
        "improved_costs": improved_costs,
        "COST_VIOLATION_WEIGHT": COST_VIOLATION_WEIGHT,
        "cap_dim": 2,
    }

    with open("rl_soft_training_log.json", "w") as f:
        json.dump(log_data, f, indent=4)

    print("Log saved to rl_soft_training_log.json")

    # ===================== PLOTS (SAVE TO img/, NO SHOW) =====================
    has_epoch0_eval = False
    def save_plot(x, y, title, xlabel, ylabel, label, path):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, linewidth=2, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        if label:
            plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    # 1) Train Reward
    train_reward = [-c for c in cost_history]
    train_epochs = list(range(1, len(cost_history) + 1))
    save_plot(
        x=train_epochs,
        y=train_reward,
        title="Training Reward (on-policy, raw)",
        xlabel="Epoch",
        ylabel="Reward",
        label="Train Avg Reward (= -cost)",
        path=os.path.join(PLOT_DIR, "train_reward.png"),
    )

    # 2) Train Cost (avg/min/max)
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, cost_history, linewidth=2, label="Train Avg Cost")
    plt.plot(train_epochs, min_cost_history, linestyle="--", label="Train Min Cost")
    plt.plot(train_epochs, max_cost_history, linestyle="--", label="Train Max Cost")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Training Cost (on-policy)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "train_cost.png"), dpi=200)
    plt.close()

    # 3) Eval Cost (greedy) with best_avg * factor line
    if len(greedy_avg_history) > 0:
        if has_epoch0_eval:
            eval_epochs = [0] + [
                i * EVAL_EVERY for i in range(1, len(greedy_avg_history))
            ]
        else:
            eval_epochs = [
                i * EVAL_EVERY for i in range(1, len(greedy_avg_history) + 1)
            ]
        best_avg = min(greedy_avg_history)
        best_line = best_avg * BEST_AVG_MULT

        plt.figure(figsize=(10, 6))
        plt.plot(eval_epochs, greedy_avg_history, linewidth=2, label="Eval Avg Cost")
        plt.plot(eval_epochs, greedy_min_history, linestyle="--", label="Eval Min Cost")
        plt.plot(eval_epochs, greedy_max_history, linestyle="--", label="Eval Max Cost")
        plt.axhline(
            best_line,
            linestyle=":",
            linewidth=2,
            label=f"BestAvg*{BEST_AVG_MULT:.2f} = {best_line:.4f}",
        )
        plt.text(eval_epochs[0], best_line, f"{best_line:.4f}", va="bottom", ha="left")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.title("Evaluation Cost (greedy)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "eval_cost_greedy.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
