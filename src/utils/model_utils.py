import torch
import torch.nn.functional as F
import copy
import math
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Any, Optional, Tuple, List, Union
from loguru import logger

def interpolate_spatial_weights(weights: torch.Tensor, old_size: int, new_size: int) -> torch.Tensor:
    """
    Interpolates spatial weights (e.g., action head weights) from old_size to new_size.
    Assumes weights are flattened spatially in the first dimension: (old_size*old_size, in_features)
    """
    if weights.shape[0] != old_size * old_size:
        raise ValueError(f"Weights first dimension {weights.shape[0]} does not match spatial size {old_size}x{old_size}")

    # weights: (spatial, features) -> (features, spatial)
    # Then reshape to (1, features, old_size, old_size) for 2D interpolation
    in_features = weights.shape[1]
    x = weights.t().view(1, in_features, old_size, old_size)
    
    # Perform bilinear interpolation
    x_new = F.interpolate(x, size=(new_size, new_size), mode='bilinear', align_corners=True)
    
    # Reshape back to (new_size*new_size, in_features)
    new_weights = x_new.view(in_features, new_size * new_size).t()
    
    # Preservation of statistics
    old_std = weights.std()
    new_std = new_weights.std()
    if new_std > 1e-6:
        new_weights = new_weights * (old_std / new_std)
        
    mean_shift = weights.mean() - new_weights.mean()
    new_weights = new_weights + mean_shift
    
    return new_weights

def decompose_action_dim(dim: int) -> Tuple[int, int, int]:
    """
    Heuristically decomposes an action dimension into (num_spatial_heads, grid_size, num_categorical).
    Supports common grid sizes: 128, 64, 32, 16.
    """
    for g in [128, 64, 32, 16]:
        g2 = g * g
        if dim >= g2:
            n = dim // g2
            c = dim % g2
            if c < 100:
                return n, g, c
    return 0, 0, dim

def graft_model_weights(state_dict: Dict[str, Any], new_in_channels: int, new_action_dim: Optional[int] = None, 
                        action_handling_mode: str = "auto", new_grid_size: Optional[int] = None,
                        target_state_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Surgically adjusts model weights to match new architectures.
    """
    new_state_dict = copy.deepcopy(state_dict)
    
    # --- 1. Handle Input Channels (CNN & Tactical CNN) ---
    # Matches first conv weights on global/tactical stacks
    _suffixes = (
        "cnn.0.weight",
        "cnn_trunk.0.weight",
        "tactical_cnn_to16.0.weight",
        "tactical_cnn_legacy.0.weight",
    )
    input_keys = [k for k in state_dict.keys() if any(k.endswith(s) for s in _suffixes)]
    
    for key in input_keys:
        old_weight = state_dict[key]
        if len(old_weight.shape) != 4: continue
        
        # Determine target channels from target_state_dict if available, else use argument
        target_in_channels = target_state_dict[key].shape[1] if (target_state_dict and key in target_state_dict) else new_in_channels
        old_in_channels = old_weight.shape[1]
        
        if old_in_channels == target_in_channels: continue
        
        logger.info(f"Grafting input weights for '{key}': {old_in_channels} -> {target_in_channels} channels")
        if target_in_channels < old_in_channels:
            new_state_dict[key] = old_weight[:, :target_in_channels, :, :]
        else:
            out_channels, _, k_h, k_w = old_weight.shape
            new_weight = torch.randn(out_channels, target_in_channels, k_h, k_w) * 0.01
            new_weight[:, :old_in_channels, :, :] = old_weight
            new_state_dict[key] = new_weight

    # --- 2. Handle Fusion Layer (The "Brain") ---
    # Matches 'linear.0.weight'
    fusion_keys = [k for k in state_dict.keys() if k.endswith("linear.0.weight")]
    for key in fusion_keys:
        if target_state_dict and key in target_state_dict:
            old_w = state_dict[key]
            new_w = target_state_dict[key]
            
            if old_w.shape == new_w.shape: continue
            
            logger.info(f"Surgically grafting Fusion Layer '{key}': {old_w.shape} -> {new_w.shape}")
            
            # Map Structure: [Global_Feat, (Tactical_Feat), Command_Feat]
            # Command is always the LAST 10 dimensions.
            old_in_dim = old_w.shape[1]
            new_in_dim = new_w.shape[1]
            
            cmd_dim = 10
            
            # Create fresh weights based on target (randomized)
            grafted_w = torch.randn_like(new_w) * 0.01
            
            # 1. Transfer Global (First chunk)
            # If we added tactical, global is still at the beginning.
            # We assume Global size doesn't change unless grid size changed (handled by linear resize if needed)
            global_size = min(old_in_dim - cmd_dim, new_in_dim - cmd_dim)
            grafted_w[:, :global_size] = old_w[:, :global_size]
            
            # 2. Transfer Command (Last chunk)
            grafted_w[:, -cmd_dim:] = old_w[:, -cmd_dim:]
            
            new_state_dict[key] = grafted_w
            
            # Handle bias
            bias_key = key.replace(".weight", ".bias")
            if bias_key in state_dict and bias_key in target_state_dict:
                new_state_dict[bias_key] = state_dict[bias_key] # features_dim is constant

    # --- 3. Handle Action/Value Heads ---
    head_keys = [
        k for k in state_dict.keys()
        if ("action_net" in k or "value_net" in k or "command_head" in k or "spatial_decoder" in k)
        and "mlp_extractor" not in k
    ]
    
    for key in head_keys:
        old_weight = state_dict[key]
        target_shape = target_state_dict[key].shape if (target_state_dict and key in target_state_dict) else None
        
        if len(old_weight.shape) == 2:
            old_out_dim, old_in_feat = old_weight.shape
            target_out_dim = target_shape[0] if target_shape else (new_action_dim if "action_net" in key else old_out_dim)
            target_in_feat = target_shape[1] if target_shape else old_in_feat
            
            if old_out_dim == target_out_dim and old_in_feat == target_in_feat:
                continue

            logger.info(f"Grafting head weights for '{key}': [{old_out_dim}, {old_in_feat}] -> [{target_out_dim}, {target_in_feat}]")
            
            # A. Handle in_features mismatch (Linear resize)
            if old_in_feat != target_in_feat:
                if target_in_feat < old_in_feat:
                    current_weight = old_weight[:, :target_in_feat]
                else:
                    current_weight = torch.randn(old_out_dim, target_in_feat) * 0.01
                    current_weight[:, :old_in_feat] = old_weight
                curr_out_dim = old_out_dim
            else:
                current_weight = old_weight
                curr_out_dim = old_out_dim

            # B. Handle output_dim (Actions) mismatch (Spatial Interpolation)
            if curr_out_dim != target_out_dim:
                n_old, g_old, c_old = decompose_action_dim(curr_out_dim)
                n_new, g_new, c_new = decompose_action_dim(target_out_dim)
                if new_grid_size:
                    g_new = new_grid_size
                    n_new = target_out_dim // (g_new * g_new)
                    c_new = target_out_dim % (g_new * g_new)

                use_interpolation = (g_old > 0 and g_new > 0 and g_new != g_old) or (action_handling_mode == "interpolate")
                
                if use_interpolation and g_old > 0:
                    logger.info(f"Interpolating spatial heads: {n_old}x{g_old}^2 -> {n_new}x{g_new}^2")
                    new_w = torch.randn(target_out_dim, target_in_feat) * 0.01
                    num_heads_to_transfer = min(n_old, n_new)
                    for i in range(num_heads_to_transfer):
                        o_s, o_e = i * (g_old * g_old), (i + 1) * (g_old * g_old)
                        n_s, n_e = i * (g_new * g_new), (i + 1) * (g_new * g_new)
                        new_w[n_s:n_e, :] = interpolate_spatial_weights(current_weight[o_s:o_e], g_old, g_new)
                    
                    # Categorical Transfer
                    o_cat_s = n_old * (g_old * g_old)
                    n_cat_s = n_new * (g_new * g_new)
                    num_cat = min(c_old, c_new)
                    if num_cat > 0:
                        new_w[n_cat_s:n_cat_s+num_cat, :] = current_weight[o_cat_s:o_cat_s+num_cat, :]
                    new_state_dict[key] = new_w
                else:
                    if target_out_dim < curr_out_dim:
                        new_state_dict[key] = current_weight[:target_out_dim, :]
                    else:
                        new_w = torch.randn(target_out_dim, target_in_feat) * 0.01
                        new_w[:curr_out_dim, :] = current_weight
                        new_state_dict[key] = new_w
            else:
                new_state_dict[key] = current_weight

        elif len(old_weight.shape) == 1:
            old_dim = old_weight.shape[0]
            target_dim = target_shape[0] if target_shape else old_dim
            if old_dim != target_dim:
                if target_dim < old_dim:
                    new_state_dict[key] = old_weight[:target_dim]
                else:
                    # Keep newly-added action logits neutral; a large positive prior
                    # can dominate policy behavior after grafting.
                    new_b = torch.zeros(target_dim, dtype=old_weight.dtype, device=old_weight.device)
                    new_b[:old_dim] = old_weight
                    new_state_dict[key] = new_b
        
    # --- 4. Handle Missing Keys (e.g., brand new tactical_cnn) ---
    if target_state_dict:
        for key in target_state_dict.keys():
            if key not in new_state_dict:
                logger.info(f"Initializing brand new component '{key}'")
                new_state_dict[key] = target_state_dict[key]

    return new_state_dict

def non_finite_state_dict_keys(
    state_dict: Dict[str, Any], *, max_keys: int = 8
) -> List[str]:
    """Return state_dict keys whose tensor/array values contain NaN or inf."""
    bad: List[str] = []
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            finite = torch.isfinite(value).all()
        elif isinstance(value, np.ndarray):
            finite = np.isfinite(value).all()
        else:
            continue
        if not finite:
            bad.append(key)
        if len(bad) >= max_keys:
            break
    return bad


def policy_state_dict_finite(state_dict: Dict[str, Any]) -> bool:
    """True when every tensor/array in a policy state_dict has only finite values."""
    return not non_finite_state_dict_keys(state_dict, max_keys=1)


def inspect_checkpoint_policy_weights(
    zip_path: str, *, max_keys: int = 8
) -> Tuple[bool, str, List[str]]:
    """
    Validate an SB3 checkpoint's policy weights.

    Returns (ok, reason, non_finite_keys). Reasons include ``ok``, ``missing_file``,
    ``not_a_zip``, ``load_failed:...``, ``no_policy_key``, and ``non_finite``.
    """
    import os
    import zipfile

    from stable_baselines3.common.save_util import load_from_zip_file

    if not os.path.isfile(zip_path):
        return False, "missing_file", []
    if not zipfile.is_zipfile(zip_path):
        return False, "not_a_zip", []
    try:
        _, params, _ = load_from_zip_file(zip_path, device="cpu", load_data=False)
    except Exception as exc:
        return False, f"load_failed:{exc}", []
    p_sd = params.get("policy")
    if p_sd is None:
        return False, "no_policy_key", []
    bad = non_finite_state_dict_keys(p_sd, max_keys=max_keys)
    if bad:
        return False, "non_finite", bad
    return True, "ok", []


def checkpoint_policy_weights_finite(zip_path: str) -> bool:
    """Check policy tensors inside an SB3 zip before hot-reloading into the collector."""
    ok, _, _ = inspect_checkpoint_policy_weights(zip_path)
    return ok


def read_checkpoint_timesteps(zip_path: str) -> Optional[int]:
    """Return ``num_timesteps`` from an SB3 ``.zip`` data blob, or None if unavailable."""
    import json
    import zipfile

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            if "data" not in archive.namelist():
                return None
            payload = json.loads(archive.read("data").decode("utf-8"))
            ts = payload.get("num_timesteps")
            return int(ts) if ts is not None else None
    except Exception:
        return None


def apply_checkpoint_timesteps(model: Any, checkpoint_path: str) -> int:
    """Copy ``num_timesteps`` from an SB3 zip onto ``model`` (returns applied count)."""
    ts = read_checkpoint_timesteps(checkpoint_path)
    if ts is not None:
        model.num_timesteps = ts
        return ts
    return int(getattr(model, "num_timesteps", 0) or 0)


def latest_step_checkpoint(paths: list[str]) -> Optional[str]:
    """Pick the checkpoint with the highest ``*_steps.zip`` step count (mtime tie-break)."""
    import re

    if not paths:
        return None

    def sort_key(path: str) -> tuple[int, float]:
        match = re.search(r"(\d+)_steps\.zip$", os.path.basename(path))
        steps = int(match.group(1)) if match else 0
        return steps, os.path.getmtime(path)

    return max(paths, key=sort_key)


def load_optimizer_state_from_zip(
    model: Any,
    zip_path: str,
    device: Union[str, torch.device] = "auto",
) -> Tuple[bool, str]:
    """
    Restore Adam state from an SB3 ``.zip`` into ``model.policy.optimizer``.

    Call after policy weights are on the model (same parameter count/order as the
    checkpoint). Returns ``(ok, reason)``; on failure the caller keeps a fresh optimizer.
    """
    if not hasattr(model, "policy") or getattr(model.policy, "optimizer", None) is None:
        return False, "no_policy_optimizer"

    from stable_baselines3.common.save_util import load_from_zip_file

    try:
        _, params, _ = load_from_zip_file(zip_path, device=device, load_data=False)
    except Exception as exc:
        return False, f"zip_read_failed:{exc}"

    opt_sd = params.get("policy.optimizer")
    if not opt_sd:
        return False, "no_optimizer_in_checkpoint"

    try:
        model.policy.optimizer.load_state_dict(opt_sd)
        return True, "ok"
    except Exception as exc:
        return False, f"load_state_dict_failed:{exc}"


def load_policy_weights_from_zip(
    policy: torch.nn.Module,
    zip_path: str,
    device: Union[str, torch.device] = "auto",
    *,
    strict: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """
    Load only the policy ``state_dict`` from an SB3 ``.zip`` checkpoint.

    Safer than ``model.set_parameters`` for rollout snapshots: avoids optimizer /
    ``exact_match`` failures leaving the learner on stale random weights.
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    _, params, _ = load_from_zip_file(zip_path, device=device, load_data=False)
    p_sd = params.get("policy")
    if p_sd is None:
        return False, [], []
    missing, unexpected = policy.load_state_dict(p_sd, strict=strict)
    return True, list(missing), list(unexpected)


def load_curriculum_checkpoint(
    algo_class: Any,
    model_path: str,
    algo_kwargs: Dict[str, Any],
    load_kwargs: Dict[str, Any],
    observation_space: Any,
    action_space: Any,
    *,
    grid_size: int = 32,
    restore_optimizer: bool = False,
) -> Any:
    """
    Load a checkpoint for curriculum continuation.

    Tries a direct SB3 load first. On observation/action shape mismatch, grafts
    compatible policy weights onto a freshly constructed model (same path as sync
    ``train_pipeline``).

    When ``restore_optimizer`` is True and parameter shapes match the checkpoint,
    Adam moments are restored for same-level interrupted resume. On curriculum
    level change, pass ``restore_optimizer=False`` so the new MDP starts with fresh Adam.
    """
    load_kwargs = {**load_kwargs, "restore_optimizer": restore_optimizer}
    try:
        model = algo_class.load(model_path, **load_kwargs)
        if restore_optimizer:
            ok, reason = load_optimizer_state_from_zip(
                model, model_path, load_kwargs.get("device", "auto")
            )
            if ok:
                logger.info(f"Restored optimizer state from {model_path!r}")
            elif reason != "no_optimizer_in_checkpoint":
                logger.warning(
                    f"Optimizer restore skipped for {model_path!r} ({reason})"
                )
        return model
    except Exception as direct_exc:
        logger.warning(
            f"Direct checkpoint load failed for {model_path!r} ({direct_exc}); "
            "attempting graft onto the new level architecture."
        )

    from stable_baselines3.common.save_util import load_from_zip_file

    model = algo_class(**algo_kwargs)
    _, params, _ = load_from_zip_file(model_path, device="cpu", load_data=False)
    saved_sd = params.get("policy")
    if saved_sd is None:
        logger.error(f"Checkpoint {model_path!r} has no policy weights; using fresh model.")
        return model

    if hasattr(observation_space, "spaces") and "map" in observation_space.spaces:
        new_in_channels = observation_space.spaces["map"].shape[0]
    else:
        new_in_channels = observation_space.shape[0]

    new_action_dim = getattr(action_space, "n", None)
    if new_action_dim is None and hasattr(action_space, "nvec"):
        # Universal MultiDiscrete([64, 1024]) — graft spatial/command heads separately.
        new_action_dim = int(np.sum(action_space.nvec))
    grafted = graft_model_weights(
        saved_sd,
        new_in_channels=new_in_channels,
        new_action_dim=new_action_dim,
        new_grid_size=grid_size,
        target_state_dict=model.policy.state_dict(),
    )
    missing, unexpected = model.policy.load_state_dict(grafted, strict=False)
    logger.info(
        f"Grafted curriculum weights from {model_path!r}: "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if restore_optimizer:
        ok, reason = load_optimizer_state_from_zip(model, model_path, "cpu")
        if ok:
            logger.info(f"Restored optimizer state after graft from {model_path!r}")
        else:
            logger.warning(
                f"Optimizer restore skipped after graft from {model_path!r} ({reason})"
            )
    apply_checkpoint_timesteps(model, model_path)
    return model


def graft_build_model_weights(
    checkpoint_path: str,
    model: Any,
    *,
    strict: bool = False,
) -> bool:
    """
    Load compatible build-agent policy weights across curriculum levels.
    Grafts CNN input channels when build_map_layers grow.
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    try:
        _, params, _ = load_from_zip_file(checkpoint_path, device="cpu", load_data=False)
    except Exception as exc:
        logger.warning(f"graft_build_model_weights: could not load {checkpoint_path}: {exc}")
        return False

    old_sd = params.get("policy")
    if old_sd is None:
        return False

    new_sd = model.policy.state_dict()
    grafted = graft_model_weights(
        old_sd,
        new_in_channels=None,
        target_state_dict=new_sd,
    )
    missing, unexpected = model.policy.load_state_dict(grafted, strict=strict)
    if missing:
        logger.info(f"graft_build_model_weights missing keys: {missing[:8]}...")
    if unexpected:
        logger.info(f"graft_build_model_weights unexpected keys: {unexpected[:8]}...")
    opt_ok, opt_reason = load_optimizer_state_from_zip(model, checkpoint_path, "cpu")
    if opt_ok:
        logger.info(f"graft_build_model_weights: restored optimizer from {checkpoint_path!r}")
    elif opt_reason != "no_optimizer_in_checkpoint":
        logger.warning(
            f"graft_build_model_weights: optimizer restore skipped ({opt_reason})"
        )
    return True


def load_build_checkpoint(
    model_path: str,
    model_kwargs: Dict[str, Any],
    *,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Load a build-agent checkpoint with policy and Adam state.

    Build curriculum keeps a fixed action space and stable map channel count, so
    weights should load directly without channel grafting. On metadata mismatch we
    copy policy weights and restore the optimizer explicitly.
    """
    from src.agent.maskable_ppo import MaskablePPO

    load_kwargs = dict(load_kwargs or {})
    load_kwargs["restore_optimizer"] = True
    device = load_kwargs.get("device", "auto")

    try:
        model = MaskablePPO.load(model_path, **load_kwargs)
        opt_ok, opt_reason = load_optimizer_state_from_zip(model, model_path, device)
        if opt_ok:
            logger.info(f"Build checkpoint: optimizer restored from {model_path!r}")
        elif opt_reason != "no_optimizer_in_checkpoint":
            logger.warning(
                f"Build checkpoint: optimizer restore skipped ({opt_reason})"
            )
        return model
    except Exception as direct_exc:
        logger.warning(
            f"Build direct load failed for {model_path!r} ({direct_exc}); "
            "loading policy and optimizer without grafting."
        )

    model = MaskablePPO(**model_kwargs)
    ok, missing, unexpected = load_policy_weights_from_zip(
        model.policy, model_path, device, strict=False
    )
    if not ok:
        raise RuntimeError(f"Build checkpoint {model_path!r} has no policy weights")
    if missing:
        logger.warning(
            "Build policy partial load "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    opt_ok, opt_reason = load_optimizer_state_from_zip(model, model_path, device)
    if opt_ok:
        logger.info(
            f"Build checkpoint: optimizer restored after policy load from {model_path!r}"
        )
    else:
        logger.warning(f"Build checkpoint: optimizer restore skipped ({opt_reason})")
    return model


_SQUAD_FE_KEY_REMAP = {
    "features_extractor.cnn_trunk.": "features_extractor.global_cnn.",
    "features_extractor.tactical_cnn_to16.": "features_extractor.tactical_cnn.",
    "features_extractor.linear.": "features_extractor.query_linear.",
    "features_extractor.reasoning.": "features_extractor.reasoning.",
}


def graft_squad_policy_from_universal(
    universal_sd: Dict[str, torch.Tensor],
    squad_sd: Dict[str, torch.Tensor],
    *,
    new_in_channels: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Map UniversalRecurrentPolicy weights onto SquadRecurrentPolicy.

    Pointer / entity modules are left at squad initialization when missing.
    """
    remapped: Dict[str, torch.Tensor] = {}
    for key, tensor in universal_sd.items():
        new_key = key
        for old_prefix, new_prefix in _SQUAD_FE_KEY_REMAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        if new_key in squad_sd and squad_sd[new_key].shape == tensor.shape:
            remapped[new_key] = tensor

    for key, tensor in squad_sd.items():
        if key not in remapped:
            remapped[key] = tensor

    return graft_model_weights(
        remapped,
        new_in_channels=new_in_channels,
        target_state_dict=squad_sd,
    )


def load_squad_checkpoint(
    model_path: str,
    algo_kwargs: Dict[str, Any],
    load_kwargs: Optional[Dict[str, Any]] = None,
    *,
    observation_space: Any = None,
    action_space: Any = None,
    grid_size: int = 32,
    restore_optimizer: bool = False,
) -> Any:
    """Load squad policy, grafting from universal checkpoint on shape mismatch."""
    from stable_baselines3.common.save_util import load_from_zip_file
    from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO

    load_kwargs = dict(load_kwargs or {})
    AlgoClass = MaskableRecurrentPPO
    try:
        model = AlgoClass.load(model_path, **load_kwargs)
        if restore_optimizer:
            opt_ok, opt_reason = load_optimizer_state_from_zip(
                model, model_path, load_kwargs.get("device", "auto")
            )
            if not opt_ok and opt_reason != "no_optimizer_in_checkpoint":
                logger.warning(f"Squad checkpoint optimizer restore skipped ({opt_reason})")
        return model
    except Exception as direct_exc:
        logger.warning(
            f"Squad direct load failed for {model_path!r} ({direct_exc}); grafting universal weights."
        )

    model = AlgoClass(**algo_kwargs)
    _, params, _ = load_from_zip_file(model_path, device="cpu", load_data=False)
    saved_sd = params.get("policy")
    if saved_sd is None:
        logger.error(f"Squad checkpoint {model_path!r} has no policy weights; using fresh model.")
        return model

    new_in = None
    if observation_space is not None and hasattr(observation_space, "spaces"):
        if "map" in observation_space.spaces:
            new_in = observation_space.spaces["map"].shape[0]

    grafted = graft_squad_policy_from_universal(
        saved_sd,
        model.policy.state_dict(),
        new_in_channels=new_in,
    )
    missing, unexpected = model.policy.load_state_dict(grafted, strict=False)
    logger.info(
        f"Grafted squad policy from {model_path!r}: missing={len(missing)} unexpected={len(unexpected)}"
    )
    if restore_optimizer:
        opt_ok, opt_reason = load_optimizer_state_from_zip(
            model, model_path, load_kwargs.get("device", "auto")
        )
        if not opt_ok and opt_reason != "no_optimizer_in_checkpoint":
            logger.warning(f"Squad graft optimizer restore skipped ({opt_reason})")
    apply_checkpoint_timesteps(model, model_path)
    return model


def calculate_distance_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def solve_optimal_assignment(points_a: np.ndarray, points_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(points_a) == 0 or len(points_b) == 0:
        return np.array([]), np.array([])
    cost_matrix = calculate_distance_matrix(points_a, points_b)
    return linear_sum_assignment(cost_matrix)
