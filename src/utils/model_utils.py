import torch
import torch.nn.functional as F
import copy
import math
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Any, Optional, Tuple, List, Union, Sequence
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
            
            # Map Structure: [Global_Feat, (Tactical_Feat), Command_Feat, optional Race]
            # Default: command is the LAST 10 dimensions.
            # With include_race: [..., command(10), race(6)] — race is a trailing append.
            old_in_dim = old_w.shape[1]
            new_in_dim = new_w.shape[1]
            
            cmd_dim = 10
            from src.agent.squad_constants import SQUAD_RACE_DIM
            
            # Create fresh weights based on target (randomized)
            grafted_w = torch.randn_like(new_w) * 0.01
            
            race_extra = new_in_dim - old_in_dim
            if (
                race_extra == SQUAD_RACE_DIM
                and old_in_dim >= cmd_dim
                and new_in_dim >= cmd_dim + SQUAD_RACE_DIM
            ):
                # Trailing race expansion (e.g. CTF 11→12): keep all prior fusion
                # weights; leave the new race columns randomly initialized.
                grafted_w[:, :old_in_dim] = old_w
            elif (
                race_extra == -SQUAD_RACE_DIM
                and new_in_dim >= cmd_dim
                and old_in_dim == new_in_dim + SQUAD_RACE_DIM
            ):
                # Trailing race removal (e.g. 144-wide checkpoint → 138-wide target).
                grafted_w[:, :new_in_dim] = old_w[:, :new_in_dim]
            else:
                # 1. Transfer Global (First chunk)
                # If we added tactical, global is still at the beginning.
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


def policy_params_checksum(policy: torch.nn.Module, *, digest_len: int = 16) -> str:
    """
    Device-invariant fingerprint of ``policy.state_dict()`` (CPU float32 bytes).

    Prefer ``rollout_policy_zip_checksum`` when pairing async IPC — that hashes the
    exact ``policy.pth`` bytes the learner will load, avoiding GPU-reduction noise
    and shared-module alias differences vs the on-disk zip.
    """
    import hashlib

    hasher = hashlib.sha1()
    with torch.no_grad():
        for name, value in sorted(policy.state_dict().items()):
            if not isinstance(value, torch.Tensor):
                continue
            arr = (
                value.detach()
                .to(dtype=torch.float32, device="cpu")
                .contiguous()
                .numpy()
            )
            hasher.update(name.encode("utf-8"))
            hasher.update(arr.tobytes())
    return hasher.hexdigest()[:digest_len]


def rollout_policy_zip_checksum(zip_path: str, *, digest_len: int = 16) -> str:
    """SHA1 of the SB3 zip's ``policy.pth`` member — stable across collector/learner."""
    import hashlib
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as archive:
        payload = archive.read("policy.pth")
    return hashlib.sha1(payload).hexdigest()[:digest_len]


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
    On tensor shape mismatch (e.g. include_race 138↔144), grafts compatible weights
    onto the target policy architecture before loading.

    Checkpoint tensors are always deserialized on CPU, then copied into ``policy``
    (whatever device it already lives on). ``device`` is kept for call-site
    compatibility and is not used as ``map_location``.
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    # Always unzip onto CPU, then copy into ``policy`` (which may already be on
    # CUDA). Loading the zip with map_location=cuda doubles peak VRAM for the
    # duration of the load — fatal in SubprocVecEnv workers that already hold
    # an opponent snapshot on the shared GPU.
    _ = device  # API compat; destination is policy's existing device.
    _, params, _ = load_from_zip_file(zip_path, device="cpu", load_data=False)
    p_sd = params.get("policy")
    if p_sd is None:
        return False, [], []

    target_sd = policy.state_dict()
    mismatched = [
        k
        for k in p_sd
        if k in target_sd and target_sd[k].shape != p_sd[k].shape
    ]
    if mismatched:
        new_in_channels = 0
        for prefix in (
            "features_extractor.",
            "pi_features_extractor.",
            "vf_features_extractor.",
        ):
            cnn_key = f"{prefix}global_cnn.0.weight"
            if cnn_key in target_sd:
                new_in_channels = int(target_sd[cnn_key].shape[1])
                break
        logger.warning(
            f"Grafting policy weights from {zip_path!r} "
            f"({len(mismatched)} shape mismatches, e.g. {mismatched[0]})"
        )
        p_sd = graft_model_weights(
            p_sd,
            new_in_channels=new_in_channels,
            target_state_dict=target_sd,
        )

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
        if new_key not in squad_sd:
            continue
        if squad_sd[new_key].shape == tensor.shape:
            remapped[new_key] = tensor
        elif new_key.endswith("linear.0.weight") or new_key.endswith("linear.0.bias"):
            # Shape mismatch (e.g. CTF include_race 138→144): keep old weights so
            # graft_model_weights can surgically expand the fusion/query layer.
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


# Blend toward the template command row (1 = full copy). Extra nats are added
# after the blend. Full copy + 1.0 made heal the default combat action.
DEFAULT_LOUD_COMMAND_MIX = 0.5
DEFAULT_LOUD_COMMAND_BIAS = 0.0
_LOUD_COMMAND_TEMPLATE = "attack_local"


def loud_command_heads_from_conf(training_conf: Optional[dict]) -> List[str]:
    """Command names to boost at curriculum graft (root or env)."""
    conf = training_conf or {}
    env = conf.get("env") if isinstance(conf.get("env"), dict) else {}
    raw = conf.get("loud_command_heads", env.get("loud_command_heads", []))
    if raw is None:
        return []
    if isinstance(raw, str):
        name = raw.strip().lower()
        return [name] if name else []
    return [str(x).strip().lower() for x in raw if str(x).strip()]


def loud_command_bias_from_conf(training_conf: Optional[dict]) -> float:
    """Extra bias (nats) after mixing toward the template. Default 0.0."""
    conf = training_conf or {}
    env = conf.get("env") if isinstance(conf.get("env"), dict) else {}
    raw = conf.get("loud_command_bias", env.get("loud_command_bias", None))
    if raw is None or raw == "":
        return DEFAULT_LOUD_COMMAND_BIAS
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_LOUD_COMMAND_BIAS


def loud_command_mix_from_conf(training_conf: Optional[dict]) -> float:
    """How far to blend the loud slot toward the template row in [0, 1]."""
    conf = training_conf or {}
    env = conf.get("env") if isinstance(conf.get("env"), dict) else {}
    raw = conf.get("loud_command_mix", env.get("loud_command_mix", None))
    if raw is None or raw == "":
        return DEFAULT_LOUD_COMMAND_MIX
    try:
        mix = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_LOUD_COMMAND_MIX
    if mix != mix:  # NaN
        return DEFAULT_LOUD_COMMAND_MIX
    return min(max(mix, 0.0), 1.0)


def configured_command_names(training_conf: Optional[dict]) -> List[str]:
    """Cumulative action_heads + categorical_heads (root preferred over env)."""
    conf = training_conf or {}
    env = conf.get("env") if isinstance(conf.get("env"), dict) else {}
    heads = list(conf.get("action_heads") or env.get("action_heads") or [])
    cats = list(conf.get("categorical_heads") or env.get("categorical_heads") or [])
    return [str(h).lower() for h in heads] + [str(c).lower() for c in cats]


def _resolve_command_head(policy_or_model: Any) -> Optional[Any]:
    obj = policy_or_model
    head = getattr(obj, "command_head", None)
    if head is None:
        inner = getattr(obj, "policy", None)
        head = getattr(inner, "command_head", None) if inner is not None else None
    if head is None or getattr(head, "weight", None) is None:
        return None
    if getattr(head, "bias", None) is None:
        return None
    return head


def _template_command_index(names: Sequence[str], loud_set: set) -> Optional[int]:
    lowered = [str(n).lower() for n in names]
    preferred = _LOUD_COMMAND_TEMPLATE
    if preferred in lowered:
        idx = lowered.index(preferred)
        if lowered[idx] not in loud_set:
            return idx
    for i, name in enumerate(lowered):
        if name not in loud_set:
            return i
    return None


def apply_loud_command_head_init(
    policy_or_model: Any,
    command_names: Sequence[str],
    loud_names: Sequence[str],
    extra_bias: float,
    mix: float = DEFAULT_LOUD_COMMAND_MIX,
) -> List[int]:
    """Blend each loud slot toward a template command row, then add extra_bias.

    Unused padded rows start near zero, so a bias bump alone cannot beat a
    peaked attack logit. Mixing toward ``attack_local`` (else the first
    non-loud configured head) gives the new slot a combat-sized direction
    without replacing attack. ``mix=1`` is a full copy; ``mix=0.5`` is a
    halfway blend. Idempotent only for ``mix=1``. Returns boosted indices.
    """
    loud = [str(n).strip().lower() for n in loud_names if str(n).strip()]
    if not loud:
        return []
    head = _resolve_command_head(policy_or_model)
    if head is None:
        logger.warning("Loud command init skipped: policy has no command_head")
        return []
    try:
        mix_f = float(mix)
    except (TypeError, ValueError):
        mix_f = DEFAULT_LOUD_COMMAND_MIX
    if mix_f != mix_f:
        mix_f = DEFAULT_LOUD_COMMAND_MIX
    mix_f = min(max(mix_f, 0.0), 1.0)
    extra_f = float(extra_bias)
    names = [str(n).lower() for n in command_names]
    name_to_idx = {n: i for i, n in enumerate(names)}
    loud_set = set(loud)
    template_idx = _template_command_index(names, loud_set)
    boosted: List[int] = []
    n_out = int(head.weight.shape[0])
    with torch.no_grad():
        for name in loud:
            idx = name_to_idx.get(name)
            if idx is None:
                logger.warning(
                    f"Loud command init: {name!r} not in configured command names"
                )
                continue
            if idx >= n_out or idx >= int(head.bias.numel()):
                logger.warning(
                    f"Loud command init: {name!r} index {idx} out of command_head range {n_out}"
                )
                continue
            if template_idx is not None and template_idx != idx:
                w_new = (1.0 - mix_f) * head.weight[idx] + mix_f * head.weight[template_idx]
                b_new = (
                    (1.0 - mix_f) * head.bias[idx]
                    + mix_f * head.bias[template_idx]
                    + extra_f
                )
                head.weight[idx].copy_(w_new)
                head.bias[idx].copy_(b_new)
            else:
                head.bias[idx].copy_(head.bias[idx] + extra_f)
            boosted.append(idx)
    return boosted


# A unit-pointer command has no counterpart in the universal attack curriculum,
# so its command_head row arrives as a near-zero padded ``_unused`` row that can
# never out-compete a peaked attack logit. Seed it from the spatial head it
# generalises, exactly like the loud-command graft prior, but automatically —
# the mapping is a property of the architecture, not of any one level's YAML.
POINTER_COMMAND_ROW_TEMPLATES: Dict[str, str] = {
    "attack_unit": "attack_local",
    "heal_unit": "heal_local",
    # load_unit is introduced with unload at ctf_level_24; no prior spatial
    # load row exists in the graft. Seed from attack_local so the new head is
    # not a near-zero padded unused logit at first sample.
    "load_unit": "attack_local",
    # snipe is a unit-targeted special; seed from attack_unit when present,
    # else attack_local (same enemy-pointer family).
    "snipe": "attack_unit",
}


def seed_pointer_command_rows(
    policy_or_model: Any,
    command_names: Sequence[str],
    *,
    templates: Optional[Dict[str, str]] = None,
) -> List[int]:
    """Copy each pointer command's template ``command_head`` row onto it.

    Full copy (not a blend): ``attack_unit`` *is* ``attack_local`` with a unit
    target instead of a cell, so the grafted attack logits transfer directly.
    Idempotent — copying the same source row twice is a no-op. Returns the
    command indices that were seeded.
    """
    table = dict(POINTER_COMMAND_ROW_TEMPLATES if templates is None else templates)
    head = _resolve_command_head(policy_or_model)
    if head is None:
        return []
    names = [str(n).lower() for n in command_names]
    name_to_idx = {n: i for i, n in enumerate(names)}
    n_out = int(head.weight.shape[0])
    seeded: List[int] = []
    with torch.no_grad():
        for new_name, template_name in table.items():
            dst = name_to_idx.get(new_name)
            src = name_to_idx.get(template_name)
            if dst is None or src is None or dst == src:
                continue
            if max(dst, src) >= n_out or max(dst, src) >= int(head.bias.numel()):
                continue
            head.weight[dst].copy_(head.weight[src])
            head.bias[dst].copy_(head.bias[src])
            seeded.append(dst)
    if seeded:
        logger.info(f"Seeded pointer command rows from templates: indices={seeded}")
    return seeded


def maybe_seed_pointer_command_rows(
    policy_or_model: Any,
    training_conf: Optional[dict],
    *,
    resume: bool,
) -> List[int]:
    """Curriculum-graft hook for pointer commands. No-op on same-level resume."""
    if resume:
        return []
    return seed_pointer_command_rows(
        policy_or_model, configured_command_names(training_conf)
    )


def maybe_apply_loud_command_head_init(
    policy_or_model: Any,
    training_conf: Optional[dict],
    *,
    resume: bool,
    command_names: Optional[Sequence[str]] = None,
) -> List[int]:
    """Curriculum-graft hook. No-op on same-level CONTINUE_TRAINING (resume)."""
    if resume:
        return []
    loud = loud_command_heads_from_conf(training_conf)
    if not loud:
        return []
    names = list(command_names) if command_names is not None else configured_command_names(
        training_conf
    )
    extra = loud_command_bias_from_conf(training_conf)
    mix = loud_command_mix_from_conf(training_conf)
    boosted = apply_loud_command_head_init(
        policy_or_model, names, loud, extra, mix=mix
    )
    if boosted:
        logger.info(
            f"Loud command init: {loud} mix={mix} extra_bias={extra} "
            f"slots={boosted} template={_LOUD_COMMAND_TEMPLATE}"
        )
    return boosted


def calculate_distance_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def solve_optimal_assignment(points_a: np.ndarray, points_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(points_a) == 0 or len(points_b) == 0:
        return np.array([]), np.array([])
    cost_matrix = calculate_distance_matrix(points_a, points_b)
    return linear_sum_assignment(cost_matrix)
