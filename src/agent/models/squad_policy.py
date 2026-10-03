"""Autoregressive maskable recurrent policy for the CTF squad architecture.

Sampling order is AlphaStar-style ``command → actors → terminal target``:

1. **command** — one categorical over ``FIXED_UNIVERSAL_COMMAND_DIM`` slots,
   masked by the env action mask (configured heads whose prerequisites exist).
2. **actors** — one binary pointer per padded squad slot, masked by aliveness
   *and* by what the sampled command needs (ready healers for ``heal_unit``,
   load-capable units for ``load_unit``, armed controllables for
   ``attack_unit``, AoE casters, loaded transports, conceal/reveal units).
3. **terminal target** — either one unit-pointer slot (allied slot for
   ``heal_unit`` / ``load_unit``, visible-enemy slot for ``attack_unit``) or
   one 32×32 world cell for spatial commands. The inactive branch is forced to
   a single legal value, so it contributes exactly zero to the joint log
   probability.

The scalar log probability is the sum of those conditional log probabilities.
``evaluate_actions`` rebuilds the conditionals from the **stored** command, so
Proximal Policy Optimization trains against exactly the distribution the
collector sampled from (watch ``train/rollout_log_prob_abs_err``).

Command-dependent legality rides in the observation (``command_modes``,
``heal_actor_mask``, ``heal_target_mask``, ``load_actor_mask``,
``load_target_mask``, ``attack_actor_mask``, ``aoe_actor_mask``,
``snipe_actor_mask``, ``snipe_target_mask``, ``unload_actor_mask``,
``cloak_actor_mask``, ``uncloak_actor_mask``, ``enemy_mask``,
``tactical_enemy_mask``, ``engage_enemy_mask``) rather than in the flat action
mask, because the flat mask is built by the env before the command has been
sampled.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, NamedTuple, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import Schedule

from src.agent.maskable_recurrent_ppo import MaskableRecurrentActorCriticPolicy

from src.agent.models.spatial_decoder import SpatialDecoder, tactical_center_spatial_bias
from src.agent.models.squad_network import SquadNetwork
from src.agent.squad_constants import (
    CMD_MODE_ALLY_TARGET,
    CMD_MODE_AOE,
    CMD_MODE_ATTACK_LOCAL,
    CMD_MODE_CLOAK,
    CMD_MODE_ENEMY_TARGET,
    CMD_MODE_LOAD_TARGET,
    CMD_MODE_SNIPE,
    CMD_MODE_SPATIAL,
    CMD_MODE_UNCLOAK,
    CMD_MODE_UNLOAD,
    MAX_SQUAD_ENEMIES,
    MAX_SQUAD_UNITS,
    SPATIAL_COMMAND_MODES,
    SQUAD_ACTION_COMMAND_IDX,
    SQUAD_ACTION_SPATIAL_IDX,
    SQUAD_ACTION_TARGET_IDX,
    SQUAD_MASK_COMMAND_OFFSET,
    SQUAD_NULL_TARGET_SLOT,
    SQUAD_TARGET_SLOTS,
)
from src.agent.universal_constants import (
    FIXED_UNIVERSAL_COMMAND_DIM,
    UNIVERSAL_DEFAULT_SPATIAL_IDX,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates

# Keep illegal logits finite and small enough that a rare collector/learner mask
# disagreement cannot explode ``exp(logπ − logπ_old)`` into Inf (which made
# ``train/approx_kl`` / ``rollout_log_prob_abs_err`` ~1e7–1e8 and NaN EV).
_MASK_NEG = -50.0
COMMAND_EMBED_DIM = 64


def _stabilize_action_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)


def _apply_categorical_mask(
    logits: torch.Tensor,
    mask: torch.Tensor,
    taken: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mask logits, falling back to "everything legal" for fully-masked rows.

    When *taken* is set (``evaluate_actions``), the stored class is always kept
    legal. That stops a slightly drifted obs mask (missing ``command_modes`` /
    heal target bits in a padded RNN row, etc.) from assigning the collected
    action log-prob ≈ ``_MASK_NEG`` and blowing up the PPO ratio.
    """
    any_legal = mask.any(dim=-1, keepdim=True)
    safe = torch.where(any_legal, mask, torch.ones_like(mask))
    if taken is not None:
        safe = safe.clone()
        idx = taken.long().view(-1, 1).clamp(0, safe.shape[-1] - 1)
        safe.scatter_(1, idx, True)
    return logits.masked_fill(~safe, _MASK_NEG)


def _categorical_sample(logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
    if deterministic:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits, dim=-1)
    invalid = (~torch.isfinite(probs).all(dim=-1)) | (probs.sum(dim=-1) <= 1e-12)
    if invalid.any():
        probs = torch.where(invalid.unsqueeze(-1), torch.ones_like(probs), probs)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _categorical_log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(1, actions.unsqueeze(-1).long()).squeeze(-1)


def _categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


class SquadDecoderStats(NamedTuple):
    """Joint log probability and entropy of one autoregressive decode."""

    actions: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


class TargetPointerHead(nn.Module):
    """Command-conditioned pointer over allied slots, enemy slots, and null."""

    def __init__(self, latent_dim: int, command_dim: int, embed_dim: int = 64):
        super().__init__()
        in_dim = latent_dim + command_dim
        self.query = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.null_logit = nn.Linear(in_dim, 1)
        self.scale = float(embed_dim) ** -0.5

    def forward(
        self,
        latent: torch.Tensor,
        command_embed: torch.Tensor,
        ally_embeddings: Optional[torch.Tensor],
        enemy_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = torch.cat([latent, command_embed], dim=-1)
        query = self.query(h)
        batch = latent.shape[0]

        def _logits(embs: Optional[torch.Tensor], width: int) -> torch.Tensor:
            if embs is None:
                return latent.new_zeros((batch, width))
            out = torch.einsum("be,bne->bn", query, embs) * self.scale
            if out.shape[1] == width:
                return out
            padded = latent.new_zeros((batch, width))
            n = min(width, out.shape[1])
            padded[:, :n] = out[:, :n]
            return padded

        ally = _logits(ally_embeddings, MAX_SQUAD_UNITS)
        enemy = _logits(enemy_embeddings, MAX_SQUAD_ENEMIES)
        return torch.cat([ally, enemy, self.null_logit(h)], dim=-1)


class SquadRecurrentPolicy(MaskableRecurrentActorCriticPolicy):
    """Autoregressive command → actors → target policy for squad control."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Any,
        *args,
        **kwargs,
    ):
        self.spatial_resolution = 32
        kwargs.pop("spatial_resolution", None)
        kwargs.setdefault("features_extractor_class", SquadNetwork)
        fe_kw = dict(kwargs.get("features_extractor_kwargs") or {})
        fe_kw.setdefault("features_dim", 512)
        fe_kw.setdefault("thinking_steps", 3)
        kwargs["features_extractor_kwargs"] = fe_kw
        if "lstm_hidden_size" not in kwargs:
            kwargs["lstm_hidden_size"] = 512
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.command_head = nn.Linear(latent_dim_pi, FIXED_UNIVERSAL_COMMAND_DIM)
        self.spatial_decoder = SpatialDecoder(
            latent_dim_pi,
            fused_channels=128,
            spatial_resolution=self.spatial_resolution,
        )
        # Conditioning modules for the autoregressive decode. The adapters are
        # zero-initialised so a freshly grafted universal attack checkpoint
        # behaves exactly as it did before the command conditioning existed.
        # Embedding itself is small-random (a zero embedding would give the
        # zero-initialised adapters no gradient to escape from).
        self.command_embed = nn.Embedding(FIXED_UNIVERSAL_COMMAND_DIM, COMMAND_EMBED_DIM)
        nn.init.normal_(self.command_embed.weight, mean=0.0, std=0.02)
        self.actor_cmd_adapter = nn.Linear(COMMAND_EMBED_DIM, MAX_SQUAD_UNITS)
        nn.init.zeros_(self.actor_cmd_adapter.weight)
        nn.init.zeros_(self.actor_cmd_adapter.bias)
        self.spatial_cmd_adapter = nn.Linear(COMMAND_EMBED_DIM, latent_dim_pi)
        nn.init.zeros_(self.spatial_cmd_adapter.weight)
        nn.init.zeros_(self.spatial_cmd_adapter.bias)
        self.target_pointer = TargetPointerHead(
            latent_dim_pi, COMMAND_EMBED_DIM, embed_dim=64
        )
        delattr(self, "action_net")
        if self.ortho_init:
            self.command_head.apply(partial(self.init_weights, gain=0.01))
            self.spatial_decoder.apply(partial(self.init_weights, gain=0.01))
            self.target_pointer.apply(partial(self.init_weights, gain=0.01))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _command_mask_from_flat(
        action_masks: Optional[torch.Tensor], batch: int, device
    ) -> torch.Tensor:
        """Command slice of the flat env mask (all legal when no mask is given)."""
        if action_masks is None:
            return torch.ones(
                (batch, FIXED_UNIVERSAL_COMMAND_DIM), dtype=torch.bool, device=device
            )
        masks = action_masks.reshape(batch, -1).bool()
        start = SQUAD_MASK_COMMAND_OFFSET
        stop = start + FIXED_UNIVERSAL_COMMAND_DIM
        return masks[:, start:stop]

    @staticmethod
    def _alive_mask_from_flat(
        action_masks: Optional[torch.Tensor],
        obs: Dict[str, torch.Tensor],
        batch: int,
        device,
    ) -> torch.Tensor:
        """Which actor slots may be set to 1 (env aliveness/padding contract)."""
        if action_masks is not None:
            masks = action_masks.reshape(batch, -1).bool()
            # Actor slice is [not-selected, selected] per slot; slot is usable
            # only when the "selected" option is legal.
            return masks[:, 1 : MAX_SQUAD_UNITS * 2 : 2]
        unit_mask = obs.get("unit_mask")
        if unit_mask is None:
            return torch.ones((batch, MAX_SQUAD_UNITS), dtype=torch.bool, device=device)
        return unit_mask.reshape(batch, -1)[:, :MAX_SQUAD_UNITS] > 0.5

    @staticmethod
    def _obs_bool(
        obs: Dict[str, torch.Tensor], key: str, batch: int, width: int, device
    ) -> torch.Tensor:
        value = obs.get(key)
        if value is None:
            return torch.zeros((batch, width), dtype=torch.bool, device=device)
        return value.reshape(batch, -1)[:, :width] > 0.5

    def _command_modes(
        self, obs: Dict[str, torch.Tensor], commands: torch.Tensor, batch: int, device
    ) -> torch.Tensor:
        """Terminal-target mode of each sampled command (see ``command_target_modes``)."""
        table = obs.get("command_modes")
        if table is None:
            return torch.full(
                (batch,), float(CMD_MODE_SPATIAL), device=device
            ).long()
        table = table.reshape(batch, -1)
        return table.gather(1, commands.unsqueeze(-1).long()).squeeze(-1).round().long()

    @staticmethod
    def _with_fallback(mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        """Use *fallback* on rows where *mask* has nothing legal."""
        any_legal = mask.any(dim=-1, keepdim=True)
        return torch.where(any_legal, mask, fallback)

    def _actor_mask(
        self,
        obs: Dict[str, torch.Tensor],
        modes: torch.Tensor,
        alive: torch.Tensor,
        batch: int,
        device,
    ) -> torch.Tensor:
        heal_actor = self._obs_bool(obs, "heal_actor_mask", batch, MAX_SQUAD_UNITS, device)
        load_actor = self._obs_bool(obs, "load_actor_mask", batch, MAX_SQUAD_UNITS, device)
        attack_actor = self._obs_bool(
            obs, "attack_actor_mask", batch, MAX_SQUAD_UNITS, device
        )
        aoe_actor = self._obs_bool(obs, "aoe_actor_mask", batch, MAX_SQUAD_UNITS, device)
        snipe_actor = self._obs_bool(
            obs, "snipe_actor_mask", batch, MAX_SQUAD_UNITS, device
        )
        unload_actor = self._obs_bool(
            obs, "unload_actor_mask", batch, MAX_SQUAD_UNITS, device
        )
        cloak_actor = self._obs_bool(
            obs, "cloak_actor_mask", batch, MAX_SQUAD_UNITS, device
        )
        uncloak_actor = self._obs_bool(
            obs, "uncloak_actor_mask", batch, MAX_SQUAD_UNITS, device
        )
        move_actor = self._obs_bool(obs, "move_actor_mask", batch, MAX_SQUAD_UNITS, device)
        is_ally_cmd = (modes == CMD_MODE_ALLY_TARGET).unsqueeze(-1)
        is_load_cmd = (modes == CMD_MODE_LOAD_TARGET).unsqueeze(-1)
        is_enemy_cmd = (modes == CMD_MODE_ENEMY_TARGET).unsqueeze(-1)
        is_snipe_cmd = (modes == CMD_MODE_SNIPE).unsqueeze(-1)
        is_attack_local = (modes == CMD_MODE_ATTACK_LOCAL).unsqueeze(-1)
        is_aoe_cmd = (modes == CMD_MODE_AOE).unsqueeze(-1)
        is_unload_cmd = (modes == CMD_MODE_UNLOAD).unsqueeze(-1)
        is_cloak_cmd = (modes == CMD_MODE_CLOAK).unsqueeze(-1)
        is_uncloak_cmd = (modes == CMD_MODE_UNCLOAK).unsqueeze(-1)
        is_move_cmd = (modes == CMD_MODE_SPATIAL).unsqueeze(-1)
        is_attack_cmd = is_enemy_cmd | is_attack_local
        capable = torch.where(
            is_ally_cmd,
            heal_actor,
            torch.where(
                is_load_cmd,
                load_actor,
                torch.where(
                    is_attack_cmd,
                    attack_actor,
                    torch.where(
                        is_snipe_cmd,
                        snipe_actor,
                        torch.where(
                            is_aoe_cmd,
                            aoe_actor,
                            torch.where(
                                is_unload_cmd,
                                unload_actor,
                                torch.where(
                                    is_cloak_cmd,
                                    cloak_actor,
                                    torch.where(
                                        is_uncloak_cmd,
                                        uncloak_actor,
                                        torch.where(is_move_cmd, move_actor, alive),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        return self._with_fallback(capable & alive, alive)

    def _target_mask(
        self, obs: Dict[str, torch.Tensor], modes: torch.Tensor, batch: int, device
    ) -> torch.Tensor:
        heal_target = self._obs_bool(
            obs, "heal_target_mask", batch, MAX_SQUAD_UNITS, device
        )
        load_target = self._obs_bool(
            obs, "load_target_mask", batch, MAX_SQUAD_UNITS, device
        )
        enemy = self._obs_bool(obs, "enemy_mask", batch, MAX_SQUAD_ENEMIES, device)
        snipe_target = self._obs_bool(
            obs, "snipe_target_mask", batch, MAX_SQUAD_ENEMIES, device
        )
        zeros_ally = torch.zeros_like(heal_target)
        zeros_enemy = torch.zeros_like(enemy)
        is_ally_cmd = (modes == CMD_MODE_ALLY_TARGET).unsqueeze(-1)
        is_load_cmd = (modes == CMD_MODE_LOAD_TARGET).unsqueeze(-1)
        is_enemy_cmd = (modes == CMD_MODE_ENEMY_TARGET).unsqueeze(-1)
        is_snipe_cmd = (modes == CMD_MODE_SNIPE).unsqueeze(-1)
        ally_part = torch.where(
            is_ally_cmd,
            heal_target,
            torch.where(is_load_cmd, load_target, zeros_ally),
        )
        enemy_part = torch.where(
            is_enemy_cmd,
            enemy,
            torch.where(is_snipe_cmd, snipe_target, zeros_enemy),
        )
        unit_target_cmd = is_ally_cmd | is_load_cmd | is_enemy_cmd | is_snipe_cmd
        null_part = ~unit_target_cmd
        mask = torch.cat([ally_part, enemy_part, null_part], dim=-1)
        # A unit-target command with no legal unit falls back to null so the
        # distribution stays normalised (the env then issues no order).
        null_only = torch.zeros_like(mask)
        null_only[:, SQUAD_NULL_TARGET_SLOT] = True
        return self._with_fallback(mask, null_only)

    @staticmethod
    def _spatial_mask(modes: torch.Tensor, batch: int, device) -> torch.Tensor:
        cells = 32 * 32
        spatial_modes = torch.zeros_like(modes, dtype=torch.bool)
        for mode in SPATIAL_COMMAND_MODES:
            spatial_modes = spatial_modes | (modes == mode)
        is_spatial = spatial_modes.unsqueeze(-1)
        all_cells = torch.ones((batch, cells), dtype=torch.bool, device=device)
        center_only = torch.zeros((batch, cells), dtype=torch.bool, device=device)
        center_only[:, UNIVERSAL_DEFAULT_SPATIAL_IDX] = True
        return torch.where(is_spatial, all_cells, center_only)

    # ------------------------------------------------------------------
    # Autoregressive decode
    # ------------------------------------------------------------------

    def _decode(
        self,
        latent_pi: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        action_masks: Optional[torch.Tensor],
        *,
        actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> SquadDecoderStats:
        """Sample (or score) one squad action and return its joint statistics."""
        batch = latent_pi.shape[0]
        device = latent_pi.device
        fe = self.features_extractor
        if not isinstance(obs, dict):
            obs = {}
        given = actions is not None
        if given:
            actions = actions.reshape(batch, -1).long()

        # --- 1. command ---
        cmd_logits = _stabilize_action_logits(self.command_head(latent_pi))
        cmd_mask = self._command_mask_from_flat(action_masks, batch, device)
        if given:
            commands = actions[:, SQUAD_ACTION_COMMAND_IDX]
            cmd_logits = _apply_categorical_mask(cmd_logits, cmd_mask, taken=commands)
        else:
            cmd_logits = _apply_categorical_mask(cmd_logits, cmd_mask)
            commands = _categorical_sample(cmd_logits, deterministic)
        log_prob = _categorical_log_prob(cmd_logits, commands)
        entropy = _categorical_entropy(cmd_logits)

        cmd_embed = self.command_embed(commands.clamp(0, FIXED_UNIVERSAL_COMMAND_DIM - 1))
        modes = self._command_modes(obs, commands, batch, device)

        # --- 2. actors (command-conditioned binary pointer per padded slot) ---
        base_pointer = getattr(fe, "_pointer_logits", None)
        if base_pointer is None:
            base_pointer = latent_pi.new_zeros((batch, MAX_SQUAD_UNITS))
        actor_logits = _stabilize_action_logits(
            base_pointer + self.actor_cmd_adapter(cmd_embed)
        )
        alive = self._alive_mask_from_flat(action_masks, obs, batch, device)
        actor_mask = self._actor_mask(obs, modes, alive, batch, device)
        logp_on = F.logsigmoid(actor_logits)
        logp_off = F.logsigmoid(-actor_logits)
        if given:
            # Score the *stored* bits. Do not re-AND with actor_mask — that
            # rewrites the action if heal/load legality drifted vs collection.
            bits = actions[:, :MAX_SQUAD_UNITS] > 0
        elif deterministic:
            bits = (actor_logits > 0.0) & actor_mask
        else:
            bits = (torch.rand_like(actor_logits) < torch.sigmoid(actor_logits)) & actor_mask
        # Sampling does ``bernoulli & actor_mask``, so a slot the command
        # forbids is bit 0 with probability 1. Charging ``log(1-sigmoid)``
        # there is wrong: a strong pointer (logit ~ +20) makes that term ~-20
        # per forbidden alive unit, the stored log-prob falls to about -150,
        # and a tiny float difference between collector and learner exceeds
        # the replay threshold so every PPO update is skipped.
        stochastic = alive & actor_mask
        if given:
            # A stored ON bit that this row's mask dropped was not deterministic.
            scored = stochastic | bits
        else:
            scored = stochastic
        actor_weight = scored.to(actor_logits.dtype)
        slot_logp = torch.where(bits, logp_on, logp_off) * actor_weight
        p_on = torch.sigmoid(actor_logits)
        slot_entropy = (
            -(p_on * logp_on + (1.0 - p_on) * logp_off)
            * stochastic.to(actor_logits.dtype)
        )
        log_prob = log_prob + slot_logp.sum(dim=-1)
        entropy = entropy + slot_entropy.sum(dim=-1)

        # --- 3a. terminal unit target ---
        target_logits = _stabilize_action_logits(
            self.target_pointer(
                latent_pi,
                cmd_embed,
                getattr(fe, "_ally_embeddings", None),
                getattr(fe, "_enemy_embeddings", None),
            )
        )
        target_mask = self._target_mask(obs, modes, batch, device)
        if given:
            targets = actions[:, SQUAD_ACTION_TARGET_IDX].clamp(0, SQUAD_TARGET_SLOTS - 1)
            target_logits = _apply_categorical_mask(
                target_logits, target_mask, taken=targets
            )
        else:
            target_logits = _apply_categorical_mask(target_logits, target_mask)
            targets = _categorical_sample(target_logits, deterministic)
        log_prob = log_prob + _categorical_log_prob(target_logits, targets)
        entropy = entropy + _categorical_entropy(target_logits)

        # --- 3b. terminal world cell ---
        fused = getattr(fe, "_spatial_fused", None)
        if fused is None:
            fused = latent_pi.new_zeros((batch, 128, 16, 16))
        spatial_latent = latent_pi + self.spatial_cmd_adapter(cmd_embed)
        spatial_logits = self.spatial_decoder(spatial_latent, fused)
        spatial_logits = spatial_logits + tactical_center_spatial_bias(
            self.spatial_resolution, latent_pi
        ).unsqueeze(0)
        spatial_logits = _stabilize_action_logits(spatial_logits)
        spatial_mask = self._spatial_mask(modes, batch, device)
        if given:
            cells = actions[:, SQUAD_ACTION_SPATIAL_IDX].clamp(
                0, self.spatial_resolution ** 2 - 1
            )
            spatial_logits = _apply_categorical_mask(
                spatial_logits, spatial_mask, taken=cells
            )
        else:
            spatial_logits = _apply_categorical_mask(spatial_logits, spatial_mask)
            cells = _categorical_sample(spatial_logits, deterministic)
        log_prob = log_prob + _categorical_log_prob(spatial_logits, cells)
        entropy = entropy + _categorical_entropy(spatial_logits)

        if given:
            out_actions = actions
        else:
            out_actions = torch.cat(
                [
                    bits.long(),
                    commands.unsqueeze(-1),
                    targets.unsqueeze(-1),
                    cells.unsqueeze(-1),
                ],
                dim=-1,
            )
        return SquadDecoderStats(
            actions=out_actions, log_prob=log_prob, entropy=entropy
        )

    # ------------------------------------------------------------------
    # Policy API
    # ------------------------------------------------------------------

    def forward(self, obs, lstm_states, episode_starts, action_masks=None, deterministic=False):
        features = self.extract_features(obs)
        pi_f, vf_f = (features, features) if self.share_features_extractor else features
        l_pi, s_pi = self._process_sequence(pi_f, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic:
            l_vf, s_vf = self._process_sequence(vf_f, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            l_vf, s_vf = l_pi.detach(), tuple(s.detach() for s in s_pi)
        else:
            l_vf, s_vf = self.critic(vf_f), s_pi

        l_pi = self.mlp_extractor.forward_actor(l_pi)
        l_vf = self.mlp_extractor.forward_critic(l_vf)
        values = self.value_net(l_vf)
        stats = self._decode(l_pi, obs, action_masks, deterministic=deterministic)
        return stats.actions, values, stats.log_prob, RNNStates(s_pi, s_vf)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts, action_masks=None):
        features = self.extract_features(obs)
        pi_f, vf_f = (features, features) if self.share_features_extractor else features
        l_pi, _ = self._process_sequence(pi_f, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic:
            l_vf = self._process_sequence(
                vf_f, lstm_states.vf, episode_starts, self.lstm_critic
            )[0]
        elif self.shared_lstm:
            l_vf = l_pi.detach()
        else:
            l_vf = self.critic(vf_f)
        l_pi = self.mlp_extractor.forward_actor(l_pi)
        l_vf = self.mlp_extractor.forward_critic(l_vf)
        stats = self._decode(l_pi, obs, action_masks, actions=actions)
        return self.value_net(l_vf), stats.log_prob, stats.entropy

    def get_distribution(self, obs, lstm_states, episode_starts, action_masks=None):
        raise NotImplementedError(
            "SquadRecurrentPolicy decodes command → actors → target autoregressively; "
            "there is no single flat distribution object. Use forward() to sample and "
            "evaluate_actions() to score stored actions."
        )
