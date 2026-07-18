"""PPO trainer with support for action masking and LSTM policies."""

import copy
import random
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TrainingConfig
from ..env.actions import (
    NUM_SLOT_INTENT_COMPONENTS,
    NUM_TERA_INTENT_COMPONENTS,
    SLOT_INTENT_KIND,
    SLOT_INTENT_MOVE,
    SLOT_INTENT_SWITCH_SPECIES,
    SLOT_INTENT_TARGET,
    TERA_INTENT_SLOT,
    TERA_INTENT_USE,
)
from ..env.observation import (
    ABILITY_KNOWN_OFFSET,
    ITEM_KNOWN_OFFSET,
    NUM_PUBLIC_TARGET_POSITIONS,
    ROLE_BENCH,
    TERA_KNOWN_OFFSET,
)
from ..field_transition import (
    FIELD_CHANGE_COMPONENT_DIM,
    SCALAR_FIELD_INDICES,
    TERRAIN_CLASS_SLICE,
    WEATHER_CLASS_SLICE,
    advanced_public_field_persistence,
    public_field_change_targets,
)
from ..model.agent import PokeAgent
from ..model.checkpoint import (
    model_checkpoint_metadata,
    validate_model_checkpoint_metadata,
)
from ..public_consequence import (
    MAX_PUBLIC_HIT_COUNT,
    NUM_MOVE_BRANCHES,
    NUM_NONMOVE_REASONS,
    NUM_TARGET_INTERACTION_REASONS,
    PUBLIC_CONSEQUENCE_FIELD_DIM,
)
from .aux_metrics import (
    hidden_info_belief_additive_metrics,
    opponent_intent_additive_metrics,
    public_consequence_additive_metrics,
)
from .rollout import RolloutBatch


# Increment when trainer-checkpoint semantics or same-shaped model inputs change.
TRAINER_CHECKPOINT_FORMAT_VERSION = 7


_PUBLIC_CONSEQUENCE_LOGIT_NAMES = (
    "move_emitted_logits",
    "nonmove_reason_logits",
    "move_branch_logits",
    "route_logits",
    "interaction_logits",
    "interaction_reason_logits",
    "hit_count_logits",
    "cant_logits",
    "field_logits",
    "field_change_logits",
    "field_alternative_logits",
)


def _copy_initialization_provenance(
    checkpoint: Mapping[str, object],
) -> dict[str, object] | None:
    provenance = checkpoint.get("initialization_provenance")
    if provenance is None:
        return None
    if not isinstance(provenance, Mapping):
        raise ValueError("Checkpoint initialization_provenance must be a mapping.")
    return copy.deepcopy(dict(provenance))


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(rng_state: Mapping[str, object]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(rng_state))
    if missing:
        raise ValueError(
            "Trainer checkpoint RNG state is missing key(s): " + ", ".join(missing)
        )

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])

    torch_cpu_state = rng_state["torch_cpu"]
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise ValueError("Trainer checkpoint torch CPU RNG state must be a tensor.")
    torch.set_rng_state(torch_cpu_state.detach().to(device="cpu"))

    cuda_states = rng_state["torch_cuda"]
    if cuda_states is None or not torch.cuda.is_available():
        return
    if not isinstance(cuda_states, (tuple, list)) or not all(
        isinstance(state, torch.Tensor) for state in cuda_states
    ):
        raise ValueError("Trainer checkpoint torch CUDA RNG state must be tensor list.")
    device_count = torch.cuda.device_count()
    if len(cuda_states) != device_count:
        raise ValueError(
            "Trainer checkpoint CUDA RNG state count does not match this runtime: "
            f"checkpoint={len(cuda_states)}, runtime={device_count}."
        )
    torch.cuda.set_rng_state_all(
        [state.detach().to(device="cpu") for state in cuda_states]
    )


def _validate_restorable_rng_state(rng_state: object) -> Mapping[str, object]:
    if not isinstance(rng_state, Mapping):
        raise ValueError("Trainer checkpoint RNG state must be a mapping.")
    current_state = _capture_rng_state()
    try:
        _restore_rng_state(rng_state)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Trainer checkpoint RNG state is invalid: {exc}") from exc
    finally:
        _restore_rng_state(current_state)
    return rng_state


class PPOTrainer:
    """Proximal Policy Optimization with clipped surrogate objective."""

    _LOG_RATIO_CLAMP = 20.0
    _HEAD_BATCH_TARGET = 1024

    def __init__(self, agent: PokeAgent, config: TrainingConfig, device: torch.device):
        self.agent = agent.to(device)
        self.config = config
        self.device = device
        self.optimizer = self._make_optimizer()
        self._trainable_params = [
            param for param in agent.parameters() if param.requires_grad
        ]
        memory = getattr(agent, "memory", None)
        self._memory_params = (
            tuple(param for param in memory.parameters() if param.requires_grad)
            if memory is not None
            else ()
        )
        opponent_representation = getattr(
            getattr(agent, "spatial", None),
            "team_opp",
            None,
        )
        self._opponent_representation_params = (
            tuple(
                param
                for param in opponent_representation.parameters()
                if param.requires_grad
            )
            if opponent_representation is not None
            else ()
        )
        self.global_step = 0
        self.completed_updates = 0
        self.loaded_training_state_id: str | None = None
        self.initialized_from_training_state_id: str | None = None
        self.initialization_provenance: dict[str, object] | None = None

    def _make_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.agent.parameters(), lr=self.config.lr, eps=1e-5)

    def _apply_update_step(self, loss: torch.Tensor) -> bool:
        if not torch.isfinite(loss).item():
            self.optimizer.zero_grad(set_to_none=True)
            return False

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        total_norm = nn.utils.clip_grad_norm_(
            self._trainable_params, self.config.max_grad_norm
        )
        if not torch.isfinite(total_norm).item():
            self.optimizer.zero_grad(set_to_none=True)
            return False

        self.optimizer.step()
        return True

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """Run PPO update epochs on a rollout batch.

        Minibatches are slices of environments, not random timesteps, so that
        the LSTM can be unrolled through the full rollout sequence with correct
        gradient flow across episode boundaries (hidden state is zeroed on done).

        Returns a dict of loss metrics for logging.
        """
        T = batch.num_steps
        E = batch.num_envs
        if self.config.num_minibatches > E:
            raise ValueError(
                f"num_minibatches ({self.config.num_minibatches}) cannot exceed num_envs ({E}); "
                "reduce num_minibatches."
            )
        batch_device = next(iter(batch.obs.values())).device
        if batch_device.type != self.device.type:
            raise ValueError(
                f"RolloutBatch is on {batch_device}, but PPOTrainer is on {self.device}. "
                "Collect rollouts on the trainer device."
            )
        if self.device.index is not None and batch_device.index != self.device.index:
            raise ValueError(
                f"RolloutBatch is on {batch_device}, but PPOTrainer is on {self.device}. "
                "Collect rollouts on the trainer device."
            )

        # Normalize advantages over the full batch once, before the update loop.
        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_seq = batch.obs
        actions_seq = batch.actions  # (T, E, 3)
        masks_seq = batch.action_masks
        old_lp_seq = batch.log_probs  # (T, E)
        adv_seq = adv  # (T, E)
        returns_seq = batch.returns  # (T, E)
        values_old_seq = batch.values  # (T, E)
        dones_seq = batch.dones  # (T, E)
        dones_seq_cpu = dones_seq.detach().to(device="cpu")

        # Per-env initial hidden states at the start of the rollout.
        if batch.initial_hidden_h is not None:
            init_h_all = batch.initial_hidden_h
            init_c_all = batch.initial_hidden_c
        else:
            init_h_all = None
            init_c_all = None

        metric_sums = {
            "policy_loss": torch.zeros((), device=self.device),
            "value_loss": torch.zeros((), device=self.device),
            "entropy": torch.zeros((), device=self.device),
            "action_entropy": torch.zeros((), device=self.device),
            "approx_kl": torch.zeros((), device=self.device),
            "clip_fraction": torch.zeros((), device=self.device),
            "value_clip_fraction": torch.zeros((), device=self.device),
        }
        max_approx_kl = torch.zeros((), device=self.device)
        base_tera_entropy_sum = torch.zeros((), device=self.device)
        base_tera_gate_entropy_sum = torch.zeros((), device=self.device)
        base_tera_slot_entropy_sum = torch.zeros((), device=self.device)
        base_tera_use_prob_sum = torch.zeros((), device=self.device)
        base_tera_none_prob_sum = torch.zeros((), device=self.device)
        base_tera_real_choice_evaluations = torch.zeros((), device=self.device)
        base_tera_slot_choice_evaluations = torch.zeros((), device=self.device)
        num_updates = 0
        skipped_nonfinite = 0
        memory_gradient_diagnostics: dict[str, float] = {}
        aux_diagnostic_sums: dict[str, float] = {}
        initial_policy_shift: dict[str, torch.Tensor] = {}

        aux_labels_all = batch.aux_labels
        aux_label_valid_all = batch.aux_label_valid
        hidden_info_coverage_metrics = self._hidden_info_coverage_metrics(
            aux_label_valid_all,
            obs_seq.get("opp_mask"),
        )
        opponent_intent_slot_labels_all = batch.opponent_intent_slot_labels
        opponent_intent_slot_masks_all = batch.opponent_intent_slot_masks
        opponent_intent_tera_labels_all = batch.opponent_intent_tera_labels
        opponent_intent_tera_masks_all = batch.opponent_intent_tera_masks
        public_consequence_structured_all = {
            name: getattr(batch, f"public_consequence_{name}")
            for name in (
                "move_emitted_targets",
                "move_emitted_masks",
                "nonmove_reason_labels",
                "nonmove_reason_masks",
                "move_branch_labels",
                "move_branch_masks",
                "route_targets",
                "route_masks",
                "interaction_targets",
                "interaction_masks",
                "interaction_reason_labels",
                "interaction_reason_masks",
                "hit_count_labels",
                "hit_count_masks",
            )
        }
        public_consequence_cant_targets_all = batch.public_consequence_cant_targets
        public_consequence_cant_masks_all = batch.public_consequence_cant_masks
        public_consequence_field_targets_all = batch.public_consequence_field_targets
        public_consequence_field_masks_all = batch.public_consequence_field_masks
        hidden_info_enabled = (
            self.config.aux_hidden_info_coef != 0.0
            and bool(aux_labels_all)
            and hasattr(self.agent, "hidden_info_head")
            and hidden_info_coverage_metrics["aux_hidden_info_valid_selected_slots"] > 0
        )
        opponent_intent_enabled = (
            self.config.aux_opponent_intent_coef != 0.0
            and hasattr(self.agent, "opponent_intent_head")
            and bool(
                opponent_intent_slot_masks_all.any().item()
                or opponent_intent_tera_masks_all.any().item()
            )
        )
        public_consequence_enabled = (
            self.config.aux_public_consequence_coef != 0.0
            and hasattr(
                getattr(self.agent, "action_head", None),
                "public_consequence_action_head",
            )
            and bool(
                public_consequence_structured_all["move_emitted_masks"].any().item()
                or public_consequence_cant_masks_all.any().item()
                or public_consequence_field_masks_all.any().item()
            )
        )
        if hidden_info_enabled:
            metric_sums["aux_hidden_info_loss"] = torch.zeros((), device=self.device)
        if opponent_intent_enabled:
            metric_sums["aux_opponent_intent_loss"] = torch.zeros(
                (),
                device=self.device,
            )
        if public_consequence_enabled:
            metric_sums["aux_public_consequence_loss"] = torch.zeros(
                (),
                device=self.device,
            )

        for epoch in range(self.config.num_epochs):
            env_order_cpu = torch.randperm(E, device="cpu")

            for mb_envs_cpu in torch.tensor_split(
                env_order_cpu,
                self.config.num_minibatches,
            ):
                if mb_envs_cpu.numel() == 0:
                    continue
                mb_envs = mb_envs_cpu.to(device=self.device)
                E_mb = mb_envs.numel()

                mb_obs_seq = {k: v[:, mb_envs] for k, v in obs_seq.items()}
                mb_actions_seq = actions_seq[:, mb_envs]  # (T, E_mb, 3)
                mb_masks_seq = {
                    key: value[:, mb_envs] for key, value in masks_seq.items()
                }
                mb_dones_seq_cpu = dones_seq_cpu[:, mb_envs_cpu]
                mb_old_lp = old_lp_seq[:, mb_envs].reshape(-1)
                mb_adv = adv_seq[:, mb_envs].reshape(-1)
                mb_returns = returns_seq[:, mb_envs].reshape(-1)
                mb_values_old = values_old_seq[:, mb_envs].reshape(-1)

                # Initial hidden for this env subset.
                if init_h_all is not None:
                    hidden: tuple | None = (
                        init_h_all[mb_envs].unsqueeze(0),
                        init_c_all[mb_envs].unsqueeze(0),
                    )
                else:
                    hidden = None

                # Spatial encoder: process all T*E_mb pokemon in one batched pass
                # (heavy attention layers run once; only the lightweight LSTM loops).
                flat_obs = {
                    k: v.reshape(T * E_mb, *v.shape[2:]) for k, v in mb_obs_seq.items()
                }
                (
                    context_flat,
                    active_reps_flat,
                    own_reps_flat,
                    opp_reps_flat,
                    opp_active_reps_flat,
                ) = self.agent.spatial(flat_obs)
                context_seq_mb = context_flat.reshape(T, E_mb, -1)
                active_reps_seq_mb = active_reps_flat.reshape(T, E_mb, 2, -1)
                own_reps_seq_mb = own_reps_flat.reshape(
                    T, E_mb, own_reps_flat.shape[1], -1
                )
                opp_reps_seq_mb = opp_reps_flat.reshape(
                    T, E_mb, opp_reps_flat.shape[1], -1
                )
                opp_active_reps_seq_mb = opp_active_reps_flat.reshape(T, E_mb, 2, -1)
                own_bench_mask_seq_mb = (
                    mb_obs_seq["own_roles"] != ROLE_BENCH
                ) | mb_obs_seq["own_mask"]
                opp_bench_mask_seq_mb = (
                    mb_obs_seq["opp_roles"] != ROLE_BENCH
                ) | mb_obs_seq["opp_mask"]

                # Pack independent episode fragments so recurrent training takes
                # one LSTM launch per explicit TBPTT window rather than one per
                # timestep. Terminal resets and detach boundaries are unchanged.
                ctx_seq_mb, _ = self.agent.memory.unroll(
                    context_seq_mb,
                    mb_dones_seq_cpu,
                    hidden,
                    bptt_chunk_steps=self.config.bptt_chunk_steps,
                )
                (
                    belief_opp_reps_seq_mb,
                    belief_opp_active_reps_seq_mb,
                ) = self.agent.condition_opponent_reps(
                    opp_reps_seq_mb,
                    opp_active_reps_seq_mb,
                    ctx_seq_mb,
                )

                # The action and value heads are independent across timesteps.
                # Chunking avoids many tiny GPU launches without evaluating the
                # whole rollout minibatch at once.
                all_log_probs: list[torch.Tensor] = []
                all_entropies: list[torch.Tensor] = []
                all_action_entropies: list[torch.Tensor] = []
                all_base_tera_entropies: list[torch.Tensor] = []
                all_base_tera_gate_entropies: list[torch.Tensor] = []
                all_base_tera_slot_entropies: list[torch.Tensor] = []
                all_base_tera_use_probs: list[torch.Tensor] = []
                all_base_tera_none_probs: list[torch.Tensor] = []
                all_base_tera_real_choices: list[torch.Tensor] = []
                all_base_tera_slot_choices: list[torch.Tensor] = []
                all_values: list[torch.Tensor] = []
                all_public_consequence_logits: dict[str, list[torch.Tensor]] = {
                    name: [] for name in _PUBLIC_CONSEQUENCE_LOGIT_NAMES
                }
                head_chunk_steps = max(1, min(T, self._HEAD_BATCH_TARGET // E_mb))
                for start in range(0, T, head_chunk_steps):
                    end = min(start + head_chunk_steps, T)
                    flat_count = (end - start) * E_mb
                    ctx_flat = ctx_seq_mb[start:end].reshape(flat_count, -1)
                    own_reps_flat = own_reps_seq_mb[start:end].reshape(
                        flat_count,
                        own_reps_seq_mb.shape[2],
                        own_reps_seq_mb.shape[-1],
                    )
                    opp_reps_flat = belief_opp_reps_seq_mb[start:end].reshape(
                        flat_count,
                        belief_opp_reps_seq_mb.shape[2],
                        belief_opp_reps_seq_mb.shape[-1],
                    )
                    opp_bench_mask_flat = opp_bench_mask_seq_mb[start:end].reshape(
                        flat_count,
                        opp_bench_mask_seq_mb.shape[-1],
                    )

                    action_out = self.agent.action_head(
                        ctx_flat,
                        active_reps_seq_mb[start:end].reshape(
                            flat_count,
                            2,
                            active_reps_seq_mb.shape[-1],
                        ),
                        own_reps_flat,
                        opp_reps_flat,
                        opp_bench_mask_flat,
                        belief_opp_active_reps_seq_mb[start:end].reshape(
                            flat_count,
                            2,
                            belief_opp_active_reps_seq_mb.shape[-1],
                        ),
                        {
                            key: value[start:end].reshape(flat_count, *value.shape[2:])
                            for key, value in mb_masks_seq.items()
                        },
                        mb_actions_seq[start:end].reshape(flat_count, 3),
                        battle_obs={
                            key: value[start:end].reshape(flat_count, *value.shape[2:])
                            for key, value in mb_obs_seq.items()
                        },
                        tera_exploration_prob=batch.tera_exploration_prob,
                        validate_actions=False,
                        move_embedding=getattr(
                            getattr(self.agent.spatial, "pokemon_encoder", None),
                            "move_emb",
                            None,
                        ),
                    )
                    value_t = self.agent.value_head(
                        ctx_flat,
                        own_reps_flat,
                        opp_reps_flat,
                        own_bench_mask_seq_mb[start:end].reshape(
                            flat_count,
                            own_bench_mask_seq_mb.shape[-1],
                        ),
                        opp_bench_mask_flat,
                    )

                    all_log_probs.append(action_out["log_prob"])
                    all_entropies.append(action_out["entropy"])
                    all_action_entropies.append(action_out["action_entropy"])
                    all_base_tera_entropies.append(action_out["base_tera_entropy"])
                    all_base_tera_gate_entropies.append(
                        action_out["base_tera_gate_entropy"]
                    )
                    all_base_tera_slot_entropies.append(
                        action_out["base_tera_slot_entropy"]
                    )
                    all_base_tera_use_probs.append(action_out["base_tera_use_prob"])
                    all_base_tera_none_probs.append(action_out["base_tera_none_prob"])
                    all_base_tera_real_choices.append(
                        action_out["base_tera_has_real_choice"]
                    )
                    all_base_tera_slot_choices.append(
                        action_out["base_tera_has_slot_choice"]
                    )
                    if public_consequence_enabled:
                        for name, chunks in all_public_consequence_logits.items():
                            chunks.append(action_out[f"public_consequence_{name}"])
                    all_values.append(value_t)

                log_probs_flat = torch.cat(all_log_probs)
                entropies_flat = torch.cat(all_entropies)
                action_entropies_flat = torch.cat(all_action_entropies)
                base_tera_entropies_flat = torch.cat(all_base_tera_entropies)
                base_tera_gate_entropies_flat = torch.cat(all_base_tera_gate_entropies)
                base_tera_slot_entropies_flat = torch.cat(all_base_tera_slot_entropies)
                base_tera_use_probs_flat = torch.cat(all_base_tera_use_probs)
                base_tera_none_probs_flat = torch.cat(all_base_tera_none_probs)
                base_tera_real_choices_flat = torch.cat(all_base_tera_real_choices)
                base_tera_slot_choices_flat = torch.cat(all_base_tera_slot_choices)
                values_flat = torch.cat(all_values)
                public_consequence_out = None
                if public_consequence_enabled:
                    public_consequence_out = {
                        f"public_consequence_{name}": torch.cat(chunks).reshape(
                            T,
                            E_mb,
                            *chunks[0].shape[1:],
                        )
                        for name, chunks in all_public_consequence_logits.items()
                    }

                # Policy loss (clipped surrogate objective).
                log_ratio = log_probs_flat - mb_old_lp
                # Rare extreme policy shifts can overflow exp(log_ratio) in float32 and
                # poison Adam with non-finite gradients. Clamping here only affects
                # already-pathological ratios far outside PPO's trust region.
                ratio = torch.exp(
                    log_ratio.clamp(
                        min=-self._LOG_RATIO_CLAMP, max=self._LOG_RATIO_CLAMP
                    )
                )
                if not initial_policy_shift:
                    with torch.no_grad():
                        approx_kl_before_battle_update = (
                            (ratio - 1)
                            - log_ratio.clamp(
                                min=-self._LOG_RATIO_CLAMP,
                                max=self._LOG_RATIO_CLAMP,
                            )
                        ).mean()
                        initial_policy_shift = {
                            "pre_ppo_approx_kl": approx_kl_before_battle_update.detach(),
                            "pre_ppo_clip_fraction": (
                                (ratio - 1.0).abs() > self.config.clip_eps
                            )
                            .to(torch.float32)
                            .mean()
                            .detach(),
                            "pre_ppo_mean_abs_log_ratio": log_ratio.abs()
                            .mean()
                            .detach(),
                            "pre_ppo_max_abs_log_ratio": log_ratio.abs().max().detach(),
                        }
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
                    )
                    * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping (prevents large value updates).
                values_clipped = mb_values_old + torch.clamp(
                    values_flat - mb_values_old,
                    -self.config.clip_eps,
                    self.config.clip_eps,
                )
                value_loss = (
                    0.5
                    * torch.max(
                        (values_flat - mb_returns).pow(2),
                        (values_clipped - mb_returns).pow(2),
                    ).mean()
                )

                # Battle actions, the use-now gate, and the conditional slot
                # choice have independent entropy coefficients. Conditional
                # slot entropy is optimized only when both slots are eligible,
                # and therefore cannot itself pressure the gate to use Tera.
                action_entropy_loss = -action_entropies_flat.mean()
                base_tera_real_choice_weights = base_tera_real_choices_flat.to(
                    base_tera_entropies_flat.dtype
                )
                base_tera_gate_support = base_tera_real_choice_weights.sum().clamp_min(
                    1.0
                )
                tera_gate_entropy_loss = (
                    -(
                        base_tera_gate_entropies_flat * base_tera_real_choice_weights
                    ).sum()
                    / base_tera_gate_support
                )
                base_tera_slot_choice_weights = base_tera_slot_choices_flat.to(
                    base_tera_slot_entropies_flat.dtype
                )
                base_tera_slot_support = base_tera_slot_choice_weights.sum().clamp_min(
                    1.0
                )
                tera_slot_entropy_loss = (
                    -(
                        base_tera_slot_entropies_flat * base_tera_slot_choice_weights
                    ).sum()
                    / base_tera_slot_support
                )

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * action_entropy_loss
                    + self.config.tera_gate_entropy_coef * tera_gate_entropy_loss
                    + self.config.tera_slot_entropy_coef * tera_slot_entropy_loss
                )

                # --- Aux losses ---
                # Hidden info: predict exact selected-team item/ability/
                # tera-type/move-set from spatial reps plus recurrent belief
                # state. Opponent intent: predict only non-forced semantic
                # choices (kind/move/target/switch/Tera) from memory plus the
                # public teams. Public consequence: predict the next observable
                # resolution/CANT/field outcome of each chosen action. All
                # targets remain trainer-private and rewards remain terminal.
                hidden_info_loss = None
                opponent_intent_loss = None
                public_consequence_loss = None
                hidden_info_out = None
                opponent_intent_out = None
                public_consequence_persistence_mb = None
                if hidden_info_enabled:
                    aux_labels_mb = {
                        key: value[:, mb_envs] for key, value in aux_labels_all.items()
                    }
                    aux_label_valid_mb = aux_label_valid_all[:, mb_envs]
                    hidden_info_out = self.agent.hidden_info_head.decode(
                        belief_opp_reps_seq_mb,
                    )
                    hidden_info_loss = self._hidden_info_loss(
                        logits=hidden_info_out,
                        labels=aux_labels_mb,
                        label_valid=aux_label_valid_mb,
                    )
                    loss = loss + self.config.aux_hidden_info_coef * hidden_info_loss

                if opponent_intent_enabled:
                    opponent_intent_out = self.agent.opponent_intent_head(
                        ctx_seq_mb.reshape(T * E_mb, -1),
                        belief_opp_reps_seq_mb.reshape(
                            T * E_mb,
                            belief_opp_reps_seq_mb.shape[2],
                            belief_opp_reps_seq_mb.shape[-1],
                        ),
                        belief_opp_active_reps_seq_mb.reshape(T * E_mb, 2, -1),
                        active_reps_seq_mb.reshape(T * E_mb, 2, -1),
                        mb_obs_seq["opp_mask"].reshape(
                            T * E_mb,
                            mb_obs_seq["opp_mask"].shape[-1],
                        ),
                    )
                    opponent_intent_slot_labels_mb = opponent_intent_slot_labels_all[
                        :, mb_envs
                    ]
                    opponent_intent_slot_masks_mb = opponent_intent_slot_masks_all[
                        :, mb_envs
                    ]
                    opponent_intent_tera_labels_mb = opponent_intent_tera_labels_all[
                        :, mb_envs
                    ]
                    opponent_intent_tera_masks_mb = opponent_intent_tera_masks_all[
                        :, mb_envs
                    ]
                    opponent_intent_loss = self._opponent_intent_loss(
                        logits=opponent_intent_out,
                        slot_labels=opponent_intent_slot_labels_mb,
                        slot_masks=opponent_intent_slot_masks_mb,
                        tera_labels=opponent_intent_tera_labels_mb,
                        tera_masks=opponent_intent_tera_masks_mb,
                    )
                    loss = (
                        loss
                        + self.config.aux_opponent_intent_coef * opponent_intent_loss
                    )

                if public_consequence_enabled:
                    public_consequence_structured_mb = {
                        name: value[:, mb_envs]
                        for name, value in public_consequence_structured_all.items()
                    }
                    public_consequence_cant_targets_mb = (
                        public_consequence_cant_targets_all[:, mb_envs]
                    )
                    public_consequence_cant_masks_mb = (
                        public_consequence_cant_masks_all[:, mb_envs]
                    )
                    public_consequence_field_targets_mb = (
                        public_consequence_field_targets_all[:, mb_envs]
                    )
                    public_consequence_field_masks_mb = (
                        public_consequence_field_masks_all[:, mb_envs]
                    )
                    public_consequence_persistence_mb = (
                        advanced_public_field_persistence(
                            mb_obs_seq["field"][..., :PUBLIC_CONSEQUENCE_FIELD_DIM]
                        )
                        .unsqueeze(-2)
                        .expand(*public_consequence_field_targets_mb.shape)
                    )
                    public_consequence_loss = self._public_consequence_loss(
                        logits=public_consequence_out,
                        **public_consequence_structured_mb,
                        cant_targets=public_consequence_cant_targets_mb,
                        cant_masks=public_consequence_cant_masks_mb,
                        field_targets=public_consequence_field_targets_mb,
                        field_masks=public_consequence_field_masks_mb,
                        persistence_fields=public_consequence_persistence_mb,
                    )
                    loss = (
                        loss
                        + self.config.aux_public_consequence_coef
                        * public_consequence_loss
                    )

                pending_aux_diagnostics: dict[str, float] = {}
                if epoch == self.config.num_epochs - 1:
                    if hidden_info_out is not None:
                        pending_aux_diagnostics.update(
                            hidden_info_belief_additive_metrics(
                                hidden_info_out,
                                aux_labels_mb,
                                aux_label_valid_mb,
                                mb_obs_seq,
                            )
                        )
                    if opponent_intent_out is not None:
                        pending_aux_diagnostics.update(
                            opponent_intent_additive_metrics(
                                {
                                    key: value.reshape(T, E_mb, *value.shape[1:])
                                    for key, value in opponent_intent_out.items()
                                },
                                opponent_intent_slot_labels_mb,
                                opponent_intent_slot_masks_mb,
                                opponent_intent_tera_labels_mb,
                                opponent_intent_tera_masks_mb,
                            )
                        )
                    if public_consequence_out is not None:
                        pending_aux_diagnostics.update(
                            public_consequence_additive_metrics(
                                logits=public_consequence_out,
                                **public_consequence_structured_mb,
                                cant_targets=public_consequence_cant_targets_mb,
                                cant_masks=public_consequence_cant_masks_mb,
                                field_targets=public_consequence_field_targets_mb,
                                field_masks=public_consequence_field_masks_mb,
                                persistence_fields=public_consequence_persistence_mb,
                            )
                        )

                pending_memory_diagnostics = None
                if (
                    self.config.diagnose_memory_gradients
                    and not memory_gradient_diagnostics
                    and self._memory_params
                ):
                    support_tensors = []
                    support_names = []
                    if hidden_info_loss is not None:
                        support_names.append("memory_grad_hidden_info_valid_slots")
                        support_tensors.append(aux_label_valid_mb.sum())
                    if opponent_intent_loss is not None:
                        support_names.append(
                            "memory_grad_opponent_intent_valid_components"
                        )
                        support_tensors.append(
                            opponent_intent_slot_masks_mb.sum()
                            + opponent_intent_tera_masks_mb.sum()
                        )
                    if public_consequence_loss is not None:
                        support_names.append(
                            "memory_grad_public_consequence_valid_components"
                        )
                        support_tensors.append(
                            sum(
                                value.sum()
                                for name, value in public_consequence_structured_mb.items()
                                if name.endswith("_masks")
                            )
                            + public_consequence_cant_masks_mb.sum()
                            + public_consequence_field_masks_mb.sum()
                        )
                    support_values = (
                        torch.stack(support_tensors).detach().to(device="cpu").tolist()
                        if support_tensors
                        else []
                    )
                    support = dict(zip(support_names, support_values, strict=True))
                    objectives = {
                        "actor": (
                            policy_loss
                            + self.config.entropy_coef * action_entropy_loss
                            + self.config.tera_gate_entropy_coef
                            * tera_gate_entropy_loss
                            + self.config.tera_slot_entropy_coef
                            * tera_slot_entropy_loss
                        ),
                        "value": self.config.value_coef * value_loss,
                    }
                    if hidden_info_loss is not None:
                        objectives["hidden_info"] = (
                            self.config.aux_hidden_info_coef * hidden_info_loss
                        )
                    if opponent_intent_loss is not None:
                        objectives["opponent_intent"] = (
                            self.config.aux_opponent_intent_coef * opponent_intent_loss
                        )
                    if public_consequence_loss is not None:
                        objectives["public_consequence"] = (
                            self.config.aux_public_consequence_coef
                            * public_consequence_loss
                        )
                    if all(value > 0 for value in support.values()):
                        pending_memory_diagnostics = self._gradient_diagnostics(
                            objectives,
                            self._memory_params,
                            prefix="memory_grad",
                        )
                        if self._opponent_representation_params:
                            pending_memory_diagnostics.update(
                                self._gradient_diagnostics(
                                    objectives,
                                    self._opponent_representation_params,
                                    prefix="opp_rep_grad",
                                )
                            )
                        pending_memory_diagnostics.update(support)
                        pending_memory_diagnostics["memory_grad_minibatch_envs"] = (
                            float(E_mb)
                        )

                if not self._apply_update_step(loss):
                    skipped_nonfinite += 1
                    continue
                if pending_memory_diagnostics is not None:
                    memory_gradient_diagnostics = pending_memory_diagnostics
                for key, value in pending_aux_diagnostics.items():
                    aux_diagnostic_sums[key] = aux_diagnostic_sums.get(key, 0.0) + value

                with torch.no_grad():
                    approx_kl = (
                        (ratio - 1)
                        - log_ratio.clamp(
                            min=-self._LOG_RATIO_CLAMP,
                            max=self._LOG_RATIO_CLAMP,
                        )
                    ).mean()
                    metric_sums["policy_loss"].add_(policy_loss.detach())
                    metric_sums["value_loss"].add_(value_loss.detach())
                    metric_sums["entropy"].add_(entropies_flat.mean().detach())
                    metric_sums["action_entropy"].add_((-action_entropy_loss).detach())
                    base_tera_entropy_sum.add_(
                        base_tera_entropies_flat[base_tera_real_choices_flat]
                        .sum()
                        .detach()
                    )
                    base_tera_gate_entropy_sum.add_(
                        base_tera_gate_entropies_flat[base_tera_real_choices_flat]
                        .sum()
                        .detach()
                    )
                    base_tera_slot_entropy_sum.add_(
                        base_tera_slot_entropies_flat[base_tera_slot_choices_flat]
                        .sum()
                        .detach()
                    )
                    base_tera_use_prob_sum.add_(
                        base_tera_use_probs_flat[base_tera_real_choices_flat]
                        .sum()
                        .detach()
                    )
                    base_tera_none_prob_sum.add_(
                        base_tera_none_probs_flat[base_tera_real_choices_flat]
                        .sum()
                        .detach()
                    )
                    base_tera_real_choice_evaluations.add_(
                        base_tera_real_choices_flat.sum().detach()
                    )
                    base_tera_slot_choice_evaluations.add_(
                        base_tera_slot_choices_flat.sum().detach()
                    )
                    metric_sums["approx_kl"].add_(approx_kl)
                    max_approx_kl = torch.maximum(max_approx_kl, approx_kl)
                    metric_sums["clip_fraction"].add_(
                        ((ratio - 1.0).abs() > self.config.clip_eps)
                        .to(torch.float32)
                        .mean()
                    )
                    metric_sums["value_clip_fraction"].add_(
                        ((values_flat - mb_values_old).abs() > self.config.clip_eps)
                        .to(torch.float32)
                        .mean()
                    )
                    if hidden_info_enabled:
                        metric_sums["aux_hidden_info_loss"].add_(
                            hidden_info_loss.detach()
                        )
                    if opponent_intent_enabled:
                        metric_sums["aux_opponent_intent_loss"].add_(
                            opponent_intent_loss.detach()
                        )
                    if public_consequence_enabled:
                        metric_sums["aux_public_consequence_loss"].add_(
                            public_consequence_loss.detach()
                        )
                num_updates += 1

        if num_updates == 0:
            raise RuntimeError(
                "PPO update skipped every minibatch because the loss or gradients became non-finite."
            )

        self.global_step += T * E
        averaged = {
            key: (value / num_updates).item() for key, value in metric_sums.items()
        }
        averaged.update(
            {key: value.item() for key, value in initial_policy_shift.items()}
        )
        averaged["max_approx_kl"] = max_approx_kl.item()
        returns_variance = returns_seq.var(unbiased=False)
        averaged["value_explained_variance"] = (
            (
                1.0
                - (returns_seq - values_old_seq).var(unbiased=False)
                / returns_variance.clamp_min(1e-8)
            ).item()
            if returns_variance.item() > 0.0
            else 0.0
        )
        base_tera_support = base_tera_real_choice_evaluations.item()
        averaged["base_tera_real_choice_evaluations"] = base_tera_support
        averaged["base_tera_entropy"] = (
            base_tera_entropy_sum.item() / base_tera_support
            if base_tera_support > 0
            else 0.0
        )
        averaged["base_tera_gate_entropy"] = (
            base_tera_gate_entropy_sum.item() / base_tera_support
            if base_tera_support > 0
            else 0.0
        )
        base_tera_slot_support = base_tera_slot_choice_evaluations.item()
        averaged["base_tera_slot_choice_evaluations"] = base_tera_slot_support
        averaged["base_tera_slot_entropy"] = (
            base_tera_slot_entropy_sum.item() / base_tera_slot_support
            if base_tera_slot_support > 0
            else 0.0
        )
        averaged["base_tera_use_prob"] = (
            base_tera_use_prob_sum.item() / base_tera_support
            if base_tera_support > 0
            else 0.0
        )
        averaged["base_tera_none_prob"] = (
            base_tera_none_prob_sum.item() / base_tera_support
            if base_tera_support > 0
            else 0.0
        )
        if aux_labels_all:
            averaged.update(hidden_info_coverage_metrics)
        averaged.update(self._finalize_aux_diagnostics(aux_diagnostic_sums))
        averaged.update(memory_gradient_diagnostics)
        averaged["policy_belief_adapter_weight_norm"] = (
            self.agent.hidden_info_head.policy_belief_adapter.weight.detach()
            .norm()
            .item()
        )
        averaged["public_consequence_adapter_weight_norm"] = (
            self.agent.action_head.public_consequence_policy_adapter.weight.detach()
            .norm()
            .item()
        )
        if skipped_nonfinite > 0:
            averaged["skipped_nonfinite_minibatches"] = float(skipped_nonfinite)
        return averaged

    @staticmethod
    def _finalize_aux_diagnostics(sums: Mapping[str, float]) -> dict[str, float]:
        """Convert additive auxiliary sufficient statistics to readable metrics."""
        result: dict[str, float] = {}
        for task in ("item", "ability", "tera"):
            prefix = f"aux_{task}_unknown"
            count = sums.get(f"{prefix}_count", 0.0)
            result[f"{prefix}_count"] = count
            for source, target in (
                ("top1_correct", "top1_accuracy"),
                ("nll_sum", "nll"),
                ("brier_sum", "brier"),
            ):
                numerator = sums.get(f"{prefix}_{source}", 0.0)
                result[f"{prefix}_{target}"] = numerator / count if count > 0.0 else 0.0

        move_count = sums.get("aux_move_unknown_recall_at_4_count", 0.0)
        result["aux_move_unknown_recall_at_4_count"] = move_count
        result["aux_move_unknown_recall_at_4"] = (
            sums.get("aux_move_unknown_recall_at_4_correct", 0.0) / move_count
            if move_count > 0.0
            else 0.0
        )

        for component in (
            "kind",
            "move",
            "target",
            "switch_species",
            "tera_use",
            "tera_slot",
        ):
            prefix = f"opp_intent_{component}"
            count = sums.get(f"{prefix}_count", 0.0)
            result[f"{prefix}_count"] = count
            for source, target in (
                ("top1_correct", "top1_accuracy"),
                ("nll_sum", "nll"),
                ("brier_sum", "brier"),
            ):
                numerator = sums.get(f"{prefix}_{source}", 0.0)
                result[f"{prefix}_{target}"] = numerator / count if count > 0.0 else 0.0

        for task in (
            "emitted",
            "route",
            "interaction",
            "cant",
            "field_change",
            "field_alternative_scalar",
        ):
            prefix = f"aux_public_consequence_{task}"
            count = sums.get(f"{prefix}_count", 0.0)
            positive = sums.get(f"{prefix}_positive", 0.0)
            negative = sums.get(f"{prefix}_negative", 0.0)
            predicted_positive = sums.get(f"{prefix}_predicted_positive", 0.0)
            result[f"{prefix}_count"] = count
            for source, target in (
                ("positive", "prevalence"),
                ("predicted_positive", "predicted_positive_rate"),
                ("correct", "accuracy"),
                ("nll_sum", "nll"),
                ("brier_sum", "brier"),
            ):
                result[f"{prefix}_{target}"] = (
                    sums.get(f"{prefix}_{source}", 0.0) / count if count > 0.0 else 0.0
                )
            result[f"{prefix}_positive_recall"] = (
                sums.get(f"{prefix}_true_positive", 0.0) / positive
                if positive > 0.0
                else 0.0
            )
            result[f"{prefix}_negative_specificity"] = (
                sums.get(f"{prefix}_true_negative", 0.0) / negative
                if negative > 0.0
                else 0.0
            )
            result[f"{prefix}_precision"] = (
                sums.get(f"{prefix}_true_positive", 0.0) / predicted_positive
                if predicted_positive > 0.0
                else 0.0
            )

        categorical_tasks = {
            "nonmove_reason": ("cant", "fainted"),
            "move_branch": (
                "action_failed",
                "no_target",
                "scoped",
                "target_phase",
            ),
            "interaction_reason": (
                "miss",
                "immune",
                "protected",
                "blocked",
                "failed",
            ),
            "hit_count": tuple(f"hit_{count}" for count in range(1, 11)),
        }
        for task, classes in categorical_tasks.items():
            prefix = f"aux_public_consequence_{task}"
            count = sums.get(f"{prefix}_count", 0.0)
            result[f"{prefix}_count"] = count
            for source, target in (
                ("top1_correct", "top1_accuracy"),
                ("nll_sum", "nll"),
                ("brier_sum", "brier"),
            ):
                result[f"{prefix}_{target}"] = (
                    sums.get(f"{prefix}_{source}", 0.0) / count if count > 0.0 else 0.0
                )
            for class_name in classes:
                class_prefix = f"{prefix}_{class_name}"
                class_count = sums.get(f"{class_prefix}_count", 0.0)
                predicted_count = sums.get(f"{class_prefix}_predicted_count", 0.0)
                result[f"{class_prefix}_count"] = class_count
                result[f"{class_prefix}_recall"] = (
                    sums.get(f"{class_prefix}_correct", 0.0) / class_count
                    if class_count > 0.0
                    else 0.0
                )
                result[f"{class_prefix}_precision"] = (
                    sums.get(f"{class_prefix}_correct", 0.0) / predicted_count
                    if predicted_count > 0.0
                    else 0.0
                )
                for source, target in (("nll_sum", "nll"), ("brier_sum", "brier")):
                    result[f"{class_prefix}_{target}"] = (
                        sums.get(f"{class_prefix}_{source}", 0.0) / class_count
                        if class_count > 0.0
                        else 0.0
                    )

        for task in (
            "field_alternative_weather",
            "field_alternative_terrain",
        ):
            prefix = f"aux_public_consequence_{task}"
            count = sums.get(f"{prefix}_count", 0.0)
            result[f"{prefix}_count"] = count
            for source, target in (
                ("top1_correct", "top1_accuracy"),
                ("nll_sum", "nll"),
                ("brier_sum", "brier"),
            ):
                result[f"{prefix}_{target}"] = (
                    sums.get(f"{prefix}_{source}", 0.0) / count if count > 0.0 else 0.0
                )

        field_prefix = "aux_public_consequence_field"
        field_count = sums.get(f"{field_prefix}_count", 0.0)
        field_mse = (
            sums.get(f"{field_prefix}_mse_sum", 0.0) / field_count
            if field_count > 0.0
            else 0.0
        )
        persistence_mse = (
            sums.get(f"{field_prefix}_persistence_mse_sum", 0.0) / field_count
            if field_count > 0.0
            else 0.0
        )
        result[f"{field_prefix}_count"] = field_count
        result[f"{field_prefix}_mse"] = field_mse
        result[f"{field_prefix}_nll"] = (
            sums.get(f"{field_prefix}_nll_sum", 0.0) / field_count
            if field_count > 0.0
            else 0.0
        )
        result[f"{field_prefix}_brier"] = (
            sums.get(f"{field_prefix}_brier_sum", 0.0) / field_count
            if field_count > 0.0
            else 0.0
        )
        result[f"{field_prefix}_persistence_mse"] = persistence_mse
        result[f"{field_prefix}_mse_skill"] = (
            1.0 - field_mse / persistence_mse if persistence_mse > 0.0 else 0.0
        )
        for metric_name in ("typed_nll", "typed_brier"):
            model_metric = (
                sums.get(f"{field_prefix}_{metric_name}_sum", 0.0) / field_count
                if field_count > 0.0
                else 0.0
            )
            persistence_metric = (
                sums.get(
                    f"{field_prefix}_persistence_{metric_name}_sum",
                    0.0,
                )
                / field_count
                if field_count > 0.0
                else 0.0
            )
            result[f"{field_prefix}_{metric_name}"] = model_metric
            result[f"{field_prefix}_persistence_{metric_name}"] = persistence_metric
            result[f"{field_prefix}_{metric_name}_skill"] = (
                1.0 - model_metric / persistence_metric
                if persistence_metric > 0.0
                else 0.0
            )
        for group_name in ("weather", "terrain"):
            result[f"{field_prefix}_{group_name}_probability_sum_error"] = (
                sums.get(
                    f"{field_prefix}_{group_name}_probability_sum_error_sum",
                    0.0,
                )
                / field_count
                if field_count > 0.0
                else 0.0
            )
        for suffix in (
            "changed",
            "trick_room",
            "trick_room_value_changed",
            "trick_room_presence_changed",
        ):
            diagnostic_prefix = f"{field_prefix}_{suffix}"
            diagnostic_count = sums.get(f"{diagnostic_prefix}_count", 0.0)
            diagnostic_mse = (
                sums.get(f"{diagnostic_prefix}_mse_sum", 0.0) / diagnostic_count
                if diagnostic_count > 0.0
                else 0.0
            )
            diagnostic_persistence_mse = (
                sums.get(f"{diagnostic_prefix}_persistence_mse_sum", 0.0)
                / diagnostic_count
                if diagnostic_count > 0.0
                else 0.0
            )
            result[f"{diagnostic_prefix}_count"] = diagnostic_count
            result[f"{diagnostic_prefix}_mse"] = diagnostic_mse
            result[f"{diagnostic_prefix}_persistence_mse"] = diagnostic_persistence_mse
            result[f"{diagnostic_prefix}_mse_skill"] = (
                1.0 - diagnostic_mse / diagnostic_persistence_mse
                if diagnostic_persistence_mse > 0.0
                else 0.0
            )
        feature_prefix = f"{field_prefix}_feature"
        feature_count = sums.get(f"{feature_prefix}_count", 0.0)
        feature_positive = sums.get(f"{feature_prefix}_positive", 0.0)
        feature_negative = sums.get(f"{feature_prefix}_negative", 0.0)
        result[f"{feature_prefix}_count"] = feature_count
        result[f"{feature_prefix}_accuracy"] = (
            sums.get(f"{feature_prefix}_correct", 0.0) / feature_count
            if feature_count > 0.0
            else 0.0
        )
        result[f"{feature_prefix}_prevalence"] = (
            feature_positive / feature_count if feature_count > 0.0 else 0.0
        )
        result[f"{feature_prefix}_positive_recall"] = (
            sums.get(f"{feature_prefix}_true_positive", 0.0) / feature_positive
            if feature_positive > 0.0
            else 0.0
        )
        result[f"{feature_prefix}_negative_specificity"] = (
            sums.get(f"{feature_prefix}_true_negative", 0.0) / feature_negative
            if feature_negative > 0.0
            else 0.0
        )
        return result

    @staticmethod
    def _hidden_info_coverage_metrics(
        label_valid: torch.Tensor,
        opponent_padding: torch.Tensor | None,
    ) -> dict[str, float]:
        """Report how much public opponent state received terminal supervision."""
        if label_valid.ndim != 3:
            raise ValueError(
                "Hidden-info validity must have shape (steps, envs, team_size)."
            )
        if opponent_padding is None or opponent_padding.shape != label_valid.shape:
            public_slots = label_valid.new_zeros((), dtype=torch.int64)
        else:
            public_slots = (~opponent_padding).sum()
        valid_slots = label_valid.sum()
        labeled_steps = label_valid.any(dim=-1).sum()
        values = (
            torch.stack((valid_slots, public_slots, labeled_steps))
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .tolist()
        )
        valid_slots_value, public_slots_value, labeled_steps_value = values
        total_steps = label_valid.shape[0] * label_valid.shape[1]
        return {
            "aux_hidden_info_valid_selected_slots": valid_slots_value,
            "aux_hidden_info_public_slots": public_slots_value,
            "aux_hidden_info_label_coverage": (
                valid_slots_value / public_slots_value
                if public_slots_value > 0
                else 0.0
            ),
            "aux_hidden_info_labeled_steps": labeled_steps_value,
            "aux_hidden_info_valid_selected_slots_per_labeled_step": (
                valid_slots_value / labeled_steps_value
                if labeled_steps_value > 0
                else 0.0
            ),
            "aux_hidden_info_labeled_step_coverage": (
                labeled_steps_value / total_steps if total_steps > 0 else 0.0
            ),
        }

    @staticmethod
    def _gradient_diagnostics(
        objectives: dict[str, torch.Tensor],
        parameters: tuple[torch.nn.Parameter, ...],
        *,
        prefix: str,
    ) -> dict[str, float]:
        """Measure objective gradient norms/cosines without consuming the graph."""
        gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        norms: dict[str, torch.Tensor] = {}
        scalar_diagnostics: dict[str, torch.Tensor] = {}
        reference = next(iter(objectives.values()))

        for name, objective in objectives.items():
            objective_gradients = (
                torch.autograd.grad(
                    objective,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                if objective.requires_grad
                else (None,) * len(parameters)
            )
            gradients[name] = objective_gradients
            squared_norm = objective.new_zeros(())
            for gradient in objective_gradients:
                if gradient is not None:
                    squared_norm = squared_norm + gradient.detach().square().sum()
            norm = torch.sqrt(squared_norm)
            norms[name] = norm
            connected = any(gradient is not None for gradient in objective_gradients)
            scalar_diagnostics[f"{prefix}_norm_{name}"] = norm
            scalar_diagnostics[f"{prefix}_connected_{name}"] = reference.new_tensor(
                float(connected)
            )
            scalar_diagnostics[f"{prefix}_nonzero_{name}"] = (norm > 0).to(
                reference.dtype
            )

        names = tuple(objectives)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                denominator = norms[left_name] * norms[right_name]
                dot = objectives[left_name].new_zeros(())
                for left_gradient, right_gradient in zip(
                    gradients[left_name],
                    gradients[right_name],
                    strict=True,
                ):
                    if left_gradient is not None and right_gradient is not None:
                        dot = (
                            dot
                            + (left_gradient.detach() * right_gradient.detach()).sum()
                        )
                valid = denominator > 0
                scalar_diagnostics[f"{prefix}_cos_{left_name}_{right_name}"] = (
                    torch.where(valid, dot / denominator.clamp_min(1e-30), dot)
                )
                scalar_diagnostics[f"{prefix}_cos_{left_name}_{right_name}_valid"] = (
                    valid.to(reference.dtype)
                )

        squared_total_norm = reference.new_zeros(())
        for parameter_index in range(len(parameters)):
            combined_gradient = None
            for name in names:
                gradient = gradients[name][parameter_index]
                if gradient is not None:
                    combined_gradient = (
                        gradient.detach()
                        if combined_gradient is None
                        else combined_gradient + gradient.detach()
                    )
            if combined_gradient is not None:
                squared_total_norm = (
                    squared_total_norm + combined_gradient.square().sum()
                )
        combined_norm = torch.sqrt(squared_total_norm)
        component_norm_sum = sum(norms.values(), start=reference.new_zeros(()))
        scalar_diagnostics[f"{prefix}_norm_combined"] = combined_norm
        scalar_diagnostics[f"{prefix}_cancellation_fraction"] = torch.where(
            component_norm_sum > 0,
            1.0 - combined_norm / component_norm_sum.clamp_min(1e-30),
            component_norm_sum,
        )

        keys = tuple(scalar_diagnostics)
        values = (
            torch.stack([scalar_diagnostics[key] for key in keys])
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .tolist()
        )
        return dict(zip(keys, values, strict=True))

    @staticmethod
    def _hidden_info_loss(
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Masked CE/BCE against exact selected-team terminal labels."""
        opp_pad = labels["opp_mask"]  # (T, E_mb, 6) True if padding slot
        opp_cont = labels["opp_continuous"]  # (T, E_mb, 6, cont_dim)
        item_known = opp_cont[..., ITEM_KNOWN_OFFSET] > 0.5
        ability_known = opp_cont[..., ABILITY_KNOWN_OFFSET] > 0.5
        tera_known = opp_cont[..., TERA_KNOWN_OFFSET] > 0.5

        if label_valid.shape != opp_pad.shape:
            raise ValueError(
                "Hidden-info validity must have shape (steps, envs, team_size)."
            )
        base_mask = label_valid & ~opp_pad

        item_mask = (base_mask & item_known).reshape(-1)
        ability_mask = (base_mask & ability_known).reshape(-1)
        # Tera-type label is types[..., 2]; treat 0 as "no tera known" rather than a valid class.
        tera_types = labels["opp_types"][..., 2].reshape(-1)
        tera_mask = (base_mask & tera_known).reshape(-1) & (tera_types > 0)

        item_logits = logits["item_logits"].reshape(-1, logits["item_logits"].shape[-1])
        ability_logits = logits["ability_logits"].reshape(
            -1, logits["ability_logits"].shape[-1]
        )
        tera_logits = logits["tera_type_logits"].reshape(
            -1, logits["tera_type_logits"].shape[-1]
        )
        move_logits = logits["move_logits"]  # (T, E_mb, 6, num_moves)

        item_labels = labels["opp_item"].reshape(-1)
        ability_labels = labels["opp_ability"].reshape(-1)

        component_losses: list[torch.Tensor] = []
        if item_mask.any():
            component_losses.append(
                F.cross_entropy(item_logits[item_mask], item_labels[item_mask])
            )
        if ability_mask.any():
            component_losses.append(
                F.cross_entropy(
                    ability_logits[ability_mask], ability_labels[ability_mask]
                )
            )
        if tera_mask.any():
            component_losses.append(
                F.cross_entropy(tera_logits[tera_mask], tera_types[tera_mask])
            )

        # Multi-hot move labels from slot indices. Slot value 0 is "no move" and
        # is not a valid move id, so we ignore it. A pokemon's label is valid if
        # at least one move slot is > 0.
        move_slots = labels["opp_moves"]  # (T, E_mb, 6, 4)
        num_moves = move_logits.shape[-1]
        multihot = torch.zeros_like(move_logits)
        valid_slot = move_slots > 0
        safe_slots = move_slots.clamp(min=0, max=num_moves - 1)
        multihot.scatter_(-1, safe_slots, valid_slot.float())
        # Zero out slot 0 contribution (it was set to 1.0 anywhere safe_slots==0).
        multihot[..., 0] = 0.0

        move_slot_mask = base_mask & valid_slot.any(dim=-1)  # (T, E_mb, 6)
        if move_slot_mask.any():
            move_loss_raw = F.binary_cross_entropy_with_logits(
                move_logits, multihot, reduction="none"
            )
            # Mean across the move vocab for each pokemon, then mask-average.
            per_pokemon_bce = move_loss_raw.mean(dim=-1)
            denom = move_slot_mask.float().sum()
            component_losses.append(
                (per_pokemon_bce * move_slot_mask.float()).sum() / denom
            )

        if not component_losses:
            return item_logits.sum() * 0.0
        # The configured coefficient weights the auxiliary task as a whole;
        # adding another prediction head must not silently increase that weight.
        return torch.stack(component_losses).mean()

    @staticmethod
    def _opponent_intent_loss(
        logits: dict[str, torch.Tensor],
        slot_labels: torch.Tensor,
        slot_masks: torch.Tensor,
        tera_labels: torch.Tensor,
        tera_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Average CE across available non-forced semantic intent tasks."""
        if slot_labels.shape != slot_masks.shape or slot_labels.shape[-2:] != (
            2,
            NUM_SLOT_INTENT_COMPONENTS,
        ):
            raise ValueError(
                "Opponent slot intent labels/masks must share shape (..., 2, 4)."
            )
        base_shape = slot_labels.shape[:-2]
        expected_tera_shape = (*base_shape, NUM_TERA_INTENT_COMPONENTS)
        if (
            tera_labels.shape != expected_tera_shape
            or tera_masks.shape != expected_tera_shape
        ):
            raise ValueError(
                "Opponent Tera intent labels/masks must share shape (..., 2)."
            )

        component_losses: list[torch.Tensor] = []
        for logits_key, component in (
            ("opp_intent_kind_logits", SLOT_INTENT_KIND),
            ("opp_intent_move_logits", SLOT_INTENT_MOVE),
            ("opp_intent_target_logits", SLOT_INTENT_TARGET),
            (
                "opp_intent_switch_species_logits",
                SLOT_INTENT_SWITCH_SPECIES,
            ),
        ):
            component_logits = logits[logits_key]
            component_mask = slot_masks[..., component].reshape(-1)
            component_targets = slot_labels[..., component].reshape(-1)
            component_logits = component_logits.reshape(
                -1,
                component_logits.shape[-1],
            )
            if component_mask.any():
                component_losses.append(
                    F.cross_entropy(
                        component_logits[component_mask],
                        component_targets[component_mask],
                    )
                )

        for logits_key, component in (
            ("opp_intent_tera_use_logits", TERA_INTENT_USE),
            ("opp_intent_tera_slot_logits", TERA_INTENT_SLOT),
        ):
            component_logits = logits[logits_key].reshape(
                -1,
                logits[logits_key].shape[-1],
            )
            component_mask = tera_masks[..., component].reshape(-1)
            component_targets = tera_labels[..., component].reshape(-1)
            if component_mask.any():
                component_losses.append(
                    F.cross_entropy(
                        component_logits[component_mask],
                        component_targets[component_mask],
                    )
                )

        if not component_losses:
            return logits["opp_intent_kind_logits"].sum() * 0.0
        return torch.stack(component_losses).mean()

    @staticmethod
    def _public_consequence_loss(
        logits: Mapping[str, torch.Tensor],
        move_emitted_targets: torch.Tensor,
        move_emitted_masks: torch.Tensor,
        nonmove_reason_labels: torch.Tensor,
        nonmove_reason_masks: torch.Tensor,
        move_branch_labels: torch.Tensor,
        move_branch_masks: torch.Tensor,
        route_targets: torch.Tensor,
        route_masks: torch.Tensor,
        interaction_targets: torch.Tensor,
        interaction_masks: torch.Tensor,
        interaction_reason_labels: torch.Tensor,
        interaction_reason_masks: torch.Tensor,
        hit_count_labels: torch.Tensor,
        hit_count_masks: torch.Tensor,
        cant_targets: torch.Tensor,
        cant_masks: torch.Tensor,
        field_targets: torch.Tensor,
        field_masks: torch.Tensor,
        persistence_fields: torch.Tensor,
    ) -> torch.Tensor:
        """Average conditional losses for the public consequence factorization.

        Every subtype is supervised only where the protocol makes that question
        meaningful.  This preserves the proper likelihood for each conditional
        distribution without amplifying rare outcomes with class weights.
        """
        required = {
            "public_consequence_move_emitted_logits",
            "public_consequence_nonmove_reason_logits",
            "public_consequence_move_branch_logits",
            "public_consequence_route_logits",
            "public_consequence_interaction_logits",
            "public_consequence_interaction_reason_logits",
            "public_consequence_hit_count_logits",
            "public_consequence_cant_logits",
            "public_consequence_field_logits",
            "public_consequence_field_change_logits",
            "public_consequence_field_alternative_logits",
        }
        missing = sorted(required.difference(logits))
        if missing:
            raise ValueError(f"logits is missing key(s): {', '.join(missing)}.")
        if move_emitted_targets.shape != move_emitted_masks.shape:
            raise ValueError("Move-emitted targets/masks must share shape.")
        if move_emitted_targets.ndim < 1 or move_emitted_targets.shape[-1] != 2:
            raise ValueError("Move-emitted targets must end with two action slots.")
        base_shape = tuple(move_emitted_targets.shape)
        target_shape = (*base_shape, NUM_PUBLIC_TARGET_POSITIONS)
        paired_tensors = {
            "nonmove reason": (nonmove_reason_labels, nonmove_reason_masks),
            "move branch": (move_branch_labels, move_branch_masks),
            "route": (route_targets, route_masks),
            "interaction": (interaction_targets, interaction_masks),
            "interaction reason": (
                interaction_reason_labels,
                interaction_reason_masks,
            ),
            "hit count": (hit_count_labels, hit_count_masks),
        }
        for name, (targets, masks) in paired_tensors.items():
            if targets.shape != masks.shape:
                raise ValueError(f"Public consequence {name} labels/masks differ.")
        for name, tensor in (
            ("nonmove reason", nonmove_reason_labels),
            ("move branch", move_branch_labels),
        ):
            if tensor.shape != base_shape:
                raise ValueError(
                    f"Public consequence {name} labels have incompatible shape."
                )
        for name, tensor in (
            ("route", route_targets),
            ("interaction", interaction_targets),
            ("interaction reason", interaction_reason_labels),
            ("hit count", hit_count_labels),
        ):
            if tensor.shape != target_shape:
                raise ValueError(
                    f"Public consequence {name} labels have incompatible shape."
                )

        expected_logit_shapes = {
            "public_consequence_move_emitted_logits": base_shape,
            "public_consequence_nonmove_reason_logits": (
                *base_shape,
                NUM_NONMOVE_REASONS,
            ),
            "public_consequence_move_branch_logits": (
                *base_shape,
                NUM_MOVE_BRANCHES,
            ),
            "public_consequence_route_logits": target_shape,
            "public_consequence_interaction_logits": target_shape,
            "public_consequence_interaction_reason_logits": (
                *target_shape,
                NUM_TARGET_INTERACTION_REASONS,
            ),
            "public_consequence_hit_count_logits": (
                *target_shape,
                MAX_PUBLIC_HIT_COUNT,
            ),
        }
        for name, expected_shape in expected_logit_shapes.items():
            if tuple(logits[name].shape) != expected_shape:
                raise ValueError(f"{name} has incompatible shape.")
        if cant_targets.shape != (*base_shape, 2) or cant_masks.shape != (
            *base_shape,
            2,
        ):
            raise ValueError(
                "Public consequence CANT targets/masks must have shape (..., 2, 2)."
            )
        cant_logits = logits["public_consequence_cant_logits"]
        if cant_logits.shape != cant_targets.shape:
            raise ValueError(
                "Public consequence CANT logits must match CANT target shape."
            )
        expected_field_shape = (*base_shape, PUBLIC_CONSEQUENCE_FIELD_DIM)
        if field_targets.shape != expected_field_shape:
            raise ValueError(
                "Public consequence field targets have incompatible shape."
            )
        if persistence_fields.shape != expected_field_shape:
            raise ValueError(
                "Public consequence persistence fields have incompatible shape."
            )
        if field_masks.shape != base_shape:
            raise ValueError(
                "Public consequence field masks must match action-slot shape."
            )
        field_logits = logits["public_consequence_field_logits"]
        if field_logits.shape != expected_field_shape:
            raise ValueError("Public consequence field logits have incompatible shape.")
        field_change_logits = logits["public_consequence_field_change_logits"]
        if field_change_logits.shape != (*base_shape, FIELD_CHANGE_COMPONENT_DIM):
            raise ValueError(
                "Public consequence field-change logits have incompatible shape."
            )
        field_alternative_logits = logits["public_consequence_field_alternative_logits"]
        if field_alternative_logits.shape != expected_field_shape:
            raise ValueError(
                "Public consequence field-alternative logits have incompatible shape."
            )

        component_losses: list[torch.Tensor] = []

        def add_binary(logit_name: str, targets: torch.Tensor, masks: torch.Tensor):
            selected = masks.to(torch.bool)
            if selected.any():
                component_losses.append(
                    F.binary_cross_entropy_with_logits(
                        logits[logit_name][selected],
                        targets.to(logits[logit_name].dtype)[selected],
                    )
                )

        def add_categorical(
            logit_name: str,
            labels: torch.Tensor,
            masks: torch.Tensor,
        ):
            selected = masks.to(torch.bool)
            if selected.any():
                component_losses.append(
                    F.cross_entropy(logits[logit_name][selected], labels[selected])
                )

        add_binary(
            "public_consequence_move_emitted_logits",
            move_emitted_targets,
            move_emitted_masks,
        )
        add_categorical(
            "public_consequence_nonmove_reason_logits",
            nonmove_reason_labels,
            nonmove_reason_masks,
        )
        add_categorical(
            "public_consequence_move_branch_logits",
            move_branch_labels,
            move_branch_masks,
        )
        add_binary("public_consequence_route_logits", route_targets, route_masks)
        add_binary(
            "public_consequence_interaction_logits",
            interaction_targets,
            interaction_masks,
        )
        add_categorical(
            "public_consequence_interaction_reason_logits",
            interaction_reason_labels,
            interaction_reason_masks,
        )
        add_categorical(
            "public_consequence_hit_count_logits",
            hit_count_labels,
            hit_count_masks,
        )

        add_binary("public_consequence_cant_logits", cant_targets, cant_masks)

        field_selected = field_masks.to(torch.bool)
        if field_selected.any():
            typed_field_targets = field_targets.to(field_logits.dtype)
            typed_persistence = persistence_fields.to(field_logits.dtype)
            change_targets = public_field_change_targets(
                typed_field_targets,
                typed_persistence,
            )
            field_component_losses: list[torch.Tensor] = []

            # Final-state likelihood remains useful because it trains the
            # mixture itself.  Categorical field classes use their proper
            # grouped likelihood; ordinal/binary scalars retain BCE.
            field_probs = field_logits.sigmoid()
            typed_final_losses = []
            for group in (WEATHER_CLASS_SLICE, TERRAIN_CLASS_SLICE):
                group_probs = field_probs[..., group]
                group_probs = group_probs / group_probs.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)
                group_labels = typed_field_targets[..., group].argmax(dim=-1)
                selected_group_probs = group_probs[field_selected]
                selected_group_labels = group_labels[field_selected]
                typed_final_losses.append(
                    -selected_group_probs.gather(1, selected_group_labels.unsqueeze(-1))
                    .clamp_min(1e-8)
                    .log()
                    .mean()
                )
            scalar_logits = field_logits[..., SCALAR_FIELD_INDICES]
            scalar_targets = typed_field_targets[..., SCALAR_FIELD_INDICES]
            typed_final_losses.append(
                F.binary_cross_entropy_with_logits(
                    scalar_logits[field_selected],
                    scalar_targets[field_selected],
                )
            )
            field_component_losses.append(torch.stack(typed_final_losses).mean())

            # A calibrated gate learns whether the deterministic transition is
            # wrong.  No class balancing is used: the probability of change is
            # itself part of the world model consumed by the policy.
            field_component_losses.append(
                F.binary_cross_entropy_with_logits(
                    field_change_logits[field_selected],
                    change_targets[field_selected],
                )
            )

            # Alternative values are conditional distributions: only teach
            # them on components whose next value differs from persistence.
            conditional_losses = []
            for component, group in enumerate(
                (WEATHER_CLASS_SLICE, TERRAIN_CLASS_SLICE)
            ):
                changed = field_selected & change_targets[..., component].bool()
                if changed.any():
                    conditional_losses.append(
                        F.cross_entropy(
                            field_alternative_logits[..., group][changed],
                            typed_field_targets[..., group].argmax(dim=-1)[changed],
                        )
                    )
            scalar_changed = (
                field_selected.unsqueeze(-1) & change_targets[..., 2:].bool()
            )
            if scalar_changed.any():
                conditional_losses.append(
                    F.binary_cross_entropy_with_logits(
                        field_alternative_logits[..., SCALAR_FIELD_INDICES][
                            scalar_changed
                        ],
                        scalar_targets[scalar_changed],
                    )
                )
            if conditional_losses:
                field_component_losses.append(torch.stack(conditional_losses).mean())

            # The structured field transition remains one auxiliary task in
            # the outer equal-task average, regardless of its internal factors.
            component_losses.append(torch.stack(field_component_losses).mean())

        if not component_losses:
            return logits["public_consequence_move_emitted_logits"].sum() * 0.0
        # Every available task contributes equally regardless of output width
        # or label frequency; the configured coefficient weights the complete
        # auxiliary objective rather than growing with added target fields.
        return torch.stack(component_losses).mean()

    def update_team_preview(
        self,
        data: list[tuple[dict, np.ndarray, float, np.ndarray]],
    ) -> dict[str, float]:
        """Update preview policy under the same eval semantics used for rollout."""
        was_training = self.agent.training
        memory = getattr(self.agent, "memory", None)
        recurrent = getattr(memory, "lstm", None)
        recurrent_was_training = (
            recurrent.training if isinstance(recurrent, nn.LSTM) else None
        )
        self.agent.eval()
        if isinstance(recurrent, nn.LSTM):
            if recurrent.dropout != 0.0:
                raise RuntimeError(
                    "Exact deployment-semantic preview PPO requires a dropout-free "
                    "LSTM when differentiating through cuDNN."
                )
            # cuDNN refuses LSTM backward after an eval-mode forward even when
            # the module has no train/eval-dependent behavior.  This one-layer
            # LSTM has zero dropout, so setting only the recurrent primitive's
            # flag keeps its numerical semantics identical while enabling the
            # differentiable preview pass.  Every other module remains in eval.
            recurrent.train(True)
        try:
            return self._update_team_preview_eval(data)
        finally:
            self.agent.train(was_training)
            if recurrent_was_training is not None:
                recurrent.train(recurrent_was_training)

    def _update_team_preview_eval(
        self,
        data: list[tuple[dict, np.ndarray, float, np.ndarray]],
    ) -> dict[str, float]:
        """Clipped on-policy update for terminal team-preview decisions.

        The policy is unchanged while a rollout is collected, and callers
        exclude preview decisions produced by an older policy version.  We can
        therefore reconstruct the exact behavior log-probability and value
        immediately before the first optimizer step instead of adding more
        work to the latency-sensitive live preview path.  Those frozen
        quantities support the same clipped sample reuse used by battle PPO.

        Each entry is ``(obs_dict, picks_np, episode_reward,
        opponent_picks_np)`` where the reward is the unshaped terminal outcome.
        """
        if not data:
            raise ValueError("Team-preview update requires at least one sample.")

        sample_count = len(data)
        obs_all = {
            key: torch.from_numpy(np.stack([sample[0][key] for sample in data])).to(
                self.device
            )
            for key in data[0][0]
        }
        picks_all = torch.from_numpy(np.stack([sample[1] for sample in data])).to(
            device=self.device,
            dtype=torch.long,
        )
        rewards_all = torch.tensor(
            [sample[2] for sample in data],
            dtype=torch.float32,
            device=self.device,
        )
        opponent_picks_all = torch.from_numpy(
            np.stack([sample[3] for sample in data])
        ).to(device=self.device, dtype=torch.long)

        requested_minibatches = min(self.config.num_minibatches, sample_count)
        target_minibatch_size = max(
            1,
            (sample_count + requested_minibatches - 1) // requested_minibatches,
        )
        minibatch_size = min(
            self.config.preview_update_minibatch_size,
            target_minibatch_size,
        )

        # Snapshot the exact behavior policy before any shared parameter is
        # updated.  Chunking bounds activations and avoids increasing live
        # simulator/reset latency with behavior metadata collection.
        old_log_prob_parts: list[torch.Tensor] = []
        old_value_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, sample_count, minibatch_size):
                stop = min(start + minibatch_size, sample_count)
                behavior = self.agent.team_preview(
                    {key: value[start:stop] for key, value in obs_all.items()},
                    picks=picks_all[start:stop],
                    include_opponent_preview=False,
                )
                old_log_prob_parts.append(behavior["log_prob"].detach())
                old_value_parts.append(behavior["preview_value"].detach())
        old_log_prob_all = torch.cat(old_log_prob_parts)
        old_value_all = torch.cat(old_value_parts)
        advantages_all = rewards_all - old_value_all
        if not all(
            torch.isfinite(value).all().item()
            for value in (
                old_log_prob_all,
                old_value_all,
                advantages_all,
            )
        ):
            return {"team_preview_skipped_nonfinite": 1.0}

        metric_sums = {
            "policy": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "bring": 0.0,
            "lead": 0.0,
            "approx_kl": 0.0,
            "clip": 0.0,
            "value_clip": 0.0,
        }
        metric_samples = 0
        metric_bring_slots = 0
        optimizer_steps = 0
        max_approx_kl = 0.0

        for _epoch in range(self.config.num_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                obs_batch = {
                    key: value.index_select(0, indices)
                    for key, value in obs_all.items()
                }
                picks_batch = picks_all.index_select(0, indices)
                rewards_batch = rewards_all.index_select(0, indices)
                opponent_picks_batch = opponent_picks_all.index_select(0, indices)
                old_log_prob = old_log_prob_all.index_select(0, indices)
                old_value = old_value_all.index_select(0, indices)
                advantages = advantages_all.index_select(0, indices)

                out = self.agent.team_preview(obs_batch, picks=picks_batch)
                log_prob = out["log_prob"]
                entropy = out["entropy"]
                values = out["preview_value"]
                bring_logits = out["opponent_bring_logits"]
                lead_pair_logits = out["opponent_lead_pair_logits"]

                log_ratio = log_prob - old_log_prob
                bounded_log_ratio = log_ratio.clamp(
                    min=-self._LOG_RATIO_CLAMP,
                    max=self._LOG_RATIO_CLAMP,
                )
                ratio = bounded_log_ratio.exp()
                unclipped_policy = -advantages * ratio
                clipped_policy = -advantages * ratio.clamp(
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                )
                policy_loss = torch.maximum(
                    unclipped_policy,
                    clipped_policy,
                ).mean()
                entropy_mean = entropy.mean()

                value_delta = values - old_value
                clipped_values = old_value + value_delta.clamp(
                    -self.config.clip_eps,
                    self.config.clip_eps,
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (values - rewards_batch).pow(2),
                        (clipped_values - rewards_batch).pow(2),
                    ).mean()
                )

                team_size = bring_logits.shape[1]
                bring_targets = torch.zeros_like(bring_logits)
                bring_targets.scatter_(1, opponent_picks_batch, 1.0)
                valid_slots = (~obs_batch["opp_mask"]).to(bring_logits.dtype)
                valid_bring_slots = int(valid_slots.sum().item())
                bring_loss = (
                    F.binary_cross_entropy_with_logits(
                        bring_logits,
                        bring_targets,
                        reduction="none",
                    )
                    * valid_slots
                ).sum() / max(valid_bring_slots, 1)

                lead_targets = (
                    opponent_picks_batch[:, 0] * team_size + opponent_picks_batch[:, 1]
                )
                lead_loss = F.cross_entropy(lead_pair_logits, lead_targets)

                loss = (
                    policy_loss
                    - self.config.entropy_coef * entropy_mean
                    + self.config.team_preview_value_coef * value_loss
                    + self.config.team_preview_bring_coef * bring_loss
                    + self.config.team_preview_lead_coef * lead_loss
                )
                if not self._apply_update_step(loss):
                    return {"team_preview_skipped_nonfinite": 1.0}

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - bounded_log_ratio).mean()
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.config.clip_eps).float().mean()
                    )
                    value_clip_fraction = (
                        (value_delta.abs() > self.config.clip_eps).float().mean()
                    )
                    batch_size = int(indices.numel())
                    metric_samples += batch_size
                    metric_bring_slots += valid_bring_slots
                    metric_sums["policy"] += policy_loss.item() * batch_size
                    metric_sums["value"] += value_loss.item() * batch_size
                    metric_sums["entropy"] += entropy_mean.item() * batch_size
                    metric_sums["bring"] += bring_loss.item() * valid_bring_slots
                    metric_sums["lead"] += lead_loss.item() * batch_size
                    metric_sums["approx_kl"] += approx_kl.item() * batch_size
                    metric_sums["clip"] += clip_fraction.item() * batch_size
                    metric_sums["value_clip"] += value_clip_fraction.item() * batch_size
                    max_approx_kl = max(max_approx_kl, approx_kl.item())
                    optimizer_steps += 1

        denominator = float(max(metric_samples, 1))
        policy_loss = torch.tensor(metric_sums["policy"] / denominator)
        value_loss = torch.tensor(metric_sums["value"] / denominator)
        entropy_mean = torch.tensor(metric_sums["entropy"] / denominator)
        bring_loss = torch.tensor(metric_sums["bring"] / max(metric_bring_slots, 1))
        lead_loss = torch.tensor(metric_sums["lead"] / denominator)
        loss = (
            policy_loss
            - self.config.entropy_coef * entropy_mean
            + self.config.team_preview_value_coef * value_loss
            + self.config.team_preview_bring_coef * bring_loss
            + self.config.team_preview_lead_coef * lead_loss
        )
        return {
            "team_preview_loss": loss.item(),
            "team_preview_policy_loss": policy_loss.item(),
            "team_preview_value_loss": value_loss.item(),
            "team_preview_entropy": entropy_mean.item(),
            "team_preview_bring_loss": bring_loss.item(),
            "team_preview_lead_loss": lead_loss.item(),
            "team_preview_approx_kl": metric_sums["approx_kl"] / denominator,
            "team_preview_max_approx_kl": max_approx_kl,
            "team_preview_clip_fraction": metric_sums["clip"] / denominator,
            "team_preview_value_clip_fraction": (
                metric_sums["value_clip"] / denominator
            ),
            "team_preview_optimizer_steps": float(optimizer_steps),
            "team_preview_epochs": float(self.config.num_epochs),
        }

    def save(self, path: str, *, training_state_id: str | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        state_id = training_state_id or uuid.uuid4().hex
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("training_state_id must be a non-empty string.")
        try:
            checkpoint: dict[str, object] = {
                "checkpoint_type": "trainer",
                "format_version": TRAINER_CHECKPOINT_FORMAT_VERSION,
                "model_metadata": model_checkpoint_metadata(self.agent),
                "agent": self.agent.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "completed_updates": self.completed_updates,
                "training_state_id": state_id,
                "rng_state": _capture_rng_state(),
            }
            if self.initialization_provenance is not None:
                checkpoint["initialization_provenance"] = copy.deepcopy(
                    self.initialization_provenance
                )
            torch.save(checkpoint, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def initialize_from_trainer_weights(self, path: str) -> None:
        """Strictly initialize model weights without inheriting training state.

        A weight-initialized policy is a new learner. It may start from a
        competent policy, but must not silently inherit its Adam moments,
        update counters, or RNG streams. This method accepts only the current
        versioned trainer format so architecture and vocabulary metadata can be
        validated before loading exact weights.
        """
        fresh_rng_state = _capture_rng_state()
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
            if not isinstance(checkpoint, dict):
                raise ValueError(
                    "Weight initializer checkpoint must be a mapping, got "
                    f"{type(checkpoint).__name__}."
                )
            if checkpoint.get("checkpoint_type") != "trainer":
                raise ValueError(
                    f"Weight initializer {path!r} must be a versioned trainer "
                    "checkpoint; league and legacy checkpoints do not carry the "
                    "strict model metadata required for a new learner."
                )
            if checkpoint.get("format_version") != TRAINER_CHECKPOINT_FORMAT_VERSION:
                raise ValueError(
                    f"Weight initializer {path!r} has unsupported format_version="
                    f"{checkpoint.get('format_version')!r}; expected "
                    f"{TRAINER_CHECKPOINT_FORMAT_VERSION}."
                )
            required = {"agent", "model_metadata", "training_state_id"}
            missing = sorted(required.difference(checkpoint))
            if missing:
                raise ValueError(
                    f"Weight initializer {path!r} is missing required key(s): "
                    + ", ".join(missing)
                )
            checkpoint_state = validate_model_checkpoint_metadata(
                path=path,
                checkpoint_state=checkpoint["agent"],
                checkpoint_metadata=checkpoint["model_metadata"],
                current_agent=self.agent,
            )
            source_state_id = checkpoint["training_state_id"]
            if not isinstance(source_state_id, str) or not source_state_id:
                raise ValueError(
                    "Weight initializer training_state_id must be non-empty string."
                )
            initialization_provenance = _copy_initialization_provenance(checkpoint)
            try:
                self.agent.load_state_dict(checkpoint_state, strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    f"Weight initializer {path!r} failed strict model loading: {exc}"
                ) from exc

            # Construct a new optimizer after loading.  No source optimizer
            # tensors, rollout boundary, or random stream is adopted.
            self.optimizer = self._make_optimizer()
            self.global_step = 0
            self.completed_updates = 0
            self.loaded_training_state_id = None
            self.initialized_from_training_state_id = source_state_id
            self.initialization_provenance = initialization_provenance
        finally:
            # Loading checkpoint bytes and validating metadata must be RNG
            # transparent.  The caller's freshly configured phase streams stay
            # in force even if initialization fails.
            _restore_rng_state(fresh_rng_state)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Trainer checkpoint must be a mapping, got {type(checkpoint).__name__}."
            )
        checkpoint_type = checkpoint.get("checkpoint_type")
        if checkpoint_type != "trainer":
            raise ValueError(
                f"Trainer checkpoint {path!r} has checkpoint_type={checkpoint_type!r}; "
                "only a versioned checkpoint_type='trainer' checkpoint can be used "
                "with --resume. Legacy and league checkpoints are unsupported."
            )
        format_version = checkpoint.get("format_version")
        if format_version != TRAINER_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Trainer checkpoint {path!r} has unsupported format_version="
                f"{format_version!r}; expected {TRAINER_CHECKPOINT_FORMAT_VERSION}. "
                "Legacy trainer checkpoints cannot be resumed."
            )
        required_keys = {
            "agent",
            "optimizer",
            "global_step",
            "completed_updates",
            "training_state_id",
            "model_metadata",
            "rng_state",
        }
        missing_keys = sorted(required_keys.difference(checkpoint))
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(
                f"Trainer checkpoint {path!r} is missing required key(s): {missing}."
            )

        checkpoint_state = validate_model_checkpoint_metadata(
            path=path,
            checkpoint_state=checkpoint["agent"],
            checkpoint_metadata=checkpoint["model_metadata"],
            current_agent=self.agent,
        )
        global_step = checkpoint["global_step"]
        if isinstance(global_step, bool) or not isinstance(global_step, int):
            raise ValueError("Trainer checkpoint global_step must be an integer.")
        if global_step < 0:
            raise ValueError("Trainer checkpoint global_step must be non-negative.")
        completed_updates = checkpoint["completed_updates"]
        if isinstance(completed_updates, bool) or not isinstance(
            completed_updates,
            int,
        ):
            raise ValueError("Trainer checkpoint completed_updates must be an integer.")
        if completed_updates < 0:
            raise ValueError(
                "Trainer checkpoint completed_updates must be non-negative."
            )
        training_state_id = checkpoint["training_state_id"]
        if not isinstance(training_state_id, str) or not training_state_id:
            raise ValueError(
                "Trainer checkpoint training_state_id must be a non-empty string."
            )
        rng_state = _validate_restorable_rng_state(checkpoint["rng_state"])
        initialization_provenance = _copy_initialization_provenance(checkpoint)

        loaded_optimizer = self._make_optimizer()
        try:
            loaded_optimizer.load_state_dict(checkpoint["optimizer"])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Trainer checkpoint {path!r} optimizer state is incompatible: {exc}"
            ) from exc

        try:
            self.agent.load_state_dict(checkpoint_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Trainer checkpoint {path!r} model state failed strict loading: {exc}"
            ) from exc
        self.optimizer = loaded_optimizer
        self.global_step = global_step
        self.completed_updates = completed_updates
        self.loaded_training_state_id = training_state_id
        self.initialized_from_training_state_id = None
        self.initialization_provenance = initialization_provenance
        _restore_rng_state(rng_state)
