"""PPO trainer with support for action masking and LSTM policies."""

import random
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TrainingConfig
from ..env.observation import (
    ABILITY_KNOWN_OFFSET,
    ITEM_KNOWN_OFFSET,
    ROLE_BENCH,
    TERA_KNOWN_OFFSET,
)
from ..model.agent import PokeAgent
from .rollout import RolloutBatch


# Increment when trainer-checkpoint semantics or same-shaped model inputs change.
TRAINER_CHECKPOINT_FORMAT_VERSION = 1


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _state_dict_schema(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
    schema: dict[str, dict[str, object]] = {}
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Model state entry {key!r} must be a tensor, got "
                f"{type(value).__name__}."
            )
        schema[key] = {
            "shape": tuple(int(dimension) for dimension in value.shape),
            "dtype": str(value.dtype),
            "layout": str(value.layout),
        }
    return schema


def _model_config_metadata(agent: nn.Module) -> dict[str, object] | None:
    config = getattr(agent, "config", None)
    if config is None:
        return None
    if is_dataclass(config) and not isinstance(config, type):
        return {
            "type": _qualified_type_name(config),
            "values": asdict(config),
        }
    if isinstance(config, Mapping):
        return {
            "type": _qualified_type_name(config),
            "values": dict(config),
        }
    return None


def _model_checkpoint_metadata(agent: nn.Module) -> dict[str, object]:
    return {
        "model_type": _qualified_type_name(agent),
        "model_config": _model_config_metadata(agent),
        "game_data_signature": getattr(agent, "game_data_signature", None),
        "state_schema": _state_dict_schema(agent.state_dict()),
    }


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


def _validate_model_checkpoint_metadata(
    *,
    path: str,
    checkpoint_state: object,
    checkpoint_metadata: object,
    current_agent: nn.Module,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint_state, Mapping):
        raise ValueError("Trainer checkpoint agent state must be a mapping.")
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("Trainer checkpoint model metadata must be a mapping.")
    required_metadata = {
        "model_type",
        "model_config",
        "game_data_signature",
        "state_schema",
    }
    missing_metadata = sorted(required_metadata.difference(checkpoint_metadata))
    if missing_metadata:
        raise ValueError(
            "Trainer checkpoint model metadata is missing key(s): "
            + ", ".join(missing_metadata)
        )

    declared_schema = checkpoint_metadata["state_schema"]
    if not isinstance(declared_schema, Mapping):
        raise ValueError("Trainer checkpoint model state schema must be a mapping.")
    actual_checkpoint_schema = _state_dict_schema(checkpoint_state)
    if actual_checkpoint_schema != dict(declared_schema):
        raise ValueError(
            f"Trainer checkpoint {path!r} agent state does not match its declared "
            "model-state schema (missing, unexpected, shape, dtype, or layout mismatch)."
        )

    current_metadata = _model_checkpoint_metadata(current_agent)
    if checkpoint_metadata["model_type"] != current_metadata["model_type"]:
        raise ValueError(
            f"Trainer checkpoint {path!r} model type mismatch: "
            f"checkpoint={checkpoint_metadata['model_type']!r}, "
            f"current={current_metadata['model_type']!r}."
        )
    if checkpoint_metadata["model_config"] != current_metadata["model_config"]:
        raise ValueError(
            f"Trainer checkpoint {path!r} model config mismatch; resume requires "
            "the exact architecture configuration used to create the checkpoint."
        )
    if (
        checkpoint_metadata["game_data_signature"]
        != current_metadata["game_data_signature"]
    ):
        raise ValueError(
            f"Trainer checkpoint {path!r} game-data vocabulary mismatch; resume "
            "requires the exact categorical index mapping used by the checkpoint."
        )

    current_schema = current_metadata["state_schema"]
    if declared_schema != current_schema:
        checkpoint_keys = set(declared_schema)
        current_keys = set(current_schema)
        missing_keys = sorted(current_keys - checkpoint_keys)
        unexpected_keys = sorted(checkpoint_keys - current_keys)
        incompatible_keys = sorted(
            key
            for key in checkpoint_keys & current_keys
            if declared_schema[key] != current_schema[key]
        )
        raise ValueError(
            f"Trainer checkpoint {path!r} model-state schema mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}, "
            f"incompatible={incompatible_keys}."
        )
    return checkpoint_state


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
        self.loaded_training_state_id: str | None = None

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
            "approx_kl": torch.zeros((), device=self.device),
        }
        base_tera_entropy_sum = torch.zeros((), device=self.device)
        base_tera_none_prob_sum = torch.zeros((), device=self.device)
        base_tera_real_choice_evaluations = torch.zeros((), device=self.device)
        num_updates = 0
        skipped_nonfinite = 0
        memory_gradient_diagnostics: dict[str, float] = {}
        initial_policy_shift: dict[str, torch.Tensor] = {}

        aux_labels_all = batch.aux_labels
        aux_label_valid_all = batch.aux_label_valid
        hidden_info_coverage_metrics = self._hidden_info_coverage_metrics(
            aux_label_valid_all,
            obs_seq.get("opp_mask"),
        )
        opp_actions_all = batch.opp_actions
        opp_actions_valid_all = batch.opp_actions_valid
        hidden_info_enabled = (
            self.config.aux_hidden_info_coef != 0.0
            and bool(aux_labels_all)
            and hasattr(self.agent, "hidden_info_head")
            and hidden_info_coverage_metrics["aux_hidden_info_valid_selected_slots"] > 0
        )
        opp_action_enabled = (
            self.config.aux_opp_action_coef != 0.0
            and hasattr(self.agent, "opponent_action_head")
            and bool(opp_actions_valid_all.any().item())
        )
        if hidden_info_enabled:
            metric_sums["aux_hidden_info_loss"] = torch.zeros((), device=self.device)
        if opp_action_enabled:
            metric_sums["aux_opp_action_loss"] = torch.zeros((), device=self.device)

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

                # The action and value heads are independent across timesteps.
                # Chunking avoids many tiny GPU launches without evaluating the
                # whole rollout minibatch at once.
                all_log_probs: list[torch.Tensor] = []
                all_entropies: list[torch.Tensor] = []
                all_base_tera_entropies: list[torch.Tensor] = []
                all_base_tera_none_probs: list[torch.Tensor] = []
                all_base_tera_real_choices: list[torch.Tensor] = []
                all_values: list[torch.Tensor] = []
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
                    opp_reps_flat = opp_reps_seq_mb[start:end].reshape(
                        flat_count,
                        opp_reps_seq_mb.shape[2],
                        opp_reps_seq_mb.shape[-1],
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
                        opp_active_reps_seq_mb[start:end].reshape(
                            flat_count,
                            2,
                            opp_active_reps_seq_mb.shape[-1],
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
                        tera_force_delay_prob=batch.tera_force_delay_prob,
                        tera_force_delay_max_turn=(batch.tera_force_delay_max_turn),
                        validate_actions=False,
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
                    all_base_tera_entropies.append(action_out["base_tera_entropy"])
                    all_base_tera_none_probs.append(action_out["base_tera_none_prob"])
                    all_base_tera_real_choices.append(
                        action_out["base_tera_has_real_choice"]
                    )
                    all_values.append(value_t)

                log_probs_flat = torch.cat(all_log_probs)
                entropies_flat = torch.cat(all_entropies)
                base_tera_entropies_flat = torch.cat(all_base_tera_entropies)
                base_tera_none_probs_flat = torch.cat(all_base_tera_none_probs)
                base_tera_real_choices_flat = torch.cat(all_base_tera_real_choices)
                values_flat = torch.cat(all_values)

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

                entropy_loss = -entropies_flat.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # --- Aux losses ---
                # Hidden info: predict exact selected-team item/ability/
                # tera-type/move-set from spatial reps plus recurrent belief
                # state. Opp action: predict the opp's chosen (a1, a2, tera)
                # using memory output plus both-side actives. Trainer-private
                # labels come from the buffer's retroactive fill on done.
                hidden_info_loss = None
                opp_action_loss = None
                if hidden_info_enabled:
                    aux_labels_mb = {
                        key: value[:, mb_envs] for key, value in aux_labels_all.items()
                    }
                    aux_label_valid_mb = aux_label_valid_all[:, mb_envs]
                    hidden_info_out = self.agent.hidden_info_head(
                        opp_reps_seq_mb,
                        ctx_seq_mb,
                    )
                    hidden_info_loss = self._hidden_info_loss(
                        logits=hidden_info_out,
                        labels=aux_labels_mb,
                        label_valid=aux_label_valid_mb,
                    )
                    loss = loss + self.config.aux_hidden_info_coef * hidden_info_loss

                if opp_action_enabled:
                    opp_action_out = self.agent.opponent_action_head(
                        ctx_seq_mb.reshape(T * E_mb, -1),
                        opp_active_reps_seq_mb.reshape(T * E_mb, 2, -1),
                        active_reps_seq_mb.reshape(T * E_mb, 2, -1),
                    )
                    opp_actions_mb = opp_actions_all[:, mb_envs]
                    opp_actions_valid_mb = opp_actions_valid_all[:, mb_envs]
                    opp_action_loss = self._opp_action_loss(
                        logits=opp_action_out,
                        labels=opp_actions_mb,
                        valid=opp_actions_valid_mb,
                    )
                    loss = loss + self.config.aux_opp_action_coef * opp_action_loss

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
                    if opp_action_loss is not None:
                        support_names.append("memory_grad_opp_action_valid_steps")
                        support_tensors.append(opp_actions_valid_mb.sum())
                    support_values = (
                        torch.stack(support_tensors).detach().to(device="cpu").tolist()
                        if support_tensors
                        else []
                    )
                    support = dict(zip(support_names, support_values, strict=True))
                    objectives = {
                        "actor": policy_loss + self.config.entropy_coef * entropy_loss,
                        "value": self.config.value_coef * value_loss,
                    }
                    if hidden_info_loss is not None:
                        objectives["hidden_info"] = (
                            self.config.aux_hidden_info_coef * hidden_info_loss
                        )
                    if opp_action_loss is not None:
                        objectives["opp_action"] = (
                            self.config.aux_opp_action_coef * opp_action_loss
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
                    metric_sums["entropy"].add_((-entropy_loss).detach())
                    base_tera_entropy_sum.add_(
                        base_tera_entropies_flat[base_tera_real_choices_flat]
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
                    metric_sums["approx_kl"].add_(approx_kl)
                    if hidden_info_enabled:
                        metric_sums["aux_hidden_info_loss"].add_(
                            hidden_info_loss.detach()
                        )
                    if opp_action_enabled:
                        metric_sums["aux_opp_action_loss"].add_(
                            opp_action_loss.detach()
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
        base_tera_support = base_tera_real_choice_evaluations.item()
        averaged["base_tera_real_choice_evaluations"] = base_tera_support
        averaged["base_tera_entropy"] = (
            base_tera_entropy_sum.item() / base_tera_support
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
        averaged.update(memory_gradient_diagnostics)
        if skipped_nonfinite > 0:
            averaged["skipped_nonfinite_minibatches"] = float(skipped_nonfinite)
        return averaged

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
    def _opp_action_loss(
        logits: dict[str, torch.Tensor],
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Masked CE on opp a1 / a2 / tera using observed opponent actions."""
        labels_flat = labels.reshape(-1, 3)
        valid_flat = valid.reshape(-1)
        a1_labels = labels_flat[:, 0]
        a2_labels = labels_flat[:, 1]
        tera_labels = labels_flat[:, 2]

        a1_mask = valid_flat & (a1_labels >= 0)
        a2_mask = valid_flat & (a2_labels >= 0)
        tera_mask = valid_flat & (tera_labels >= 0)

        component_losses: list[torch.Tensor] = []
        if a1_mask.any():
            component_losses.append(
                F.cross_entropy(logits["opp_a1_logits"][a1_mask], a1_labels[a1_mask])
            )
        if a2_mask.any():
            component_losses.append(
                F.cross_entropy(logits["opp_a2_logits"][a2_mask], a2_labels[a2_mask])
            )
        if tera_mask.any():
            component_losses.append(
                F.cross_entropy(
                    logits["opp_tera_logits"][tera_mask], tera_labels[tera_mask]
                )
            )
        if not component_losses:
            return logits["opp_a1_logits"].sum() * 0.0
        return torch.stack(component_losses).mean()

    def update_team_preview(
        self,
        data: list[tuple[dict, np.ndarray, float, np.ndarray]],
    ) -> dict[str, float]:
        """REINFORCE update for team preview decisions.

        Called once per PPO update cycle using all team preview decisions made
        during the rollout. Each entry is ``(obs_dict, picks_np, episode_reward,
        opponent_picks_np)`` where episode_reward is +1 for a win and -1 for a loss.

        Uses the value function as a baseline to reduce gradient variance.

        Returns a dict of preview-loss metrics for logging.
        """
        obs_batch = {
            k: torch.from_numpy(np.stack([d[0][k] for d in data])).to(self.device)
            for k in data[0][0]
        }
        picks_batch = (
            torch.from_numpy(np.stack([d[1] for d in data])).long().to(self.device)
        )
        rewards_batch = torch.tensor(
            [d[2] for d in data], dtype=torch.float32, device=self.device
        )
        opponent_picks_batch = (
            torch.from_numpy(np.stack([d[3] for d in data])).long().to(self.device)
        )

        out = self.agent.team_preview(obs_batch, picks=picks_batch)
        log_prob = out["log_prob"]
        entropy = out["entropy"]
        baselines = out["preview_value"]
        bring_logits = out["opponent_bring_logits"]
        lead_pair_logits = out["opponent_lead_pair_logits"]

        # Preview actor-critic: use a dedicated preview value head as the baseline.
        advantages = rewards_batch - baselines.detach()
        policy_loss = -(log_prob * advantages).mean()
        entropy_bonus = self.config.entropy_coef * entropy.mean()
        value_loss = 0.5 * (baselines - rewards_batch).pow(2).mean()

        team_size = bring_logits.shape[1]
        bring_targets = torch.zeros_like(bring_logits)
        bring_targets.scatter_(1, opponent_picks_batch, 1.0)
        valid_slots = (~obs_batch["opp_mask"]).to(bring_logits.dtype)
        bring_loss_raw = F.binary_cross_entropy_with_logits(
            bring_logits, bring_targets, reduction="none"
        )
        bring_loss = (bring_loss_raw * valid_slots).sum() / valid_slots.sum().clamp(
            min=1.0
        )

        lead_targets = (
            opponent_picks_batch[:, 0] * team_size + opponent_picks_batch[:, 1]
        )
        lead_loss = F.cross_entropy(lead_pair_logits, lead_targets)

        loss = (
            policy_loss
            - entropy_bonus
            + self.config.team_preview_value_coef * value_loss
            + self.config.team_preview_bring_coef * bring_loss
            + self.config.team_preview_lead_coef * lead_loss
        )

        if not self._apply_update_step(loss):
            return {"team_preview_skipped_nonfinite": 1.0}

        return {
            "team_preview_loss": loss.item(),
            "team_preview_policy_loss": policy_loss.item(),
            "team_preview_value_loss": value_loss.item(),
            "team_preview_entropy": entropy.mean().item(),
            "team_preview_bring_loss": bring_loss.item(),
            "team_preview_lead_loss": lead_loss.item(),
        }

    def save(self, path: str, *, training_state_id: str | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        state_id = training_state_id or uuid.uuid4().hex
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("training_state_id must be a non-empty string.")
        try:
            torch.save(
                {
                    "checkpoint_type": "trainer",
                    "format_version": TRAINER_CHECKPOINT_FORMAT_VERSION,
                    "model_metadata": _model_checkpoint_metadata(self.agent),
                    "agent": self.agent.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "global_step": self.global_step,
                    "training_state_id": state_id,
                    "rng_state": _capture_rng_state(),
                },
                temporary,
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

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

        checkpoint_state = _validate_model_checkpoint_metadata(
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
        training_state_id = checkpoint["training_state_id"]
        if not isinstance(training_state_id, str) or not training_state_id:
            raise ValueError(
                "Trainer checkpoint training_state_id must be a non-empty string."
            )
        rng_state = _validate_restorable_rng_state(checkpoint["rng_state"])

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
        self.loaded_training_state_id = training_state_id
        _restore_rng_state(rng_state)
