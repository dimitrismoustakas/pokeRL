"""Synchronous Regulation F rollouts from Pikapro player views."""

from collections import Counter
from time import perf_counter

import numpy as np
from pikapro import rl
import torch

from pokerl.env.observation import build_packed_team_aux_labels
from pokerl.env.vector_inputs import (
    DECODED_CHOICE,
    HOLD,
    RESET,
    VectorActionCodec,
    VectorInputRow,
    configure_player_inputs,
    forced_passes,
    model_inputs,
    opponent_intent_labels,
    remember_actions,
)
from pokerl.model.preview_cache import TeamPreviewCache
from pokerl.training.cuda_execution import cuda_model_execution
from pokerl.training.preview import TeamPreviewEpisodeStart


_INTENT_KEYS = (
    "opponent_intent_slot_labels",
    "opponent_intent_slot_masks",
    "opponent_intent_tera_labels",
    "opponent_intent_tera_masks",
)


class PikaproVectorEnv:
    def __init__(
        self,
        config,
        data,
        league,
        preview_batcher,
        team_pair_provider,
        assignments,
        *,
        seed,
        collect_timing,
    ):
        self.num_envs = config.num_envs
        self.threads = min(4, self.num_envs)
        self.data = data
        self.league = league
        self.preview_batcher = preview_batcher
        self.team_pair_provider = team_pair_provider
        self.assignments = dict(assignments)
        self.groups = np.arange(self.num_envs) % len(assignments)
        self.active_paths = [None] * self.num_envs
        self.opponent_hidden = [None] * self.num_envs
        self.pending_intents = [None] * self.num_envs
        self.previous = np.zeros((self.num_envs, 2, 2, 5), dtype=np.int64)
        self.preview_starts = []
        self.preview_cache = TeamPreviewCache()
        self.preview_version = preview_batcher.policy_version
        self.episode_start = np.zeros(self.num_envs)
        self.first_tera = np.full(self.num_envs, -1, dtype=np.int64)
        self.opponent_turn_calls = [Counter() for _ in range(self.num_envs)]
        self.collect_timing = collect_timing
        self.timing = Counter()
        self.pairs = [team_pair_provider() for _ in range(self.num_envs)]
        self.env = rl.VecEnv(
            "gen9vgc",
            n=self.num_envs,
            teams=self.pairs,
            threads=self.threads,
            masks="factored",
            switch_by="position",
            hp="percent",
            reward="terminal",
            opponent="external",
            spin_us=0,
        )
        try:
            configure_player_inputs(self.env, data)
            self.actions = VectorActionCodec(self.env.space)
            self.turn_column = self.env.observation["columns"].index("field.turn")
            self.state = self.env.reset(seed=seed)
            self._project()
            self._team_previews(np.arange(self.num_envs))
        except BaseException:
            self.env.close()
            raise

    def _project(self):
        started = perf_counter()
        self.block = self.env._env.v6_player_inputs()
        self.obs = model_inputs(self.block, self.previous.reshape(-1, 2, 5))
        if self.collect_timing:
            self.timing["native_observation_sec"] += perf_counter() - started

    def _current(self):
        return {k: v[::2] for k, v in self.obs.items()}, {
            k: self.block[k][:, 0] for k in ("joint", "can_tera")
        }

    def _hold(self):
        return np.full((self.num_envs, 2, 2), HOLD, dtype=np.int64)

    def _advance(self, commands):
        started = perf_counter()
        stepped = commands[:, 0, 0] != HOLD
        self.state = self.env.step(commands)
        errors = self.env.take_errors()
        if errors or self.state.truncated[stepped].any():
            raise RuntimeError(f"Pikapro interrupted a training battle: {errors!r}")
        rejected = (self.state.status == rl.STATUS["rejected"]) & stepped[:, None]
        if rejected.any():
            lane, side = map(int, np.argwhere(rejected)[0])
            code = int(commands[lane, side, 0])
            choice = (
                self.env.space.decode(code - DECODED_CHOICE)
                if code >= DECODED_CHOICE
                else code
            )
            request = self.env.battle(lane).request_json(side)
            raise ValueError(
                f"Lane {lane}, player {side + 1}, rejected {choice!r} on {request}"
            )
        if self.collect_timing:
            self.timing["native_step_and_checks_sec"] += perf_counter() - started
            self.timing["native_step_calls"] += 1
        self._project()

    def _rows(self, lanes, side):
        return [
            {key: array[2 * lane + side] for key, array in self.obs.items()}
            for lane in lanes
        ]

    @torch.no_grad()
    def _team_previews(self, lanes):
        started = perf_counter()
        if (
            not self.state.needs_action[lanes].all()
            or not self.state.masks[lanes, :, 2 * self.env.space.slot_size :]
            .any(axis=-1)
            .all()
        ):
            raise RuntimeError("Pikapro did not request both previews at reset.")
        version = self.preview_batcher.policy_version
        if self.preview_version != version:
            self.preview_cache = TeamPreviewCache()
            self.preview_version = version
        paths = [self.assignments[int(self.groups[lane])] for lane in lanes]
        for lane, path in zip(lanes, paths, strict=True):
            self.active_paths[lane] = path
            self.opponent_hidden[lane] = None
            self.opponent_turn_calls[lane].clear()
        self.episode_start[lanes] = started
        self.first_tera[lanes] = -1
        rows = self._rows(lanes, 0)
        with cuda_model_execution(self.league.device):
            result = self.preview_cache.predict(
                self.league.agent_template, self.league.device, rows, include_value=True
            )
            picks = result["picks"].cpu().numpy()
            log_probs = result["log_prob"].cpu().numpy()
            values = result["preview_value"].cpu().numpy().reshape(-1)
        opponent_started = perf_counter()
        opponent_picks = self.league.vector_previews(paths, self._rows(lanes, 1))
        if self.collect_timing:
            self.timing["native_opponent_preview_sec"] += (
                perf_counter() - opponent_started
            )
        commands = self._hold()
        for side, orders in enumerate((picks, opponent_picks)):
            commands[lanes, side, 0] = self.actions.preview[tuple(orders.T)]
            commands[lanes, side, 1] = -1
        for i, lane in enumerate(lanes):
            labels = build_packed_team_aux_labels(
                self.data, 6, self.pairs[lane][1], selected_indices=opponent_picks[i]
            )
            self.preview_starts.append(
                (
                    int(lane),
                    TeamPreviewEpisodeStart(
                        rows[i],
                        picks[i],
                        log_probs[i],
                        values[i],
                        version,
                        opponent_picks[i],
                        labels,
                    ),
                )
            )
        self._advance(commands)
        if not (self.state.status[lanes] == rl.STATUS["submitted"]).all():
            raise RuntimeError("Pikapro did not accept a team preview.")
        if self.collect_timing:
            self.timing["native_preview_sec"] += perf_counter() - started
        self._resolve_opening(lanes)

    def _resolve_opening(self, lanes):
        # Intimidate can trigger an opposing Eject Pack before turn one. Let
        # that player choose its replacement before asking the learner to act.
        while len(lanes):
            forced = forced_passes(self.block, 0)
            lanes = lanes[~self.state.needs_action[lanes, 0] | forced[lanes]]
            if not len(lanes):
                return
            playing = np.zeros(self.num_envs, dtype=bool)
            playing[lanes] = True
            commands = self._hold()
            commands[lanes] = -1
            commands[playing & self.state.needs_action[:, 0], 0, 0] = (
                self.actions.both_pass
            )
            self._opponents(commands, playing)
            self._advance(commands)
            if self.state.terminated[lanes].any():
                raise RuntimeError("A battle ended before its first learner decision.")
            # These opponent actions preceded every learner action in the game.
            for lane in lanes:
                self.pending_intents[lane] = None

    def _issue(self, commands, side, lanes, actions):
        rows = np.arange(len(lanes))
        joint = self.block["joint"][lanes, side]
        can_tera = self.block["can_tera"][lanes, side]
        used = actions[:, 2] != 0
        if not (
            joint[rows, 0, actions[:, 0]].all()
            and joint[rows, 1 + actions[:, 0], actions[:, 1]].all()
            and can_tera[rows[used], actions[used, 2] - 1].all()
        ):
            raise ValueError("The policy selected an illegal action.")
        moves = self.obs["own_request_moves"][2 * lanes + side]
        species = self.obs["own_species"][2 * lanes + side]
        previous = remember_actions(actions, moves, species)
        self.previous[lanes, side] = previous
        commands[lanes, side, 0] = self.actions.battle_choices(
            actions, self.block["teams"][lanes, side, 0, :, 12]
        )
        if side == 1:
            targets = opponent_intent_labels(
                actions, previous, moves, species, joint, can_tera
            )
            for i, lane in enumerate(lanes):
                self.pending_intents[lane] = {
                    key: values[i]
                    for key, values in zip(_INTENT_KEYS, targets, strict=True)
                }

    def _turns(self, lanes):
        # The public basic observation caps the turn at 100.
        turns = np.rint(self.state.obs[lanes, 0, self.turn_column] * 100).astype(
            np.int64
        )
        for i in np.flatnonzero(turns == 100):
            turns[i] = self.env.view(int(lanes[i]), 0).turn
        return turns

    def _opponents(self, commands, playing):
        ready = playing & self.state.needs_action[:, 1]
        forced = ready & forced_passes(self.block, 1)
        commands[forced, 1, 0] = self.actions.both_pass
        self.previous[forced, 1] = 0
        for lane in np.flatnonzero(forced):
            self.pending_intents[lane] = None
        lanes = np.flatnonzero(ready & ~forced)
        if not len(lanes):
            return
        started = perf_counter()
        for lane, turn in zip(lanes, self._turns(lanes), strict=True):
            self.opponent_turn_calls[lane][int(turn)] += 1
        actions = []
        for lane, (action, hidden) in zip(
            lanes, self._opponent_actions(lanes), strict=True
        ):
            self.opponent_hidden[lane] = hidden
            actions.append(action)
        self._issue(commands, 1, lanes, np.stack(actions))
        if self.collect_timing:
            self.timing["native_opponent_battle_sec"] += perf_counter() - started

    def _opponent_actions(self, lanes):
        obs = {key: array[1::2] for key, array in self.obs.items()}
        masks = {key: self.block[key][:, 1] for key in ("joint", "can_tera")}
        return self.league.vector_actions(
            [self.active_paths[lane] for lane in lanes],
            [VectorInputRow(obs, lane) for lane in lanes],
            [VectorInputRow(masks, lane) for lane in lanes],
            [self.opponent_hidden[lane] for lane in lanes],
        )

    def _finish(self, lanes, rewards, infos):
        for lane, turn in zip(lanes, self._turns(lanes), strict=True):
            calls = self.opponent_turn_calls[lane]
            total = sum(calls.values())
            infos[lane].update(
                episode_reward=float(rewards[lane]),
                episode_opponent_checkpoint_path=self.active_paths[lane],
                episode_sec=perf_counter() - self.episode_start[lane],
                battle_tag=f"pikapro-{lane}-{self.state.episode[lane]}",
                battle_turn=int(turn),
                episode_first_tera_turn=int(self.first_tera[lane])
                if self.first_tera[lane] >= 0
                else None,
                episode_opponent_choice_calls=total,
                episode_opponent_repeated_turn_requests=total - len(calls),
                episode_opponent_max_calls_same_turn=max(calls.values(), default=0),
                final_observation={k: v[2 * lane].copy() for k, v in self.obs.items()},
            )
            self.pairs[lane] = self.team_pair_provider()
        self.env.set_pairs(lanes.tolist(), [self.pairs[i] for i in lanes])
        self.previous[lanes] = 0
        commands = self._hold()
        commands[lanes] = RESET
        self._advance(commands)
        self._team_previews(lanes)

    def step(self, actions):
        """Advance every lane to its next learner decision, resolving opponent-only requests."""
        started = perf_counter()
        tera = np.flatnonzero((actions[:, 2] != 0) & (self.first_tera < 0))
        self.first_tera[tera] = self._turns(tera)
        playing = np.ones(self.num_envs, dtype=bool)
        awaiting_resolution = playing.copy()
        rewards = np.zeros(self.num_envs, np.float32)
        dones = np.zeros(self.num_envs, bool)
        infos = [{} for _ in range(self.num_envs)]
        first = True
        while playing.any():
            commands = self._hold()
            commands[playing] = -1
            if first:
                self._issue(commands, 0, np.arange(self.num_envs), actions)
                first = False
            else:
                default = playing & self.state.needs_action[:, 0]
                if not forced_passes(self.block, 0)[default].all():
                    raise RuntimeError(
                        "A learner decision was skipped while resolving an opponent request."
                    )
                commands[default, 0, 0] = self.actions.both_pass
            self._opponents(commands, playing)
            self._advance(commands)
            resolved = playing & ~(self.state.status == rl.STATUS["unavailable"]).any(
                axis=1
            )
            for lane in np.flatnonzero(resolved):
                # A private reselection replaces the pending opponent target.
                if awaiting_resolution[lane] and self.pending_intents[lane] is not None:
                    infos[lane].update(self.pending_intents[lane])
                self.pending_intents[lane] = None
            awaiting_resolution[resolved] = False
            ended = np.flatnonzero(playing & self.state.terminated)
            if len(ended):
                if (
                    dones[ended].any()
                    or (self.state.reward[ended].sum(axis=1) != 0).any()
                ):
                    raise RuntimeError(
                        "Expected one zero-sum terminal result per game."
                    )
                dones[ended] = True
                rewards[ended] = self.state.reward[ended, 0]
                self._finish(ended, rewards, infos)
            playing &= ~(self.state.needs_action[:, 0] & ~forced_passes(self.block, 0))
        if self.collect_timing:
            self.timing["native_environment_sec"] += perf_counter() - started
        return *self._current(), rewards, dones, infos

    def reset(self):
        return self._current()

    def drain_team_preview_starts(self):
        starts, self.preview_starts = self.preview_starts, []
        return starts

    def wait_for_prepared_team_previews(self):
        if self.preview_starts:
            raise RuntimeError("A team preview was omitted from the rollout.")

    def refresh_prepared_team_previews(self):
        # No preview is prepared ahead of its episode on the synchronous path.
        pass

    def retained_battle_counts(self):
        return self.num_envs, self.num_envs

    def reset_timing(self):
        self.timing.clear()

    def snapshot_timing(self):
        return dict(self.timing)

    def close(self):
        self.env.close()
