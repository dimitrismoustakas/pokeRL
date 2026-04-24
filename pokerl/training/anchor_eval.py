"""Anchor evaluation harness for detecting strategic cycling in self-play.

Background: LEAGUE_IMPROVEMENTS.md §A — before changing the training algorithm
to address suspected non-transitive cycling, we first measure whether it is
actually happening. Anchors are frozen checkpoints taken at specified fractions
of the training run and kept permanently outside the rolling league pool. Every
``anchor_eval_interval_updates`` PPO updates, the live agent plays
``anchor_eval_games_per_anchor`` games against each existing anchor, and the
per-anchor win-rate curve over training time is logged for later inspection.

If an anchor's win-rate curve flatlines or decreases, cycling is real. If every
anchor's curve keeps rising, self-play is working and further interventions can
be deferred.
"""

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import EnvConfig, TrainingConfig
from ..data import GameData
from ..env.showdown_env import BattleEnv, ParallelBattleEnv
from ..model.agent import PokeAgent
from .league import League


_ANCHOR_SUBDIR = "anchors"
_ANCHOR_STATE_FILE = "anchor_state.json"


@dataclass
class AnchorRecord:
    """A permanent reference checkpoint."""

    fraction: float
    step: int
    path: str


@dataclass
class AnchorEvalRow:
    """One row in the per-anchor evaluation log."""

    anchor_fraction: float
    anchor_step: int
    anchor_path: str
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    ci_low: float
    ci_high: float


def _wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI on a win rate. Avoids degenerate 0/1 CIs that the
    normal approximation produces when wins == 0 or wins == games.
    """
    if games == 0:
        return (0.0, 0.0)
    p = wins / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    half = (z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


class AnchorEvaluator:
    """Manages anchor snapshots and periodic win-rate evaluation vs each anchor.

    Owns a dedicated ``ParallelBattleEnv`` whose opponents are frozen
    ``LeaguePlayer`` instances swapped between anchors via
    ``stage_next_opponent_checkpoint``. Inference for anchor checkpoints flows
    through the League's shared ``_LeagueInferenceService`` so multi-env anchor
    forward passes are still batched.

    Lifecycle:
        evaluator = AnchorEvaluator(...)
        # in training loop:
        evaluator.maybe_snapshot(agent, global_step, total_timesteps)
        if update % interval == 0:
            result = evaluator.evaluate(update=update, global_step=global_step)
    """

    def __init__(
        self,
        train_config: TrainingConfig,
        env_config: EnvConfig,
        game_data: GameData,
        agent: PokeAgent,
        league: League,
        server_urls: list[str],
        team_preview_fn,
        team_preview_batcher,
        team_pair_provider,
        device: torch.device,
    ):
        self._train_config = train_config
        self._env_config = env_config
        self._game_data = game_data
        self._agent = agent
        self._league = league
        self._device = device
        self._server_urls = server_urls
        self._team_preview_fn = team_preview_fn
        self._team_preview_batcher = team_preview_batcher
        self._team_pair_provider = team_pair_provider

        self._anchor_dir = Path(train_config.checkpoint_dir) / _ANCHOR_SUBDIR
        self._anchor_dir.mkdir(parents=True, exist_ok=True)
        self._eval_log_path = Path(train_config.checkpoint_dir) / "anchor_eval.jsonl"
        self._eval_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._anchors: list[AnchorRecord] = []
        self._load_state()

        self._parallel_env: ParallelBattleEnv | None = None
        self._initial_reset_done = False
        self._eval_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_obs = None
        self._last_masks = None

    # ------------------------------------------------------------------
    # Snapshotting
    # ------------------------------------------------------------------

    def maybe_snapshot(
        self,
        agent: PokeAgent,
        global_step: int,
        total_timesteps: int,
    ) -> AnchorRecord | None:
        """Save a new anchor if ``global_step`` has just crossed an unsaved
        fractional threshold. Returns the new AnchorRecord, or None.
        """
        if not self._train_config.anchor_eval_enabled:
            return None

        taken_fractions = {record.fraction for record in self._anchors}
        for fraction in self._train_config.anchor_fractions:
            if fraction in taken_fractions:
                continue
            threshold = int(fraction * total_timesteps)
            if global_step < threshold:
                continue

            path = str(self._anchor_dir / f"step_{global_step}.pt")
            torch.save({"agent": agent.state_dict()}, path)
            record = AnchorRecord(fraction=fraction, step=global_step, path=path)
            self._anchors.append(record)
            self._league.protect_path(path)
            self._league.inference_service.ensure_checkpoint(path)
            self._save_state()
            return record

        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def has_anchors(self) -> bool:
        return bool(self._anchors)

    def evaluate(self, update: int, global_step: int) -> dict | None:
        """Play ``games_per_anchor`` games of live agent vs each existing anchor
        and write one JSONL row + return a summary dict. No-op if there are no
        anchors yet.

        Anchors are evaluated sequentially, sharing one persistent
        ``ParallelBattleEnv``. Between anchors the staged checkpoint swap is
        picked up only by the next ``BattleEnv.reset()``, which happens on the
        auto-reset that fires when the previous anchor's in-flight battle
        ends. Explicitly calling ``parallel_env.reset()`` mid-eval would start
        a new battle while the previous one is still running, which poke-env
        does not handle. Completions whose recorded opponent checkpoint path
        does not match the anchor currently under evaluation are carryover
        from the previous anchor and are skipped.
        """
        if not self._train_config.anchor_eval_enabled or not self._anchors:
            return None

        if self._parallel_env is None:
            self._build_eval_env()

        eval_started = time.perf_counter()
        rows: list[AnchorEvalRow] = []
        was_training = self._agent.training
        self._agent.eval()
        try:
            for index, record in enumerate(self._anchors):
                row = self._evaluate_anchor(record, is_first=(index == 0))
                rows.append(row)
        finally:
            if was_training:
                self._agent.train()

        eval_sec = time.perf_counter() - eval_started
        summary = {
            "timestamp": time.time(),
            "update": update,
            "global_step": global_step,
            "eval_sec": eval_sec,
            "per_anchor": [row.__dict__ for row in rows],
        }
        with open(self._eval_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")

        return summary

    def _evaluate_anchor(self, record: AnchorRecord, is_first: bool) -> AnchorEvalRow:
        assert self._parallel_env is not None
        target_games = self._train_config.anchor_eval_games_per_anchor

        for env in self._parallel_env.envs:
            if not env.stage_next_opponent_checkpoint(record.path):
                raise RuntimeError(
                    "AnchorEvaluator: eval env did not accept staged anchor "
                    f"checkpoint (opponent type unexpected). path={record.path}"
                )

        if is_first and not self._initial_reset_done:
            # First-ever evaluate() call — envs haven't started a battle yet.
            obs, masks = self._parallel_env.reset()
            self._eval_hidden = self._agent.initial_hidden(len(self._parallel_env.envs))
            self._initial_reset_done = True
        else:
            obs = self._last_obs
            masks = self._last_masks
            if obs is None or masks is None:
                # Defensive: should not happen after the first evaluate(), but
                # fall back to an explicit reset rather than step with None.
                obs, masks = self._parallel_env.reset()
                self._eval_hidden = self._agent.initial_hidden(len(self._parallel_env.envs))

        wins = 0
        losses = 0
        draws = 0
        games = 0
        while games < target_games:
            obs_t = self._obs_to_tensors(obs)
            masks_t = self._masks_to_tensors(masks)
            with torch.inference_mode():
                out = self._agent(
                    obs_t,
                    self._eval_hidden,
                    action_masks=masks_t,
                    tera_exploration_prob=0.0,
                )
            actions = out["actions"].cpu().numpy()
            self._eval_hidden = out["hidden"]

            obs, masks, rewards, dones, infos = self._parallel_env.step(actions)

            if self._eval_hidden is not None:
                h, c = self._eval_hidden
                not_done = (
                    torch.from_numpy(~dones)
                    .to(h.dtype)
                    .view(1, dones.shape[0], 1)
                    .to(h.device)
                )
                self._eval_hidden = (h * not_done, c * not_done)

            for info in infos:
                if "episode_reward" not in info:
                    continue
                # Ignore carryover battles from the previous anchor. The env
                # records the opponent checkpoint path as of reset() time, so
                # games labelled with a different path were started before
                # our stage_next_opponent_checkpoint call took effect.
                if info.get("episode_opponent_checkpoint_path") != record.path:
                    continue
                if games >= target_games:
                    break
                reward = float(info["episode_reward"])
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
                games += 1

        self._last_obs = obs
        self._last_masks = masks

        win_rate = wins / games if games else 0.0
        ci_low, ci_high = _wilson_interval(wins, games)
        return AnchorEvalRow(
            anchor_fraction=record.fraction,
            anchor_step=record.step,
            anchor_path=record.path,
            games=games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=win_rate,
            ci_low=ci_low,
            ci_high=ci_high,
        )

    # ------------------------------------------------------------------
    # Env construction
    # ------------------------------------------------------------------

    def _build_eval_env(self) -> None:
        from dataclasses import replace

        eval_num_envs = self._train_config.anchor_eval_num_envs
        envs: list[BattleEnv] = []
        for i in range(eval_num_envs):
            env_server_url = self._server_urls[i % len(self._server_urls)]
            env_instance_config = replace(self._env_config, server_url=env_server_url)
            opponent_factory = self._make_opponent_factory(env_instance_config)
            env = BattleEnv(
                env_instance_config,
                self._game_data,
                opponent_factory,
                team_preview_fn=self._team_preview_fn,
                team_preview_batcher=self._team_preview_batcher,
                team_pair_provider=self._team_pair_provider,
            )
            envs.append(env)
        self._parallel_env = ParallelBattleEnv(envs)

    def _make_opponent_factory(self, env_instance_config: EnvConfig):
        """Build an opponent factory bound to the first anchor. The real
        anchor swap happens via ``stage_next_opponent_checkpoint`` before each
        anchor's eval, so the initial binding only matters as a placeholder
        that makes the underlying ``LeaguePlayer`` swappable.
        """
        league = self._league
        server_url = env_instance_config.server_url

        def factory(loop: asyncio.AbstractEventLoop):
            # Anchors are registered with the inference service by
            # maybe_snapshot(), so ensure_checkpoint() is cheap.
            bootstrap_path = self._anchors[0].path if self._anchors else league.sample_checkpoint_path()
            if bootstrap_path is None:
                raise RuntimeError(
                    "AnchorEvaluator: cannot build opponent — no anchors and empty league."
                )
            opponent = league.make_opponent(
                loop,
                server_url=server_url,
                checkpoint_path=bootstrap_path,
            )
            if opponent is None:
                raise RuntimeError(
                    "AnchorEvaluator: league.make_opponent returned None with a valid checkpoint path."
                )
            return opponent

        return factory

    # ------------------------------------------------------------------
    # Tensor plumbing (mirrors what scripts/train.py does for training)
    # ------------------------------------------------------------------

    def _obs_to_tensors(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(np.ascontiguousarray(value)).to(self._device)
            for key, value in obs.items()
        }

    def _masks_to_tensors(self, masks: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(np.ascontiguousarray(value)).to(self._device).bool()
            for key, value in masks.items()
        }

    # ------------------------------------------------------------------
    # Persistence and teardown
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        return self._anchor_dir / _ANCHOR_STATE_FILE

    def _save_state(self) -> None:
        payload = [
            {"fraction": r.fraction, "step": r.step, "path": r.path}
            for r in self._anchors
        ]
        self._state_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        state_path = self._state_path()
        if not state_path.exists():
            return
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        missing = 0
        for entry in payload:
            path = entry["path"]
            if not Path(path).exists():
                missing += 1
                continue
            self._anchors.append(
                AnchorRecord(
                    fraction=float(entry["fraction"]),
                    step=int(entry["step"]),
                    path=path,
                )
            )
            self._league.protect_path(path)
            self._league.inference_service.ensure_checkpoint(path)
        if missing:
            print(f"AnchorEvaluator: {missing} anchor file(s) missing on disk, skipping.")

    def close(self) -> None:
        if self._parallel_env is not None:
            self._parallel_env.close()
            self._parallel_env = None

    @property
    def anchors(self) -> list[AnchorRecord]:
        return list(self._anchors)

    @property
    def eval_log_path(self) -> Path:
        return self._eval_log_path
