"""Compare two battle policies on reciprocal Regulation F matchups."""

import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path

import numpy as np
from pikapro import rl
import torch

from pokerl.data import GameData
from pokerl.env.vector_inputs import (
    HOLD,
    VectorActionCodec,
    configure_player_inputs,
    forced_passes,
    model_inputs,
    remember_actions,
)
from pokerl.eval.fixed_bot import load_agent_for_evaluation
from pokerl.eval.metrics import outcome_summary
from pokerl.eval.schedule import balanced_distinct_team_pairs
from pokerl.team_pool import load_team_pool


def _commands_for_seat(
    env, state, block, obs, previous, codec, active, commands, side, policy, hidden
):
    ready = active & state.needs_action[:, side]
    preview = ready & state.masks[:, side, 2 * env.space.slot_size :].any(axis=-1)
    automatic = ready & ~preview & forced_passes(block, side)
    commands[automatic, side, 0] = codec.both_pass
    for selecting_team, lanes in (
        (True, np.flatnonzero(preview)),
        (False, np.flatnonzero(ready & ~preview & ~automatic)),
    ):
        if not len(lanes):
            continue
        rows = 2 * lanes + side
        inputs = {key: torch.from_numpy(values[rows]) for key, values in obs.items()}
        if selecting_team:
            picks = policy.team_preview(
                inputs,
                include_value=False,
                include_policy_stats=False,
                include_opponent_preview=False,
            )["picks"].numpy()
            commands[lanes, side, 0] = codec.preview[tuple(picks.T)]
        else:
            masks = {
                key: torch.from_numpy(block[key][lanes, side])
                for key in ("joint", "can_tera")
            }
            out = policy(
                inputs,
                tuple(h[:, lanes] for h in hidden) if hidden is not None else None,
                action_masks=masks,
                include_value=False,
                include_policy_stats=False,
            )
            actions = out["actions"].numpy()
            previous[lanes, side] = remember_actions(
                actions, obs["own_request_moves"][rows], obs["own_species"][rows]
            )
            if hidden is not None:
                for h, updated in zip(hidden, out["hidden"], strict=True):
                    h[:, lanes] = updated
            commands[lanes, side, 0] = codec.battle_choices(
                actions, block["teams"][lanes, side, 0, :, 12]
            )


@torch.inference_mode()
def play_batch(policies, data, pairs, battle_seed, candidate_seat):
    n = len(pairs)
    games = [None] * n
    with closing(
        rl.VecEnv(
            "gen9vgc",
            n=n,
            teams=[(a.packed_team, b.packed_team) for a, b in pairs],
            threads=2,
            masks="factored",
            switch_by="position",
            hp="percent",
            reward="terminal",
            opponent="external",
            record=True,
            spin_us=0,
        )
    ) as env:
        configure_player_inputs(env, data)
        codec = VectorActionCodec(env.space)
        previous = np.zeros((n, 2, 2, 5), dtype=np.int64)
        hidden = [policy.initial_hidden(n) for policy in policies]
        active = np.ones(n, dtype=bool)
        state = env.reset(seed=battle_seed)
        while active.any():
            block = env._env.v6_player_inputs()
            obs = model_inputs(block, previous.reshape(-1, 2, 5))
            commands = np.full((n, 2, 2), HOLD, dtype=np.int64)
            commands[active] = -1
            for side, policy in enumerate(policies):
                _commands_for_seat(
                    env,
                    state,
                    block,
                    obs,
                    previous,
                    codec,
                    active,
                    commands,
                    side,
                    policy,
                    hidden[side],
                )
            state = env.step(commands)
            errors = env.take_errors()
            if (
                errors
                or state.truncated[active].any()
                or (state.status[active] == rl.STATUS["rejected"]).any()
            ):
                raise RuntimeError(f"Pikapro interrupted a match: {errors!r}")
            for lane in np.flatnonzero(active & state.terminated):
                record = env.record(int(lane))
                replay = rl.replay("gen9vgc", record, log=False)
                reward = int(state.reward[lane, candidate_seat])
                winner = replay.winner
                expected = (
                    0 if winner is None else (1 if winner == candidate_seat else -1)
                )
                if (
                    not replay.over
                    or reward != expected
                    or state.reward[lane].sum() != 0
                ):
                    raise RuntimeError(
                        "The replay and vector environment disagree on the result."
                    )
                games[lane] = {
                    "teams": [team.file for team in pairs[lane]],
                    "candidate_seat": candidate_seat,
                    "seed": record["seed"],
                    "turns": replay.turn,
                    "reward": reward,
                }
                active[lane] = False
    return games


def compare(candidate, opponent, manifest, pairs, seed):
    if pairs < 1:
        raise ValueError("At least one reciprocal matchup is required.")
    data = GameData()
    policies = [
        load_agent_for_evaluation(
            checkpoint_path=str(path),
            device=torch.device("cpu"),
            game_data=data,
        )
        for path in (candidate, opponent)
    ]
    pool = load_team_pool(manifest, usage="evaluation")
    schedule = balanced_distinct_team_pairs(pool.entries, games=pairs, seed=seed)
    seeds = np.random.default_rng(seed)
    games = []
    for start in range(0, len(schedule), 32):
        batch = schedule[start : start + 32]
        battle_seed, forward_seed, reverse_seed = map(
            int, seeds.integers(2**31, size=3)
        )
        for side, policy_seed in enumerate((forward_seed, reverse_seed)):
            torch.manual_seed(policy_seed)
            seats = policies if side == 0 else policies[::-1]
            games.extend(play_batch(seats, data, batch, battle_seed, side))
    # Resample entire unordered roster pairs, preserving seats and repeats.
    scores = defaultdict(list)
    for game in games:
        scores[tuple(sorted(game["teams"]))].append((game["reward"] + 1) / 2)
    totals = np.array([(sum(group), len(group)) for group in scores.values()])
    rng = np.random.default_rng(541)
    sample = totals[rng.integers(len(totals), size=(10000, len(totals)))].sum(axis=1)
    summary = outcome_summary(game["reward"] for game in games)
    return {
        "candidate": str(candidate),
        "opponent": str(opponent),
        "team_pool_manifest": str(manifest),
        "pairs": pairs,
        "seed": seed,
        **summary,
        "roster_pair_bootstrap_95": np.quantile(
            sample[:, 0] / sample[:, 1], [0.025, 0.975]
        ).tolist(),
        "games_detail": games,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("opponent", type=Path)
    parser.add_argument(
        "--pairs", type=int, default=210, help="Team matchups, two games per matchup."
    )
    parser.add_argument("--seed", type=int, default=771)
    parser.add_argument(
        "--team-pool-manifest",
        type=Path,
        default=Path("data/team_pools/gen9vgc2026regf_heldout/manifest.json"),
    )
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = compare(
        args.candidate, args.opponent, args.team_pool_manifest, args.pairs, args.seed
    )
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_path.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
    print(
        json.dumps({k: v for k, v in result.items() if k != "games_detail"}, indent=2)
    )


if __name__ == "__main__":
    main()
