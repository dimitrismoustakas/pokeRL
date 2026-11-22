from contextlib import closing
from types import SimpleNamespace

import numpy as np
from pikapro import rl
import pytest
import torch

from pokerl.data import GameData
from pokerl.env.actions import SWITCH_OFFSET
from pokerl.env.vector import PikaproVectorEnv
from pokerl.env.vector_inputs import (
    HOLD,
    RESET,
    VectorActionCodec,
    configure_player_inputs,
    model_inputs,
)
from pokerl.team_pool import load_team_pool


@pytest.fixture(scope="module")
def teams():
    pool = load_team_pool("data/team_pools/gen9vgc2026regf/manifest.json", seed=51)
    return pool.sample_packed_pair()


def test_reset_stream_distinguishes_lanes_and_episodes_on_identical_teams(teams):
    with closing(
        rl.VecEnv("gen9vgc", n=3, teams=[teams] * 3, masks="factored", record=True)
    ) as env:

        def episode_seeds():
            env.reset(seed=73)
            seeds = [env.record(lane)["seed"] for lane in range(3)]
            commands = np.full((3, 2, 2), HOLD, dtype=np.int64)
            commands[1] = RESET
            env.step(commands)
            assert [env.record(i)["seed"] for i in (0, 2)] == [seeds[0], seeds[2]]
            return [*seeds, env.record(1)["seed"]]

        first = episode_seeds()
        assert len(set(first)) == 4
        assert episode_seeds() == first


@pytest.mark.parametrize("owner", (0, 1))
def test_private_item_changes_only_its_owners_view(teams, owner):
    changed = list(teams)
    members = changed[owner].split("]")
    # The sixth member stays on the bench throughout this replay.
    fields = members[5].split("|")
    fields[2] = "leftovers" if fields[2].lower() != "leftovers" else "lifeorb"
    members[5] = "|".join(fields)
    changed[owner] = "]".join(members)
    with closing(
        rl.VecEnv(
            "gen9vgc",
            n=2,
            teams=[teams, tuple(changed)],
            masks="factored",
            switch_by="position",
            hp="percent",
            opponent="external",
        )
    ) as env:
        configure_player_inputs(env, GameData())
        for lane in range(2):
            env._env.set_battle_seed(lane, [23, 9, 12, 19])
        env.reset(seed=138)
        codec = VectorActionCodec(env.space)
        previous = np.zeros((4, 2, 5), dtype=np.int64)
        for preview in (True, False):
            block = env._env.v6_player_inputs()
            obs = model_inputs(block, previous)
            viewer = 1 - owner
            for name, rows in obs.items():
                np.testing.assert_array_equal(
                    rows[viewer], rows[2 + viewer], err_msg=name
                )
            for name in ("joint", "can_tera"):
                np.testing.assert_array_equal(
                    block[name][0, viewer], block[name][1, viewer]
                )
            if preview:
                assert not np.array_equal(
                    obs["own_item"][owner], obs["own_item"][2 + owner]
                )
                commands = np.full((2, 2, 2), -1, dtype=np.int64)
                commands[:, :, 0] = codec.preview[0, 1, 2, 3]
                env.step(commands)


def test_opening_eject_pack_replacement_precedes_first_learner_decision(monkeypatch):
    pool = load_team_pool("data/team_pools/gen9vgc2026regf/manifest.json")
    eject = next(e.packed_team for e in pool.entries if "052_sun-euic" in e.file)
    intimidate = pool.entries[0].packed_team
    assert (
        "ejectpack" in eject.split("]")[1].lower()
        and "intimidate" in intimidate.split("]")[3].lower()
    )
    eject_lead = [1, 0, 2, 3]
    intimidate_lead = [3, 0, 1, 2]
    replacements = []

    def choose_replacement(paths, rows, masks, hidden):
        assert len(paths) == 1
        joint = masks[0]["joint"]
        first = np.flatnonzero(joint[0])[0]
        second = np.flatnonzero(joint[1 + first])[0]
        replacements.append([first, second, 0])
        return [(replacements[-1], None)]

    league = SimpleNamespace(
        device=torch.device("cpu"),
        vector_previews=lambda paths, rows: np.array([intimidate_lead, eject_lead]),
        vector_actions=choose_replacement,
    )
    monkeypatch.setattr(
        "pokerl.env.vector.TeamPreviewCache.predict",
        lambda *args, **kwargs: {
            "picks": torch.tensor([eject_lead, intimidate_lead]),
            "log_prob": torch.zeros(2),
            "preview_value": torch.zeros(2),
        },
    )
    data = GameData()
    pairs = iter([(eject, intimidate), (intimidate, eject)])
    with closing(
        PikaproVectorEnv(
            2,
            data,
            league,
            None,
            0,
            lambda: next(pairs),
            {0: "frozen"},
            seed=732,
            collect_timing=False,
        )
    ) as env:
        assert env.state.needs_action[:, 0].all()
        assert len(replacements) == 1
        assert env.pending_intents == [None, None]
        starts = env.drain_team_preview_starts()
        assert len(starts) == 2
        for lane, sample in starts:
            opponent = (
                pool.entries[0]
                if lane == 0
                else next(e for e in pool.entries if e.packed_team == eject)
            )
            picks = [intimidate_lead, eject_lead][lane]
            slots = [mon.split("|") for mon in opponent.packed_team.split("]")]
            labels = sample.opponent_aux_labels
            np.testing.assert_array_equal(
                labels["opp_mask"], ~np.isin(np.arange(6), picks)
            )
            np.testing.assert_array_equal(
                labels["opp_item"], [data.item_to_idx[mon[2].lower()] for mon in slots]
            )
            assert not sample.obs["opp_item"].any()

        # The learner chooses its own forced switch in lane zero. In lane one
        # the opponent has already switched and both players choose a move.
        _, masks = env.reset()
        assert not masks["joint"][0, 0, :SWITCH_OFFSET].any()
        assert masks["joint"][1, 0, :SWITCH_OFFSET].any()
