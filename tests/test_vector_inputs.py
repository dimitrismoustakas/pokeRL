from contextlib import closing
from types import SimpleNamespace

import numpy as np
from pikapro import rl
import pytest
import torch

from pokerl.data import GameData
from pokerl.env.actions import SWITCH_OFFSET
from pokerl.env.observation import ITEM_LOST_OFFSET, PARADOX_BOOST_OFFSET
from pokerl.env.vector import PikaproVectorEnv
from pokerl.env.vector_inputs import (
    DECODED_CHOICE,
    HOLD,
    RESET,
    VectorActionCodec,
    configure_player_inputs,
    model_inputs,
    player_env,
)
from pokerl.team_pool import load_team_pool


@pytest.fixture(scope="module")
def teams():
    pool = load_team_pool("data/team_pools/gen9vgc2026regf/manifest.json", seed=51)
    return pool.sample_packed_pair()


def test_paradox_announcement_item_loss_and_damage_are_public():
    pool = load_team_pool("data/team_pools/gen9vgc2026regf/manifest.json")
    team = next(
        entry.packed_team for entry in pool.entries if "009_battleroom" in entry.file
    )
    lead, rest = team.split("]", 1)
    pairs = []
    for evs in ("4,,252,,,252", "4,,,252,,252"):
        fields = lead.split("|")
        fields[6] = evs
        pairs.append((team, "|".join(fields) + "]" + rest))
    data = GameData()
    previous = np.zeros((4, 2, 5), dtype=np.int64)
    with closing(player_env(pairs, threads=1)) as env:
        configure_player_inputs(env, data)
        for lane in range(2):
            env._env.set_battle_seed(lane, [23, 9, 12, 19])
        env.reset(seed=138)
        preview = model_inputs(env._env.v6_player_inputs(), previous)
        for rows in preview.values():
            np.testing.assert_array_equal(rows[0], rows[2])

        codec = VectorActionCodec(env.space)
        orders = np.full((2, 2, 2), -1, dtype=np.int64)
        orders[:, 0, 0] = codec.preview[5, 1, 2, 3]
        orders[:, 1, 0] = codec.preview[0, 1, 2, 3]
        env.step(orders)
        opening = model_inputs(env._env.v6_player_inputs(), previous)
        for seat, side in enumerate(("opp_", "own_")):
            for lane, boosted_stat in enumerate((4, 2)):  # Speed or Sp. Atk
                row = 2 * lane + seat
                slots = opening[side + "species"][row]
                mon = np.flatnonzero(slots == data.species_to_idx["fluttermane"]).item()
                state = opening[side + "continuous"][row, mon]
                assert state[ITEM_LOST_OFFSET] == 1
                assert (
                    opening[side + "item"][row, mon]
                    == data.item_to_idx["boosterenergy"]
                )
                np.testing.assert_array_equal(
                    state[PARADOX_BOOST_OFFSET:ITEM_LOST_OFFSET],
                    np.eye(5)[boosted_stat],
                )

        # Two attacks into Ogerpon take its public bar from 100 to 46 to 0.
        # In the other lane Flutter Mane switches out and keeps the item loss.
        for lane, choices in enumerate(
            (
                ("move 1 2, move 2 2", "move 1 2, move 2 1"),
                ("move 4, move 4", "switch 3, move 4"),
            )
        ):
            battle = env.battle(lane)
            for seat, choice in enumerate(choices):
                orders[lane, seat, 0] = DECODED_CHOICE + env.space.encode(
                    battle.parse_choice(seat, choice)
                )
        env.step(orders)
        assert not env.take_errors()
        block = env._env.v6_player_inputs()
        for seat, target in enumerate((3, 1)):
            damage = block["move_damage"][0, seat, :, target]
            np.testing.assert_allclose(damage[damage >= 0], [0.54, 0.46], atol=1e-6)
        switched = model_inputs(block, previous)
        for row, side in ((2, "opp_"), (3, "own_")):
            mon = np.flatnonzero(
                switched[side + "species"][row] == data.species_to_idx["fluttermane"]
            ).item()
            state = switched[side + "continuous"][row, mon]
            assert state[ITEM_LOST_OFFSET] == 1
            assert not state[PARADOX_BOOST_OFFSET:ITEM_LOST_OFFSET].any()


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
        vector_previews=lambda paths, observations, indices: np.array(
            [intimidate_lead, eject_lead]
        ),
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
