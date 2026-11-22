"""Map Pikapro's interleaved player views into the battle policy's named arrays."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pikapro

from pokerl.data import GameData
from pokerl.env.actions import (
    ACTIONS_PER_SLOT,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SWITCH,
    MAX_TEAM_SLOTS,
    NUM_MOVES,
    NUM_SLOT_INTENT_COMPONENTS,
    NUM_TARGETS,
    PASS_ACTION,
    SWITCH_OFFSET,
)
from pokerl.env.observation import ObservationBuilder


# Factored VecEnv commands, distinct from indices into its legal menu.
HOLD = np.iinfo(np.int64).min
RESET = HOLD + 1
DECODED_CHOICE = 1 << 32


@dataclass(slots=True)
class VectorInputRow(Mapping[str, np.ndarray]):
    """One battle lane in a shared batch of player observations or legal masks."""

    columns: dict[str, np.ndarray]
    lane: int

    def __getitem__(self, key):
        return self.columns[key][self.lane]

    def __iter__(self):
        return iter(self.columns)

    def __len__(self):
        return len(self.columns)


def forced_passes(block, side):
    joint = block["joint"][:, side]
    return (
        joint[:, 0, PASS_ACTION]
        & joint[:, 1 + PASS_ACTION, PASS_ACTION]
        & (joint.sum(axis=(1, 2)) == 2)
        & ~block["can_tera"][:, side].any(axis=1)
    )


def configure_player_inputs(env, data: GameData) -> None:
    builder = ObservationBuilder(data)
    metadata = np.stack((builder._move_types, builder._move_categories), axis=1)
    env._env.set_v6_vocab(
        data.move_to_idx,
        metadata.tolist(),
        data.type_to_idx,
        data.species_to_idx,
        data.abilities_to_idx,
        data.item_to_idx,
        builder._move_target_scopes.tolist(),
    )


def model_inputs(block, previous):
    def players(key):
        values = block[key]
        return values.reshape(-1, *values.shape[2:])

    teams = players("teams")
    obs = {}
    for side, prefix in enumerate(("own_", "opp_")):
        team = teams[:, side]
        for name, column in (
            ("species", 0),
            ("ability", 1),
            ("item", 2),
            ("roles", 10),
        ):
            obs[prefix + name] = team[:, :, column]
        obs[prefix + "types"] = team[:, :, 3:6]
        obs[prefix + "moves"] = team[:, :, 6:10]
        obs[prefix + "mask"] = team[:, :, 11].astype(bool)
        obs[prefix + "continuous"] = players("continuous")[:, side]
        obs[prefix + "last_move"] = players("last_moves")[:, side]
        active = team[:, :, 10, None] == np.arange(2)
        obs[prefix + "active_present"] = active.any(axis=1)
        obs[prefix + "active_indices"] = active.argmax(axis=1).astype(
            np.int64, copy=False
        )
    obs["opp_source_index"] = teams[:, 1, :, 12]
    for key in (
        "own_request_moves",
        "own_request_move_metadata",
        "field",
        "recent_event_flags",
        "recent_event_target_outcomes",
        "recent_event_target_activation_move",
    ):
        obs[key] = players(key)
    events = players("events")
    for column, name in enumerate(
        (
            "kind",
            "actor",
            "target",
            "move",
            "source_move",
            "move_scope",
            "species",
            "tera_type",
        )
    ):
        obs["recent_event_" + name] = events[:, :, column]
    obs["recent_event_mask"] = events[:, :, 0] == 0
    obs["recent_event_meta"] = players("meta")
    for column, name in enumerate(("kind", "move", "target", "switch_species")):
        obs["prev_own_" + name] = previous[:, :, column]
    obs["prev_own_tera"] = previous[:, :, 4].astype(bool)
    obs["prev_own_valid"] = previous[:, :, 0] != 0
    obs["active_mechanics"] = players("active_mechanics")
    obs["active_mechanics_move"] = players("active_mechanics_move")
    return obs


class VectorActionCodec:
    """Translate policy move targets, stable team slots and bring-four orders into Pikapro choices."""

    def __init__(self, space):
        self.move_codes = np.full((2, ACTIONS_PER_SLOT, 2), -1, dtype=np.int64)
        self.switch_codes = np.array(
            [
                space.encode_slot(pikapro.SlotAction("switch", i))
                for i in range(MAX_TEAM_SLOTS)
            ]
        )
        self.pass_code = space.encode_slot(pikapro.SlotAction("pass"))
        self.slot_size = space.slot_size
        self.both_pass = DECODED_CHOICE + self.pass_code * (space.slot_size + 1)
        self.move_codes[:, PASS_ACTION, :] = self.pass_code
        for slot in range(2):
            targets = (1, 2, -(2 - slot), -(1 + slot), None)
            for move in range(NUM_MOVES):
                for target, target_location in enumerate(targets):
                    for tera, gimmick in enumerate((None, "terastallize")):
                        choice = pikapro.SlotAction(
                            "move", move, target_location, gimmick
                        )
                        self.move_codes[slot, move * NUM_TARGETS + target, tera] = (
                            space.encode_slot(choice)
                        )
        self.preview = np.full((MAX_TEAM_SLOTS,) * 4, -1, dtype=np.int64)
        for order in permutations(range(MAX_TEAM_SLOTS), 4):
            self.preview[order] = (
                space.encode(pikapro.Choice.team(list(order))) - space.joint_size
            )

    def battle_choices(self, actions, team_positions):
        rows = np.arange(len(actions))
        codes = np.empty((len(actions), 2), dtype=np.int64)
        for slot in range(2):
            action = actions[:, slot]
            tera = (actions[:, 2] == slot + 1).astype(np.int64)
            codes[:, slot] = self.move_codes[slot, action, tera]
            switching = (SWITCH_OFFSET <= action) & (action < PASS_ACTION)
            positions = team_positions[
                rows[switching], action[switching] - SWITCH_OFFSET
            ]
            if (positions < 0).any():
                raise ValueError(
                    "A selected switch has no position in the current team."
                )
            codes[switching, slot] = self.switch_codes[positions]
        if (codes < 0).any():
            raise ValueError("A policy action has no native slot choice.")
        return DECODED_CHOICE + codes[:, 0] * self.slot_size + codes[:, 1]


def remember_actions(actions, moves, species):
    """Record the acting player's submitted move, target, switch species and Tera in request order."""
    rows = np.arange(len(actions))
    previous = np.zeros((len(actions), 2, 5), dtype=np.int64)
    for slot in range(2):
        action = actions[:, slot]
        moved = action < SWITCH_OFFSET
        switched = (action >= SWITCH_OFFSET) & (action < PASS_ACTION)
        previous[:, slot, 0] = np.where(
            moved, INTENT_MOVE, np.where(switched, INTENT_SWITCH, INTENT_PASS)
        )
        previous[moved, slot, 1] = moves[
            rows[moved], slot, action[moved] // NUM_TARGETS
        ]
        targets = np.array((3, 4, 2 - slot, 1 + slot, 5))
        previous[moved, slot, 2] = targets[action[moved] % NUM_TARGETS]
        previous[switched, slot, 3] = species[
            rows[switched], action[switched] - SWITCH_OFFSET
        ]
        previous[:, slot, 4] = moved & (actions[:, 2] == slot + 1)
    return previous


def opponent_intent_labels(actions, previous, moves, species, joint, can_tera):
    """Trainer-only targets. Supervise each component only when the opponent could choose it."""
    rows = np.arange(len(actions))
    labels = np.full((len(actions), 2, NUM_SLOT_INTENT_COMPONENTS), -1, dtype=np.int64)
    masks = np.zeros_like(labels, dtype=bool)
    for slot in range(2):
        action = actions[:, slot]
        legal = joint[:, 0] if slot == 0 else joint[rows, 1 + actions[:, 0]]
        moved = action < SWITCH_OFFSET
        switched = (action >= SWITCH_OFFSET) & (action < PASS_ACTION)
        move_choices = legal[:, :SWITCH_OFFSET].reshape(-1, NUM_MOVES, NUM_TARGETS)
        switch_choices = legal[:, SWITCH_OFFSET:PASS_ACTION]
        move = previous[:, slot, 1]
        switch = previous[:, slot, 3]
        target = action % NUM_TARGETS
        labels[:, slot, 0] = previous[:, slot, 0]
        labels[moved, slot, 1] = move[moved]
        labels[moved, slot, 2] = target[moved]
        labels[switched, slot, 3] = switch[switched]
        kinds = np.stack(
            (
                move_choices.any(axis=(1, 2)),
                switch_choices.any(axis=1),
                legal[:, PASS_ACTION],
            ),
            axis=1,
        )
        masks[:, slot, 0] = kinds.sum(axis=1) > 1
        same_move = moves[:, slot] == move[:, None]
        masks[:, slot, 1] = moved & (move_choices.any(axis=2) & ~same_move).any(axis=1)
        alternative_target = np.arange(NUM_TARGETS) != target[:, None]
        masks[:, slot, 2] = moved & (
            move_choices & same_move[:, :, None] & alternative_target[:, None, :]
        ).any(axis=(1, 2))
        masks[:, slot, 3] = switched & (
            switch_choices & (species != switch[:, None])
        ).any(axis=1)
    eligible = can_tera & (actions[:, :2] < SWITCH_OFFSET)
    used = actions[:, 2] != 0
    tera_labels = np.stack((used.astype(np.int64), actions[:, 2] - 1), axis=1)
    tera_masks = np.stack((eligible.any(axis=1), used & eligible.all(axis=1)), axis=1)
    return labels, masks, tera_labels, tera_masks
