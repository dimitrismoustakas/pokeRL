"""Action indices used by the battle policy and the Pikapro vector collector."""

# --- Action layout per active slot ---
#
#   0..19:  move_i * 5 + target_j        (4 moves x 5 targets)
#  20..25:  switch to own_team_slot_k    (stable team slot, 0..5)
#  26:      pass                         (slot has no action this turn)
#
# Terastallization is a separate joint decision made after both slot actions:
#   0: no tera
#   1: tera slot 0
#   2: tera slot 1
#
# Target order:
#   0: opposing_1
#   1: opposing_2
#   2: ally
#   3: self
#   4: no explicit target

NUM_MOVES = 4
NUM_TARGETS = 5
MAX_TEAM_SLOTS = 6
TARGET_OPPONENT_1 = 0
TARGET_OPPONENT_2 = 1
TARGET_ALLY = 2
TARGET_SELF = 3
TARGET_NONE = 4
SWITCH_OFFSET = NUM_MOVES * NUM_TARGETS  # 20
PASS_ACTION = SWITCH_OFFSET + MAX_TEAM_SLOTS  # 26
ACTIONS_PER_SLOT = PASS_ACTION + 1  # 27
NUM_TERA_CHOICES = 3
TERA_NONE = 0
TERA_SLOT_1 = 1
TERA_SLOT_2 = 2
FIRST_SLOT_MASK_ROW = 0

INTENT_NONE = 0
INTENT_PASS = 1
INTENT_MOVE = 2
INTENT_SWITCH = 3
NUM_INTENT_KINDS = 4

SLOT_INTENT_KIND = 0
SLOT_INTENT_MOVE = 1
SLOT_INTENT_TARGET = 2
SLOT_INTENT_SWITCH_SPECIES = 3
NUM_SLOT_INTENT_COMPONENTS = 4
TERA_INTENT_USE = 0
TERA_INTENT_SLOT = 1
NUM_TERA_INTENT_COMPONENTS = 2


def decode_action(action_idx: int) -> tuple[str, int, int]:
    """Decode a flat action index into (kind, move_or_switch_idx, target)."""
    if action_idx == PASS_ACTION:
        return "pass", -1, TARGET_NONE

    if action_idx >= SWITCH_OFFSET:
        return "switch", action_idx - SWITCH_OFFSET, TARGET_NONE

    move_idx = action_idx // NUM_TARGETS
    target_idx = action_idx % NUM_TARGETS
    return "move", move_idx, target_idx
