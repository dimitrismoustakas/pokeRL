"""Export battle replays for visualization in Pokemon Showdown."""

import json
from pathlib import Path
from poke_env.environment import DoubleBattle


def save_replay(battle: DoubleBattle, output_dir: str = "replays") -> str | None:
    """Save a battle's log as a Showdown-compatible replay file.

    The replay can be viewed by uploading it to replay.pokemonshowdown.com
    or opening it in a local Showdown client.

    Returns the file path, or None if no log available.
    """
    if not hasattr(battle, "battle_msg_history") or not battle.battle_msg_history:
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tag = battle.battle_tag.replace("[", "").replace("]", "").replace(" ", "_")
    path = f"{output_dir}/{tag}.log"

    with open(path, "w") as f:
        f.write(battle.battle_msg_history)

    return path


def save_replay_json(battle: DoubleBattle, output_dir: str = "replays") -> str | None:
    """Save battle metadata as JSON for programmatic analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tag = battle.battle_tag.replace("[", "").replace("]", "").replace(" ", "_")
    path = f"{output_dir}/{tag}.json"

    data = {
        "battle_tag": battle.battle_tag,
        "format": battle.battle_tag.split("]")[0].lstrip("[") if battle.battle_tag else "",
        "won": battle.won,
        "lost": battle.lost,
        "turn": battle.turn,
        "team": [p.species for p in battle.team.values()],
        "opponent_team": [p.species for p in battle.opponent_team.values()],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path
