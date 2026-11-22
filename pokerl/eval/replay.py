"""Export battle replays for visualization in Pokemon Showdown."""

import json
from pathlib import Path
from poke_env.battle import DoubleBattle


def save_replay(battle: DoubleBattle, output_dir: str = "replays") -> str | None:
    """Save a battle's replay HTML using poke-env's public replay API."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tag = battle.battle_tag.replace("[", "").replace("]", "").replace(" ", "_")
    path = Path(output_dir) / f"{tag}.html"
    battle.save_replay(path)
    return str(path)


def save_replay_json(battle: DoubleBattle, output_dir: str = "replays") -> str | None:
    """Save battle metadata as JSON for programmatic analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tag = battle.battle_tag.replace("[", "").replace("]", "").replace(" ", "_")
    path = f"{output_dir}/{tag}.json"

    data = {
        "battle_tag": battle.battle_tag,
        "format": battle.battle_tag.split("]")[0].lstrip("[")
        if battle.battle_tag
        else "",
        "won": battle.won,
        "lost": battle.lost,
        "turn": battle.turn,
        "team": [p.species for p in battle.team.values()],
        "opponent_team": [p.species for p in battle.opponent_team.values()],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path
