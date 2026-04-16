from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..config import EnvConfig, ModelConfig
from .showdown import fetch_format_metadata


@dataclass(frozen=True)
class RegulationMechanics:
    tera: bool
    mega: bool
    zmove: bool
    dynamax: bool


@dataclass(frozen=True)
class RegulationProfile:
    format_id: str
    name: str
    gen: int
    mod: str
    built_in_team_kind: str | None
    team_size: int
    pick_size: int
    min_team_size: int
    level: int | None
    has_team_preview: bool
    item_clause: bool
    species_clause: bool
    max_restricted: int | None
    mechanics: RegulationMechanics


@lru_cache(maxsize=None)
def load_regulation_profile(format_id: str) -> RegulationProfile:
    raw = fetch_format_metadata(format_id)
    mechanics = RegulationMechanics(
        tera=bool(raw["mechanics"]["tera"]),
        mega=bool(raw["mechanics"]["mega"]),
        zmove=bool(raw["mechanics"]["zmove"]),
        dynamax=bool(raw["mechanics"]["dynamax"]),
    )
    return RegulationProfile(
        format_id=raw["id"],
        name=raw["name"],
        gen=int(raw["gen"]),
        mod=raw["mod"],
        built_in_team_kind=raw["team"],
        team_size=int(raw["teamSize"]),
        pick_size=int(raw["pickSize"]),
        min_team_size=int(raw["minTeamSize"]),
        level=raw["level"],
        has_team_preview=bool(raw["hasTeamPreview"]),
        item_clause=bool(raw["itemClause"]),
        species_clause=bool(raw["speciesClause"]),
        max_restricted=raw["maxRestricted"],
        mechanics=mechanics,
    )


def apply_regulation_profile(model_config: ModelConfig, env_config: EnvConfig) -> RegulationProfile:
    profile = load_regulation_profile(env_config.battle_format)
    env_config.team_size = profile.team_size
    env_config.pick_size = profile.pick_size
    model_config.pick_size = profile.pick_size
    return profile