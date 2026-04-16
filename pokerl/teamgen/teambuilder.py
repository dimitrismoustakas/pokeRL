from __future__ import annotations

import random
import secrets
from dataclasses import dataclass
from functools import lru_cache

from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

from .regulation import RegulationProfile, load_regulation_profile
from .showdown import validate_set, validate_team


_STAT_NAMES = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")


@dataclass(frozen=True)
class TeamSetTemplate:
    species: str
    item: str
    ability: str
    moves: tuple[str, str, str, str]
    nature: str
    evs: tuple[int, int, int, int, int, int]
    ivs: tuple[int, int, int, int, int, int] = (31, 31, 31, 31, 31, 31)
    tera_type: str | None = None
    level: int | None = None
    restricted: bool = False

    def to_showdown_text(self, default_level: int | None) -> str:
        level = self.level if self.level is not None else default_level
        lines = [f"{self.species} @ {self.item}", f"Ability: {self.ability}"]
        if level is not None:
            lines.append(f"Level: {level}")
        if self.tera_type:
            lines.append(f"Tera Type: {self.tera_type}")
        if any(self.evs):
            lines.append(_spread_line("EVs", self.evs, default=0))
        lines.append(f"{self.nature} Nature")
        if any(iv != 31 for iv in self.ivs):
            lines.append(_spread_line("IVs", self.ivs, default=31))
        for move in self.moves:
            lines.append(f"- {move}")
        return "\n".join(lines)

    def to_teambuilder_pokemon(self, default_level: int | None) -> TeambuilderPokemon:
        level = self.level if self.level is not None else default_level
        return TeambuilderPokemon(
            species=self.species,
            item=self.item,
            ability=self.ability,
            moves=list(self.moves),
            nature=self.nature,
            evs=list(self.evs),
            ivs=list(self.ivs),
            level=level,
            tera_type=self.tera_type,
        )


class PoolSamplingTeambuilder(Teambuilder):
    def __init__(self, packed_teams: tuple[str, ...], seed: int | None = None):
        self._packed_teams = packed_teams
        self._rng = random.Random(seed if seed is not None else secrets.randbits(64))

    def yield_team(self) -> str:
        return self._rng.choice(self._packed_teams)


def maybe_build_teambuilder(format_id: str, pool_size: int = 32) -> Teambuilder | None:
    profile = load_regulation_profile(format_id)
    if profile.built_in_team_kind is not None:
        return None
    packed_teams = _build_packed_team_pool(format_id, pool_size)
    return PoolSamplingTeambuilder(packed_teams)


@lru_cache(maxsize=None)
def _validated_candidates(format_id: str) -> tuple[TeamSetTemplate, ...]:
    profile = load_regulation_profile(format_id)
    candidates = _candidate_sets_for_format(format_id)
    invalid: dict[str, list[str]] = {}
    valid: list[TeamSetTemplate] = []

    for template in candidates:
        problems = validate_set(format_id, template.to_showdown_text(profile.level))
        if problems:
            invalid[template.species] = problems
            continue
        valid.append(template)

    if invalid:
        details = "; ".join(f"{species}: {', '.join(problems)}" for species, problems in invalid.items())
        raise RuntimeError(f"Invalid set templates for {format_id}: {details}")
    if len(valid) < profile.team_size:
        raise RuntimeError(
            f"Only {len(valid)} valid team templates are available for {format_id}, "
            f"which is fewer than the required team size of {profile.team_size}."
        )
    return tuple(valid)


@lru_cache(maxsize=None)
def _build_packed_team_pool(format_id: str, pool_size: int) -> tuple[str, ...]:
    profile = load_regulation_profile(format_id)
    candidates = _validated_candidates(format_id)
    rng = random.Random(f"{format_id}:{pool_size}")
    packed_teams: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(pool_size * 200, 1000)

    while len(packed_teams) < pool_size and attempts < max_attempts:
        attempts += 1
        team = _sample_team(rng, profile, candidates)
        if team is None:
            continue

        team_text = "\n\n".join(template.to_showdown_text(profile.level) for template in team)
        problems = validate_team(format_id, team_text)
        if problems:
            continue

        packed = Teambuilder.join_team(
            [template.to_teambuilder_pokemon(profile.level) for template in team]
        )
        if packed in seen:
            continue
        seen.add(packed)
        packed_teams.append(packed)

    min_expected = min(pool_size, 8)
    if len(packed_teams) < min_expected:
        raise RuntimeError(
            f"Only generated {len(packed_teams)} valid teams for {format_id} after {attempts} attempts."
        )
    return tuple(packed_teams)


def _sample_team(
    rng: random.Random,
    profile: RegulationProfile,
    candidates: tuple[TeamSetTemplate, ...],
) -> list[TeamSetTemplate] | None:
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    team: list[TeamSetTemplate] = []
    used_species: set[str] = set()
    used_items: set[str] = set()
    restricted_count = 0

    for template in shuffled:
        if template.species in used_species:
            continue
        if profile.item_clause and template.item and template.item in used_items:
            continue
        if profile.max_restricted is not None and template.restricted and restricted_count >= profile.max_restricted:
            continue

        team.append(template)
        used_species.add(template.species)
        if profile.item_clause and template.item:
            used_items.add(template.item)
        if template.restricted:
            restricted_count += 1

        if len(team) == profile.team_size:
            return team
    return None


def _spread_line(label: str, values: tuple[int, ...], default: int) -> str:
    parts = [f"{value} {name}" for value, name in zip(values, _STAT_NAMES) if value != default]
    return f"{label}: {' / '.join(parts)}"


def _candidate_sets_for_format(format_id: str) -> tuple[TeamSetTemplate, ...]:
    if format_id != "gen9vgc2026regi":
        raise NotImplementedError(
            f"No regulation-backed team generator is configured for {format_id!r}. "
            "Start by adding a candidate set library for that format."
        )

    return (
        TeamSetTemplate(
            species="Incineroar",
            item="Safety Goggles",
            ability="Intimidate",
            moves=("Fake Out", "Parting Shot", "Flare Blitz", "Knock Off"),
            nature="Careful",
            evs=(252, 4, 116, 0, 132, 4),
            tera_type="Grass",
        ),
        TeamSetTemplate(
            species="Amoonguss",
            item="Sitrus Berry",
            ability="Regenerator",
            moves=("Spore", "Rage Powder", "Pollen Puff", "Protect"),
            nature="Sassy",
            evs=(236, 0, 156, 0, 116, 0),
            ivs=(31, 0, 31, 31, 31, 0),
            tera_type="Water",
        ),
        TeamSetTemplate(
            species="Rillaboom",
            item="Assault Vest",
            ability="Grassy Surge",
            moves=("Grassy Glide", "Fake Out", "Wood Hammer", "High Horsepower"),
            nature="Adamant",
            evs=(236, 252, 4, 0, 12, 4),
            tera_type="Fire",
        ),
        TeamSetTemplate(
            species="Urshifu-Rapid-Strike",
            item="Mystic Water",
            ability="Unseen Fist",
            moves=("Surging Strikes", "Close Combat", "Aqua Jet", "Protect"),
            nature="Jolly",
            evs=(4, 252, 0, 0, 0, 252),
            tera_type="Water",
        ),
        TeamSetTemplate(
            species="Gholdengo",
            item="Choice Specs",
            ability="Good as Gold",
            moves=("Make It Rain", "Shadow Ball", "Thunderbolt", "Trick"),
            nature="Timid",
            evs=(4, 0, 0, 252, 0, 252),
            tera_type="Steel",
        ),
        TeamSetTemplate(
            species="Dragonite",
            item="Choice Band",
            ability="Inner Focus",
            moves=("Extreme Speed", "Stomping Tantrum", "Aerial Ace", "Ice Spinner"),
            nature="Adamant",
            evs=(4, 252, 0, 0, 0, 252),
            tera_type="Normal",
        ),
        TeamSetTemplate(
            species="Pelipper",
            item="Focus Sash",
            ability="Drizzle",
            moves=("Hurricane", "Hydro Pump", "Tailwind", "Protect"),
            nature="Timid",
            evs=(4, 0, 0, 252, 0, 252),
            tera_type="Ghost",
        ),
        TeamSetTemplate(
            species="Archaludon",
            item="Power Herb",
            ability="Stamina",
            moves=("Electro Shot", "Draco Meteor", "Flash Cannon", "Protect"),
            nature="Modest",
            evs=(252, 0, 4, 252, 0, 0),
            tera_type="Dragon",
        ),
        TeamSetTemplate(
            species="Farigiraf",
            item="Throat Spray",
            ability="Armor Tail",
            moves=("Hyper Voice", "Psychic", "Trick Room", "Protect"),
            nature="Quiet",
            evs=(252, 0, 4, 252, 0, 0),
            ivs=(31, 0, 31, 31, 31, 0),
            tera_type="Fairy",
        ),
        TeamSetTemplate(
            species="Iron Hands",
            item="Booster Energy",
            ability="Quark Drive",
            moves=("Fake Out", "Drain Punch", "Wild Charge", "Volt Switch"),
            nature="Adamant",
            evs=(116, 156, 4, 0, 228, 4),
            tera_type="Grass",
        ),
        TeamSetTemplate(
            species="Flutter Mane",
            item="Choice Scarf",
            ability="Protosynthesis",
            moves=("Moonblast", "Shadow Ball", "Icy Wind", "Power Gem"),
            nature="Timid",
            evs=(4, 0, 0, 252, 0, 252),
            tera_type="Fairy",
        ),
        TeamSetTemplate(
            species="Landorus-Therian",
            item="Life Orb",
            ability="Intimidate",
            moves=("Earthquake", "Rock Slide", "U-turn", "Protect"),
            nature="Jolly",
            evs=(4, 252, 0, 0, 0, 252),
            tera_type="Flying",
        ),
        TeamSetTemplate(
            species="Whimsicott",
            item="Mental Herb",
            ability="Prankster",
            moves=("Tailwind", "Moonblast", "Encore", "Protect"),
            nature="Timid",
            evs=(4, 0, 0, 252, 0, 252),
            tera_type="Ghost",
        ),
        TeamSetTemplate(
            species="Kingambit",
            item="Black Glasses",
            ability="Defiant",
            moves=("Kowtow Cleave", "Sucker Punch", "Iron Head", "Protect"),
            nature="Adamant",
            evs=(236, 252, 4, 0, 12, 4),
            tera_type="Dark",
        ),
        TeamSetTemplate(
            species="Indeedee-F",
            item="Psychic Seed",
            ability="Psychic Surge",
            moves=("Psychic", "Follow Me", "Helping Hand", "Protect"),
            nature="Bold",
            evs=(252, 0, 156, 0, 100, 0),
            tera_type="Fairy",
        ),
        TeamSetTemplate(
            species="Armarouge",
            item="Weakness Policy",
            ability="Flash Fire",
            moves=("Armor Cannon", "Expanding Force", "Trick Room", "Protect"),
            nature="Quiet",
            evs=(252, 0, 4, 252, 0, 0),
            ivs=(31, 0, 31, 31, 31, 0),
            tera_type="Grass",
        ),
    )