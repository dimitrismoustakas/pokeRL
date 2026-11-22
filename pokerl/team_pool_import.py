"""Validate and import published Regulation F teams and measure roster coverage."""

from __future__ import annotations

import itertools
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pikapro import to_id

from .team_pool import pack_team, repo_root


USER_AGENT = "pokeRL-published-team-import/1.0"
VALID_SPLITS = frozenset({"training", "heldout"})
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class PublishedTeamSource:
    source_order: int
    title: str
    result: str
    export_link: str
    recreation: bool
    split: str
    source_page: str
    spread_caveat: str

    def provenance(self) -> dict[str, object]:
        return {
            "source_order": self.source_order,
            "title": self.title,
            "result": self.result,
            "source_page": self.source_page,
            "export_link": self.export_link,
            "recreation": self.recreation,
            "spread_caveat": self.spread_caveat,
        }


@dataclass(frozen=True, slots=True)
class ValidatedPublishedTeam:
    source: PublishedTeamSource
    team_text: str
    packed_team: str


@dataclass(frozen=True, slots=True)
class ManifestTeam:
    split: str
    title: str
    file: str
    team_text: str
    packed_team: str


class TeamValidator(Protocol):
    def validate(self, team_text: str) -> list[str]: ...


class ShowdownTeamValidator:
    """Check Regulation F legality with the repository's local Showdown build."""

    def __init__(self, showdown_dir: Path, format_id: str):
        self.showdown_dir = showdown_dir.expanduser().resolve()
        self.format_id = format_id

    def validate(self, team_text: str) -> list[str]:
        query = """
            const {Teams, TeamValidator} = require('./dist/sim');
            const text = require('node:fs').readFileSync(0, 'utf8');
            const problems = TeamValidator.get(process.argv[1]).validateTeam(Teams.import(text));
            console.log(JSON.stringify(problems || []));
        """
        result = subprocess.run(
            ["node", "-e", query, self.format_id],
            input=team_text,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            cwd=self.showdown_dir,
            check=True,
        )
        return json.loads(result.stdout)


def load_published_source_spec(path: str | Path) -> list[PublishedTeamSource]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    source_page = raw.get("source_page")
    spread_caveat = raw.get("spread_caveat")
    entries = raw.get("entries")
    if not isinstance(source_page, str) or not source_page:
        raise ValueError("Published-team source spec requires source_page.")
    if not isinstance(spread_caveat, str) or not spread_caveat:
        raise ValueError("Published-team source spec requires spread_caveat.")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Published-team source spec requires a non-empty entries list."
        )

    parsed: list[PublishedTeamSource] = []
    seen_orders: set[int] = set()
    seen_links: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Published-team entries must be objects.")
        source_order = raw_entry.get("source_order")
        title = raw_entry.get("title")
        result = raw_entry.get("result")
        export_link = raw_entry.get("export_link")
        recreation = raw_entry.get("recreation")
        split = raw_entry.get("split")
        if isinstance(source_order, bool) or not isinstance(source_order, int):
            raise ValueError("Every source_order must be an integer.")
        if source_order <= 0 or source_order in seen_orders:
            raise ValueError(f"Duplicate or invalid source_order {source_order!r}.")
        if not isinstance(title, str) or not title:
            raise ValueError(f"Source {source_order} requires a title.")
        if not isinstance(result, str) or not result:
            raise ValueError(f"Source {source_order} requires result metadata.")
        if not isinstance(export_link, str) or not export_link:
            raise ValueError(f"Source {source_order} requires an export_link.")
        if urlparse(export_link).netloc.lower() != "pokepast.es":
            raise ValueError(f"Source {source_order} is not an approved Pokepaste URL.")
        if export_link in seen_links:
            raise ValueError(f"Duplicate export_link {export_link!r}.")
        if type(recreation) is not bool:
            raise ValueError(f"Source {source_order} requires boolean recreation.")
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Source {source_order} split must be one of {sorted(VALID_SPLITS)}."
            )
        seen_orders.add(source_order)
        seen_links.add(export_link)
        parsed.append(
            PublishedTeamSource(
                source_order=source_order,
                title=title,
                result=result,
                export_link=export_link.rstrip("/"),
                recreation=recreation,
                split=split,
                source_page=source_page,
                spread_caveat=spread_caveat,
            )
        )
    return sorted(parsed, key=lambda entry: entry.source_order)


def fetch_pokepaste_raw(url: str, timeout: int = 30) -> str:
    raw_url = url.rstrip("/") + "/raw"
    request = Request(raw_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return payload.decode(charset)


def normalize_team_text(team_text: str) -> str:
    normalized = team_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("Published team export is empty.")
    pokemon = re.split(r"\n(?:[ \t]*\n)+", normalized)
    if len(pokemon) != 6:
        raise ValueError(
            f"Published team export must contain exactly 6 Pokemon, found {len(pokemon)}."
        )
    return normalized + "\n"


def load_manifest_teams(manifest_path: str | Path, split: str) -> list[ManifestTeam]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = repo_root(manifest_path.parent)
    records: list[ManifestTeam] = []
    seen_packed: set[str] = set()
    for raw_entry in manifest["pool"]:
        file_value = raw_entry["file"]
        team_path = root / file_value
        team_text = normalize_team_text(team_path.read_text(encoding="utf-8"))
        packed = pack_team(team_text)
        if packed in seen_packed:
            raise ValueError(
                f"Manifest {manifest_path} contains duplicate packed team {file_value}."
            )
        seen_packed.add(packed)
        records.append(
            ManifestTeam(
                split=split,
                title=str(raw_entry.get("title") or team_path.stem),
                file=file_value,
                team_text=team_text,
                packed_team=packed,
            )
        )
    return records


def assert_disjoint_splits(
    training: Iterable[ManifestTeam], heldout: Iterable[ManifestTeam]
) -> None:
    training_by_packed = {team.packed_team: team for team in training}
    heldout_by_packed = {team.packed_team: team for team in heldout}
    duplicates = sorted(set(training_by_packed).intersection(heldout_by_packed))
    if not duplicates:
        return
    details = [
        {
            "training": training_by_packed[packed].file,
            "heldout": heldout_by_packed[packed].file,
        }
        for packed in duplicates
    ]
    raise ValueError(f"Training and heldout pools share exact teams: {details!r}")


def _counter_payload(counter: Counter[str]) -> dict[str, object]:
    return {
        "unique": len(counter),
        "values": sorted(counter),
        "counts": dict(sorted(counter.items())),
    }


def coverage_summary(team_texts: Iterable[str]) -> dict[str, object]:
    species: Counter[str] = Counter()
    moves: Counter[str] = Counter()
    items: Counter[str] = Counter()
    abilities: Counter[str] = Counter()
    tera_types: Counter[str] = Counter()
    species_pairs: Counter[str] = Counter()
    exact_sets: set[str] = set()
    team_count = 0

    for team_text in team_texts:
        team_count += 1
        pokemon = pack_team(team_text).split("]")
        roster: list[str] = []
        for slot in pokemon:
            mon = slot.split("|")
            species_id = to_id(mon[1] or mon[0])
            roster.append(species_id)
            species[species_id] += 1
            item_id = to_id(mon[2])
            ability_id = to_id(mon[3])
            tera_id = to_id(mon[11].split(",")[5]) if mon[11].count(",") >= 5 else ""
            if item_id:
                items[item_id] += 1
            if ability_id:
                abilities[ability_id] += 1
            if tera_id:
                tera_types[tera_id] += 1
            move_ids = sorted(to_id(move) for move in mon[4].split(",") if move)
            moves.update(move_ids)
            exact_sets.add(
                "|".join([species_id, item_id, ability_id, tera_id, *move_ids])
            )
        for first, second in itertools.combinations(sorted(set(roster)), 2):
            species_pairs[f"{first}+{second}"] += 1

    return {
        "teams": team_count,
        "species": _counter_payload(species),
        "moves": _counter_payload(moves),
        "items": _counter_payload(items),
        "abilities": _counter_payload(abilities),
        "tera_types": _counter_payload(tera_types),
        "species_pairs": _counter_payload(species_pairs),
        "exact_sets": {"unique": len(exact_sets)},
    }


def compare_pool_coverage(
    training: Iterable[ManifestTeam], heldout: Iterable[ManifestTeam]
) -> dict[str, object]:
    training = list(training)
    heldout = list(heldout)
    assert_disjoint_splits(training, heldout)
    training_summary = coverage_summary(team.team_text for team in training)
    heldout_summary = coverage_summary(team.team_text for team in heldout)
    overlap: dict[str, object] = {}
    for category in ("species", "moves", "items", "abilities", "tera_types"):
        training_values = set(training_summary[category]["values"])
        heldout_values = set(heldout_summary[category]["values"])
        overlap[category] = {
            "shared": sorted(training_values & heldout_values),
            "training_only": sorted(training_values - heldout_values),
            "heldout_only": sorted(heldout_values - training_values),
        }
    return {
        "training": training_summary,
        "heldout": heldout_summary,
        "cross_split_exact_team_duplicates": [],
        "vocabulary_overlap": overlap,
    }


def _slugify(value: str) -> str:
    return _NON_ALNUM_RE.sub("_", value.lower()).strip("_") or "team"


def _published_manifest_entry(
    team: ValidatedPublishedTeam, *, file_value: str
) -> dict[str, object]:
    source = team.source
    return {
        "title": source.title,
        "subset": "curated",
        "file": file_value,
        "spread_provenance": "smogon_2026_sample_public_paste",
        "spread_caveat": source.spread_caveat,
        "result": source.result,
        "source_page": source.source_page,
        "export_link": source.export_link,
        "recreation": source.recreation,
        "published_sources": [source.provenance()],
        "source_links": [source.export_link],
        "sources": [
            {
                "section": "VGC Regulation F 2.0 Sample Teams",
                "country": "",
                "player": "",
                "result": source.result,
                "team_summary": source.title,
                "export_links": [source.export_link],
                "source_page": source.source_page,
                "raw_cells": [],
            }
        ],
        "seeds": [],
    }


def _next_team_filename(
    manifest: Mapping[str, object], team: ValidatedPublishedTeam
) -> str:
    curated = manifest.get("curated")
    if not isinstance(curated, list):
        raise ValueError("Manifest curated section must be a list.")
    return f"{len(curated) + 1:03d}_{_slugify(team.source.title)}.txt"


def _find_manifest_entries_by_packed(
    manifest: Mapping[str, object], root: Path, packed_team: str
) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    for section_name in ("curated", "pool"):
        section = manifest.get(section_name)
        if not isinstance(section, list):
            continue
        for raw_entry in section:
            if not isinstance(raw_entry, dict):
                continue
            file_value = raw_entry.get("file")
            if not isinstance(file_value, str):
                continue
            path = Path(file_value)
            if not path.is_absolute():
                path = root / path
            text = normalize_team_text(path.read_text(encoding="utf-8"))
            if pack_team(text) == packed_team:
                matched.append(raw_entry)
    return matched


def _add_provenance(entry: dict[str, object], source: PublishedTeamSource) -> None:
    provenance = source.provenance()
    published_sources = entry.setdefault("published_sources", [])
    if not isinstance(published_sources, list):
        raise ValueError("Manifest published_sources must be a list when present.")
    if any(
        isinstance(existing, dict)
        and existing.get("export_link") == source.export_link
        and existing.get("source_page") == source.source_page
        for existing in published_sources
    ):
        return
    published_sources.append(provenance)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_validated_team(
    manifest_path: Path,
    manifest: dict[str, object],
    team: ValidatedPublishedTeam,
) -> None:
    root = repo_root(manifest_path.parent)
    filename = _next_team_filename(manifest, team)
    curated_path = manifest_path.parent / "curated" / filename
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    curated_path.write_text(team.team_text, encoding="utf-8")
    curated_file = curated_path.relative_to(root).as_posix()
    curated_entry = _published_manifest_entry(team, file_value=curated_file)
    manifest["curated"].append(curated_entry)

    if team.source.split == "training":
        pool_path = manifest_path.parent / "pool" / filename
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool_path.write_text(team.team_text, encoding="utf-8")
        pool_file = pool_path.relative_to(root).as_posix()
        pool_entry = _published_manifest_entry(team, file_value=pool_file)
    else:
        pool_entry = dict(curated_entry)
    manifest["pool"].append(pool_entry)


def _refresh_manifest_stats(manifest: dict[str, object]) -> None:
    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        stats = {}
        manifest["stats"] = stats
    stats["curated_valid"] = len(manifest["curated"])
    generated = manifest.get("generated")
    stats["generated_valid"] = len(generated) if isinstance(generated, list) else 0
    stats["pool_valid"] = len(manifest["pool"])
    manifest["last_extended_at"] = datetime.now(timezone.utc).isoformat()


def import_published_teams(
    *,
    sources: Iterable[PublishedTeamSource],
    training_manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    validator: TeamValidator,
    fetcher: Callable[[str], str] = fetch_pokepaste_raw,
    apply: bool = False,
) -> dict[str, object]:
    """Fetch and validate every source, then merge novel exact teams by split."""

    training_manifest_path = Path(training_manifest_path)
    heldout_manifest_path = Path(heldout_manifest_path)
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    heldout_manifest = json.loads(heldout_manifest_path.read_text(encoding="utf-8"))
    training_existing = load_manifest_teams(training_manifest_path, "training")
    heldout_existing = load_manifest_teams(heldout_manifest_path, "heldout")
    assert_disjoint_splits(training_existing, heldout_existing)
    before_coverage = compare_pool_coverage(training_existing, heldout_existing)

    existing_by_packed = {
        team.packed_team: team for team in [*training_existing, *heldout_existing]
    }
    validated_new: list[ValidatedPublishedTeam] = []
    validated_by_packed: dict[str, ValidatedPublishedTeam] = {}
    results: list[dict[str, object]] = []

    for source in sorted(sources, key=lambda entry: entry.source_order):
        result: dict[str, object] = source.provenance()
        result["requested_split"] = source.split
        try:
            team_text = normalize_team_text(fetcher(source.export_link))
        except Exception as exc:  # noqa: BLE001
            result.update(status="fetch_failed", problems=[str(exc)])
            results.append(result)
            continue
        problems = validator.validate(team_text)
        if problems:
            result.update(status="invalid", problems=problems)
            results.append(result)
            continue
        packed_team = pack_team(team_text)

        existing = existing_by_packed.get(packed_team)
        if existing is not None:
            result.update(
                status=f"duplicate_{existing.split}",
                existing_file=existing.file,
                effective_split=existing.split,
            )
            if apply:
                manifest_path = (
                    training_manifest_path
                    if existing.split == "training"
                    else heldout_manifest_path
                )
                manifest = (
                    training_manifest
                    if existing.split == "training"
                    else heldout_manifest
                )
                root = repo_root(manifest_path.parent)
                for entry in _find_manifest_entries_by_packed(
                    manifest, root, packed_team
                ):
                    _add_provenance(entry, source)
            results.append(result)
            continue

        prior_source = validated_by_packed.get(packed_team)
        if prior_source is not None:
            if prior_source.source.split != source.split:
                raise ValueError(
                    "Published source spec assigns one exact packed team to both splits: "
                    f"{prior_source.source.export_link} and {source.export_link}."
                )
            result.update(
                status="duplicate_source",
                effective_split=source.split,
                duplicate_of=prior_source.source.export_link,
            )
            results.append(result)
            continue

        validated = ValidatedPublishedTeam(source, team_text, packed_team)
        validated_new.append(validated)
        validated_by_packed[packed_team] = validated
        result.update(status=f"added_{source.split}", effective_split=source.split)
        results.append(result)

    proposed_training = list(training_existing)
    proposed_heldout = list(heldout_existing)
    for team in validated_new:
        record = ManifestTeam(
            split=team.source.split,
            title=team.source.title,
            file=f"<pending:{team.source.export_link}>",
            team_text=team.team_text,
            packed_team=team.packed_team,
        )
        if team.source.split == "training":
            proposed_training.append(record)
        else:
            proposed_heldout.append(record)
    assert_disjoint_splits(proposed_training, proposed_heldout)
    proposed_coverage = compare_pool_coverage(proposed_training, proposed_heldout)

    if apply:
        for team in validated_new:
            manifest_path = (
                training_manifest_path
                if team.source.split == "training"
                else heldout_manifest_path
            )
            manifest = (
                training_manifest
                if team.source.split == "training"
                else heldout_manifest
            )
            _append_validated_team(manifest_path, manifest, team)
        _refresh_manifest_stats(training_manifest)
        _refresh_manifest_stats(heldout_manifest)
        _write_json_atomic(training_manifest_path, training_manifest)
        _write_json_atomic(heldout_manifest_path, heldout_manifest)
        final_training = load_manifest_teams(training_manifest_path, "training")
        final_heldout = load_manifest_teams(heldout_manifest_path, "heldout")
        assert_disjoint_splits(final_training, final_heldout)
        after_coverage = compare_pool_coverage(final_training, final_heldout)
    else:
        after_coverage = proposed_coverage

    status_counts = Counter(str(result["status"]) for result in results)
    return {
        "applied": apply,
        "source_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "results": results,
        "coverage_before": before_coverage,
        "coverage_after": after_coverage,
    }
