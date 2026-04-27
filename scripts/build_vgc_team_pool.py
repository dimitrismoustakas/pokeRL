"""Build a validated VGC team pool from public team archives.

The initial source is Victory Road's Regulation Set I archive page, with direct
Pokepaste exports as the highest-yield ingest path. Teams are validated against
the local Pokemon Showdown checkout already used by this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_FORMAT = "gen9vgc2026regi"
DEFAULT_SOURCE_PAGES = {
    "gen9vgc2026regf": "https://victoryroad.pro/sv-rental-teams-reg-set-f/",
    "gen9vgc2026regi": "https://victoryroad.pro/sv-rental-teams-reg-set-i/",
}
DEFAULT_SOURCE_PAGE = DEFAULT_SOURCE_PAGES[DEFAULT_FORMAT]
DEFAULT_SHOWDOWN_DIR = Path.home() / "AppData" / "Local" / "pokeRL" / "pokemon-showdown"
USER_AGENT = "pokeRL-team-pool/1.0"
SUPPORTED_FETCH_HOSTS = {"liberty-note.com", "pokepast.es"}
KNOWN_EXPORT_HOSTS = {
    "pokepast.es",
    "www.vrpastes.com",
    "note.com",
    "liberty-note.com",
    "carrefourpokemon.hatenablog.com",
    "cona5757.hatenablog.com",
    "ryouma9901.hatenadiary.jp",
    "zezerozero.hatenablog.com",
}
BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)
ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
HEADING_RE = re.compile(r"<h([1-4])\b[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
HREF_RE = re.compile(r"href=\"([^\"]+)\"", re.IGNORECASE)
TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
STAT_SEGMENT_RE = re.compile(r"([HABCDS])[^()-]*(?:\((\d+)\))?")

STAT_LABELS = {
    "H": "HP",
    "A": "Atk",
    "B": "Def",
    "C": "SpA",
    "D": "SpD",
    "S": "Spe",
}

LIBERTY_REQUIRED_ROWS = {"持ち物", "テラスタイプ", "性格", "実数値(努力値)", "技"}

JP_SPECIES_TO_EN = {
    "ウネルミナモ": "Walking Wake",
    "オーガポン(いしずえのめん)": "Ogerpon-Cornerstone",
    "コライドン": "Koraidon",
    "ザシアン": "Zacian-Crowned",
    "パオジアン": "Chien-Pao",
    "化身ボルトロス": "Thundurus-Incarnate",
}

JP_ABILITIES_TO_EN = {
    "いたずらごころ": "Prankster",
    "がんじょう": "Sturdy",
    "こだいかっせい": "Protosynthesis",
    "ひひいろのこどう": "Orichalcum Pulse",
    "ふとうのけん": "Intrepid Sword",
    "わざわいのつるぎ": "Sword of Ruin",
}

JP_ITEMS_TO_EN = {
    "いしずえのめん": "Cornerstone Mask",
    "いのちのたま": "Life Orb",
    "オボンのみ": "Sitrus Berry",
    "きあいのタスキ": "Focus Sash",
    "くちたけん": "Rusted Sword",
    "クリアチャーム": "Clear Amulet",
}

JP_TYPES_TO_EN = {
    "あく": "Dark",
    "エスパー": "Psychic",
    "かくとう": "Fighting",
    "くさ": "Grass",
    "こおり": "Ice",
    "ゴースト": "Ghost",
    "じめん": "Ground",
    "でんき": "Electric",
    "ドラゴン": "Dragon",
    "どく": "Poison",
    "いわ": "Rock",
    "はがね": "Steel",
    "ほのお": "Fire",
    "ひこう": "Flying",
    "フェアリー": "Fairy",
    "むし": "Bug",
    "みず": "Water",
    "ステラ": "Stellar",
    "ノーマル": "Normal",
}

JP_NATURES_TO_EN = {
    "いじっぱり": "Adamant",
    "おくびょう": "Timid",
    "おだやか": "Calm",
    "ようき": "Jolly",
}

JP_MOVES_TO_EN = {
    "アイアンヘッド(きょじゅうざん)": "Behemoth Blade",
    "かいでんぱ": "Eerie Impulse",
    "かみくだく": "Crunch",
    "かみなりあらし": "Wildbolt Storm",
    "このゆびとまれ": "Follow Me",
    "じゃれつく": "Play Rough",
    "テラバースト": "Tera Blast",
    "つたこんぼう": "Ivy Cudgel",
    "ツタこんぼう": "Ivy Cudgel",
    "つららおとし": "Icicle Crash",
    "でんじは": "Thunder Wave",
    "にらみつける": "Leer",
    "ニードルガード": "Spiky Shield",
    "バークアウト": "Snarl",
    "パワーウィップ": "Power Whip",
    "ハイドロスチーム": "Hydro Steam",
    "ふいうち": "Sucker Punch",
    "フレアドライブ": "Flare Blitz",
    "みがわり": "Substitute",
    "まもる": "Protect",
    "りゅうのはどう": "Dragon Pulse",
    "りゅうせいぐん": "Draco Meteor",
    "スケイルショット": "Scale Shot",
    "アクセルブレイク": "Collision Course",
}


def normalize_id(value: str) -> str:
    return NON_ALNUM_RE.sub("", value.lower())


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def strip_tags(value: str) -> str:
    return normalize_text(unescape(TAG_RE.sub(" ", value)))


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def slugify(value: str, fallback: str = "team") -> str:
    slug = NON_ALNUM_RE.sub("-", value.lower()).strip("-")
    return slug or fallback


def fetch_text(url: str, timeout: int = 30, attempts: int = 1) -> str:
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                payload = response.read()
            return payload.decode(charset, errors="replace")
        except (TimeoutError, URLError) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


@dataclass(slots=True)
class SourceRow:
    section: str
    country: str
    player: str
    result: str
    team_summary: str
    export_links: list[str]
    source_page: str
    raw_cells: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "country": self.country,
            "player": self.player,
            "result": self.result,
            "team_summary": self.team_summary,
            "export_links": self.export_links,
            "source_page": self.source_page,
            "raw_cells": self.raw_cells,
        }


@dataclass(slots=True)
class TeamBlock:
    species: str
    species_id: str
    text: str


@dataclass(slots=True)
class TeamRecord:
    title: str
    team_text: str
    packed_team: str
    source_rows: list[SourceRow]
    source_links: list[str]
    subset: str
    file: str = ""
    seeds: list[str] = field(default_factory=list)

    @property
    def packed_sha1(self) -> str:
        return hashlib.sha1(self.packed_team.encode("utf-8")).hexdigest()

    def blocks(self) -> list[TeamBlock]:
        return parse_team_blocks(self.team_text)


class VictoryRoadArchiveParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.current_section = ""
        self._skip_depth = 0
        self._heading_depth = 0
        self._heading_parts: list[str] = []
        self._in_row = False
        self._in_cell = False
        self._row_cells: list[str] = []
        self._row_links: list[str] = []
        self._cell_parts: list[str] = []
        self.rows: list[tuple[str, list[str], list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        attr_map = dict(attrs)
        if tag in {"h1", "h2", "h3", "h4"}:
            self._heading_depth += 1
            if self._heading_depth == 1:
                self._heading_parts = []
            return

        if tag == "tr":
            self._in_row = True
            self._row_cells = []
            self._row_links = []
            return

        if self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []
            return

        if self._in_row and tag == "a":
            href = attr_map.get("href")
            if href:
                self._row_links.append(urljoin(self.base_url, href))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return

        if tag in {"h1", "h2", "h3", "h4"}:
            if self._heading_depth == 1:
                heading = normalize_text("".join(self._heading_parts))
                if heading:
                    self.current_section = heading
                self._heading_parts = []
            self._heading_depth = max(0, self._heading_depth - 1)
            return

        if self._in_row and tag in {"td", "th"} and self._in_cell:
            cell = normalize_text("".join(self._cell_parts))
            self._row_cells.append(cell)
            self._cell_parts = []
            self._in_cell = False
            return

        if tag == "tr" and self._in_row:
            if self._row_cells or self._row_links:
                self.rows.append(
                    (
                        self.current_section,
                        [cell for cell in self._row_cells if cell],
                        unique_preserve_order(self._row_links),
                    )
                )
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._heading_depth:
            self._heading_parts.append(data)
        if self._in_cell:
            self._cell_parts.append(data)


class PokepasteParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self._in_pre = False
        self._title_parts: list[str] = []
        self._pre_parts: list[str] = []
        self.title = ""
        self.pre_blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "title":
            self._in_title = True
            self._title_parts = []
        elif tag == "pre":
            self._in_pre = True
            self._pre_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._in_title:
            self.title = normalize_text("".join(self._title_parts))
            self._in_title = False
        elif tag == "pre" and self._in_pre:
            block = "".join(self._pre_parts).replace("\r", "").strip()
            if block:
                self.pre_blocks.append(block)
            self._in_pre = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        if self._in_pre:
            self._pre_parts.append(data)


class ShowdownCliValidator:
    def __init__(self, showdown_dir: Path, format_id: str):
        self.showdown_dir = showdown_dir
        self.format_id = format_id
        self.entrypoint = showdown_dir / "pokemon-showdown"
        if not self.entrypoint.exists():
            raise FileNotFoundError(
                f"Could not find Pokemon Showdown entrypoint at {self.entrypoint}. "
                "Set --showdown-dir to a Pokemon Showdown checkout."
            )

    def _run(self, args: list[str], team_text: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["node", str(self.entrypoint), *args],
            input=team_text,
            text=True,
            capture_output=True,
            cwd=self.showdown_dir,
            timeout=180,
            check=False,
        )

    def validate_team(self, team_text: str) -> tuple[bool, list[str]]:
        result = self._run(["validate-team", self.format_id], team_text)
        if result.returncode == 0:
            return True, []

        problem_text = result.stderr.strip() or result.stdout.strip() or "Unknown validator error"
        return False, [line for line in problem_text.splitlines() if line.strip()]

    def pack_team(self, team_text: str) -> str:
        result = self._run(["pack-team"], team_text)
        if result.returncode != 0:
            problem_text = result.stderr.strip() or result.stdout.strip() or "Unknown pack-team error"
            raise RuntimeError(problem_text)
        return result.stdout.strip()

    def validate_and_pack(self, team_text: str) -> tuple[bool, str | None, list[str]]:
        valid, problems = self.validate_team(team_text)
        if not valid:
            return False, None, problems
        try:
            return True, self.pack_team(team_text), []
        except RuntimeError as exc:
            return False, None, [str(exc)]


def extract_victory_road_rows(page_html: str, source_page: str) -> list[SourceRow]:
    sections = [
        (match.start(), strip_tags(match.group(2)))
        for match in HEADING_RE.finditer(page_html)
        if strip_tags(match.group(2))
    ]
    rows: list[SourceRow] = []
    for match in ROW_RE.finditer(page_html):
        row_html = match.group(1)
        section = ""
        for section_pos, section_name in sections:
            if section_pos > match.start():
                break
            section = section_name

        cells = [strip_tags(cell_html) for cell_html in CELL_RE.findall(row_html)]
        links = [urljoin(source_page, href) for href in HREF_RE.findall(row_html)]
        export_links = [link for link in links if urlparse(link).netloc.lower() in KNOWN_EXPORT_HOSTS]
        if not export_links or len(cells) < 4:
            continue
        rows.append(
            SourceRow(
                section=section,
                country=cells[0],
                player=cells[1],
                result=cells[2],
                team_summary=cells[3],
                export_links=unique_preserve_order(export_links),
                source_page=source_page,
                raw_cells=cells,
            )
        )
    return rows


def parse_pokepaste_team(page_html: str) -> tuple[str, str]:
    parser = PokepasteParser()
    parser.feed(page_html)
    team_text = "\n\n".join(block.strip() for block in parser.pre_blocks if block.strip())
    if not team_text:
        raise ValueError("Pokepaste page did not contain any exported team blocks.")
    return parser.title or "Pokepaste team", team_text + "\n"


def extract_html_title(page_html: str) -> str:
    match = TITLE_RE.search(page_html)
    if not match:
        return ""
    return strip_tags(match.group(1)).split(" - ", 1)[0].strip()


def translate_exact(value: str, mapping: dict[str, str], kind: str) -> str:
    translated = mapping.get(normalize_text(value))
    if translated is None:
        raise ValueError(f"Unsupported {kind} translation: {value}")
    return translated


def canonical_jp_key(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def parse_evs_from_stat_line(stat_line: str) -> str:
    ev_parts: list[str] = []
    for stat, value in STAT_SEGMENT_RE.findall(normalize_text(stat_line)):
        if not value:
            continue
        ev_value = int(value)
        if ev_value <= 0:
            continue
        ev_parts.append(f"{ev_value} {STAT_LABELS[stat]}")
    if not ev_parts:
        raise ValueError(f"Could not parse EVs from stat line: {stat_line}")
    return " / ".join(ev_parts)


def parse_liberty_moves(move_text: str) -> list[str]:
    raw_moves: list[str] = []
    for chunk in normalize_text(move_text).split():
        raw_moves.extend(part for part in chunk.split("/") if part)
    if len(raw_moves) != 4:
        raise ValueError(f"Expected 4 moves in Liberty Note set, found {len(raw_moves)}: {move_text}")
    return [translate_exact(move, JP_MOVES_TO_EN, "move") for move in raw_moves]


def parse_liberty_note_team(page_html: str) -> tuple[str, str]:
    team_blocks: list[str] = []
    translated_species: list[str] = []

    headings = [
        (match.start(), strip_tags(match.group(2)))
        for match in HEADING_RE.finditer(page_html)
        if strip_tags(match.group(2))
    ]

    for table_match in TABLE_RE.finditer(page_html):
        table_html = table_match.group(1)
        species_heading = ""
        for heading_pos, heading_text in headings:
            if heading_pos > table_match.start():
                break
            species_heading = heading_text

        table_rows = []
        for row_html in ROW_RE.findall(table_html):
            cells = [strip_tags(cell_html) for cell_html in CELL_RE.findall(row_html)]
            cells = [cell for cell in cells if cell]
            if cells:
                table_rows.append(cells)

        if not table_rows:
            continue

        if species_heading not in JP_SPECIES_TO_EN:
            continue

        first_row = table_rows[0]
        if len(first_row) < 2 or canonical_jp_key(first_row[0]) != "特性":
            continue

        row_map: dict[str, str] = {
            canonical_jp_key(cells[0]): cells[1] for cells in table_rows[1:] if len(cells) >= 2
        }
        if not LIBERTY_REQUIRED_ROWS.issubset(row_map):
            continue

        species = translate_exact(species_heading, JP_SPECIES_TO_EN, "species")
        ability = translate_exact(first_row[1], JP_ABILITIES_TO_EN, "ability")
        item = translate_exact(row_map["持ち物"], JP_ITEMS_TO_EN, "item")
        tera_type = translate_exact(row_map["テラスタイプ"], JP_TYPES_TO_EN, "tera type")
        nature = translate_exact(row_map["性格"], JP_NATURES_TO_EN, "nature")
        evs = parse_evs_from_stat_line(row_map["実数値(努力値)"])
        moves = parse_liberty_moves(row_map["技"])

        block_lines = [
            f"{species} @ {item}",
            f"Ability: {ability}",
            f"Tera Type: {tera_type}",
            f"EVs: {evs}",
            f"{nature} Nature",
        ]
        block_lines.extend(f"- {move}" for move in moves)
        team_blocks.append("\n".join(block_lines))
        translated_species.append(species)

    if len(team_blocks) != 6:
        raise ValueError(f"Expected 6 Liberty Note team entries, found {len(team_blocks)}.")

    species_title = " / ".join(translated_species[:3])
    title = f"Liberty Note - {species_title}"
    return title, "\n\n".join(team_blocks) + "\n"


def parse_exported_team(export_link: str) -> tuple[str, str]:
    host = urlparse(export_link).netloc.lower()
    timeout = 120 if host == "liberty-note.com" else 30
    attempts = 3 if host == "liberty-note.com" else 1
    page_html = fetch_text(export_link, timeout=timeout, attempts=attempts)
    if host == "pokepast.es":
        return parse_pokepaste_team(page_html)
    if host == "liberty-note.com":
        return parse_liberty_note_team(page_html)
    raise ValueError(f"No parser implemented for host: {host}")


def default_source_page_for_format(format_id: str) -> str | None:
    return DEFAULT_SOURCE_PAGES.get(format_id)


def extract_species(first_line: str) -> str:
    left = first_line.split(" @ ", 1)[0].strip()
    if left.endswith(" (M)") or left.endswith(" (F)"):
        left = left[:-4]
    if left.endswith(")") and "(" in left:
        return left.rsplit("(", 1)[1][:-1].strip()
    return left.strip()


def parse_team_blocks(team_text: str) -> list[TeamBlock]:
    blocks: list[TeamBlock] = []
    for raw_block in BLOCK_SPLIT_RE.split(team_text.strip()):
        block = raw_block.strip()
        if not block:
            continue
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        species = extract_species(lines[0])
        blocks.append(
            TeamBlock(
                species=species,
                species_id=normalize_id(species),
                text="\n".join(lines),
            )
        )
    return blocks


def choose_candidate_blocks(
    curated_teams: list[TeamRecord],
    team_blocks_by_packed: dict[str, list[TeamBlock]],
    all_blocks: list[tuple[TeamRecord, TeamBlock]],
    rng: random.Random,
) -> tuple[list[TeamBlock], list[str]]:
    if not curated_teams:
        return [], []

    sampled_teams = rng.sample(curated_teams, k=min(2, len(curated_teams)))
    split_patterns = ((3, 3), (4, 2), (5, 1))
    take_a, take_b = split_patterns[rng.randrange(len(split_patterns))]
    targets = [take_a]
    if len(sampled_teams) > 1:
        targets.append(take_b)

    chosen: list[TeamBlock] = []
    used_species: set[str] = set()
    seeds: list[str] = []

    for team, target_count in zip(sampled_teams, targets, strict=False):
        seeds.append(team.title)
        team_blocks = list(team_blocks_by_packed[team.packed_team])
        rng.shuffle(team_blocks)
        taken_from_team = 0
        for block in team_blocks:
            if block.species_id in used_species:
                continue
            chosen.append(block)
            used_species.add(block.species_id)
            taken_from_team += 1
            if taken_from_team >= target_count:
                break

    filler_pool = list(all_blocks)
    rng.shuffle(filler_pool)
    for _, block in filler_pool:
        if len(chosen) >= 6:
            break
        if block.species_id in used_species:
            continue
        chosen.append(block)
        used_species.add(block.species_id)

    return chosen[:6], seeds


def build_generated_teams(
    curated_teams: list[TeamRecord],
    validator: ShowdownCliValidator,
    count: int,
    seed: int,
) -> tuple[list[TeamRecord], int]:
    if count <= 0 or not curated_teams:
        return [], 0

    rng = random.Random(seed)
    generated: list[TeamRecord] = []
    seen_packed = {team.packed_team for team in curated_teams}
    team_blocks_by_packed = {team.packed_team: team.blocks() for team in curated_teams}
    all_blocks = [
        (team, block)
        for team in curated_teams
        for block in team_blocks_by_packed[team.packed_team]
    ]
    seen_candidate_signatures: set[tuple[str, ...]] = set()
    max_attempts = max(2000, count * 400)
    attempts = 0

    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        blocks, seeds = choose_candidate_blocks(curated_teams, team_blocks_by_packed, all_blocks, rng)
        if len(blocks) != 6:
            continue

        candidate_signature = tuple(
            sorted(hashlib.sha1(block.text.encode("utf-8")).hexdigest() for block in blocks)
        )
        if candidate_signature in seen_candidate_signatures:
            continue
        seen_candidate_signatures.add(candidate_signature)

        team_text = "\n\n".join(block.text for block in blocks) + "\n"
        valid, packed_team, problems = validator.validate_and_pack(team_text)
        if not valid or packed_team is None or packed_team in seen_packed:
            del problems
            continue

        seen_packed.add(packed_team)
        title = " / ".join(block.species for block in blocks[:3])
        generated.append(
            TeamRecord(
                title=f"Generated remix - {title}",
                team_text=team_text,
                packed_team=packed_team,
                source_rows=[],
                source_links=[],
                subset="generated",
                seeds=seeds,
            )
        )

    return generated, attempts


def write_team_records(
    team_records: list[TeamRecord],
    subset_dir: Path,
    repo_root: Path,
) -> list[dict]:
    subset_dir.mkdir(parents=True, exist_ok=True)
    for existing_file in subset_dir.glob("*.txt"):
        existing_file.unlink()
    manifest_entries: list[dict] = []

    for index, team in enumerate(team_records, start=1):
        slug_base = slugify(team.title)
        file_name = f"{index:03d}_{slug_base}_{team.packed_sha1[:8]}.txt"
        file_path = subset_dir / file_name
        file_path.write_text(team.team_text, encoding="utf-8")
        team.file = file_path.relative_to(repo_root).as_posix()
        manifest_entries.append(
            {
                "title": team.title,
                "subset": team.subset,
                "file": team.file,
                "packed_sha1": team.packed_sha1,
                "source_links": team.source_links,
                "sources": [row.to_dict() for row in team.source_rows],
                "seeds": team.seeds,
            }
        )

    return manifest_entries


def build_curated_teams(
    rows: list[SourceRow],
    validator: ShowdownCliValidator,
) -> tuple[list[TeamRecord], list[dict], list[dict]]:
    curated_by_packed: dict[str, TeamRecord] = {}
    unsupported: list[dict] = []
    invalid: list[dict] = []
    processed_links: dict[str, str | None] = {}

    for row in rows:
        for export_link in row.export_links:
            if export_link in processed_links:
                packed_team = processed_links[export_link]
                if packed_team and packed_team in curated_by_packed:
                    existing = curated_by_packed[packed_team]
                    existing.source_rows.append(row)
                    existing.source_links = unique_preserve_order(existing.source_links + [export_link])
                continue

            host = urlparse(export_link).netloc.lower()
            if host not in SUPPORTED_FETCH_HOSTS:
                processed_links[export_link] = None
                unsupported.append({"url": export_link, "row": row.to_dict()})
                continue

            try:
                title, team_text = parse_exported_team(export_link)
            except Exception as exc:  # noqa: BLE001
                processed_links[export_link] = None
                invalid.append(
                    {
                        "url": export_link,
                        "row": row.to_dict(),
                        "problems": [f"Fetch/parsing failed: {exc}"],
                    }
                )
                continue

            valid, packed_team, problems = validator.validate_and_pack(team_text)
            if not valid or packed_team is None:
                processed_links[export_link] = None
                invalid.append(
                    {
                        "url": export_link,
                        "row": row.to_dict(),
                        "problems": problems,
                    }
                )
                continue

            processed_links[export_link] = packed_team
            existing = curated_by_packed.get(packed_team)
            if existing is None:
                curated_by_packed[packed_team] = TeamRecord(
                    title=title,
                    team_text=team_text,
                    packed_team=packed_team,
                    source_rows=[row],
                    source_links=[export_link],
                    subset="curated",
                )
                continue

            existing.source_rows.append(row)
            existing.source_links = unique_preserve_order(existing.source_links + [export_link])

    curated = sorted(curated_by_packed.values(), key=lambda team: team.title.lower())
    return curated, unsupported, invalid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validated VGC team pool from online archives.")
    parser.add_argument("--format", default=DEFAULT_FORMAT, help="Showdown format id to validate against.")
    parser.add_argument(
        "--source-page",
        default=None,
        help="Archive page to crawl for team exports. Defaults to a built-in source page for known formats.",
    )
    parser.add_argument(
        "--showdown-dir",
        default=str(DEFAULT_SHOWDOWN_DIR),
        help="Path to the local Pokemon Showdown checkout used for validation.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to data/team_pools/<format> under the repo root.",
    )
    parser.add_argument(
        "--generate-remixes",
        type=int,
        default=100,
        help="How many additional legal remixes to generate from the curated pool.",
    )
    parser.add_argument(
        "--target-team-count",
        type=int,
        default=None,
        help="Desired total pool size including curated teams. If set, only enough remixes are generated to reach this total.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for legal remix generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "data" / "team_pools" / args.format
    showdown_dir = Path(args.showdown_dir)
    source_page = args.source_page or default_source_page_for_format(args.format)
    if source_page is None:
        raise SystemExit(f"No default source page is configured for format {args.format}; pass --source-page.")

    validator = ShowdownCliValidator(showdown_dir=showdown_dir, format_id=args.format)

    print(f"Fetching archive page: {source_page}")
    archive_html = fetch_text(source_page)
    rows = extract_victory_road_rows(archive_html, source_page)
    print(f"Found {len(rows)} candidate archive rows with export links.")

    curated, unsupported, invalid = build_curated_teams(rows, validator)
    print(f"Validated {len(curated)} unique curated teams.")
    if unsupported:
        print(f"Skipped {len(unsupported)} unsupported export links.")
    if invalid:
        print(f"Rejected {len(invalid)} invalid or unreadable sources.")

    generated_target = args.generate_remixes
    if args.target_team_count is not None:
        generated_target = max(0, args.target_team_count - len(curated))
        print(
            f"Requested total pool size {args.target_team_count}; "
            f"generating up to {generated_target} remix teams after keeping all curated teams."
        )

    generated, generation_attempts = build_generated_teams(
        curated_teams=curated,
        validator=validator,
        count=generated_target,
        seed=args.seed,
    )
    if generated_target:
        print(
            f"Generated {len(generated)} additional legal remixes "
            f"from {generation_attempts} attempts."
        )
    elif args.target_team_count is not None:
        print("Curated teams already satisfy the requested total pool size.")

    curated_manifest = write_team_records(curated, out_dir / "curated", repo_root)
    generated_manifest = write_team_records(generated, out_dir / "generated", repo_root)
    pool_manifest = write_team_records(curated + generated, out_dir / "pool", repo_root)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "source_page": source_page,
        "showdown_dir": str(showdown_dir),
        "stats": {
            "archive_rows": len(rows),
            "curated_valid": len(curated_manifest),
            "generated_valid": len(generated_manifest),
            "pool_valid": len(pool_manifest),
            "generated_requested": generated_target,
            "target_team_count": args.target_team_count,
            "unsupported_links": len(unsupported),
            "invalid_sources": len(invalid),
            "generation_attempts": generation_attempts,
        },
        "curated": curated_manifest,
        "generated": generated_manifest,
        "pool": pool_manifest,
        "unsupported": unsupported,
        "invalid": invalid,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote team pool manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
