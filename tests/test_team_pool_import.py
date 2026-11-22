import json
from pathlib import Path

import pytest
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from pokerl.team_pool_import import (
    PublishedTeamSource,
    assert_disjoint_splits,
    compare_pool_coverage,
    coverage_summary,
    import_published_teams,
    load_manifest_teams,
    load_published_source_spec,
)


SPECIES_A = ("Pikachu", "Charizard", "Blastoise", "Venusaur", "Snorlax", "Eevee")
SPECIES_B = ("Raichu", "Arcanine", "Lapras", "Gengar", "Dragonite", "Ditto")
SPECIES_C = ("Mew", "Scizor", "Tyranitar", "Pelipper", "Ludicolo", "Rotom-Wash")
SPECIES_D = ("Mimikyu", "Torkoal", "Indeedee-F", "Hatterene", "Ursaluna", "Farigiraf")


def _team(species: tuple[str, ...], move: str = "Protect") -> str:
    return (
        "\n\n".join(
            "\n".join(
                (
                    f"{name} @ Leftovers",
                    "Ability: Pressure",
                    "Tera Type: Normal",
                    f"- {move}",
                )
            )
            for name in species
        )
        + "\n"
    )


class _AcceptingValidator:
    def validate_and_pack(self, team_text: str) -> tuple[bool, str | None, list[str]]:
        # Showdown preserves display case for ids while poke-env lowercases
        # them; the importer must still canonicalize to poke-env's bytes.
        return True, ConstantTeambuilder(team_text).yield_team().swapcase(), []


def _write_manifest(root: Path, name: str, team_text: str, *, training: bool) -> Path:
    manifest_dir = root / "data" / name
    team_path = manifest_dir / "curated" / "001_existing.txt"
    team_path.parent.mkdir(parents=True)
    team_path.write_text(team_text, encoding="utf-8")
    relative = team_path.relative_to(root).as_posix()
    entry = {"title": "Existing", "subset": "curated", "file": relative}
    manifest = {
        "purpose": "training" if training else "evaluation_only",
        "allow_training": training,
        "format": "test-format",
        "stats": {"curated_valid": 1, "generated_valid": 0, "pool_valid": 1},
        "curated": [dict(entry)],
        "generated": [],
        "pool": [dict(entry)],
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _source(order: int, split: str, link: str) -> PublishedTeamSource:
    return PublishedTeamSource(
        source_order=order,
        title=f"Published {order}",
        result="Published tournament result",
        export_link=link,
        recreation=False,
        split=split,
        source_page="https://example.test/source",
        spread_caveat="Published sample-spread caveat.",
    )


def test_source_spec_requires_unique_approved_exports(tmp_path: Path) -> None:
    spec = {
        "schema_version": 1,
        "source_page": "https://example.test/source",
        "spread_caveat": "Caveat",
        "entries": [
            {
                "source_order": 1,
                "title": "One",
                "result": "Result",
                "export_link": "https://pokepast.es/abc",
                "recreation": False,
                "split": "training",
            },
            {
                "source_order": 2,
                "title": "Two",
                "result": "Result",
                "export_link": "https://pokepast.es/abc",
                "recreation": False,
                "split": "heldout",
            },
        ],
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate export_link"):
        load_published_source_spec(path)


def test_coverage_reports_species_moves_and_synergy_pairs() -> None:
    summary = coverage_summary([_team(SPECIES_A), _team(SPECIES_B, "Fake Out")])

    assert summary["teams"] == 2
    assert summary["species"]["unique"] == 12
    assert summary["moves"]["counts"] == {"fakeout": 6, "protect": 6}
    assert summary["species_pairs"]["unique"] == 30
    assert summary["exact_sets"]["unique"] == 12


def test_cross_split_exact_duplicates_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    training_path = _write_manifest(
        tmp_path, "training", _team(SPECIES_A), training=True
    )
    heldout_path = _write_manifest(
        tmp_path, "heldout", _team(SPECIES_A), training=False
    )

    training = load_manifest_teams(training_path, "training")
    heldout = load_manifest_teams(heldout_path, "heldout")

    with pytest.raises(ValueError, match="share exact teams"):
        assert_disjoint_splits(training, heldout)


def test_import_preserves_dedupes_and_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    training_path = _write_manifest(
        tmp_path, "training", _team(SPECIES_A), training=True
    )
    heldout_path = _write_manifest(
        tmp_path, "heldout", _team(SPECIES_B), training=False
    )
    sources = [
        _source(1, "heldout", "https://pokepast.es/duplicate"),
        _source(2, "training", "https://pokepast.es/training"),
        _source(3, "heldout", "https://pokepast.es/heldout"),
    ]
    fetched = {
        sources[0].export_link: _team(SPECIES_A),
        sources[1].export_link: _team(SPECIES_C),
        sources[2].export_link: _team(SPECIES_D),
    }

    report = import_published_teams(
        sources=sources,
        training_manifest_path=training_path,
        heldout_manifest_path=heldout_path,
        validator=_AcceptingValidator(),
        fetcher=fetched.__getitem__,
        apply=True,
    )

    assert report["status_counts"] == {
        "added_heldout": 1,
        "added_training": 1,
        "duplicate_training": 1,
    }
    training = load_manifest_teams(training_path, "training")
    heldout = load_manifest_teams(heldout_path, "heldout")
    assert len(training) == 2
    assert len(heldout) == 2
    compare_pool_coverage(training, heldout)
    training_manifest = json.loads(training_path.read_text(encoding="utf-8"))
    assert (
        training_manifest["curated"][0]["published_sources"][0]["export_link"]
        == sources[0].export_link
    )

    repeated = import_published_teams(
        sources=sources,
        training_manifest_path=training_path,
        heldout_manifest_path=heldout_path,
        validator=_AcceptingValidator(),
        fetcher=fetched.__getitem__,
        apply=True,
    )

    assert repeated["status_counts"] == {
        "duplicate_heldout": 1,
        "duplicate_training": 2,
    }
    assert len(load_manifest_teams(training_path, "training")) == 2
    assert len(load_manifest_teams(heldout_path, "heldout")) == 2
