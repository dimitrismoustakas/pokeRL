"""Import the checked-in list of published Regulation F 2.0 sample teams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokerl.team_pool_import import (
    ShowdownTeamValidator,
    import_published_teams,
    load_published_source_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "data" / "team_pools" / "regf_2_0_published_sources.json"
DEFAULT_TRAINING_MANIFEST = (
    REPO_ROOT / "data" / "team_pools" / "gen9vgc2026regf" / "manifest.json"
)
DEFAULT_HELDOUT_MANIFEST = (
    REPO_ROOT / "data" / "team_pools" / "gen9vgc2026regf_heldout" / "manifest.json"
)
DEFAULT_SHOWDOWN_DIR = Path.home() / "AppData" / "Local" / "pokeRL" / "pokemon-showdown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch, validate, exact-deduplicate, and optionally merge the "
            "published Regulation F 2.0 sample-team source spec."
        )
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--training-manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST
    )
    parser.add_argument(
        "--heldout-manifest", type=Path, default=DEFAULT_HELDOUT_MANIFEST
    )
    parser.add_argument("--showdown-dir", type=Path, default=DEFAULT_SHOWDOWN_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write validated novel teams and provenance. The default is a dry run.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path. Output is always printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = load_published_source_spec(args.spec)
    format_id = json.loads(args.spec.read_text(encoding="utf-8"))["format"]
    validator = ShowdownTeamValidator(args.showdown_dir, format_id)
    report = import_published_teams(
        sources=sources,
        training_manifest_path=args.training_manifest,
        heldout_manifest_path=args.heldout_manifest,
        validator=validator,
        apply=args.apply,
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
