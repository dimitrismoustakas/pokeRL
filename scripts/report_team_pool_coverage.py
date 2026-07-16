"""Report exact-team isolation and public-mechanic vocabulary coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokerl.team_pool_import import compare_pool_coverage, load_manifest_teams


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_MANIFEST = (
    REPO_ROOT / "data" / "team_pools" / "gen9vgc2026regf" / "manifest.json"
)
DEFAULT_HELDOUT_MANIFEST = (
    REPO_ROOT
    / "data"
    / "team_pools"
    / "gen9vgc2026regf_heldout"
    / "manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize train/heldout team, species, move, and set coverage."
    )
    parser.add_argument(
        "--training-manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST
    )
    parser.add_argument(
        "--heldout-manifest", type=Path, default=DEFAULT_HELDOUT_MANIFEST
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = load_manifest_teams(args.training_manifest, "training")
    heldout = load_manifest_teams(args.heldout_manifest, "heldout")
    report = compare_pool_coverage(training, heldout)
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
