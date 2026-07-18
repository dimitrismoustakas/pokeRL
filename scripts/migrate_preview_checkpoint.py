"""Convert a version-3 trainer checkpoint into an exact-preview warm start."""

from __future__ import annotations

import argparse
import json

from pokerl.data import GameData
from pokerl.training.preview_migration import migrate_v3_preview_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fresh version-4 trainer initialization from a strict "
            "version-3 checkpoint. Compatible features are copied, preview "
            "policy outputs are zeroed to exact uniformity, and the stale "
            "preview value head is reset."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--team-size", type=int, default=6)
    parser.add_argument("--gen", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = migrate_v3_preview_checkpoint(
        source_path=args.source,
        output_path=args.output,
        game_data=GameData(gen=args.gen),
        team_size=args.team_size,
        seed=args.seed,
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
