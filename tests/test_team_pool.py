from pathlib import Path

import pytest

from pokerl.team_pool import load_team_pool


def test_load_team_pool_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "format": "test-format",
          "pool": [
            {
              "title": "duplicate",
              "file": "first.txt",
              "file": "second.txt"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_team_pool(manifest_path)


def test_checked_in_team_pool_manifest_loads() -> None:
    manifest_path = Path("data/team_pools/gen9vgc2026regf/manifest.json")

    pool = load_team_pool(manifest_path, subset="pool", seed=0)

    assert pool.size == 77
