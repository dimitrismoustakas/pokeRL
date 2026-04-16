from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path


def default_showdown_dir() -> Path:
    override = os.getenv("POKERL_SHOWDOWN_DIR")
    if override:
        return Path(override)

    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError(
            "LOCALAPPDATA is not set, so the local Pokemon Showdown install could not be located. "
            "Set POKERL_SHOWDOWN_DIR explicitly."
        )
    return Path(local_app_data) / "pokeRL" / "pokemon-showdown"


def _helper_script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "showdown_team_tools.cjs"


def _run_showdown_tool(payload: dict) -> dict:
    payload = dict(payload)
    payload.setdefault("showdownDir", str(default_showdown_dir()))

    helper_path = _helper_script_path()
    if not helper_path.exists():
        raise RuntimeError(f"Showdown helper script was not found: {helper_path}")

    result = subprocess.run(
        ["node", str(helper_path)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "no stderr"
        raise RuntimeError(f"Showdown helper failed with exit code {result.returncode}: {stderr}")

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("Showdown helper returned no output.")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Showdown helper returned invalid JSON: {stdout}") from exc


@lru_cache(maxsize=None)
def fetch_format_metadata(format_id: str) -> dict:
    return _run_showdown_tool({"command": "profile", "formatId": format_id})


def validate_set(format_id: str, team_text: str) -> list[str]:
    result = _run_showdown_tool(
        {
            "command": "validateSet",
            "formatId": format_id,
            "teamText": team_text,
        }
    )
    return list(result.get("problems", []))


def validate_team(format_id: str, team_text: str) -> list[str]:
    result = _run_showdown_tool(
        {
            "command": "validateTeam",
            "formatId": format_id,
            "teamText": team_text,
        }
    )
    return list(result.get("problems", []))