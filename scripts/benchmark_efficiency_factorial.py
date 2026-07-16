"""Run a restart-safe Q/L/C/B pokeRL throughput experiment.

The four cells use one code revision and differ only in two diagnostic runtime
switches:

* Q: eager construction of all legal battle orders, deadline-only coalescing
* L: lazy construction of selected battle orders, deadline-only coalescing
* C: eager construction of all legal battle orders, target-aware coalescing
* B: both optimizations enabled

Every production cell resumes the same one-update bootstrap checkpoint.  This
pins model weights, optimizer state, RNG streams, league membership, opponent
assignments, and team-pool state.  Interrupted cells are intentionally rerun
from that common boundary rather than resumed partway through: process startup,
CUDA graph capture, and warm-up must be comparable in every cell.

The harness itself depends only on the Python standard library.  It does not
import torch or pokeRL, so orchestration cannot consume CPU/GPU resources while
the child trainer is running.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
ENV_BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_env_throughput.py"
TEAM_MANIFEST = REPO_ROOT / "data" / "team_pools" / "gen9vgc2026regf" / "manifest.json"
SCHEMA_VERSION = 1
SUCCESS_SCHEMA_VERSION = 1
VARIANT_FACTORS: dict[str, dict[str, bool]] = {
    "Q": {"lazy_action_orders": False, "target_aware_coalescing": False},
    "L": {"lazy_action_orders": True, "target_aware_coalescing": False},
    "C": {"lazy_action_orders": False, "target_aware_coalescing": True},
    "B": {"lazy_action_orders": True, "target_aware_coalescing": True},
}
# A four-treatment Williams design.  Every treatment appears once in every
# position and every ordered carry-over pair occurs exactly once.
WILLIAMS_ORDERS: tuple[tuple[str, ...], ...] = (
    ("Q", "L", "B", "C"),
    ("L", "C", "Q", "B"),
    ("C", "B", "L", "Q"),
    ("B", "Q", "C", "L"),
)
_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class PlannedRun:
    phase: str
    block: int
    position: int
    variant: str
    seed: int | None = None

    @property
    def cell_id(self) -> str:
        return (
            f"{self.phase}/block-{self.block:02d}/"
            f"position-{self.position:02d}-{self.variant}"
        )


def planned_run_payload(planned: PlannedRun) -> dict[str, Any]:
    """Serialize only seeds that are actually applied by a child command."""
    payload = asdict(planned)
    if planned.seed is None:
        payload.pop("seed")
    return payload


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(_json_ready(payload), indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_capture_result(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
) -> tuple[bool, str]:
    """Return command success separately from possibly empty stdout."""
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    if completed.returncode != 0:
        return False, completed.stdout.strip()
    return True, completed.stdout.strip()


def run_capture(command: Sequence[str], *, cwd: Path | None = None) -> str | None:
    succeeded, value = run_capture_result(command, cwd=cwd)
    if not succeeded:
        return None
    return value or None


def git_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    sha_ok, sha = run_capture_result(["git", "-C", str(resolved), "rev-parse", "HEAD"])
    if not sha_ok or not sha:
        return {
            "path": str(resolved),
            "git_sha": None,
            "worktree_dirty": None,
            "status_verified": False,
        }
    status_ok, status = run_capture_result(
        ["git", "-C", str(resolved), "status", "--porcelain", "--untracked-files=all"]
    )
    return {
        "path": str(resolved),
        "git_sha": sha,
        "worktree_dirty": bool(status) if status_ok else None,
        "status_verified": status_ok,
    }


def require_clean_git_identity(identity: Mapping[str, Any], *, name: str) -> None:
    if not identity.get("git_sha"):
        raise RuntimeError(f"{name} must be a Git checkout with a resolvable HEAD.")
    if identity.get("status_verified") is not True:
        raise RuntimeError(f"Could not verify the {name} Git worktree status.")
    if identity.get("worktree_dirty") is not False:
        raise RuntimeError(
            f"The {name} Git checkout has tracked or untracked changes. "
            "Commit, remove, or ignore them before benchmarking."
        )


def _tracked_git_root(origin: Path) -> Path | None:
    """Return a containing Git root only when *origin* is tracked by it."""
    root_ok, root_text = run_capture_result(
        ["git", "-C", str(origin.parent), "rev-parse", "--show-toplevel"]
    )
    if not root_ok or not root_text:
        return None
    root = Path(root_text).resolve()
    try:
        relative = origin.relative_to(root).as_posix()
    except ValueError:
        return None
    tracked_ok, _ = run_capture_result(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative]
    )
    return root if tracked_ok else None


def deterministic_package_tree_identity(package_root: Path) -> dict[str, Any]:
    """Hash importable package bytes when no tracked source checkout owns them."""
    root = package_root.resolve()
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() not in {".pyc", ".pyo"}
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise RuntimeError(f"Installed poke-env package tree is empty: {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        total_bytes += size
    return {
        "path": str(root),
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "total_bytes": total_bytes,
    }


def discover_poke_env_identity() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("poke-env")
    except importlib.metadata.PackageNotFoundError:
        version = None
    spec = importlib.util.find_spec("poke_env")
    origin = (
        None if spec is None or spec.origin is None else Path(spec.origin).resolve()
    )
    source_root = None if origin is None else _tracked_git_root(origin)
    package_root = None
    if spec is not None and spec.submodule_search_locations:
        locations = list(spec.submodule_search_locations)
        if len(locations) == 1:
            package_root = Path(locations[0]).resolve()
    if package_root is None and origin is not None:
        package_root = origin.parent
    identity = {
        "distribution_version": version,
        "module_origin": None if origin is None else str(origin),
        "source": None,
        "installed_package_tree": None,
    }
    if source_root is not None:
        identity["source"] = git_identity(source_root)
    elif package_root is not None:
        identity["installed_package_tree"] = deterministic_package_tree_identity(
            package_root
        )
    return identity


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def discover_node_identity() -> dict[str, str] | None:
    executable = shutil.which("node")
    if executable is None:
        return None
    version = run_capture([executable, "--version"])
    if version is None:
        return None
    return {"executable": str(Path(executable).resolve()), "version": version}


def discover_gpu_identity() -> list[dict[str, str]] | None:
    output = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return None
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            dict(
                zip(
                    ("index", "name", "uuid", "driver_version", "memory_total_mib"),
                    parts,
                    strict=True,
                )
            )
        )
    return rows or None


def discover_nvidia_identity() -> dict[str, Any] | None:
    executable = shutil.which("nvidia-smi")
    gpus = discover_gpu_identity()
    if executable is None or gpus is None:
        return None
    header = run_capture([executable])
    cuda_match = (
        None if header is None else re.search(r"CUDA Version:\s*([0-9.]+)", header)
    )
    return {
        "executable": str(Path(executable).resolve()),
        "reported_cuda_version": None if cuda_match is None else cuda_match.group(1),
        "gpus": gpus,
    }


def resolve_showdown_dir(raw: str | None) -> Path:
    if raw:
        path = Path(raw).expanduser().resolve()
    elif os.environ.get("POKERL_SHOWDOWN_DIR"):
        path = Path(os.environ["POKERL_SHOWDOWN_DIR"]).expanduser().resolve()
    elif os.environ.get("LOCALAPPDATA"):
        path = (
            Path(os.environ["LOCALAPPDATA"]) / "pokeRL" / "pokemon-showdown"
        ).resolve()
    else:
        raise ValueError(
            "--showdown-dir is required when POKERL_SHOWDOWN_DIR and LOCALAPPDATA "
            "are unavailable."
        )
    if not path.is_dir():
        raise ValueError(f"Pokemon Showdown checkout not found: {path}")
    return path


def build_production_schedule(blocks: int) -> list[PlannedRun]:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    schedule: list[PlannedRun] = []
    for block_index in range(blocks):
        order = WILLIAMS_ORDERS[block_index % len(WILLIAMS_ORDERS)]
        for position, variant in enumerate(order, start=1):
            schedule.append(
                PlannedRun(
                    phase="production",
                    block=block_index + 1,
                    position=position,
                    variant=variant,
                )
            )
    return schedule


def build_simulator_schedule(rounds: int) -> list[PlannedRun]:
    if rounds < 0:
        raise ValueError("simulator rounds must be non-negative")
    schedule: list[PlannedRun] = []
    for round_index in range(rounds):
        order = ("Q", "L") if round_index % 2 == 0 else ("L", "Q")
        seed = 72_000 + round_index
        for position, variant in enumerate(order, start=1):
            schedule.append(
                PlannedRun(
                    phase="simulator",
                    block=round_index + 1,
                    position=position,
                    variant=variant,
                    seed=seed,
                )
            )
    return schedule


def _percent(log_effect: float) -> float:
    return math.expm1(log_effect) * 100.0


def mean_log_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_log": None,
            "effect_percent": None,
            "ci_low_percent": None,
            "ci_high_percent": None,
        }
    mean_value = statistics.fmean(values)
    if len(values) == 1:
        low = high = None
    else:
        standard_error = statistics.stdev(values) / math.sqrt(len(values))
        critical = _T_975.get(len(values) - 1, 1.960)
        low = mean_value - critical * standard_error
        high = mean_value + critical * standard_error
    return {
        "count": len(values),
        "mean_log": mean_value,
        "effect_percent": _percent(mean_value),
        "ci_low_percent": None if low is None else _percent(low),
        "ci_high_percent": None if high is None else _percent(high),
    }


def factorial_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    by_block: dict[int, dict[str, float]] = {}
    for row in rows:
        if row.get("phase") != "production":
            continue
        value = float(row[metric])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{metric} must be finite and positive (got {value})")
        block = int(row["block"])
        variant = str(row["variant"])
        if variant in by_block.setdefault(block, {}):
            raise ValueError(f"Duplicate variant {variant} in block {block}")
        by_block[block][variant] = value

    block_rows: list[dict[str, Any]] = []
    lazy_blocks: list[float] = []
    coalesce_blocks: list[float] = []
    interaction_blocks: list[float] = []
    lazy_constituents: list[float] = []
    coalesce_constituents: list[float] = []
    for block in sorted(by_block):
        values = by_block[block]
        if set(values) != set(VARIANT_FACTORS):
            continue
        logs = {variant: math.log(value) for variant, value in values.items()}
        lazy_pair = (logs["L"] - logs["Q"], logs["B"] - logs["C"])
        coalesce_pair = (logs["C"] - logs["Q"], logs["B"] - logs["L"])
        lazy = statistics.fmean(lazy_pair)
        coalesce = statistics.fmean(coalesce_pair)
        interaction = logs["B"] - logs["L"] - logs["C"] + logs["Q"]
        lazy_blocks.append(lazy)
        coalesce_blocks.append(coalesce)
        interaction_blocks.append(interaction)
        lazy_constituents.extend(lazy_pair)
        coalesce_constituents.extend(coalesce_pair)
        block_rows.append(
            {
                "block": block,
                "lazy_effect_percent": _percent(lazy),
                "coalescing_effect_percent": _percent(coalesce),
                "interaction_percent": _percent(interaction),
            }
        )

    return {
        "metric": metric,
        "complete_blocks": len(block_rows),
        "blocks": block_rows,
        "lazy": {
            **mean_log_interval(lazy_blocks),
            "positive_constituents": sum(value > 0.0 for value in lazy_constituents),
            "constituent_count": len(lazy_constituents),
        },
        "coalescing": {
            **mean_log_interval(coalesce_blocks),
            "positive_constituents": sum(
                value > 0.0 for value in coalesce_constituents
            ),
            "constituent_count": len(coalesce_constituents),
        },
        "interaction": mean_log_interval(interaction_blocks),
    }


def simulator_pair_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_round: dict[int, dict[str, float]] = {}
    for row in rows:
        if row.get("phase") != "simulator":
            continue
        by_round.setdefault(int(row["block"]), {})[str(row["variant"])] = float(
            row["env_steps_per_second"]
        )
    log_ratios = []
    pairs = []
    for round_index in sorted(by_round):
        values = by_round[round_index]
        if set(values) != {"Q", "L"}:
            continue
        ratio = math.log(values["L"]) - math.log(values["Q"])
        log_ratios.append(ratio)
        pairs.append(
            {
                "round": round_index,
                "q_env_steps_per_second": values["Q"],
                "l_env_steps_per_second": values["L"],
                "effect_percent": _percent(ratio),
            }
        )
    return {
        "complete_pairs": len(pairs),
        "pairs": pairs,
        "lazy": {
            **mean_log_interval(log_ratios),
            "positive_pairs": sum(value > 0.0 for value in log_ratios),
            "pair_count": len(log_ratios),
        },
    }


def _safe_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite (got {value!r})")
    return result


def _action_cache(update: Mapping[str, Any]) -> Mapping[str, Any]:
    league = update.get("league", {})
    if not isinstance(league, Mapping):
        return {}
    inference_cache = league.get("inference_cache", {})
    if not isinstance(inference_cache, Mapping):
        return {}
    action = inference_cache.get("action", {})
    return action if isinstance(action, Mapping) else {}


def summarize_training_summary(
    summary: Mapping[str, Any],
    *,
    expected_invocation_updates: int,
    discard_updates: int,
    expected_initial_completed_updates: int | None = None,
    expected_initial_global_step: int | None = None,
) -> dict[str, Any]:
    if summary.get("status") != "completed":
        raise ValueError(f"Training summary status is {summary.get('status')!r}")
    updates_raw = summary.get("updates")
    if not isinstance(updates_raw, list):
        raise ValueError("Training summary updates must be a list")
    if len(updates_raw) != expected_invocation_updates:
        raise ValueError(
            "Training invocation produced an unexpected number of updates: "
            f"expected {expected_invocation_updates}, got {len(updates_raw)}"
        )
    if not 0 <= discard_updates < len(updates_raw):
        raise ValueError(f"discard_updates must be in [0, {len(updates_raw) - 1}]")
    updates: list[Mapping[str, Any]] = []
    for index, update in enumerate(updates_raw):
        if not isinstance(update, Mapping):
            raise ValueError(f"Update {index} is not a mapping")
        updates.append(update)
    selected = updates[discard_updates:]
    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Training summary config must be a mapping")
    num_envs = int(config["num_envs"])
    rollout_steps = int(config["rollout_steps"])
    steps_per_update = num_envs * rollout_steps
    measured_steps = steps_per_update * len(selected)

    if (expected_initial_completed_updates is None) != (
        expected_initial_global_step is None
    ):
        raise ValueError(
            "Expected update and global-step boundaries must be provided together"
        )
    if expected_initial_completed_updates is not None:
        expected_updates = list(
            range(
                expected_initial_completed_updates + 1,
                expected_initial_completed_updates + len(updates) + 1,
            )
        )
        actual_updates = [int(update.get("update", -1)) for update in updates]
        if actual_updates != expected_updates:
            raise ValueError(
                "Training summary update boundary mismatch: "
                f"expected {expected_updates}, got {actual_updates}"
            )
        expected_global_steps = [
            expected_initial_global_step + steps_per_update * (index + 1)
            for index in range(len(updates))
        ]
        actual_global_steps = [int(update.get("global_step", -1)) for update in updates]
        if actual_global_steps != expected_global_steps:
            raise ValueError(
                "Training summary global-step boundary mismatch: "
                f"expected {expected_global_steps}, got {actual_global_steps}"
            )
        if int(summary.get("update_count", -1)) != expected_updates[-1]:
            raise ValueError(
                "Training summary update_count does not match its last row"
            )
        if int(summary.get("global_step", -1)) != expected_global_steps[-1]:
            raise ValueError("Training summary global_step does not match its last row")
        final = summary.get("final")
        if not isinstance(final, Mapping):
            raise ValueError("Completed training summary omitted final metrics")
        expected_steps_this_run = steps_per_update * len(updates)
        if int(final.get("steps_this_run", -1)) != expected_steps_this_run:
            raise ValueError(
                "Training summary final.steps_this_run does not match the invocation"
            )
        if int(final.get("global_step", -1)) != expected_global_steps[-1]:
            raise ValueError("Training summary final.global_step does not match")

    normalized_config = json.loads(json.dumps(_json_ready(config), sort_keys=True))
    normalized_config.pop("lazy_action_orders", None)
    normalized_config.pop("league_target_aware_coalescing", None)
    battlestream = normalized_config.get("battlestream")
    if isinstance(battlestream, dict):
        battlestream.pop("lazy_action_orders", None)
    config_fingerprint = hashlib.sha256(
        json.dumps(
            normalized_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    wall_secs = [
        _safe_float(update["update_wall_sec"], name="update_wall_sec")
        for update in selected
    ]
    rollout_secs = [
        _safe_float(update["rollout_sec"], name="rollout_sec") for update in selected
    ]
    ppo_secs = [_safe_float(update["ppo_sec"], name="ppo_sec") for update in selected]
    preview_secs = [
        _safe_float(update.get("preview_refresh_sec", 0.0), name="preview_refresh_sec")
        for update in selected
    ]
    overhead_secs = [
        _safe_float(update.get("update_overhead_sec", 0.0), name="update_overhead_sec")
        for update in selected
    ]
    completed_battles = sum(
        int(update.get("completed_battles", 0)) for update in selected
    )
    wall_total = sum(wall_secs)
    rollout_total = sum(rollout_secs)
    learning_total = rollout_total + sum(ppo_secs)
    if min(wall_total, rollout_total, learning_total) <= 0.0:
        raise ValueError("Measured training durations must be positive")

    per_update_wall_sps = [steps_per_update / value for value in wall_secs]
    first_cache = _action_cache(updates[discard_updates - 1]) if discard_updates else {}
    last_cache = _action_cache(selected[-1])

    def cache_delta(name: str) -> int:
        return int(last_cache.get(name, 0)) - int(first_cache.get(name, 0))

    retained_player = [
        int(update.get("retained_player_battles", 0)) for update in selected
    ]
    retained_opponent = [
        int(update.get("retained_opponent_battles", 0)) for update in selected
    ]
    cuda_allocated = [
        int(update["cuda_memory_allocated_bytes"])
        for update in selected
        if "cuda_memory_allocated_bytes" in update
    ]
    cuda_reserved = [
        int(update["cuda_memory_reserved_bytes"])
        for update in selected
        if "cuda_memory_reserved_bytes" in update
    ]
    return {
        "invocation_update_count": len(updates),
        "final_update": int(updates[-1].get("update", -1)),
        "final_global_step": int(updates[-1].get("global_step", -1)),
        "discarded_update_count": discard_updates,
        "measured_update_count": len(selected),
        "steps_per_update": steps_per_update,
        "measured_steps": measured_steps,
        "wall_steps_per_second": measured_steps / wall_total,
        "rollout_steps_per_second": measured_steps / rollout_total,
        "rollout_plus_ppo_steps_per_second": measured_steps / learning_total,
        "rollout_battles_per_second": completed_battles / rollout_total,
        "completed_battles": completed_battles,
        "wall_sec": wall_total,
        "rollout_sec": rollout_total,
        "ppo_sec": sum(ppo_secs),
        "preview_refresh_sec": sum(preview_secs),
        "overhead_sec": sum(overhead_secs),
        "mean_update_wall_sps": statistics.fmean(per_update_wall_sps),
        "median_update_wall_sps": statistics.median(per_update_wall_sps),
        "min_update_wall_sps": min(per_update_wall_sps),
        "max_update_wall_sps": max(per_update_wall_sps),
        "retained_player_battles_min": min(retained_player),
        "retained_player_battles_max": max(retained_player),
        "retained_opponent_battles_min": min(retained_opponent),
        "retained_opponent_battles_max": max(retained_opponent),
        "cuda_memory_allocated_min": min(cuda_allocated) if cuda_allocated else None,
        "cuda_memory_allocated_max": max(cuda_allocated) if cuda_allocated else None,
        "cuda_memory_reserved_min": min(cuda_reserved) if cuda_reserved else None,
        "cuda_memory_reserved_max": max(cuda_reserved) if cuda_reserved else None,
        "config_fingerprint": config_fingerprint,
        "coalesce_target_batch_size": last_cache.get("coalesce_target_batch_size"),
        "coalesce_target_wakeups": cache_delta("coalesce_target_wakeups"),
        "cuda_graphs_enabled": bool(last_cache.get("cuda_graphs_enabled", False)),
        "cuda_graph_replays": cache_delta("cuda_graph_replays"),
        "cuda_graph_fallbacks": cache_delta("cuda_graph_fallbacks"),
        "cuda_graph_captures": cache_delta("cuda_graph_captures"),
        "cuda_graph_real_rows": cache_delta("cuda_graph_real_rows"),
        "cuda_graph_padded_rows": cache_delta("cuda_graph_padded_rows"),
        "cuda_graph_last_error": last_cache.get("cuda_graph_last_error"),
        "resident_cuda_graphs": int(last_cache.get("resident_cuda_graphs", 0)),
    }


def summarize_simulator_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    config = summary.get("config")
    results = summary.get("results")
    if not isinstance(config, Mapping) or not isinstance(results, Mapping):
        raise ValueError("Simulator summary requires config and results mappings")
    lazy_action_orders = config.get("lazy_action_orders")
    if not isinstance(lazy_action_orders, bool):
        raise ValueError(
            "Simulator summary must record boolean config.lazy_action_orders"
        )
    total_steps = int(results["total_env_steps"])
    run_sec = _safe_float(results["run_sec"], name="run_sec")
    env_sps = _safe_float(results["env_steps_per_sec"], name="env_steps_per_sec")
    if total_steps <= 0 or run_sec <= 0.0 or env_sps <= 0.0:
        raise ValueError("Simulator summary reports non-positive work or throughput")
    return {
        "total_env_steps": total_steps,
        "run_sec": run_sec,
        "env_steps_per_second": env_sps,
        "completed_battles": int(results.get("completed_battles", 0)),
        "avg_episode_sec": _safe_float(
            results.get("avg_episode_sec", 0.0), name="avg_episode_sec"
        ),
        "avg_episode_turns": _safe_float(
            results.get("avg_episode_turns", 0.0), name="avg_episode_turns"
        ),
        "lazy_action_orders": lazy_action_orders,
    }


def validate_factor_config(
    summary: Mapping[str, Any],
    *,
    variant: str,
) -> None:
    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Training summary config must be a mapping")
    factors = VARIANT_FACTORS[variant]
    actual_lazy = config.get("lazy_action_orders")
    actual_target = config.get("league_target_aware_coalescing")
    if actual_lazy is not factors["lazy_action_orders"]:
        raise ValueError(
            f"Variant {variant} expected lazy_action_orders="
            f"{factors['lazy_action_orders']}, got {actual_lazy!r}"
        )
    if actual_target is not factors["target_aware_coalescing"]:
        raise ValueError(
            f"Variant {variant} expected league_target_aware_coalescing="
            f"{factors['target_aware_coalescing']}, got {actual_target!r}"
        )


def _success_path(cell_dir: Path) -> Path:
    return cell_dir / "SUCCESS.json"


def validated_success_marker(cell_dir: Path) -> dict[str, Any] | None:
    marker_path = _success_path(cell_dir)
    if not marker_path.exists():
        return None
    marker = read_json(marker_path)
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != SUCCESS_SCHEMA_VERSION
    ):
        raise ValueError(f"Invalid success marker: {marker_path}")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Success marker has no artifact map: {marker_path}")
    for relative_path, expected_hash in artifacts.items():
        artifact = cell_dir / relative_path
        if not artifact.is_file():
            raise ValueError(f"Successful cell is missing artifact: {artifact}")
        actual_hash = sha256_file(artifact)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Successful cell artifact changed: {artifact}; "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return marker


def next_attempt_dir(cell_dir: Path) -> Path:
    cell_dir.mkdir(parents=True, exist_ok=True)
    attempt_numbers = []
    for child in cell_dir.iterdir():
        if child.is_dir() and child.name.startswith("attempt-"):
            try:
                attempt_numbers.append(int(child.name.removeprefix("attempt-")))
            except ValueError:
                continue
    attempt_number = max(attempt_numbers, default=0) + 1
    attempt = cell_dir / f"attempt-{attempt_number:03d}"
    attempt.mkdir()
    return attempt


def ensure_experiment_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        current = read_json(path)
        if current != _json_ready(expected):
            raise ValueError(
                f"Experiment manifest mismatch at {path}. Use the exact original "
                "command/output directory, or choose a new output directory."
            )
        return
    atomic_write_json(path, expected)


class ExperimentLock:
    """OS-owned non-blocking lock; process death or reboot releases it."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def __enter__(self) -> "ExperimentLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self._handle.write(b"0")
            self._handle.flush()
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"Another benchmark harness owns {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _process_rows_windows() -> list[dict[str, Any]]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    output = run_capture(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ]
    )
    if output is None:
        return []
    payload = json.loads(output)
    if isinstance(payload, dict):
        return [payload]
    return [row for row in payload if isinstance(row, dict)]


def find_conflicting_processes(showdown_dir: Path) -> list[dict[str, Any]]:
    """Find another pokeRL trainer or orphaned local Showdown worker."""
    if os.name != "nt":
        output = run_capture(["ps", "-eo", "pid=,args="])
        if output is None:
            return []
        rows = []
        for line in output.splitlines():
            pid_text, _, command = line.strip().partition(" ")
            if pid_text.isdigit():
                rows.append(
                    {"ProcessId": int(pid_text), "Name": "", "CommandLine": command}
                )
    else:
        rows = _process_rows_windows()

    repo_token = str(REPO_ROOT.resolve()).lower().replace("/", "\\")
    showdown_token = str(showdown_dir.resolve()).lower().replace("/", "\\")
    conflicts = []
    for row in rows:
        pid = int(row.get("ProcessId") or 0)
        if pid in {0, os.getpid(), os.getppid()}:
            continue
        command = str(row.get("CommandLine") or "")
        normalized = command.lower().replace("/", "\\")
        name = str(row.get("Name") or "").lower()
        trainer = repo_token in normalized and "scripts\\train.py" in normalized
        harness = (
            repo_token in normalized
            and "benchmark_efficiency_factorial.py" in normalized
        )
        showdown_worker = (
            (name.startswith("node") or "node" in normalized)
            and showdown_token in normalized
            and any(
                token in normalized
                for token in ("worker", "battlestream", "battle-stream")
            )
        )
        if trainer or harness or showdown_worker:
            conflicts.append(
                {"pid": pid, "name": row.get("Name"), "command_line": command}
            )
    return conflicts


def common_training_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    summary_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--device",
        "cuda",
        "--seed",
        str(args.bootstrap_seed),
        "--battle-format",
        "gen9vgc2026regf",
        "--showdown-dir",
        str(args.showdown_dir),
        "--team-pool-format",
        "gen9vgc2026regf",
        "--team-pool-subset",
        "pool",
        "--team-pool-seed",
        str(args.team_pool_seed),
        "--num-envs",
        str(args.num_envs),
        "--opponent-groups",
        str(args.opponent_groups),
        "--worker-count",
        str(args.worker_count),
        "--battles-per-worker",
        str(args.battles_per_worker),
        "--runtime-loop-count",
        str(args.runtime_loop_count),
        "--rollout-steps",
        str(args.rollout_steps),
        "--num-epochs",
        str(args.num_epochs),
        "--num-minibatches",
        str(args.num_minibatches),
        "--log-interval",
        "1",
        "--save-interval",
        "1000000",
        "--league-update-interval",
        "10",
        "--league-refresh-interval",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--summary-path",
        str(summary_path),
    ]


def factor_flags(variant: str) -> list[str]:
    factors = VARIANT_FACTORS[variant]
    return [
        "--lazy-action-orders"
        if factors["lazy_action_orders"]
        else "--no-lazy-action-orders",
        "--league-target-aware-coalescing"
        if factors["target_aware_coalescing"]
        else "--no-league-target-aware-coalescing",
    ]


def bootstrap_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    summary_path: Path,
) -> list[str]:
    return common_training_command(
        args,
        checkpoint_dir=checkpoint_dir,
        summary_path=summary_path,
    ) + ["--num-updates", "1", *factor_flags("B")]


def production_command(
    args: argparse.Namespace,
    planned: PlannedRun,
    *,
    bootstrap_checkpoint: Path,
    checkpoint_dir: Path,
    summary_path: Path,
) -> list[str]:
    # --resume restores the bootstrap's Python/NumPy/torch RNG streams, team
    # sampler, PFSP sampler, and exact opponent assignments. Showdown does not
    # expose an internal battle seed, so production blocks intentionally carry
    # no fictitious seed metadata.
    return common_training_command(
        args,
        checkpoint_dir=checkpoint_dir,
        summary_path=summary_path,
    ) + [
        "--num-updates",
        str(args.updates_per_cell),
        "--resume",
        str(bootstrap_checkpoint),
        *factor_flags(planned.variant),
    ]


def simulator_command(
    args: argparse.Namespace,
    planned: PlannedRun,
    *,
    summary_path: Path,
) -> list[str]:
    if planned.seed is None:
        raise ValueError("Simulator cells require an explicit host RNG seed.")
    return [
        sys.executable,
        str(ENV_BENCHMARK_SCRIPT),
        "--battle-format",
        "gen9vgc2026regf",
        "--showdown-dir",
        str(args.showdown_dir),
        "--num-envs",
        str(args.num_envs),
        "--worker-count",
        str(args.worker_count),
        "--battles-per-worker",
        str(args.battles_per_worker),
        "--runtime-loop-count",
        str(args.runtime_loop_count),
        "--steps",
        str(args.simulator_steps),
        "--scheduling",
        "barrier",
        "--policy",
        "random-legal",
        "--opponent",
        "max-base-power",
        "--team-pool-format",
        "gen9vgc2026regf",
        "--team-pool-subset",
        "pool",
        "--team-pool-seed",
        str(planned.seed),
        "--seed",
        str(planned.seed),
        "--summary-path",
        str(summary_path),
        "--lazy-action-orders" if planned.variant == "L" else "--no-lazy-action-orders",
    ]


def execute_attempt(
    *,
    cell_dir: Path,
    command: Sequence[str],
    metadata: Mapping[str, Any],
    summary_kind: str,
    expected_updates: int = 0,
    discard_updates: int = 0,
    expected_initial_completed_updates: int | None = None,
    expected_initial_global_step: int | None = None,
    variant: str | None = None,
    keep_checkpoints: bool = False,
    preserve_checkpoint: bool = False,
) -> dict[str, Any]:
    existing = validated_success_marker(cell_dir)
    if existing is not None:
        print(f"SKIP {metadata['cell_id']} (verified success)", flush=True)
        return existing

    attempt = next_attempt_dir(cell_dir)
    stdout_path = attempt / "stdout.log"
    stderr_path = attempt / "stderr.log"
    summary_path = attempt / "summary.json"
    checkpoint_dir = attempt / "checkpoints"
    actual_command = [
        str(checkpoint_dir)
        if token == "{CHECKPOINT_DIR}"
        else str(summary_path)
        if token == "{SUMMARY_PATH}"
        else token
        for token in command
    ]
    command_metadata = {
        "schema_version": SCHEMA_VERSION,
        **dict(metadata),
        "attempt": attempt.name,
        "status": "running",
        "started_at": utc_now(),
        "cwd": str(REPO_ROOT),
        "command": actual_command,
    }
    atomic_write_json(attempt / "run.json", command_metadata)
    print(
        f"RUN {metadata['cell_id']} ({attempt.name})\n  "
        + subprocess.list2cmdline(actual_command),
        flush=True,
    )
    started = time.perf_counter()
    try:
        with (
            stdout_path.open("w", encoding="utf-8", newline="") as stdout_handle,
            stderr_path.open("w", encoding="utf-8", newline="") as stderr_handle,
        ):
            completed = subprocess.run(
                actual_command,
                cwd=REPO_ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
    except BaseException as exc:
        command_metadata.update(
            {
                "status": "interrupted",
                "finished_at": utc_now(),
                "elapsed_sec": time.perf_counter() - started,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        atomic_write_json(attempt / "run.json", command_metadata)
        raise

    command_metadata.update(
        {
            "status": "completed" if completed.returncode == 0 else "failed",
            "finished_at": utc_now(),
            "elapsed_sec": time.perf_counter() - started,
            "return_code": completed.returncode,
        }
    )
    atomic_write_json(attempt / "run.json", command_metadata)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cell {metadata['cell_id']} failed with exit code "
            f"{completed.returncode}; see {stderr_path}"
        )
    if not summary_path.is_file():
        raise RuntimeError(f"Cell did not write its summary: {summary_path}")
    summary = read_json(summary_path)
    if not isinstance(summary, Mapping):
        raise RuntimeError(f"Cell summary is not a mapping: {summary_path}")

    if summary_kind == "training":
        metrics = summarize_training_summary(
            summary,
            expected_invocation_updates=expected_updates,
            discard_updates=discard_updates,
            expected_initial_completed_updates=expected_initial_completed_updates,
            expected_initial_global_step=expected_initial_global_step,
        )
        if variant is not None:
            validate_factor_config(summary, variant=variant)
        final_checkpoint = checkpoint_dir / "final.pt"
        league_state = checkpoint_dir / "final.league_state.json"
        if not final_checkpoint.is_file() or not league_state.is_file():
            raise RuntimeError(
                f"Completed training cell omitted final checkpoint state in {checkpoint_dir}"
            )
        checkpoint_hashes = {
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            "final_league_state_sha256": sha256_file(league_state),
        }
    elif summary_kind == "simulator":
        metrics = summarize_simulator_summary(summary)
        expected_lazy = variant == "L"
        if metrics["lazy_action_orders"] is not expected_lazy:
            raise RuntimeError(
                f"Simulator cell {variant} did not record its requested lazy-order mode"
            )
        checkpoint_hashes = {}
    else:
        raise ValueError(f"Unknown summary kind: {summary_kind}")

    relative_artifacts = {
        str(path.relative_to(cell_dir)): sha256_file(path)
        for path in (attempt / "run.json", stdout_path, stderr_path, summary_path)
    }
    marker = {
        "schema_version": SUCCESS_SCHEMA_VERSION,
        "cell_id": metadata["cell_id"],
        "phase": metadata["phase"],
        "block": metadata.get("block"),
        "position": metadata.get("position"),
        "variant": variant,
        "attempt": attempt.name,
        "completed_at": utc_now(),
        "artifacts": relative_artifacts,
        "metrics": metrics,
        "stderr_contains_traceback": (
            "Traceback (most recent call last)"
            in stderr_path.read_text(encoding="utf-8", errors="replace")
        ),
        **checkpoint_hashes,
    }
    atomic_write_json(_success_path(cell_dir), marker)

    if summary_kind == "training" and not preserve_checkpoint and not keep_checkpoints:
        shutil.rmtree(checkpoint_dir)
        marker["checkpoint_dir_removed_after_validation"] = str(checkpoint_dir)
        atomic_write_json(_success_path(cell_dir), marker)
    print(f"DONE {metadata['cell_id']}", flush=True)
    return marker


def collect_success_rows(
    output_dir: Path,
    schedules: Iterable[PlannedRun],
) -> list[dict[str, Any]]:
    rows = []
    for planned in schedules:
        marker = validated_success_marker(output_dir / planned.cell_id)
        if marker is None:
            continue
        rows.append(
            {
                "phase": planned.phase,
                "block": planned.block,
                "position": planned.position,
                "variant": planned.variant,
                "stderr_contains_traceback": bool(
                    marker.get("stderr_contains_traceback", False)
                ),
                **dict(marker["metrics"]),
            }
        )
    return rows


def build_warnings(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    warnings = []
    for row in rows:
        if row.get("phase") != "production":
            continue
        label = f"block {row['block']} variant {row['variant']}"
        if row.get("stderr_contains_traceback"):
            warnings.append(f"{label}: stderr contains a traceback")
        factors = VARIANT_FACTORS[str(row["variant"])]
        wakeups = int(row.get("coalesce_target_wakeups", 0))
        if factors["target_aware_coalescing"] and wakeups <= 0:
            warnings.append(f"{label}: target-aware mode recorded no early wakeups")
        if not factors["target_aware_coalescing"] and wakeups != 0:
            warnings.append(f"{label}: deadline-only mode recorded {wakeups} wakeups")
        fallbacks = int(row.get("cuda_graph_fallbacks", 0))
        if fallbacks:
            warnings.append(f"{label}: {fallbacks} CUDA graph fallback(s)")
        error = row.get("cuda_graph_last_error")
        if error:
            warnings.append(f"{label}: CUDA graph error: {error}")
        if not row.get("cuda_graphs_enabled"):
            warnings.append(f"{label}: CUDA graphs were not enabled at measurement end")
        if int(row.get("cuda_graph_replays", 0)) <= 0:
            warnings.append(f"{label}: measured window recorded no CUDA graph replays")
        if int(row.get("cuda_graph_real_rows", 0)) <= 0:
            warnings.append(f"{label}: measured window recorded no CUDA graph rows")
        if row.get("retained_player_battles_min") != row.get(
            "retained_player_battles_max"
        ):
            warnings.append(f"{label}: retained player battle count changed")
        if row.get("retained_opponent_battles_min") != row.get(
            "retained_opponent_battles_max"
        ):
            warnings.append(f"{label}: retained opponent battle count changed")
    return warnings


def build_retention_decision(
    *,
    operationally_complete: bool,
    production: Sequence[Mapping[str, Any]],
    primary_effects: Mapping[str, Any],
    warnings: Sequence[str],
) -> dict[str, Any]:
    """Apply the experiment's prespecified evidence rule without hiding raw data."""
    fingerprints = {
        str(row.get("config_fingerprint"))
        for row in production
        if row.get("config_fingerprint") is not None
    }
    configuration_matched = len(fingerprints) == 1 and bool(production)
    scientifically_valid = (
        operationally_complete and configuration_matched and not warnings
    )
    factors: dict[str, Any] = {}
    for name in ("lazy", "coalescing"):
        effect = primary_effects[name]
        checks = {
            "effect_at_least_one_percent": (
                effect.get("effect_percent") is not None
                and float(effect["effect_percent"]) >= 1.0
            ),
            "ci_lower_bound_above_zero": (
                effect.get("ci_low_percent") is not None
                and float(effect["ci_low_percent"]) > 0.0
            ),
            "at_least_six_of_eight_constituents_positive": (
                int(effect.get("constituent_count", 0)) == 8
                and int(effect.get("positive_constituents", 0)) >= 6
            ),
        }
        factors[name] = {
            "retain_optimization": scientifically_valid and all(checks.values()),
            "checks": checks,
        }
    return {
        "scientifically_valid": scientifically_valid,
        "configuration_fingerprints_match": configuration_matched,
        "configuration_fingerprints": sorted(fingerprints),
        "rule": {
            "minimum_effect_percent": 1.0,
            "ci_lower_bound_must_exceed_percent": 0.0,
            "minimum_positive_constituents": 6,
            "expected_constituents": 8,
            "requires_no_runtime_warnings": True,
        },
        "factors": factors,
    }


def build_aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_production_cells: int,
    expected_simulator_cells: int,
) -> dict[str, Any]:
    production = [dict(row) for row in rows if row.get("phase") == "production"]
    simulator = [dict(row) for row in rows if row.get("phase") == "simulator"]
    metrics = (
        "wall_steps_per_second",
        "rollout_steps_per_second",
        "rollout_battles_per_second",
    )
    effects = {
        metric: factorial_effects(production, metric=metric) for metric in metrics
    }
    operationally_complete = (
        len(production) == expected_production_cells
        and len(simulator) == expected_simulator_cells
    )
    warnings = build_warnings(production)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "completed" if operationally_complete else "partial",
        "completed_production_cells": len(production),
        "expected_production_cells": expected_production_cells,
        "completed_simulator_cells": len(simulator),
        "expected_simulator_cells": expected_simulator_cells,
        "production_runs": production,
        "simulator_runs": simulator,
        "factorial_effects": effects,
        "simulator_pairs": simulator_pair_effects(simulator),
        "warnings": warnings,
        "primary_metric": "wall_steps_per_second",
        "retention_decision": build_retention_decision(
            operationally_complete=operationally_complete,
            production=production,
            primary_effects=effects["wall_steps_per_second"],
            warnings=warnings,
        ),
        "notes": [
            "Production effects use log contrasts within balanced Williams blocks.",
            "Confidence intervals treat blocks, not individual updates, as independent.",
            "Showdown simulator RNG is not seeded; pairing pins every exposed host stream.",
            "Inference is conditional on one bootstrap checkpoint, this host, and this "
            "temporal schedule; unseeded Showdown battles are not reproducible "
            "battle-seed replicates.",
            "The first configured invocation updates are excluded as process/CUDA warm-up.",
        ],
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_report(aggregate: Mapping[str, Any]) -> str:
    lines = [
        "# pokeRL Q/L/C/B efficiency experiment",
        "",
        f"Status: **{aggregate['status']}**",
        "",
        "Q is the eager/deadline control, L enables lazy battle orders, C enables "
        "target-aware coalescing, and B enables both.",
        "",
        "## Production cells",
        "",
        "| Block | Pos | Variant | Wall SPS | Rollout SPS | Battles/s | Wakeups | Graph rows | Padded | Fallbacks |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        aggregate["production_runs"],
        key=lambda item: (int(item["block"]), int(item["position"])),
    ):
        lines.append(
            "| {block} | {position} | {variant} | {wall} | {rollout} | "
            "{battles} | {wakeups} | {rows} | {padded} | {fallbacks} |".format(
                block=row["block"],
                position=row["position"],
                variant=row["variant"],
                wall=_fmt(row["wall_steps_per_second"]),
                rollout=_fmt(row["rollout_steps_per_second"]),
                battles=_fmt(row["rollout_battles_per_second"], 3),
                wakeups=row.get("coalesce_target_wakeups", 0),
                rows=row.get("cuda_graph_real_rows", 0),
                padded=row.get("cuda_graph_padded_rows", 0),
                fallbacks=row.get("cuda_graph_fallbacks", 0),
            )
        )

    lines.extend(["", "## Factor effects", ""])
    for metric, result in aggregate["factorial_effects"].items():
        lines.extend([f"### {metric}", ""])
        for name in ("lazy", "coalescing", "interaction"):
            effect = result[name]
            signs = ""
            if name in {"lazy", "coalescing"}:
                signs = (
                    f"; positive constituents {effect['positive_constituents']}/"
                    f"{effect['constituent_count']}"
                )
            lines.append(
                f"- {name}: {_fmt(effect['effect_percent'])}% "
                f"(95% CI {_fmt(effect['ci_low_percent'])}% to "
                f"{_fmt(effect['ci_high_percent'])}%){signs}"
            )
        lines.append("")

    simulator = aggregate["simulator_pairs"]["lazy"]
    decision = aggregate["retention_decision"]
    lines.extend(
        [
            "## Prespecified retention decision",
            "",
            "Scientific validity: **{}**.".format(
                "valid" if decision["scientifically_valid"] else "invalid/incomplete"
            ),
            "",
            "- lazy order construction: **{}**".format(
                "retain"
                if decision["factors"]["lazy"]["retain_optimization"]
                else "not established"
            ),
            "- target-aware coalescing: **{}**".format(
                "retain"
                if decision["factors"]["coalescing"]["retain_optimization"]
                else "not established"
            ),
            "",
            "## Simulator-only lazy-order pairs",
            "",
            f"Effect: {_fmt(simulator['effect_percent'])}% "
            f"(95% CI {_fmt(simulator['ci_low_percent'])}% to "
            f"{_fmt(simulator['ci_high_percent'])}%); positive pairs "
            f"{simulator['positive_pairs']}/{simulator['pair_count']}.",
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = aggregate["warnings"]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None detected in completed summaries.")
    lines.extend(
        [
            "",
            "The JSON and CSV beside this report contain the complete aggregate. "
            "Each cell directory retains its raw command metadata, stdout, stderr, "
            "and trainer or simulator summary.",
            "",
        ]
    )
    return "\n".join(lines)


def write_aggregate_files(output_dir: Path, aggregate: Mapping[str, Any]) -> None:
    atomic_write_json(output_dir / "aggregate.json", aggregate)
    atomic_write_text(output_dir / "report.md", render_report(aggregate))
    rows = [*aggregate["simulator_runs"], *aggregate["production_runs"]]
    fieldnames = sorted({key for row in rows for key in row})
    temporary = output_dir / f".aggregate.csv.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(temporary, output_dir / "aggregate.csv")
    finally:
        temporary.unlink(missing_ok=True)


def write_progress(
    output_dir: Path,
    *,
    status: str,
    completed: int,
    total: int,
    current_cell: str | None = None,
    error: str | None = None,
) -> None:
    atomic_write_json(
        output_dir / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "completed_cells": completed,
            "total_cells": total,
            "current_cell": current_cell,
            "error": error,
            "updated_at": utc_now(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Stable experiment directory. Reuse exactly this path after an "
            "interruption to verify and skip completed cells."
        ),
    )
    parser.add_argument("--showdown-dir", default=None)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--updates-per-cell", type=int, default=10)
    parser.add_argument("--discard-updates", type=int, default=3)
    parser.add_argument("--simulator-rounds", type=int, default=6)
    parser.add_argument("--simulator-steps", type=int, default=128)
    parser.add_argument("--cooldown-sec", type=float, default=30.0)
    parser.add_argument("--bootstrap-seed", type=int, default=93_001)
    parser.add_argument("--team-pool-seed", type=int, default=93_002)
    parser.add_argument("--num-envs", type=int, default=72)
    parser.add_argument("--opponent-groups", type=int, default=2)
    parser.add_argument("--worker-count", type=int, default=24)
    parser.add_argument("--battles-per-worker", type=int, default=3)
    parser.add_argument("--runtime-loop-count", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep every completed factor-cell checkpoint (roughly gigabytes).",
    )
    parser.add_argument(
        "--allow-process-collisions",
        action="store_true",
        help="Run despite another detected pokeRL trainer/Showdown worker.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the immutable plan without creating files or launching children.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.blocks <= 0 or args.blocks % len(WILLIAMS_ORDERS) != 0:
        raise ValueError("--blocks must be a positive multiple of 4")
    if args.updates_per_cell <= 0:
        raise ValueError("--updates-per-cell must be positive")
    if not 0 <= args.discard_updates < args.updates_per_cell:
        raise ValueError("--discard-updates must be smaller than updates per cell")
    if args.simulator_rounds < 0 or args.simulator_rounds % 2:
        raise ValueError("--simulator-rounds must be a non-negative even integer")
    for name in (
        "simulator_steps",
        "num_envs",
        "opponent_groups",
        "worker_count",
        "battles_per_worker",
        "runtime_loop_count",
        "rollout_steps",
        "num_epochs",
        "num_minibatches",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.opponent_groups > args.num_envs:
        raise ValueError("--opponent-groups cannot exceed --num-envs")
    if not math.isfinite(args.cooldown_sec) or args.cooldown_sec < 0.0:
        raise ValueError("--cooldown-sec must be finite and non-negative")


def experiment_manifest(
    args: argparse.Namespace,
    *,
    production_schedule: Sequence[PlannedRun],
    simulator_schedule: Sequence[PlannedRun],
) -> dict[str, Any]:
    repo = git_identity(REPO_ROOT)
    require_clean_git_identity(repo, name="pokeRL")
    poke_env = discover_poke_env_identity()
    poke_env_source = poke_env.get("source")
    poke_env_tree = poke_env.get("installed_package_tree")
    if isinstance(poke_env_source, Mapping):
        require_clean_git_identity(poke_env_source, name="active poke-env")
    elif not isinstance(poke_env_tree, Mapping) or not poke_env_tree.get("sha256"):
        raise RuntimeError(
            "Could not pin active poke-env to clean tracked source or an installed "
            "package-tree digest."
        )
    showdown = git_identity(args.showdown_dir)
    require_clean_git_identity(showdown, name="Pokemon Showdown")
    torch_version = distribution_version("torch")
    if torch_version is None:
        raise RuntimeError("The active interpreter has no torch distribution metadata.")
    node = discover_node_identity()
    if node is None:
        raise RuntimeError("Node.js is unavailable or its version could not be read.")
    nvidia = discover_nvidia_identity()
    if nvidia is None:
        raise RuntimeError("nvidia-smi could not verify the CUDA/driver/GPU runtime.")
    if not TEAM_MANIFEST.is_file():
        raise RuntimeError(f"Training team manifest not found: {TEAM_MANIFEST}")
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "pokerl": repo,
            "poke_env": poke_env,
            "showdown": showdown,
            "team_manifest_path": str(TEAM_MANIFEST.resolve()),
            "team_manifest_sha256": sha256_file(TEAM_MANIFEST),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "runtime": {
                "torch_distribution_version": torch_version,
                "node": node,
                "nvidia": nvidia,
            },
        },
        "factor_definitions": VARIANT_FACTORS,
        "config": {
            "showdown_dir": str(args.showdown_dir),
            "blocks": args.blocks,
            "updates_per_cell": args.updates_per_cell,
            "discard_updates": args.discard_updates,
            "simulator_rounds": args.simulator_rounds,
            "simulator_steps": args.simulator_steps,
            "cooldown_sec": args.cooldown_sec,
            "bootstrap_seed": args.bootstrap_seed,
            "team_pool_seed": args.team_pool_seed,
            "num_envs": args.num_envs,
            "opponent_groups": args.opponent_groups,
            "worker_count": args.worker_count,
            "battles_per_worker": args.battles_per_worker,
            "runtime_loop_count": args.runtime_loop_count,
            "rollout_steps": args.rollout_steps,
            "num_epochs": args.num_epochs,
            "num_minibatches": args.num_minibatches,
            "keep_checkpoints": args.keep_checkpoints,
            "allow_process_collisions": args.allow_process_collisions,
            "collect_timing": False,
            "simulator_rng_seeded": False,
        },
        "bootstrap": {
            "updates": 1,
            "variant": "B",
            "purpose": "common weights/optimizer/RNG/league/team-sampler boundary",
        },
        "production_schedule": [
            planned_run_payload(item) for item in production_schedule
        ],
        "simulator_schedule": [
            planned_run_payload(item) for item in simulator_schedule
        ],
        "statistics": {
            "primary_metric": "steps / sum(update_wall_sec) after warm-up",
            "independent_unit": "Williams block",
            "confidence_interval": "two-sided 95% Student t over block log effects",
            "inference_scope": (
                "conditional on one bootstrap checkpoint, this host, and this "
                "temporal schedule; Showdown battle RNG is not seed-controlled"
            ),
        },
    }


def _sleep_between_cells(seconds: float) -> None:
    if seconds <= 0.0:
        return
    print(f"Cooling down for {seconds:g}s...", flush=True)
    time.sleep(seconds)


def _enforce_cooldown_since_last_success(output_dir: Path, seconds: float) -> None:
    """Preserve the thermal cooldown even if the harness was interrupted in it."""
    if seconds <= 0.0:
        return
    completed_times = []
    for marker_path in output_dir.rglob("SUCCESS.json"):
        marker = read_json(marker_path)
        completed_at = (
            marker.get("completed_at") if isinstance(marker, Mapping) else None
        )
        if not isinstance(completed_at, str):
            continue
        completed_times.append(datetime.fromisoformat(completed_at).timestamp())
    if not completed_times:
        return
    remaining = seconds - (time.time() - max(completed_times))
    _sleep_between_cells(max(0.0, remaining))


def _cell_count(output_dir: Path, all_runs: Sequence[PlannedRun]) -> int:
    count = int(validated_success_marker(output_dir / "bootstrap") is not None)
    return count + sum(
        validated_success_marker(output_dir / planned.cell_id) is not None
        for planned in all_runs
    )


def _refresh_outputs(
    output_dir: Path,
    production: Sequence[PlannedRun],
    simulator: Sequence[PlannedRun],
) -> dict[str, Any]:
    rows = collect_success_rows(output_dir, [*simulator, *production])
    aggregate = build_aggregate(
        rows,
        expected_production_cells=len(production),
        expected_simulator_cells=len(simulator),
    )
    write_aggregate_files(output_dir, aggregate)
    return aggregate


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.showdown_dir = resolve_showdown_dir(args.showdown_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    production = build_production_schedule(args.blocks)
    simulator = build_simulator_schedule(args.simulator_rounds)
    all_runs = [*simulator, *production]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "showdown_dir": str(args.showdown_dir),
                    "bootstrap_updates": 1,
                    "production": [planned_run_payload(item) for item in production],
                    "simulator": [planned_run_payload(item) for item in simulator],
                    "factor_definitions": VARIANT_FACTORS,
                },
                indent=2,
            )
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with ExperimentLock(output_dir / ".experiment.lock"):
        manifest = experiment_manifest(
            args,
            production_schedule=production,
            simulator_schedule=simulator,
        )
        ensure_experiment_manifest(output_dir / "experiment.json", manifest)
        conflicts = find_conflicting_processes(args.showdown_dir)
        if conflicts and not args.allow_process_collisions:
            atomic_write_json(output_dir / "process_collisions.json", conflicts)
            raise RuntimeError(
                "Another pokeRL trainer or local Showdown worker is running. "
                f"See {output_dir / 'process_collisions.json'}; wait for it to stop, "
                "or explicitly use --allow-process-collisions."
            )

        total_cells = 1 + len(all_runs)
        completed = _cell_count(output_dir, all_runs)
        write_progress(
            output_dir,
            status="running",
            completed=completed,
            total=total_cells,
        )
        try:
            bootstrap_cell = output_dir / "bootstrap"
            bootstrap_was_done = validated_success_marker(bootstrap_cell) is not None
            if not bootstrap_was_done:
                _enforce_cooldown_since_last_success(output_dir, args.cooldown_sec)
            bootstrap_marker = execute_attempt(
                cell_dir=bootstrap_cell,
                command=bootstrap_command(
                    args,
                    checkpoint_dir=Path("{CHECKPOINT_DIR}"),
                    summary_path=Path("{SUMMARY_PATH}"),
                ),
                metadata={
                    "cell_id": "bootstrap",
                    "phase": "bootstrap",
                    "block": 0,
                    "position": 0,
                },
                summary_kind="training",
                expected_updates=1,
                discard_updates=0,
                expected_initial_completed_updates=0,
                expected_initial_global_step=0,
                variant="B",
                preserve_checkpoint=True,
            )
            bootstrap_attempt = bootstrap_cell / str(bootstrap_marker["attempt"])
            bootstrap_checkpoint = bootstrap_attempt / "checkpoints" / "final.pt"
            bootstrap_league = (
                bootstrap_attempt / "checkpoints" / "final.league_state.json"
            )
            for path, hash_key in (
                (bootstrap_checkpoint, "final_checkpoint_sha256"),
                (bootstrap_league, "final_league_state_sha256"),
            ):
                if (
                    not path.is_file()
                    or sha256_file(path) != bootstrap_marker[hash_key]
                ):
                    raise RuntimeError(
                        f"Bootstrap artifact is missing or changed: {path}"
                    )
            for planned in simulator:
                cell_dir = output_dir / planned.cell_id
                was_done = validated_success_marker(cell_dir) is not None
                if not was_done:
                    _enforce_cooldown_since_last_success(output_dir, args.cooldown_sec)
                write_progress(
                    output_dir,
                    status="running",
                    completed=_cell_count(output_dir, all_runs),
                    total=total_cells,
                    current_cell=planned.cell_id,
                )
                execute_attempt(
                    cell_dir=cell_dir,
                    command=simulator_command(
                        args,
                        planned,
                        summary_path=Path("{SUMMARY_PATH}"),
                    ),
                    metadata={
                        "cell_id": planned.cell_id,
                        **planned_run_payload(planned),
                    },
                    summary_kind="simulator",
                    variant=planned.variant,
                )
                _refresh_outputs(output_dir, production, simulator)

            for planned in production:
                cell_dir = output_dir / planned.cell_id
                was_done = validated_success_marker(cell_dir) is not None
                if not was_done:
                    _enforce_cooldown_since_last_success(output_dir, args.cooldown_sec)
                write_progress(
                    output_dir,
                    status="running",
                    completed=_cell_count(output_dir, all_runs),
                    total=total_cells,
                    current_cell=planned.cell_id,
                )
                execute_attempt(
                    cell_dir=cell_dir,
                    command=production_command(
                        args,
                        planned,
                        bootstrap_checkpoint=bootstrap_checkpoint,
                        checkpoint_dir=Path("{CHECKPOINT_DIR}"),
                        summary_path=Path("{SUMMARY_PATH}"),
                    ),
                    metadata={
                        "cell_id": planned.cell_id,
                        **planned_run_payload(planned),
                    },
                    summary_kind="training",
                    expected_updates=args.updates_per_cell,
                    discard_updates=args.discard_updates,
                    expected_initial_completed_updates=int(
                        bootstrap_marker["metrics"]["final_update"]
                    ),
                    expected_initial_global_step=int(
                        bootstrap_marker["metrics"]["final_global_step"]
                    ),
                    variant=planned.variant,
                    keep_checkpoints=args.keep_checkpoints,
                )
                _refresh_outputs(output_dir, production, simulator)

            aggregate = _refresh_outputs(output_dir, production, simulator)
            write_progress(
                output_dir,
                status=str(aggregate["status"]),
                completed=_cell_count(output_dir, all_runs),
                total=total_cells,
            )
            print(f"Experiment complete. Read {output_dir / 'report.md'}", flush=True)
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            write_progress(
                output_dir,
                status=status,
                completed=_cell_count(output_dir, all_runs),
                total=total_cells,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


if __name__ == "__main__":
    main()
