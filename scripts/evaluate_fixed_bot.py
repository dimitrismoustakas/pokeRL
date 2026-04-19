"""Evaluate a random or trained policy against a fixed max-BST max-power bot."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import numpy as np
import torch

from pokerl.config import EnvConfig, ModelConfig
from pokerl.data import GameData
from pokerl.eval.fixed_bot import (
    build_species_bst_lookup,
    evaluate_policy_vs_fixed_bot,
    load_agent_for_evaluation,
    select_max_bst_team_spec,
    select_team_spec_by_title,
)
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pokeRL against a MaxBasePowerPlayer using real 6-to-4 team preview battles."
    )
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path; omit for random initialization.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument("--battle-format", default="gen9vgc2026regf", help="Showdown battle format.")
    parser.add_argument("--server-url", default="localhost:8000", help="Showdown server host:port.")
    parser.add_argument(
        "--team-pool-format",
        default="gen9vgc2026regf",
        help="Load teams from data/team_pools/<format>/manifest.json.",
    )
    parser.add_argument(
        "--team-pool-subset",
        choices=["curated", "generated", "pool"],
        default="pool",
        help="Which manifest subset to use for both the fixed model roster and the varied opponent teams.",
    )
    parser.add_argument("--team-pool-seed", type=int, default=0, help="Seed used when loading the team pool.")
    parser.add_argument(
        "--model-team-title",
        default=None,
        help="Optional exact team title to use as the model's fixed 6-mon roster. Defaults to the strongest pool team.",
    )
    parser.add_argument("--games", type=int, default=32, help="Number of evaluation battles.")
    parser.add_argument(
        "--action-mode",
        choices=["sample", "argmax"],
        default="argmax",
        help="Whether to sample from the policy or play greedily.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch/numpy seed for evaluation reproducibility.")
    parser.add_argument("--max-turns", type=int, default=200, help="Fail if a battle exceeds this many decisions.")
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="Optional output directory for representative replay HTML/JSON/trace artifacts.",
    )
    parser.add_argument(
        "--replay-count",
        type=int,
        default=1,
        help="How many completed battles to save as replay artifacts.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path for the JSON evaluation summary.",
    )
    return parser.parse_args()


def _port_open(server_url: str) -> bool:
    host, port_raw = server_url.split(":", maxsplit=1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, int(port_raw))) == 0


def _write_json(path: str, payload: dict) -> Path:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    if not _port_open(args.server_url):
        raise SystemExit(
            f"Showdown server {args.server_url} is not reachable. Start it first with scripts/start_showdown_server.ps1."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    env_config = EnvConfig(
        battle_format=args.battle_format,
        server_url=args.server_url,
        num_envs=1,
        team_size=6,
        pick_size=4,
    )
    model_config = ModelConfig(pick_size=env_config.pick_size)
    game_data = GameData(gen=9)
    species_bst_lookup = build_species_bst_lookup(gen=game_data.gen)

    manifest_path = default_team_pool_manifest_path(REPO_ROOT, args.team_pool_format)
    team_pool = load_team_pool(
        manifest_path,
        subset=args.team_pool_subset,
        seed=args.team_pool_seed,
    )
    if args.model_team_title:
        model_team_spec = select_team_spec_by_title(
            team_pool,
            title=args.model_team_title,
            species_bst_lookup=species_bst_lookup,
            pick_size=env_config.pick_size,
        )
    else:
        model_team_spec = select_max_bst_team_spec(
            team_pool,
            species_bst_lookup=species_bst_lookup,
            pick_size=env_config.pick_size,
        )
    agent = load_agent_for_evaluation(
        checkpoint_path=args.checkpoint,
        device=device,
        model_config=model_config,
        game_data=game_data,
    )

    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "device": str(device),
            "battle_format": env_config.battle_format,
            "server_url": env_config.server_url,
            "games": args.games,
            "action_mode": args.action_mode,
            "opponent": "max-base-power",
            "team_size": env_config.team_size,
            "pick_size": env_config.pick_size,
            "seed": args.seed,
            "team_pool_format": args.team_pool_format,
            "team_pool_subset": args.team_pool_subset,
            "team_pool_size": team_pool.size,
            "model_team_title": model_team_spec.title,
        },
        "model_team": model_team_spec.to_summary(),
        "results": evaluate_policy_vs_fixed_bot(
            agent=agent,
            device=device,
            env_config=env_config,
            model_team_spec=model_team_spec,
            opponent_team_pool=team_pool,
            species_bst_lookup=species_bst_lookup,
            num_games=args.games,
            action_mode=args.action_mode,
            seed=args.seed,
            max_turns=args.max_turns,
            replay_dir=args.replay_dir,
            replay_count=args.replay_count,
        ),
    }

    if args.summary_path:
        out_path = _write_json(args.summary_path, summary)
        print(f"Wrote summary to {out_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()