"""Benchmark short PPO training runs across env/server configurations."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pokeRL env/server scaling.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to train.py auto-detection")
    parser.add_argument("--battle-format", default="gen9vgc2026regf", help="Showdown format to benchmark.")
    parser.add_argument(
        "--team-pool-format",
        default="gen9vgc2026regf",
        help="Team pool format under data/team_pools/<format>/.",
    )
    parser.add_argument(
        "--team-pool-subset",
        choices=["curated", "generated", "pool"],
        default="pool",
        help="Which team pool subset to sample from.",
    )
    parser.add_argument("--team-pool-seed", type=int, default=0, help="Seed for fixed-team sampling.")
    parser.add_argument("--env-counts", default="4,8,16", help="Comma-separated env counts to benchmark.")
    parser.add_argument("--server-counts", default="1,2,4", help="Comma-separated server counts to benchmark.")
    parser.add_argument("--base-port", type=int, default=8000, help="First localhost port in the server list.")
    parser.add_argument("--rollout-steps", type=int, default=16, help="Rollout steps collected per update.")
    parser.add_argument("--num-updates", type=int, default=2, help="How many rollout/PPO updates to run per config.")
    parser.add_argument("--num-epochs", type=int, default=4, help="PPO epochs per update.")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Maximum PPO minibatches per update.")
    parser.add_argument(
        "--league-update-interval",
        type=int,
        default=1,
        help="How often to refresh the league during the short benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for logs and summary JSON. Defaults to benchmarks/<timestamp>.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep per-run checkpoint directories instead of deleting them with the benchmark folder cleanup policy.",
    )
    return parser.parse_args()


def parse_csv_ints(raw_value: str) -> list[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def ensure_servers_ready(server_urls: list[str]) -> None:
    missing = [url for url in server_urls if not _port_open(url)]
    if missing:
        missing_ports = ", ".join(missing)
        raise SystemExit(
            f"The following Showdown servers are not reachable: {missing_ports}. "
            "Start them first with scripts/start_showdown_server.ps1."
        )


def _port_open(server_url: str) -> bool:
    host, port_raw = server_url.split(":", maxsplit=1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, int(port_raw))) == 0


def benchmark_config(
    *,
    args: argparse.Namespace,
    env_count: int,
    server_count: int,
    output_dir: Path,
) -> dict:
    server_urls = [f"localhost:{args.base_port + idx}" for idx in range(server_count)]
    ensure_servers_ready(server_urls)

    run_name = f"env{env_count}_srv{server_count}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    checkpoint_dir = run_dir / "checkpoints"
    effective_minibatches = min(args.num_minibatches, env_count)

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--battle-format",
        args.battle_format,
        "--team-pool-format",
        args.team_pool_format,
        "--team-pool-subset",
        args.team_pool_subset,
        "--team-pool-seed",
        str(args.team_pool_seed),
        "--num-envs",
        str(env_count),
        "--rollout-steps",
        str(args.rollout_steps),
        "--num-updates",
        str(args.num_updates),
        "--num-epochs",
        str(args.num_epochs),
        "--num-minibatches",
        str(effective_minibatches),
        "--log-interval",
        "1",
        "--save-interval",
        "1000000",
        "--league-update-interval",
        str(args.league_update_interval),
        "--server-urls",
        ",".join(server_urls),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--summary-path",
        str(summary_path),
    ]
    if args.device:
        command.extend(["--device", args.device])

    print(
        f"Starting {run_name} | "
        f"rollout_steps={args.rollout_steps} updates={args.num_updates} "
        f"epochs={args.num_epochs} minibatches={effective_minibatches}"
    )

    combined_output: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        combined_output.append(line)
    return_code = process.wait()

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    stdout_log.write_text("".join(combined_output), encoding="utf-8")
    stderr_log.write_text("", encoding="utf-8")

    if return_code != 0:
        raise RuntimeError(
            f"Benchmark run {run_name} failed with exit code {return_code}. "
            f"See {stdout_log}."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    updates = summary.get("updates", [])
    total_rollout = sum(float(update["rollout_sec"]) for update in updates)
    total_ppo = sum(float(update["ppo_sec"]) for update in updates)
    final_wall = float(summary["final"]["wall_time_sec"])
    total_steps = int(summary["final"]["global_step"])
    total_env_step_calls = sum(
        float(update.get("env_profile", {}).get("env_step_calls", 0.0)) for update in updates
    )
    total_env_step_sec = sum(
        float(update.get("env_profile", {}).get("env_step_total_sec", 0.0)) for update in updates
    )
    total_queue_wait_sec = sum(
        float(update.get("env_profile", {}).get("env_queue_wait_sec", 0.0)) for update in updates
    )
    total_policy_forward_sec = sum(
        float(update.get("rollout_profile", {}).get("policy_forward_sec", 0.0)) for update in updates
    )
    total_obs_to_device_sec = sum(
        float(update.get("rollout_profile", {}).get("obs_to_device_sec", 0.0)) for update in updates
    )
    total_mask_to_device_sec = sum(
        float(update.get("rollout_profile", {}).get("mask_to_device_sec", 0.0)) for update in updates
    )

    result = {
        "envs": env_count,
        "servers": server_count,
        "server_urls": server_urls,
        "rollout_steps": args.rollout_steps,
        "num_updates": args.num_updates,
        "num_epochs": args.num_epochs,
        "num_minibatches": effective_minibatches,
        "initial_reset_sec": float(summary.get("initial_reset_sec", 0.0)),
        "total_rollout_sec": total_rollout,
        "total_ppo_sec": total_ppo,
        "total_wall_sec": final_wall,
        "steps": total_steps,
        "steps_per_sec": total_steps / final_wall if final_wall > 0 else 0.0,
        "env_step_avg_ms": (total_env_step_sec / total_env_step_calls * 1000.0) if total_env_step_calls else 0.0,
        "queue_wait_share": (total_queue_wait_sec / total_env_step_sec) if total_env_step_sec > 0 else 0.0,
        "policy_forward_share": (total_policy_forward_sec / total_rollout) if total_rollout > 0 else 0.0,
        "device_transfer_share": ((total_obs_to_device_sec + total_mask_to_device_sec) / total_rollout) if total_rollout > 0 else 0.0,
        "summary_path": str(summary_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    if not args.keep_checkpoints:
        result["checkpoint_dir"] = str(checkpoint_dir)
    return result


def main() -> None:
    args = parse_args()
    env_counts = parse_csv_ints(args.env_counts)
    server_counts = parse_csv_ints(args.server_counts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "benchmarks" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for env_count in env_counts:
        for server_count in server_counts:
            if server_count > env_count:
                continue

            result = benchmark_config(
                args=args,
                env_count=env_count,
                server_count=server_count,
                output_dir=output_dir,
            )
            results.append(result)
            print(
                f"envs={env_count} servers={server_count} | "
                f"SPS {result['steps_per_sec']:.1f} | "
                f"init {result['initial_reset_sec']:.2f}s | "
                f"rollout {result['total_rollout_sec']:.2f}s | "
                f"ppo {result['total_ppo_sec']:.2f}s | "
                f"queue {result['queue_wait_share'] * 100.0:.1f}%"
            )

    ranked = sorted(results, key=lambda item: item["steps_per_sec"], reverse=True)
    summary = {
        "battle_format": args.battle_format,
        "team_pool_format": args.team_pool_format,
        "generated_at": datetime.now().isoformat(),
        "results": results,
        "ranked_by_steps_per_sec": ranked,
        "best": ranked[0] if ranked else None,
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()