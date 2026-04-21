"""Diagnostic: run battles until RuntimeError, then print battle state."""
from __future__ import annotations
import pathlib, traceback, sys
import numpy as np

# Write to both stdout and a log file for visibility when backgrounded
_LOG = open("diag_out.txt", "w", buffering=1)  # line-buffered

def log(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, file=_LOG, **kwargs)
    _LOG.flush()

from pokerl.data import GameData
from pokerl.env.battle_stream_env import BattleStreamEnv
from pokerl.env.observation import ObservationBuilder
from pokerl.env.actions import build_legal_action_state, ACTIONS_PER_SLOT, _build_slot_action_state, _ordered_own_team
from pokerl.team_pool import load_team_pool, default_team_pool_manifest_path


def dump_battle(battle, label="battle"):
    print(f"\n--- {label} state ---")
    print(f"  _wait={battle._wait}")
    print(f"  force_switch={battle.force_switch}")
    try:
        print(f"  active_pokemon={[str(p) if p else None for p in battle.active_pokemon]}")
    except Exception as e:
        print(f"  active_pokemon ERROR: {e}")
    try:
        print(f"  available_switches={[[str(p) for p in sl] for sl in battle.available_switches]}")
    except Exception as e:
        print(f"  available_switches ERROR: {e}")
    try:
        print(f"  available_moves={[[m.id for m in sl] for sl in battle.available_moves]}")
    except Exception as e:
        print(f"  available_moves ERROR: {e}")
    req = getattr(battle, '_last_request', None)
    if req:
        print(f"  last_request keys={list(req.keys())}")
        if 'forceSwitch' in req:
            print(f"  forceSwitch={req['forceSwitch']}")
        if 'wait' in req:
            print(f"  wait={req['wait']}")
        if 'active' in req:
            print(f"  active count={len(req['active'])}")

    try:
        ordered_team = _ordered_own_team(battle)
        team_slot_by_id = {id(p): i for i, p in enumerate(ordered_team) if p is not None}
        for slot in range(2):
            try:
                ss = _build_slot_action_state(battle, slot, team_slot_by_id)
                print(f"  slot {slot} action_orders keys={list(ss.action_orders.keys())}")
            except Exception as e2:
                print(f"  slot {slot} _build_slot_action_state ERROR: {e2}")
    except Exception as e:
        print(f"  slot state ERROR: {e}")


print("Loading GameData...", flush=True)
gd = GameData(gen=9)
print("Loading team pool...", flush=True)
pool = load_team_pool(default_team_pool_manifest_path(pathlib.Path("."), "gen9vgc2026regf"))
obs_builder = ObservationBuilder(gd)
print("Setup done.", flush=True)

rng = np.random.default_rng(0)
errors = 0
battles = 0

print("Starting diagnostic...")

import sys

import queue as _queue
import time as _time

def probe_process(bsp, trial):
    """Check subprocess and queue state."""
    rc = bsp._proc.poll()
    print(f"  [{trial}] proc.poll()={rc} (None=running)", flush=True)
    # Drain up to 10 lines with 5s timeout each
    for i in range(10):
        try:
            line = bsp._line_queue.get(timeout=5.0)
            print(f"  [{trial}] queue line {i}: {line!r}", flush=True)
            if line is None:
                print(f"  [{trial}] EOF sentinel received!", flush=True)
                break
        except _queue.Empty:
            rc2 = bsp._proc.poll()
            print(f"  [{trial}] queue empty after 2s, proc.poll()={rc2}", flush=True)
            break

def traced_reset(env, trial):
    print(f"  [{trial}] closing old stream...", flush=True)
    if env._stream is not None:
        env._stream.close()
        env._stream = None
    print(f"  [{trial}] sampling team pair...", flush=True)
    p1_team, p2_team = env._pool.sample_packed_pair()
    print(f"  [{trial}] creating BattleStreamProcess...", flush=True)
    from pokerl.env.battle_stream_env import BattleStreamProcess
    env._stream = BattleStreamProcess(
        env._format, p1_team, p2_team,
        gen=env._gen, showdown_dir=env._showdown_dir,
    )
    print(f"  [{trial}] BattleStreamProcess created, probing process...", flush=True)
    probe_process(env._stream, trial)
    print(f"  [{trial}] calling read_until_requests...", flush=True)
    env._stream.read_until_requests()
    print(f"  [{trial}] read_until_requests done, teampreview={env._stream.battle_p1.teampreview}", flush=True)
    if env._stream.battle_p1.teampreview:
        p1_order = env._preview_order(env._stream.battle_p1)
        p2_order = env._preview_order(env._stream.battle_p2)
        env._stream.send_team_preview(p1_order, p2_order)
        print(f"  [{trial}] sent team preview, reading requests again...", flush=True)
        env._stream.read_until_requests()
    battle = env._stream.battle_p1
    from pokerl.env.actions import build_legal_action_state
    obs = env._obs_builder.build(battle)
    action_state = build_legal_action_state(battle)
    return obs, action_state.joint_mask

for trial in range(10):
    env = BattleStreamEnv(
        battle_format="gen9vgc2026regf",
        team_pool=pool,
        obs_builder=obs_builder,
    )
    print(f"battle {trial}: starting reset...", flush=True)
    obs, mask = traced_reset(env, trial)
    battles += 1
    print(f"battle {trial}: reset ok, mask shape={mask.shape}", flush=True)
    turn = 0
    done = False

    while not done:
        a0 = int(rng.choice(np.flatnonzero(mask[0])))
        legal1 = np.flatnonzero(mask[a0 + 1])
        a1 = int(rng.choice(legal1)) if legal1.size else a0

        try:
            obs, mask, reward, done, _ = env.step(np.array([a0, a1], dtype=np.int64))
            turn += 1
        except RuntimeError as e:
            errors += 1
            print(f"\n[ERROR] battle {trial}, turn {turn}: {e}")
            traceback.print_exc()
            dump_battle(env._stream.battle_p1, "p1")
            dump_battle(env._stream.battle_p2, "p2")
            done = True

    env.close()
    print(f"  battle {trial} done in {turn} turns, reward={reward:.0f}")
    if errors:
        break

print(f"\n{battles} battles, {errors} errors")
