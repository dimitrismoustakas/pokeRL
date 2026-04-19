const fs = require("fs");
const path = require("path");
const readline = require("readline");
const os = require("os");

function parseArgs(argv) {
  const args = {
    installDir: path.join(os.homedir(), "AppData", "Local", "pokeRL", "pokemon-showdown-srv8600"),
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--install-dir" && next) {
      args.installDir = path.resolve(next);
      index += 1;
    } else {
      throw new Error(`Unknown or incomplete argument: ${arg}`);
    }
  }
  return args;
}

function normalizeChoice(choice) {
  const trimmed = String(choice || "").trim();
  if (trimmed.startsWith("/choose ")) {
    return trimmed.slice("/choose ".length);
  }
  if (trimmed.startsWith("/team ")) {
    return `team ${trimmed.slice("/team ".length)}`;
  }
  if (trimmed.startsWith("/")) {
    return trimmed.slice(1);
  }
  return trimmed;
}

function createEmitter() {
  return (payload) => {
    process.stdout.write(`${JSON.stringify(payload)}\n`);
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sim = require(path.join(args.installDir, "dist", "sim"));
  const { BattleStream, getPlayerStreams } = sim;
  const emit = createEmitter();
  const battles = new Map();

  async function watchSide(battleId, side, stream) {
    try {
      for await (const chunk of stream) {
        emit({ type: "chunk", battle_id: battleId, side, chunk });
      }
      emit({ type: "side_end", battle_id: battleId, side });
    } catch (error) {
      emit({
        type: "worker_error",
        battle_id: battleId,
        side,
        error: error instanceof Error ? error.stack || error.message : String(error),
      });
    }
  }

  async function createBattle(command) {
    if (battles.has(command.battle_id)) {
      throw new Error(`Battle ${command.battle_id} already exists.`);
    }
    const battleStream = new BattleStream({ keepAlive: true });
    const streams = getPlayerStreams(battleStream);
    const battle = {
      streams,
      tasks: [
        watchSide(command.battle_id, "p1", streams.p1),
        watchSide(command.battle_id, "p2", streams.p2),
      ],
    };
    battles.set(command.battle_id, battle);
    await streams.omniscient.write(
      `>start ${JSON.stringify({ formatid: command.format })}\n` +
        `>player p1 ${JSON.stringify({ name: command.p1_name, team: command.p1_team || undefined })}\n` +
        `>player p2 ${JSON.stringify({ name: command.p2_name, team: command.p2_team || undefined })}`
    );
    emit({ type: "created", battle_id: command.battle_id });
  }

  async function closeBattle(battleId) {
    const battle = battles.get(battleId);
    if (!battle) {
      return;
    }
    battles.delete(battleId);
    try {
      await battle.streams.omniscient.writeEnd();
    } catch (_error) {
      // best-effort shutdown
    }
    await Promise.allSettled(battle.tasks);
    emit({ type: "closed", battle_id: battleId });
  }

  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (!line.trim()) {
      continue;
    }
    let command;
    try {
      command = JSON.parse(line);
    } catch (error) {
      process.stderr.write(`Invalid JSON command: ${line}\n`);
      continue;
    }
    try {
      if (command.type === "create") {
        await createBattle(command);
      } else if (command.type === "choice") {
        const battle = battles.get(command.battle_id);
        if (!battle) {
          throw new Error(`Unknown battle ${command.battle_id}.`);
        }
        const choice = normalizeChoice(command.choice);
        await battle.streams[command.side].write(choice);
      } else if (command.type === "close_battle") {
        await closeBattle(command.battle_id);
      } else if (command.type === "shutdown") {
        for (const battleId of Array.from(battles.keys())) {
          await closeBattle(battleId);
        }
        break;
      } else {
        throw new Error(`Unknown command type: ${command.type}`);
      }
    } catch (error) {
      emit({
        type: "worker_error",
        battle_id: command.battle_id || null,
        error: error instanceof Error ? error.stack || error.message : String(error),
      });
    }
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack || error.message : String(error)}\n`);
  process.exitCode = 1;
});
