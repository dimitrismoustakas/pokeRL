const fs = require("fs");
const path = require("path");
const os = require("os");

function parseArgs(argv) {
  const args = {
    installDir: path.join(os.homedir(), "AppData", "Local", "pokeRL", "pokemon-showdown-srv8600"),
    manifest: path.resolve(__dirname, "..", "data", "team_pools", "gen9vgc2026regf", "manifest.json"),
    subset: "pool",
    format: "gen9vgc2026regf",
    numBattles: 128,
    concurrency: 32,
    seed: 0,
    summaryPath: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--install-dir" && next) {
      args.installDir = path.resolve(next);
      index += 1;
    } else if (arg === "--manifest" && next) {
      args.manifest = path.resolve(next);
      index += 1;
    } else if (arg === "--subset" && next) {
      args.subset = next;
      index += 1;
    } else if (arg === "--format" && next) {
      args.format = next;
      index += 1;
    } else if (arg === "--num-battles" && next) {
      args.numBattles = Number(next);
      index += 1;
    } else if (arg === "--concurrency" && next) {
      args.concurrency = Number(next);
      index += 1;
    } else if (arg === "--seed" && next) {
      args.seed = Number(next);
      index += 1;
    } else if (arg === "--summary-path" && next) {
      args.summaryPath = path.resolve(next);
      index += 1;
    } else {
      throw new Error(`Unknown or incomplete argument: ${arg}`);
    }
  }
  if (!Number.isInteger(args.numBattles) || args.numBattles <= 0) {
    throw new Error("--num-battles must be a positive integer.");
  }
  if (!Number.isInteger(args.concurrency) || args.concurrency <= 0) {
    throw new Error("--concurrency must be a positive integer.");
  }
  return args;
}

function makeRng(seed) {
  let state = (seed >>> 0) || 1;
  return {
    next() {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0x100000000;
    },
    int(maxExclusive) {
      return Math.floor(this.next() * maxExclusive);
    },
  };
}

function makeShowdownSeed(seed) {
  const base = seed >>> 0;
  return [
    base,
    (base ^ 0x9e3779b9) >>> 0,
    (base ^ 0x85ebca6b) >>> 0,
    (base ^ 0xc2b2ae35) >>> 0,
  ];
}

function loadShowdown(installDir) {
  const sim = require(path.join(installDir, "dist", "sim"));
  const { RandomPlayerAI } = require(path.join(installDir, "dist", "sim", "tools", "random-player-ai"));
  return {
    BattleStream: sim.BattleStream,
    Teams: sim.Teams,
    getPlayerStreams: sim.getPlayerStreams,
    RandomPlayerAI,
  };
}

function loadPackedTeams(manifestPath, subset, Teams) {
  const repoRoot = path.resolve(path.dirname(manifestPath), "..", "..", "..");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const entries = manifest[subset];
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error(`Manifest ${manifestPath} does not contain any ${subset} entries.`);
  }
  return entries.map((entry) => {
    const filePath = path.isAbsolute(entry.file) ? entry.file : path.join(repoRoot, entry.file);
    const teamText = fs.readFileSync(filePath, "utf8");
    return Teams.pack(Teams.import(teamText));
  });
}

async function runBattle({ BattleStream, RandomPlayerAI, getPlayerStreams }, format, team1, team2, seed) {
  const streams = getPlayerStreams(new BattleStream());
  const p1 = new RandomPlayerAI(streams.p1, { seed: makeShowdownSeed(seed) });
  const p2 = new RandomPlayerAI(streams.p2, { seed: makeShowdownSeed(seed ^ 0x517cc1b7) });
  const p1Task = p1.start();
  const p2Task = p2.start();

  let turns = 0;
  let winner = null;
  const battleTask = (async () => {
    for await (const chunk of streams.omniscient) {
      for (const line of chunk.split("\n")) {
        if (line.startsWith("|turn|")) {
          turns += 1;
        } else if (line.startsWith("|win|")) {
          winner = line.slice(5);
        }
      }
    }
  })();

  await streams.omniscient.write(
    `>start ${JSON.stringify({ formatid: format })}\n` +
      `>player p1 ${JSON.stringify({ name: "P1", team: team1 })}\n` +
      `>player p2 ${JSON.stringify({ name: "P2", team: team2 })}`
  );
  await battleTask;
  await Promise.all([p1Task, p2Task]);
  return { turns, winner };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const showdown = loadShowdown(args.installDir);
  const teams = loadPackedTeams(args.manifest, args.subset, showdown.Teams);
  const rng = makeRng(args.seed);
  const started = process.hrtime.bigint();
  let completed = 0;
  let totalTurns = 0;

  async function worker(workerIndex) {
    while (true) {
      const battleIndex = completed;
      if (battleIndex >= args.numBattles) {
        return;
      }
      completed += 1;
      const team1 = teams[rng.int(teams.length)];
      const team2 = teams[rng.int(teams.length)];
      const result = await runBattle(showdown, args.format, team1, team2, args.seed + workerIndex * 100000 + battleIndex);
      totalTurns += result.turns;
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(args.concurrency, args.numBattles) }, (_, index) => worker(index))
  );

  const wallSec = Number(process.hrtime.bigint() - started) / 1e9;
  const summary = {
    format: args.format,
    subset: args.subset,
    num_battles: args.numBattles,
    concurrency: Math.min(args.concurrency, args.numBattles),
    wall_time_sec: wallSec,
    battles_per_sec: args.numBattles / wallSec,
    avg_turns: totalTurns / args.numBattles,
    install_dir: args.installDir,
    manifest: args.manifest,
  };

  if (args.summaryPath) {
    fs.mkdirSync(path.dirname(args.summaryPath), { recursive: true });
    fs.writeFileSync(args.summaryPath, JSON.stringify(summary, null, 2));
  }
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
