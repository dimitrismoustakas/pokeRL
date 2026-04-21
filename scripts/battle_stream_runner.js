/**
 * Thin wrapper around Pokemon Showdown's BattleStream.
 *
 * Reads battle commands from stdin line by line, writes protocol output to
 * stdout with explicit flushes after each chunk so Python's readline() is
 * never blocked by libuv's pipe-output buffer.
 *
 * Output format mirrors simulate-battle: 'update' / 'sideupdate' chunks
 * separated by a single blank line.
 *
 * Usage: node scripts/battle_stream_runner.js
 * (run from the pokemon-showdown install directory)
 */

'use strict';

const path = require('path');
// Support being run from the showdown dir or from the pokeRL repo root.
const showdownDir = process.env.SHOWDOWN_DIR || process.cwd();
const { BattleStream } = require(path.join(showdownDir, 'dist', 'sim', 'battle-stream'));

const stream = new BattleStream();

// ── stdin → stream ──────────────────────────────────────────────────────────
let stdinBuf = '';
process.stdin.setEncoding('utf8');

process.stdin.on('data', chunk => {
    stdinBuf += chunk;
    let nl;
    while ((nl = stdinBuf.indexOf('\n')) !== -1) {
        const line = stdinBuf.slice(0, nl).trimEnd();
        stdinBuf = stdinBuf.slice(nl + 1);
        if (line) stream.write(line);
    }
});

process.stdin.on('end', () => stream.destroy());

// ── stream → stdout (flushed after every chunk) ──────────────────────────────
// Each chunk ends with a '.' sentinel line so the Python reader knows exactly
// when the chunk ends without blocking on the next chunk header.
// After all chunks are written we emit 'BATTLE_DONE' so Python can drain the
// pipe without relying on the process to exit promptly.
(async () => {
    for await (const chunk of stream) {
        // chunk is a multi-line string for one event group (e.g. "update\n|move|...")
        const payload = chunk + '\n.\n';
        const flushed = process.stdout.write(payload);
        if (!flushed) {
            // Back-pressure: wait for drain before continuing.
            await new Promise(resolve => process.stdout.once('drain', resolve));
        }
    }
    process.exit(0);
})();
