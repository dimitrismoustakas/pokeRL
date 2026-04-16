const fs = require('node:fs');
const path = require('node:path');

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
  input += chunk;
});
process.stdin.on('end', main);
process.stdin.resume();

function problemsToArray(result) {
  if (!result) {
    return [];
  }
  if (Array.isArray(result)) {
    return result.map(problem => String(problem));
  }
  return [String(result)];
}

function inferMechanics(gen) {
  return {
    tera: gen >= 9,
    mega: gen >= 6 && gen <= 7,
    zmove: gen === 7,
    dynamax: gen === 8,
  };
}

function main() {
  try {
    const payload = JSON.parse(input || '{}');
    const showdownDir = payload.showdownDir;
    if (!showdownDir) {
      throw new Error('showdownDir is required');
    }
    if (!fs.existsSync(showdownDir)) {
      throw new Error(`Showdown directory does not exist: ${showdownDir}`);
    }

    const dexPath = path.resolve(showdownDir, 'dist', 'sim', 'dex.js');
    const validatorPath = path.resolve(showdownDir, 'dist', 'sim', 'team-validator.js');
    const teamsPath = path.resolve(showdownDir, 'dist', 'sim', 'teams.js');

    const {Dex} = require(dexPath);
    const {TeamValidator} = require(validatorPath);
    const {Teams} = require(teamsPath);

    if (payload.command === 'profile') {
      const format = Dex.formats.get(payload.formatId);
      const ruleTable = Dex.formats.getRuleTable(format);
      const modDex = Dex.forFormat(payload.formatId);

      const maxTeamSize = ruleTable.maxTeamSize || ruleTable.minTeamSize || 6;
      const pickSize = ruleTable.pickedTeamSize || maxTeamSize;
      const level = ruleTable.adjustLevel ?? ruleTable.adjustLevelDown ?? null;
      let maxRestricted = null;
      if (ruleTable.has('limitonerestricted')) {
        maxRestricted = 1;
      } else if (ruleTable.has('limittworestricted')) {
        maxRestricted = 2;
      }

      process.stdout.write(JSON.stringify({
        id: format.id,
        name: format.name,
        gen: modDex.gen,
        mod: format.mod,
        team: format.team || null,
        teamSize: maxTeamSize,
        pickSize,
        minTeamSize: ruleTable.minTeamSize || maxTeamSize,
        level,
        hasTeamPreview: ruleTable.has('teampreview'),
        itemClause: ruleTable.has('itemclause'),
        speciesClause: ruleTable.has('speciesclause'),
        maxRestricted,
        mechanics: inferMechanics(modDex.gen),
      }));
      return;
    }

    const validator = TeamValidator.get(payload.formatId);
    if (payload.command === 'validateSet') {
      const team = Teams.import(payload.teamText);
      if (!team || !team.length) {
        throw new Error('validateSet requires a non-empty teamText');
      }
      const problems = problemsToArray(validator.validateSet(team[0], {}));
      process.stdout.write(JSON.stringify({problems}));
      return;
    }

    if (payload.command === 'validateTeam') {
      const team = Teams.import(payload.teamText);
      if (!team || !team.length) {
        throw new Error('validateTeam requires a non-empty teamText');
      }
      const problems = problemsToArray(validator.validateTeam(team));
      process.stdout.write(JSON.stringify({problems}));
      return;
    }

    throw new Error(`Unknown command: ${payload.command}`);
  } catch (error) {
    const message = error && error.stack ? error.stack : String(error);
    process.stderr.write(message);
    process.exit(1);
  }
}