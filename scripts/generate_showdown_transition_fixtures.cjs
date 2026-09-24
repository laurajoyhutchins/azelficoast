#!/usr/bin/env node
"use strict";

const path = require("node:path");
const {execFileSync} = require("node:child_process");

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(2);
}

const [showdownRoot] = process.argv.slice(2);
if (!showdownRoot) {
  fail("usage: generate_showdown_transition_fixtures.cjs SHOWDOWN_ROOT");
}

const showdownCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();

const common = require(path.join(showdownRoot, "test", "common.js"));

const ITEMS = ["Choice Scarf", "Choice Specs"];
const BENCH_ITEMS = [
  "Leftovers",
  "Choice Band",
  "Choice Specs",
  "Choice Scarf",
  "Life Orb",
  "Lum Berry",
  "Sitrus Berry",
  "Heavy-Duty Boots",
];

const SEEDS = Array.from({length: 8}, (_, index) => [
  1000 + index,
  2000 + index,
  3000 + index,
  4000 + index,
]);

const INTERESTING_LOG_PREFIXES = [
  "|move|",
  "|-damage|",
  "|-activate|",
  "|-singleturn|",
  "|-fail|",
  "|-supereffective|",
  "|-resisted|",
  "|-immune|",
  "|upkeep|",
  "|turn|",
];

function transitionLog(lines) {
  return lines.filter(line =>
    INTERESTING_LOG_PREFIXES.some(prefix => String(line).startsWith(prefix))
  );
}

function laprasSet(moves, item = "") {
  return {
    species: "Lapras",
    ability: "Shell Armor",
    item,
    level: 100,
    nature: "Calm",
    evs: {hp: 252, atk: 0, def: 0, spa: 0, spd: 252, spe: 0},
    ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
    moves,
  };
}

function mewSet(item) {
  return {
    species: "Mew",
    ability: "Synchronize",
    item,
    level: 100,
    nature: "Modest",
    evs: {hp: 0, atk: 0, def: 0, spa: 252, spd: 0, spe: 252},
    ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
    moves: ["Aura Sphere"],
  };
}

function benchSet(item) {
  return {
    species: "Blissey",
    ability: "Natural Cure",
    item,
    level: 100,
    nature: "Serious",
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0},
    ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
    moves: ["Splash"],
  };
}

function opponentBench() {
  return {
    species: "Chansey",
    ability: "Natural Cure",
    item: "Eviolite",
    level: 100,
    nature: "Serious",
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0},
    ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
    moves: ["Splash"],
  };
}

function runFixture({scenario, item, benchItem, benchSignature, seed, seedIndex}) {
  const ownMoves = scenario === "protect"
    ? ["Protect", "Splash"]
    : ["Splash", "Protect"];

  const battle = common.createBattle(
    {seed, preview: false},
    [
      [laprasSet(ownMoves), benchSet(benchItem)],
      [mewSet(item), opponentBench()],
    ]
  );

  const beforeHp = battle.p1.active[0].hp;
  const logStart = battle.log.length;

  if (scenario === "protect") {
    battle.makeChoices("move protect", "move aurasphere");
  } else if (scenario === "damage") {
    battle.makeChoices("move splash", "move aurasphere");
  } else {
    battle.destroy();
    fail(`unknown scenario: ${scenario}`);
  }

  const afterHp = battle.p1.active[0].hp;
  const log = transitionLog(battle.log.slice(logStart));

  const fixture = {
    scenario,
    seed_index: seedIndex,
    seed,
    opponent_item: item,
    opponent_move: "Aura Sphere",
    own_species: "Lapras",
    opponent_species: "Mew",
    bench_signature: benchSignature,
    bench_item: benchItem,
    before_hp: beforeHp,
    after_hp: afterHp,
    hp_delta: afterHp - beforeHp,
    transition_log: log,
  };

  battle.destroy();
  return fixture;
}

const fixtures = [];
for (const scenario of ["protect", "damage"]) {
  for (let seedIndex = 0; seedIndex < SEEDS.length; seedIndex++) {
    const seed = SEEDS[seedIndex];
    for (const item of ITEMS) {
      for (let benchSignature = 0; benchSignature < BENCH_ITEMS.length; benchSignature++) {
        fixtures.push(
          runFixture({
            scenario,
            item,
            benchItem: BENCH_ITEMS[benchSignature],
            benchSignature,
            seed,
            seedIndex,
          })
        );
      }
    }
  }
}

process.stdout.write(
  JSON.stringify(
    {
      schema: "azelficoast.showdown-transition-fixtures",
      schema_version: 1,
      showdown_commit: showdownCommit,
      source: "pokemon-showdown Battle",
      format: "gen9 test Anything Goes without Team Preview",
      seed_count: SEEDS.length,
      item_count: ITEMS.length,
      bench_variant_count: BENCH_ITEMS.length,
      fixture_count: fixtures.length,
      fixtures,
    },
    null,
    2
  ) + "\n"
);
