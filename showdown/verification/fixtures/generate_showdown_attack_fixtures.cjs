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
  fail("usage: generate_showdown_attack_fixtures.cjs SHOWDOWN_ROOT");
}

const showdownCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();

const common = require(path.join(showdownRoot, "test", "common.js"));

const ITEMS = ["", "Choice Specs", "Life Orb"];
const BENCH_ITEMS = ["Leftovers", "Lum Berry"];
const ACCURACY_ROLLS = [0, 79, 80, 99];
const DAMAGE_ROLLS = [0, 7, 15];
const STATE_VARIANTS = [
  {id: "normal", attackerHp: null, defenderHp: null},
  {id: "defender-low", attackerHp: null, defenderHp: 25},
  {id: "attacker-low", attackerHp: 20, defenderHp: null},
];

const DEFAULT_IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};
const INTERESTING_LOG_PREFIXES = [
  "|move|",
  "|-miss|",
  "|-damage|",
  "|faint|",
  "|turn|",
];

function transitionLog(lines) {
  return lines.filter(line =>
    INTERESTING_LOG_PREFIXES.some(prefix => String(line).startsWith(prefix))
  );
}

function attackerSet(item) {
  return {
    species: "Mew",
    ability: "Synchronize",
    item,
    level: 100,
    nature: "Modest",
    evs: {hp: 0, atk: 0, def: 0, spa: 252, spd: 0, spe: 252},
    ivs: DEFAULT_IVS,
    moves: ["Hydro Pump", "Protect"],
  };
}

function defenderSet() {
  return {
    species: "Lapras",
    ability: "Shell Armor",
    item: "",
    level: 100,
    nature: "Calm",
    evs: {hp: 252, atk: 0, def: 0, spa: 0, spd: 252, spe: 0},
    ivs: DEFAULT_IVS,
    moves: ["Splash", "Protect"],
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
    ivs: DEFAULT_IVS,
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
    ivs: DEFAULT_IVS,
    moves: ["Splash"],
  };
}

function naturePercent(battle, pokemon, stat) {
  const nature = battle.dex.natures.get(pokemon.set.nature);
  if (nature.plus === stat) return 110;
  if (nature.minus === stat) return 90;
  return 100;
}

function ppFor(pokemon, moveId) {
  const slot = pokemon.moveSlots.find(candidate => candidate.id === moveId);
  if (!slot) fail(`move slot missing: ${moveId}`);
  return slot.pp;
}

function runFixture({
  item,
  benchItem,
  benchSignature,
  stateVariant,
  accuracyRoll,
  damageRoll,
}) {
  const battle = common.createBattle(
    {preview: false, seed: [31, 41, 59, 26]},
    [
      [attackerSet(item), benchSet(benchItem)],
      [defenderSet(), opponentBench()],
    ]
  );

  const source = battle.p1.active[0];
  const target = battle.p2.active[0];
  const move = battle.dex.getActiveMove("Hydro Pump");
  if (typeof move.accuracy !== "number") {
    battle.destroy();
    fail("Hydro Pump must have numeric accuracy");
  }

  if (stateVariant.attackerHp !== null) {
    source.hp = stateVariant.attackerHp;
  }
  if (stateVariant.defenderHp !== null) {
    target.hp = stateVariant.defenderHp;
  }

  const attackStat = "spa";
  const defenseStat = "spd";
  move.willCrit = false;
  move.critRatio = 0;
  const typeMod = target.runEffectiveness(move);

  const context = {
    attacker_level: source.level,
    defender_level: target.level,
    base_power: move.basePower,
    category: move.category,
    move_id: move.id,
    move_type: move.type,
    attacker_types: source.getTypes(false, true),
    tera_type: null,
    attacker_base_stat: source.species.baseStats[attackStat],
    attacker_iv: source.set.ivs[attackStat],
    attacker_ev: source.set.evs[attackStat],
    attacker_nature_percent: naturePercent(battle, source, attackStat),
    defender_base_stat: target.species.baseStats[defenseStat],
    defender_iv: target.set.ivs[defenseStat],
    defender_ev: target.set.evs[defenseStat],
    defender_nature_percent: naturePercent(battle, target, defenseStat),
    attacker_item: source.getItem().name || "",
    type_mod: typeMod,
    burned: false,
    defender_stat_modifier: 4096,
  };

  const before = {
    attacker_hp: source.hp,
    attacker_max_hp: source.baseMaxhp,
    defender_hp: target.hp,
    move_pp: ppFor(source, move.id),
  };

  const requests = [];
  const originalPrngRandom = battle.prng.random;
  battle.prng.random = function (m, n) {
    if (n !== undefined) {
      battle.prng.random = originalPrngRandom;
      battle.destroy();
      fail(`unexpected ranged RNG request: m=${m} n=${n}`);
    }
    requests.push(m);
    if (m === 100) return accuracyRoll;
    if (m === 16) return damageRoll;
    if (m === 24) return 23; // Force the non-critical branch if queried.
    battle.prng.random = originalPrngRandom;
    battle.destroy();
    fail(`unexpected RNG request: m=${m}`);
  };

  const logStart = battle.log.length;
  battle.makeChoices("move hydropump", "move splash");
  battle.prng.random = originalPrngRandom;

  const after = {
    attacker_hp: source.hp,
    defender_hp: target.hp,
    move_pp: ppFor(source, move.id),
    attacker_fainted: !!source.fainted,
    defender_fainted: !!target.fainted,
  };
  const hit = after.defender_hp < before.defender_hp;
  const log = transitionLog(battle.log.slice(logStart));

  const fixture = {
    item: item || "None",
    bench_signature: benchSignature,
    bench_item: benchItem,
    state_variant: stateVariant.id,
    accuracy_roll: accuracyRoll,
    damage_roll: damageRoll,
    move_accuracy: move.accuracy,
    context,
    before,
    after,
    hit,
    rng_requests: requests,
    transition_log: log,
  };

  battle.destroy();
  return fixture;
}

const fixtures = [];
for (const item of ITEMS) {
  for (let benchSignature = 0; benchSignature < BENCH_ITEMS.length; benchSignature++) {
    const benchItem = BENCH_ITEMS[benchSignature];
    for (const stateVariant of STATE_VARIANTS) {
      for (const accuracyRoll of ACCURACY_ROLLS) {
        for (const damageRoll of DAMAGE_ROLLS) {
          fixtures.push(
            runFixture({
              item,
              benchItem,
              benchSignature,
              stateVariant,
              accuracyRoll,
              damageRoll,
            })
          );
        }
      }
    }
  }
}

process.stdout.write(
  JSON.stringify(
    {
      schema: "azelficoast.showdown-attack-transition-fixtures",
      schema_version: 1,
      showdown_commit: showdownCommit,
      source: "pokemon-showdown Battle.makeChoices",
      format: "gen9 test Anything Goes without Team Preview",
      move: "Hydro Pump",
      item_count: ITEMS.length,
      bench_variant_count: BENCH_ITEMS.length,
      state_variant_count: STATE_VARIANTS.length,
      accuracy_rolls: ACCURACY_ROLLS,
      damage_rolls: DAMAGE_ROLLS,
      fixture_count: fixtures.length,
      fixtures,
    },
    null,
    2
  ) + "\n"
);
