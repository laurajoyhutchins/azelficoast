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
  fail("usage: generate_showdown_stateful_protect_fixtures.cjs SHOWDOWN_ROOT");
}

const showdownCommit = execFileSync(
  "git", ["-C", showdownRoot, "rev-parse", "HEAD"], {encoding: "utf8"}
).trim();
const common = require(path.join(showdownRoot, "test", "common.js"));

const COUNTERS = [1, 3, 9, 27, 81, 243, 729];
const ITEMS = ["", "Choice Specs"];
const DAMAGE_ROLLS = [0, 15];
const CANONICAL_ROLLS = {
  1: [0],
  3: [0, 728],
  9: [0, 728],
  27: [0, 728],
  81: [0, 728],
  243: [0, 728],
  729: [0, 728],
};
const IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};

function p1Set() {
  return {
    species: "Mew",
    ability: "Synchronize",
    item: "",
    level: 100,
    nature: "Serious",
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0},
    ivs: IVS,
    moves: ["Protect"],
  };
}

function p2Set(item) {
  return {
    species: "Mew",
    ability: "Synchronize",
    item,
    level: 100,
    nature: "Serious",
    evs: {hp: 0, atk: 0, def: 0, spa: 252, spd: 0, spe: 0},
    ivs: IVS,
    moves: ["Aura Sphere"],
  };
}

function naturePercent(battle, pokemon, stat) {
  const nature = battle.dex.natures.get(pokemon.set.nature);
  if (nature.plus === stat) return 110;
  if (nature.minus === stat) return 90;
  return 100;
}

function moveSlot(pokemon, moveId) {
  const slot = pokemon.moveSlots.find(candidate => candidate.id === moveId);
  if (!slot) fail("missing move slot " + moveId);
  return slot;
}

function damageContext(battle, source, target, move) {
  const attackStat = move.category === "Physical" ? "atk" : "spa";
  const defenseStat = move.category === "Physical" ? "def" : "spd";
  return {
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
    type_mod: target.runEffectiveness(move),
    burned: false,
    defender_stat_modifier: 4096,
  };
}

function relevantLog(lines) {
  return lines.filter(line => {
    const value = String(line);
    return value.startsWith("|move|") ||
      value.startsWith("|-fail|") ||
      value.startsWith("|-activate|") ||
      value.startsWith("|-damage|");
  });
}

function canonicalToShowdown(counter, canonicalRoll) {
  return Math.floor(canonicalRoll * counter / 729);
}

function runFixture({counter, canonicalRoll, item, damageRoll}) {
  const battle = common.createBattle(
    {preview: false, seed: [43, 47, 53, 59]},
    [[p1Set()], [p2Set(item)]]
  );
  const p1 = battle.p1.active[0];
  const p2 = battle.p2.active[0];
  const protect = battle.dex.getActiveMove("Protect");
  const auraSphere = battle.dex.getActiveMove("Aura Sphere");

  moveSlot(p1, protect.id).pp = 5;
  moveSlot(p2, auraSphere.id).pp = 5;

  if (counter > 1) {
    if (!p1.addVolatile("stall")) fail("could not seed stall volatile");
    p1.volatiles.stall.counter = counter;
    p1.volatiles.stall.duration = 1;
  }

  const before = {
    p1_hp: p1.hp,
    p2_hp: p2.hp,
    p1_pp: moveSlot(p1, protect.id).pp,
    p2_pp: moveSlot(p2, auraSphere.id).pp,
    p2_spa_stage: p2.boosts.spa,
    stall_counter: counter,
  };

  const rawProtectRoll = canonicalToShowdown(counter, canonicalRoll);
  const expectedSuccess = counter === 1 || rawProtectRoll === 0;
  const requests = [];
  const originalRandom = battle.prng.random;
  const originalRandomChance = battle.randomChance;

  battle.randomChance = function (numerator, denominator) {
    if (counter > 1 && numerator === 1 && denominator === counter) {
      requests.push({kind: "stall", numerator, denominator, value: rawProtectRoll});
      return rawProtectRoll === 0;
    }
    return originalRandomChance.call(battle, numerator, denominator);
  };

  battle.prng.random = function (from, to) {
    if (to !== undefined) {
      return originalRandom.call(battle.prng, from, to);
    }
    if (from === 24) {
      requests.push({kind: "crit", from});
      return 23;
    }
    if (from === 16) {
      if (expectedSuccess) {
        fail("successful Protect unexpectedly requested damage RNG");
      }
      requests.push({kind: "damage", from, value: damageRoll});
      return damageRoll;
    }
    if (from === 100) {
      requests.push({kind: "accuracy", from});
      return 0;
    }
    return originalRandom.call(battle.prng, from);
  };

  const logStart = battle.log.length;
  battle.makeChoices("move protect", "move aurasphere");
  battle.prng.random = originalRandom;
  battle.randomChance = originalRandomChance;

  const log = relevantLog(battle.log.slice(logStart));
  const moveLines = log.filter(line => String(line).startsWith("|move|"));
  const p1Acted = moveLines.some(line => String(line).includes("|p1a:"));
  const p2Acted = moveLines.some(line => String(line).includes("|p2a:"));
  const protectFailed = log.some(line => {
    const value = String(line);
    return value.startsWith("|-fail|") && value.includes("p1a:");
  });

  const afterCounter = p1.volatiles.stall?.counter || 1;
  const after = {
    p1_hp: p1.hp,
    p2_hp: p2.hp,
    p1_pp: moveSlot(p1, protect.id).pp,
    p2_pp: moveSlot(p2, auraSphere.id).pp,
    p2_spa_stage: p2.boosts.spa,
    stall_counter: afterCounter,
    p1_acted: p1Acted,
    p2_acted: p2Acted,
    protect_succeeded: !protectFailed,
  };

  const fixture = {
    stall_counter: counter,
    canonical_protect_roll: canonicalRoll,
    showdown_protect_roll: rawProtectRoll,
    expected_success: expectedSuccess,
    p2_item: item || "None",
    p2_damage_roll: damageRoll,
    p1_priority: protect.priority,
    p2_priority: auraSphere.priority,
    p1_speed: p1.getActionSpeed(),
    p2_speed: p2.getActionSpeed(),
    p1_context: damageContext(battle, p1, p2, protect),
    p2_context: damageContext(battle, p2, p1, auraSphere),
    before,
    after,
    rng_requests: requests,
    transition_log: log,
  };

  battle.destroy();
  return fixture;
}

const fixtures = [];
for (const counter of COUNTERS) {
  for (const canonicalRoll of CANONICAL_ROLLS[counter]) {
    for (const item of ITEMS) {
      for (const damageRoll of DAMAGE_ROLLS) {
        fixtures.push(runFixture({counter, canonicalRoll, item, damageRoll}));
      }
    }
  }
}

process.stdout.write(JSON.stringify({
  schema: "azelficoast.showdown-stateful-protect-fixtures",
  schema_version: 1,
  showdown_commit: showdownCommit,
  source: "pokemon-showdown Battle.makeChoices with seeded stall volatile",
  canonical_roll_denominator: 729,
  fixture_count: fixtures.length,
  fixtures,
}, null, 2) + "\n");
