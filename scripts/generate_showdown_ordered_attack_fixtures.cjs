#!/usr/bin/env node
"use strict";

const path = require("node:path");
const {execFileSync} = require("node:child_process");

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(2);
}

const [showdownRoot] = process.argv.slice(2);
if (!showdownRoot) fail("usage: generate_showdown_ordered_attack_fixtures.cjs SHOWDOWN_ROOT");

const showdownCommit = execFileSync(
  "git", ["-C", showdownRoot, "rev-parse", "HEAD"], {encoding: "utf8"}
).trim();
const common = require(path.join(showdownRoot, "test", "common.js"));

const ITEMS = ["", "Choice Specs", "Life Orb"];
const BENCH_ITEMS = ["Leftovers", "Lum Berry"];
const DAMAGE_ROLLS = [0, 7, 15];
const SECONDARY_ROLLS = [0, 19, 20, 99];
const ORDER_CASES = [
  {id: "speed-fast", move: "Shadow Ball", attackerSpeed: 300, opponentSpeed: 200},
  {id: "speed-slow", move: "Shadow Ball", attackerSpeed: 100, opponentSpeed: 200},
  {id: "speed-tie", move: "Shadow Ball", attackerSpeed: 200, opponentSpeed: 200},
  {id: "priority", move: "Quick Attack", attackerSpeed: 100, opponentSpeed: 200},
];
const IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};

function attackerSet(item) {
  return {
    species: "Mew",
    ability: "Synchronize",
    item,
    level: 100,
    nature: "Serious",
    evs: {hp: 0, atk: 252, def: 0, spa: 252, spd: 0, spe: 0},
    ivs: IVS,
    moves: ["Shadow Ball", "Quick Attack"],
  };
}

function defenderSet() {
  return {
    species: "Lapras",
    ability: "Shell Armor",
    item: "",
    level: 100,
    nature: "Serious",
    evs: {hp: 252, atk: 0, def: 0, spa: 0, spd: 252, spe: 0},
    ivs: IVS,
    moves: ["Splash"],
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
    ivs: IVS,
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
    ivs: IVS,
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

function relevantLog(lines) {
  return lines.filter(line =>
    String(line).startsWith("|move|") ||
    String(line).startsWith("|-damage|") ||
    String(line).startsWith("|-unboost|") ||
    String(line).startsWith("|faint|")
  );
}

function runFixture({orderCase, item, benchItem, benchSignature, tieRoll, damageRoll, secondaryRoll}) {
  const battle = common.createBattle(
    {preview: false, seed: [27, 18, 28, 45]},
    [
      [attackerSet(item), benchSet(benchItem)],
      [defenderSet(), opponentBench()],
    ]
  );
  const source = battle.p1.active[0];
  const target = battle.p2.active[0];

  source.storedStats.spe = orderCase.attackerSpeed;
  target.storedStats.spe = orderCase.opponentSpeed;
  source.updateSpeed();
  target.updateSpeed();

  const move = battle.dex.getActiveMove(orderCase.move);
  move.willCrit = false;
  move.critRatio = 0;

  const attackStat = move.category === "Physical" ? "atk" : "spa";
  const defenseStat = move.category === "Physical" ? "def" : "spd";
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
    defender_spd_stage: target.boosts.spd,
    attacker_speed: source.getActionSpeed(),
    opponent_speed: target.getActionSpeed(),
  };

  const requests = [];
  let hundredCall = 0;
  const originalRandom = battle.prng.random;
  const originalShuffle = battle.prng.shuffle;
  battle.prng.random = function (from, to) {
    if (to !== undefined) {
      battle.prng.random = originalRandom;
      battle.prng.shuffle = originalShuffle;
      battle.destroy();
      fail(`unexpected ranged RNG request outside delegated shuffle: ${from},${to}`);
    }
    requests.push({from});
    if (from === 24) return 23;
    if (from === 16) return damageRoll;
    if (from === 100) {
      const value = hundredCall === 0 ? 0 : secondaryRoll;
      hundredCall++;
      return value;
    }
    battle.prng.random = originalRandom;
    battle.prng.shuffle = originalShuffle;
    battle.destroy();
    fail(`unexpected RNG request: ${from}`);
  };
  battle.prng.shuffle = function (items, start = 0, end = items.length) {
    const segment = items.slice(start, end);
    const isMoveOrderTie =
      segment.length === 2 &&
      segment.every(candidate => candidate && candidate.choice === "move");
    if (isMoveOrderTie) {
      requests.push({shuffle: "move-order", tieRoll});
      const p1Action = segment.find(candidate => candidate.pokemon?.side?.id === "p1");
      const p2Action = segment.find(candidate => candidate.pokemon?.side?.id === "p2");
      if (!p1Action || !p2Action) {
        battle.prng.random = originalRandom;
        battle.prng.shuffle = originalShuffle;
        battle.destroy();
        fail("move-order tie segment did not contain p1 and p2 actions");
      }
      items[start] = tieRoll === 0 ? p1Action : p2Action;
      items[start + 1] = tieRoll === 0 ? p2Action : p1Action;
      return;
    }

    const controlledRandom = this.random;
    this.random = originalRandom;
    try {
      return originalShuffle.call(this, items, start, end);
    } finally {
      this.random = controlledRandom;
    }
  };

  const logStart = battle.log.length;
  const p1Choice = `move ${move.id}`;
  battle.makeChoices(p1Choice, "move splash");
  battle.prng.random = originalRandom;
  battle.prng.shuffle = originalShuffle;

  const log = relevantLog(battle.log.slice(logStart));
  const moveLines = log.filter(line => String(line).startsWith("|move|"));
  const attackerActedFirst = moveLines.length > 0 && String(moveLines[0]).includes("|p1a:");

  const after = {
    attacker_hp: source.hp,
    defender_hp: target.hp,
    move_pp: ppFor(source, move.id),
    defender_spd_stage: target.boosts.spd,
    attacker_fainted: !!source.fainted,
    defender_fainted: !!target.fainted,
    attacker_acted_first: attackerActedFirst,
  };

  const fixture = {
    case: orderCase.id,
    move: move.name,
    move_id: move.id,
    move_priority: move.priority,
    opponent_priority: 0,
    move_accuracy: move.accuracy,
    secondary_chance: move.secondaries?.[0]?.chance || move.secondary?.chance || 0,
    item: item || "None",
    bench_item: benchItem,
    bench_signature: benchSignature,
    order_tie_roll: tieRoll,
    accuracy_roll: 0,
    damage_roll: damageRoll,
    secondary_roll: secondaryRoll,
    context,
    before,
    after,
    rng_requests: requests,
    transition_log: log,
  };
  battle.destroy();
  return fixture;
}

const fixtures = [];
for (const orderCase of ORDER_CASES) {
  const secondaryRolls = orderCase.move === "Shadow Ball" ? SECONDARY_ROLLS : [99];
  for (const item of ITEMS) {
    for (let benchSignature = 0; benchSignature < BENCH_ITEMS.length; benchSignature++) {
      const benchItem = BENCH_ITEMS[benchSignature];
      for (const tieRoll of [0, 1]) {
        for (const damageRoll of DAMAGE_ROLLS) {
          for (const secondaryRoll of secondaryRolls) {
            fixtures.push(runFixture({
              orderCase, item, benchItem, benchSignature, tieRoll, damageRoll, secondaryRoll,
            }));
          }
        }
      }
    }
  }
}

process.stdout.write(JSON.stringify({
  schema: "azelficoast.showdown-ordered-attack-fixtures",
  schema_version: 1,
  showdown_commit: showdownCommit,
  source: "pokemon-showdown Battle.makeChoices",
  format: "gen9 test Anything Goes without Team Preview",
  fixture_count: fixtures.length,
  fixtures,
}, null, 2) + "\n");
