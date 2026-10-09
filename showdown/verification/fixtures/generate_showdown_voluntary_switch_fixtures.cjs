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
  fail("usage: generate_showdown_voluntary_switch_fixtures.cjs SHOWDOWN_ROOT");
}

const showdownCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();
const common = require(path.join(showdownRoot, "test", "common.js"));

const ITEMS = ["", "Choice Specs"];
const INCOMING_SLOTS = [1, 2];
const ACCURACY_ROLLS = [0, 79, 80, 99];
const DAMAGE_ROLLS = [0, 7, 15];
const IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};

function outgoingSet() {
  return {
    species: "Mew",
    name: "Outgoing",
    ability: "Synchronize",
    item: "",
    level: 100,
    nature: "Serious",
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0},
    ivs: IVS,
    moves: ["Splash"],
  };
}

function laprasSet() {
  return {
    species: "Lapras",
    name: "LaprasTarget",
    ability: "Shell Armor",
    item: "",
    level: 100,
    nature: "Calm",
    evs: {hp: 252, atk: 0, def: 0, spa: 0, spd: 252, spe: 0},
    ivs: IVS,
    moves: ["Splash"],
  };
}

function blisseySet() {
  return {
    species: "Blissey",
    name: "BlisseyTarget",
    ability: "Natural Cure",
    item: "",
    level: 100,
    nature: "Calm",
    evs: {hp: 252, atk: 0, def: 0, spa: 0, spd: 252, spe: 0},
    ivs: IVS,
    moves: ["Splash"],
  };
}

function opponentSet(item) {
  return {
    species: "Mew",
    name: "Opponent",
    ability: "Synchronize",
    item,
    level: 100,
    nature: "Modest",
    evs: {hp: 0, atk: 0, def: 0, spa: 252, spd: 0, spe: 252},
    ivs: IVS,
    moves: ["Hydro Pump"],
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
  if (!slot) fail("missing move slot " + moveId);
  return slot.pp;
}

function transitionLog(lines) {
  return lines.filter(line => {
    const value = String(line);
    return value.startsWith("|switch|") ||
      value.startsWith("|move|") ||
      value.startsWith("|-miss|") ||
      value.startsWith("|-damage|") ||
      value.startsWith("|faint|");
  });
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

function runFixture({item, incomingSlot, accuracyRoll, damageRoll}) {
  const battle = common.createBattle(
    {preview: false, seed: [61, 67, 71, 73]},
    [
      [outgoingSet(), laprasSet(), blisseySet()],
      [opponentSet(item)],
    ]
  );

  const outgoing = battle.p1.active[0];
  const incoming = battle.p1.pokemon[incomingSlot];
  const opponent = battle.p2.active[0];
  const move = battle.dex.getActiveMove("Hydro Pump");
  if (typeof move.accuracy !== "number") {
    battle.destroy();
    fail("Hydro Pump must have numeric accuracy");
  }

  outgoing.hp = Math.min(200, outgoing.baseMaxhp);
  move.willCrit = false;
  move.critRatio = 0;

  const before = {
    outgoing_slot: 0,
    outgoing_hp: outgoing.hp,
    outgoing_max_hp: outgoing.baseMaxhp,
    incoming_slot: incomingSlot,
    incoming_species: incoming.species.name,
    incoming_hp: incoming.hp,
    incoming_max_hp: incoming.baseMaxhp,
    opponent_hp: opponent.hp,
    opponent_max_hp: opponent.baseMaxhp,
    opponent_move_pp: ppFor(opponent, move.id),
  };
  const context = damageContext(battle, opponent, incoming, move);

  const requests = [];
  const originalRandom = battle.prng.random;
  battle.prng.random = function (from, to) {
    if (to !== undefined) {
      return originalRandom.call(battle.prng, from, to);
    }
    if (from === 100) {
      requests.push({kind: "accuracy", from, value: accuracyRoll});
      return accuracyRoll;
    }
    if (from === 24) {
      requests.push({kind: "crit", from, value: 23});
      return 23;
    }
    if (from === 16) {
      requests.push({kind: "damage", from, value: damageRoll});
      return damageRoll;
    }
    return originalRandom.call(battle.prng, from);
  };

  const logStart = battle.log.length;
  battle.makeChoices(
    "switch " + String(incomingSlot + 1),
    "move hydropump"
  );
  battle.prng.random = originalRandom;

  if (battle.p1.active[0] !== incoming) {
    battle.destroy();
    fail("selected teammate did not become active");
  }

  const after = {
    active_slot: incomingSlot,
    active_species: incoming.species.name,
    active_hp: incoming.hp,
    active_max_hp: incoming.baseMaxhp,
    bench_slot: 0,
    bench_hp: outgoing.hp,
    bench_max_hp: outgoing.baseMaxhp,
    opponent_hp: opponent.hp,
    opponent_move_pp: ppFor(opponent, move.id),
    hit: incoming.hp < before.incoming_hp,
    active_fainted: !!incoming.fainted,
    opponent_fainted: !!opponent.fainted,
  };

  const fixture = {
    opponent_item: item || "None",
    incoming_slot: incomingSlot,
    incoming_species: incoming.species.name,
    accuracy_roll: accuracyRoll,
    damage_roll: damageRoll,
    move_accuracy: move.accuracy,
    context,
    before,
    after,
    rng_requests: requests,
    transition_log: transitionLog(battle.log.slice(logStart)),
  };

  battle.destroy();
  return fixture;
}

const fixtures = [];
for (const item of ITEMS) {
  for (const incomingSlot of INCOMING_SLOTS) {
    for (const accuracyRoll of ACCURACY_ROLLS) {
      for (const damageRoll of DAMAGE_ROLLS) {
        fixtures.push(
          runFixture({item, incomingSlot, accuracyRoll, damageRoll})
        );
      }
    }
  }
}

process.stdout.write(JSON.stringify({
  schema: "azelficoast.showdown-voluntary-switch-fixtures",
  schema_version: 1,
  showdown_commit: showdownCommit,
  source: "pokemon-showdown Battle.makeChoices",
  format: "gen9 test Anything Goes without Team Preview",
  move: "Hydro Pump",
  opponent_items: ITEMS.map(item => item || "None"),
  incoming_slots: INCOMING_SLOTS,
  accuracy_rolls: ACCURACY_ROLLS,
  damage_rolls: DAMAGE_ROLLS,
  fixture_count: fixtures.length,
  fixtures,
}, null, 2) + "\n");
