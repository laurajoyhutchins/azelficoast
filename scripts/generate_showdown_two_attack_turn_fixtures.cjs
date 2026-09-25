#!/usr/bin/env node
"use strict";

const path = require("node:path");
const {execFileSync} = require("node:child_process");

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(2);
}

const [showdownRoot] = process.argv.slice(2);
if (!showdownRoot) fail("usage: generate_showdown_two_attack_turn_fixtures.cjs SHOWDOWN_ROOT");

const showdownCommit = execFileSync(
  "git", ["-C", showdownRoot, "rev-parse", "HEAD"], {encoding: "utf8"}
).trim();
const common = require(path.join(showdownRoot, "test", "common.js"));

const P2_ITEMS = ["", "Choice Specs"];
const BENCH_ITEMS = ["Leftovers", "Lum Berry"];
const DAMAGE_ROLLS = [0, 15];
const MOONBLAST_SECONDARY_ROLLS = [0, 29, 30, 99];
const HP_CASES = [
  {id: "full", p1Hp: null, p2Hp: null},
  {id: "p2-low", p1Hp: null, p2Hp: 1},
  {id: "p1-low", p1Hp: 1, p2Hp: null},
];
const ORDER_CASES = [
  {id: "p1-fast", p1Move: "Moonblast", p1Speed: 300, p2Speed: 200},
  {id: "p2-fast", p1Move: "Moonblast", p1Speed: 100, p2Speed: 200},
  {id: "speed-tie", p1Move: "Moonblast", p1Speed: 200, p2Speed: 200},
  {id: "p1-priority", p1Move: "Quick Attack", p1Speed: 100, p2Speed: 300},
];
const IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};

function p1Set() {
  return {
    species: "Mew",
    ability: "Synchronize",
    item: "",
    level: 100,
    nature: "Serious",
    evs: {hp: 0, atk: 252, def: 0, spa: 252, spd: 0, spe: 0},
    ivs: IVS,
    moves: ["Moonblast", "Quick Attack"],
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

function benchSet(item, species) {
  return {
    species,
    ability: "Natural Cure",
    item,
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

function moveSlot(pokemon, moveId) {
  const slot = pokemon.moveSlots.find(candidate => candidate.id === moveId);
  if (!slot) fail(`move slot missing: ${moveId}`);
  return slot;
}

function setPP(pokemon, moveId, pp) {
  moveSlot(pokemon, moveId).pp = pp;
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
  return lines.filter(line =>
    String(line).startsWith("|move|") ||
    String(line).startsWith("|-damage|") ||
    String(line).startsWith("|-unboost|") ||
    String(line).startsWith("|faint|")
  );
}

function p1ActsFirst(orderCase, p1Priority, p2Priority, tieRoll) {
  if (p1Priority > p2Priority) return true;
  if (p1Priority < p2Priority) return false;
  if (orderCase.p1Speed > orderCase.p2Speed) return true;
  if (orderCase.p1Speed < orderCase.p2Speed) return false;
  return tieRoll === 0;
}

function runFixture({
  orderCase,
  hpCase,
  p2Item,
  benchItem,
  benchSignature,
  tieRoll,
  p1DamageRoll,
  p2DamageRoll,
  secondaryRoll,
}) {
  const battle = common.createBattle(
    {preview: false, seed: [29, 31, 37, 41]},
    [
      [p1Set(), benchSet(benchItem, "Blissey")],
      [p2Set(p2Item), benchSet("Eviolite", "Chansey")],
    ]
  );
  const p1 = battle.p1.active[0];
  const p2 = battle.p2.active[0];

  p1.storedStats.spe = orderCase.p1Speed;
  p2.storedStats.spe = orderCase.p2Speed;
  p1.updateSpeed();
  p2.updateSpeed();

  if (hpCase.p1Hp !== null) p1.hp = hpCase.p1Hp;
  if (hpCase.p2Hp !== null) p2.hp = hpCase.p2Hp;

  const p1Move = battle.dex.getActiveMove(orderCase.p1Move);
  const p2Move = battle.dex.getActiveMove("Aura Sphere");
  setPP(p1, p1Move.id, 5);
  setPP(p2, p2Move.id, 5);

  const p1Context = damageContext(battle, p1, p2, p1Move);
  const p2Context = damageContext(battle, p2, p1, p2Move);
  const p1Priority = p1Move.priority;
  const p2Priority = p2Move.priority;
  const expectedP1First = p1ActsFirst(orderCase, p1Priority, p2Priority, tieRoll);

  const before = {
    p1_hp: p1.hp,
    p2_hp: p2.hp,
    p1_max_hp: p1.baseMaxhp,
    p2_max_hp: p2.baseMaxhp,
    p1_pp: moveSlot(p1, p1Move.id).pp,
    p2_pp: moveSlot(p2, p2Move.id).pp,
    p2_spa_stage: p2.boosts.spa,
    p1_speed: p1.getActionSpeed(),
    p2_speed: p2.getActionSpeed(),
  };

  const requests = [];
  const damageQueue = expectedP1First
    ? [p1DamageRoll, p2DamageRoll]
    : [p2DamageRoll, p1DamageRoll];
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
    if (from === 16) {
      if (!damageQueue.length) {
        battle.prng.random = originalRandom;
        battle.prng.shuffle = originalShuffle;
        battle.destroy();
        fail("more damage RNG calls than admitted actions");
      }
      return damageQueue.shift();
    }
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
    const isMoveOrderPair =
      segment.length === 2 &&
      segment.every(candidate => candidate && candidate.choice === "move");
    if (isMoveOrderPair) {
      requests.push({shuffle: "move-order", tieRoll});
      const p1Action = segment.find(candidate => candidate.pokemon?.side?.id === "p1");
      const p2Action = segment.find(candidate => candidate.pokemon?.side?.id === "p2");
      if (!p1Action || !p2Action) {
        battle.prng.random = originalRandom;
        battle.prng.shuffle = originalShuffle;
        battle.destroy();
        fail("move-order segment did not contain p1 and p2 actions");
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
  battle.makeChoices(`move ${p1Move.id}`, `move ${p2Move.id}`);
  battle.prng.random = originalRandom;
  battle.prng.shuffle = originalShuffle;

  const log = relevantLog(battle.log.slice(logStart));
  const moveLines = log.filter(line => String(line).startsWith("|move|"));
  const actorSide = line => {
    const actor = String(line).split("|")[2] || "";
    if (actor.startsWith("p1a:")) return "p1";
    if (actor.startsWith("p2a:")) return "p2";
    return null;
  };
  const p1Acted = moveLines.some(line => actorSide(line) === "p1");
  const p2Acted = moveLines.some(line => actorSide(line) === "p2");
  const firstActor = moveLines.length ? actorSide(moveLines[0]) : null;

  const after = {
    p1_hp: p1.hp,
    p2_hp: p2.hp,
    p1_pp: moveSlot(p1, p1Move.id).pp,
    p2_pp: moveSlot(p2, p2Move.id).pp,
    p2_spa_stage: p2.boosts.spa,
    p1_acted: p1Acted,
    p2_acted: p2Acted,
    first_actor: firstActor,
  };

  const fixture = {
    order_case: orderCase.id,
    hp_case: hpCase.id,
    p1_move: p1Move.name,
    p1_move_id: p1Move.id,
    p2_move: p2Move.name,
    p2_move_id: p2Move.id,
    p1_priority: p1Priority,
    p2_priority: p2Priority,
    p1_accuracy: p1Move.accuracy === true ? 100 : p1Move.accuracy,
    p2_accuracy: p2Move.accuracy === true ? 100 : p2Move.accuracy,
    p1_secondary_chance:
      p1Move.secondaries?.[0]?.chance || p1Move.secondary?.chance || 0,
    p2_item: p2Item || "None",
    bench_item: benchItem,
    bench_signature: benchSignature,
    order_tie_roll: tieRoll,
    p1_accuracy_roll: 0,
    p1_damage_roll: p1DamageRoll,
    p1_secondary_roll: secondaryRoll,
    p2_accuracy_roll: 0,
    p2_damage_roll: p2DamageRoll,
    p1_context: p1Context,
    p2_context: p2Context,
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
  const secondaryRolls =
    orderCase.p1Move === "Moonblast" ? MOONBLAST_SECONDARY_ROLLS : [99];
  for (const hpCase of HP_CASES) {
    for (const p2Item of P2_ITEMS) {
      for (let benchSignature = 0; benchSignature < BENCH_ITEMS.length; benchSignature++) {
        const benchItem = BENCH_ITEMS[benchSignature];
        for (const tieRoll of [0, 1]) {
          for (const p1DamageRoll of DAMAGE_ROLLS) {
            for (const p2DamageRoll of DAMAGE_ROLLS) {
              for (const secondaryRoll of secondaryRolls) {
                fixtures.push(runFixture({
                  orderCase,
                  hpCase,
                  p2Item,
                  benchItem,
                  benchSignature,
                  tieRoll,
                  p1DamageRoll,
                  p2DamageRoll,
                  secondaryRoll,
                }));
              }
            }
          }
        }
      }
    }
  }
}

process.stdout.write(JSON.stringify({
  schema: "azelficoast.showdown-two-attack-turn-fixtures",
  schema_version: 1,
  showdown_commit: showdownCommit,
  source: "pokemon-showdown Battle.makeChoices",
  format: "gen9 test Anything Goes without Team Preview",
  fixture_count: fixtures.length,
  fixtures,
}, null, 2) + "\n");
