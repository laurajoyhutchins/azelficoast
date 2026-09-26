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
  fail("usage: generate_showdown_switch_intimidate_fixtures.cjs SHOWDOWN_ROOT");
}

const showdownCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();
const common = require(path.join(showdownRoot, "test", "common.js"));

const ABILITIES = ["Flash Fire", "Intimidate"];
const INCOMING_ITEMS = ["", "Heavy-Duty Boots"];
const OPPONENT_ITEMS = ["", "Choice Band"];
const INITIAL_ATTACK_STAGES = [0, 1];
const HAZARDS = [
  {id: "none", stealthRock: false, spikes: 0},
  {id: "stealth-rock-plus-spikes", stealthRock: true, spikes: 1},
];
const HP_CASES = [
  {id: "full", hp: null},
  {id: "low", hp: 20},
];
const ACCURACY_ROLLS = [0, 99];
const DAMAGE_ROLLS = [0, 15];
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

function incomingSet(ability, item) {
  return {
    species: "Arcanine",
    name: "Incoming",
    ability,
    item,
    level: 100,
    nature: "Bold",
    evs: {hp: 252, atk: 0, def: 252, spa: 0, spd: 0, spe: 0},
    ivs: IVS,
    moves: ["Splash"],
  };
}

function spareSet() {
  return {
    species: "Chansey",
    name: "Spare",
    ability: "Natural Cure",
    item: "",
    level: 100,
    nature: "Serious",
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0},
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
    nature: "Adamant",
    evs: {hp: 0, atk: 252, def: 0, spa: 0, spd: 0, spe: 252},
    ivs: IVS,
    moves: ["High Horsepower"],
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
      value.startsWith("|faint|") ||
      value.startsWith("|-ability|") ||
      value.startsWith("|-unboost|");
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

function addHazards(battle, hazard, source) {
  if (hazard.stealthRock) {
    const move = battle.dex.getActiveMove("Stealth Rock");
    if (!battle.p1.addSideCondition("stealthrock", source, move)) {
      fail("could not add Stealth Rock");
    }
  }
  if (hazard.spikes) {
    const move = battle.dex.getActiveMove("Spikes");
    for (let layer = 0; layer < hazard.spikes; layer++) {
      if (!battle.p1.addSideCondition("spikes", source, move)) {
        fail("could not add Spikes layer " + String(layer + 1));
      }
    }
  }
}

function runFixture({
  ability,
  incomingItem,
  opponentItem,
  initialAttackStage,
  hazard,
  hpCase,
  accuracyRoll,
  damageRoll,
}) {
  const battle = common.createBattle(
    {preview: false, seed: [101, 103, 107, 109]},
    [
      [outgoingSet(), incomingSet(ability, incomingItem), spareSet()],
      [opponentSet(opponentItem)],
    ]
  );

  const outgoing = battle.p1.active[0];
  const incoming = battle.p1.pokemon[1];
  const opponent = battle.p2.active[0];
  const highHorsepower = battle.dex.getActiveMove("High Horsepower");
  const stealthRock = battle.dex.getActiveMove("Stealth Rock");

  outgoing.hp = Math.min(200, outgoing.baseMaxhp);
  if (hpCase.hp !== null) {
    incoming.hp = Math.min(hpCase.hp, incoming.maxhp);
  }
  opponent.boosts.atk = initialAttackStage;

  addHazards(battle, hazard, opponent);

  const before = {
    outgoing_slot: 0,
    outgoing_hp: outgoing.hp,
    outgoing_max_hp: outgoing.baseMaxhp,
    incoming_slot: 1,
    incoming_hp: incoming.hp,
    incoming_max_hp: incoming.maxhp,
    incoming_ability: incoming.getAbility().name,
    incoming_item: incoming.getItem().name || "None",
    incoming_has_heavy_duty_boots: incoming.item === "heavydutyboots",
    incoming_grounded: !!incoming.isGrounded(),
    stealth_rock_type_mod: incoming.runEffectiveness(stealthRock),
    stealth_rock: hazard.stealthRock,
    spikes_layers: hazard.spikes,
    opponent_hp: opponent.hp,
    opponent_max_hp: opponent.baseMaxhp,
    opponent_move_pp: ppFor(opponent, highHorsepower.id),
    opponent_attack_stage: initialAttackStage,
  };
  const context = damageContext(battle, opponent, incoming, highHorsepower);

  const requests = [];
  const hazardDamageEvents = [];
  const originalDamage = battle.damage;
  battle.damage = function (amount, target, source, effect, instafaint) {
    const activeEffect = effect || this.effect;
    const dealt = originalDamage.call(this, amount, target, source, effect, instafaint);
    if (
      activeEffect &&
      (activeEffect.id === "stealthrock" || activeEffect.id === "spikes") &&
      typeof dealt === "number"
    ) {
      hazardDamageEvents.push({
        effect: activeEffect.id,
        damage: dealt,
      });
    }
    return dealt;
  };

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
  battle.makeChoices("switch 2", "move highhorsepower");
  battle.prng.random = originalRandom;
  battle.damage = originalDamage;

  const turnLog = battle.log.slice(logStart).map(String);
  const moveLines = turnLog.filter(line => line.startsWith("|move|"));
  const opponentAttackExecuted = moveLines.some(line => line.includes("|p2a:"));
  const intimidateActivated = turnLog.some(
    line => line.startsWith("|-ability|") && line.includes("|Intimidate|")
  );
  const intimidateUnboost = turnLog.some(
    line => line.startsWith("|-unboost|") &&
      line.includes("|p2a:") &&
      line.includes("|atk|")
  );
  const hazardDamageTotal = hazardDamageEvents.reduce(
    (total, event) => total + event.damage,
    0
  );
  const hazardFainted = hazardDamageTotal >= before.incoming_hp;

  const after = {
    active_slot: 1,
    active_hp: incoming.hp,
    active_max_hp: incoming.maxhp,
    bench_slot: 0,
    bench_hp: outgoing.hp,
    bench_max_hp: outgoing.baseMaxhp,
    opponent_hp: opponent.hp,
    opponent_move_pp: ppFor(opponent, highHorsepower.id),
    hazard_damage: hazardDamageTotal,
    hazard_fainted: hazardFainted,
    intimidate_activated: intimidateActivated,
    intimidate_changed_stage: intimidateUnboost,
    opponent_attack_stage: opponent.boosts.atk,
    attack_executed: opponentAttackExecuted,
    hit: opponentAttackExecuted && requests.some(request => request.kind === "damage"),
    active_fainted: !!incoming.fainted,
    opponent_fainted: !!opponent.fainted,
  };

  const fixture = {
    ability,
    incoming_item: incomingItem || "None",
    opponent_item: opponentItem || "None",
    initial_attack_stage: initialAttackStage,
    hazard_case: hazard.id,
    hp_case: hpCase.id,
    accuracy_roll: accuracyRoll,
    damage_roll: damageRoll,
    move_accuracy: highHorsepower.accuracy,
    context,
    before,
    after,
    rng_requests: requests,
    hazard_damage_events: hazardDamageEvents,
    transition_log: transitionLog(turnLog),
  };

  battle.destroy();
  return fixture;
}

const fixtures = [];
for (const ability of ABILITIES) {
  for (const incomingItem of INCOMING_ITEMS) {
    for (const opponentItem of OPPONENT_ITEMS) {
      for (const initialAttackStage of INITIAL_ATTACK_STAGES) {
        for (const hazard of HAZARDS) {
          for (const hpCase of HP_CASES) {
            for (const accuracyRoll of ACCURACY_ROLLS) {
              for (const damageRoll of DAMAGE_ROLLS) {
                fixtures.push(runFixture({
                  ability,
                  incomingItem,
                  opponentItem,
                  initialAttackStage,
                  hazard,
                  hpCase,
                  accuracyRoll,
                  damageRoll,
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
  schema: "azelficoast.showdown-switch-intimidate-fixtures",
  schema_version: 1,
  showdown_commit: showdownCommit,
  source: "pokemon-showdown Battle.makeChoices with SwitchIn event handlers",
  format: "gen9 test Anything Goes without Team Preview",
  fixture_count: fixtures.length,
  fixtures,
}, null, 2) + "\n");
