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
  fail("usage: generate_showdown_damage_fixtures.cjs SHOWDOWN_ROOT");
}

const showdownCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();

const common = require(path.join(showdownRoot, "test", "common.js"));

const DEFAULT_IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};

function attackerSet(species, move, category, item = "", level = 100) {
  const physical = category === "Physical";
  return {
    species,
    ability: species === "Lucario" ? "Steadfast" : "Synchronize",
    item,
    level,
    nature: physical ? "Adamant" : "Modest",
    evs: {
      hp: 0,
      atk: physical ? 252 : 0,
      def: 0,
      spa: physical ? 0 : 252,
      spd: 0,
      spe: 252,
    },
    ivs: DEFAULT_IVS,
    moves: [move, "Protect"],
  };
}

function defenderSet(species, category, level = 100) {
  const physical = category === "Physical";
  const ability = species === "Lapras" ? "Shell Armor" :
    species === "Blastoise" ? "Torrent" : "Synchronize";
  return {
    species,
    ability,
    item: "",
    level,
    nature: physical ? "Bold" : "Calm",
    evs: {
      hp: 252,
      atk: 0,
      def: physical ? 252 : 0,
      spa: 0,
      spd: physical ? 0 : 252,
      spe: 0,
    },
    ivs: DEFAULT_IVS,
    moves: ["Splash", "Protect"],
  };
}

const SCENARIOS = [
  {
    id: "mew-aura-lapras",
    attacker: "Mew",
    defender: "Lapras",
    move: "Aura Sphere",
    attackerLevel: 78,
    defenderLevel: 88,
  },
  {id: "lucario-aura-lapras", attacker: "Lucario", defender: "Lapras", move: "Aura Sphere"},
  {
    id: "mew-tera-fighting-aura-lapras",
    attacker: "Mew",
    defender: "Lapras",
    move: "Aura Sphere",
    tera: "Fighting",
  },
  {
    id: "lucario-tera-fighting-aura-lapras",
    attacker: "Lucario",
    defender: "Lapras",
    move: "Aura Sphere",
    tera: "Fighting",
  },
  {
    id: "lucario-specs-aura-lapras",
    attacker: "Lucario",
    defender: "Lapras",
    move: "Aura Sphere",
    item: "Choice Specs",
  },
  {
    id: "lucario-lifeorb-aura-lapras",
    attacker: "Lucario",
    defender: "Lapras",
    move: "Aura Sphere",
    item: "Life Orb",
  },
  {
    id: "lucario-aura-gardevoir",
    attacker: "Lucario",
    defender: "Gardevoir",
    move: "Aura Sphere",
  },
  {
    id: "lucario-aura-blastoise",
    attacker: "Lucario",
    defender: "Blastoise",
    move: "Aura Sphere",
    attackerLevel: 82,
    defenderLevel: 74,
  },
  {
    id: "lucario-closecombat-lapras",
    attacker: "Lucario",
    defender: "Lapras",
    move: "Close Combat",
  },
  {
    id: "lucario-burned-closecombat-lapras",
    attacker: "Lucario",
    defender: "Lapras",
    move: "Close Combat",
    burned: true,
  },
  {
    id: "lucario-band-closecombat-lapras",
    attacker: "Lucario",
    defender: "Lapras",
    move: "Close Combat",
    item: "Choice Band",
  },
];

function naturePercent(battle, pokemon, stat) {
  const nature = battle.dex.natures.get(pokemon.set.nature);
  if (nature.plus === stat) return 110;
  if (nature.minus === stat) return 90;
  return 100;
}

function runScenario(scenario) {
  const dexMove = common.dex.moves.get(scenario.move);
  const category = dexMove.category;
  if (!["Physical", "Special"].includes(category)) {
    fail(`unsupported category for ${scenario.id}: ${category}`);
  }

  const battle = common.createBattle(
    {preview: false, seed: [11, 22, 33, 44]},
    [
      [
        attackerSet(
          scenario.attacker,
          scenario.move,
          category,
          scenario.item || "",
          scenario.attackerLevel || 100
        ),
      ],
      [defenderSet(scenario.defender, category, scenario.defenderLevel || 100)],
    ]
  );

  const source = battle.p1.active[0];
  const target = battle.p2.active[0];

  if (scenario.tera) source.terastallized = scenario.tera;
  if (scenario.burned) source.status = "brn";

  const attackStat = category === "Physical" ? "atk" : "spa";
  const defenseStat = category === "Physical" ? "def" : "spd";
  const moveForContext = battle.dex.getActiveMove(scenario.move);
  moveForContext.willCrit = false;
  moveForContext.critRatio = 0;
  const typeMod = target.runEffectiveness(moveForContext);

  const context = {
    attacker_level: source.level,
    defender_level: target.level,
    base_power: moveForContext.basePower,
    category,
    move_id: moveForContext.id,
    move_type: moveForContext.type,
    attacker_types: source.getTypes(false, true),
    tera_type: scenario.tera || null,
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
    burned: !!scenario.burned,
  };

  const originalRandom = battle.random;
  const rolls = [];
  for (let roll = 0; roll < 16; roll++) {
    const activeMove = battle.dex.getActiveMove(scenario.move);
    activeMove.willCrit = false;
    activeMove.critRatio = 0;

    battle.random = function (m, n) {
      if (n !== undefined || m !== 16) {
        throw new Error(
          `unexpected RNG request in ${scenario.id}: m=${m} n=${n}`
        );
      }
      return roll;
    };

    const result = battle.actions.getDamage(source, target, activeMove, true);
    if (typeof result !== "number") {
      battle.random = originalRandom;
      battle.destroy();
      fail(`non-numeric damage for ${scenario.id} roll ${roll}: ${result}`);
    }
    rolls.push({roll, damage: result});
  }
  battle.random = originalRandom;

  const fixture = {scenario: scenario.id, context, rolls};
  battle.destroy();
  return fixture;
}

const fixtures = SCENARIOS.map(runScenario);

process.stdout.write(
  JSON.stringify(
    {
      schema: "azelficoast.showdown-gen9-damage-fixtures",
      schema_version: 1,
      showdown_commit: showdownCommit,
      source: "pokemon-showdown BattleActions.getDamage",
      scenario_count: fixtures.length,
      roll_count_per_scenario: 16,
      fixtures,
    },
    null,
    2
  ) + "\n"
);
