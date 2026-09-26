#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(2);
}

const [showdownRoot, fixturePath] = process.argv.slice(2);
if (!showdownRoot || !fixturePath) {
  fail("usage: probe_protect_speed_forks.cjs SHOWDOWN_ROOT FIXTURES_JSON");
}

const common = require(path.join(showdownRoot, "test", "common"));
const fixtures = JSON.parse(fs.readFileSync(fixturePath, "utf8"));

function neutralSpread() {
  return {
    hp: 85,
    atk: 85,
    def: 85,
    spa: 85,
    spd: 85,
    spe: 85,
  };
}

function maxIvs() {
  return {
    hp: 31,
    atk: 31,
    def: 31,
    spa: 31,
    spd: 31,
    spe: 31,
  };
}

function pokemonSet(side, itemOverride) {
  const set = {
    species: side.species,
    level: side.level,
    item: itemOverride || side.item || "",
    moves: side.moves || [side.locked_move],
    nature: "Serious",
    evs: neutralSpread(),
    ivs: maxIvs(),
  };
  if (side.ability) set.ability = side.ability;
  return set;
}

function buildBattle(entry, opponentItem) {
  const battle = common.createBattle(
    {seed: [1, 2, 3, 4]},
    [
      [pokemonSet(entry.own)],
      [pokemonSet(
        {
          ...entry.opponent,
          moves: [entry.opponent.locked_move],
        },
        opponentItem
      )],
    ]
  );

  const own = battle.p1.active[0];
  const opponent = battle.p2.active[0];

  own.hp = entry.own.current_hp;
  own.boosts = {...entry.own.boosts};
  opponent.boosts = {...entry.opponent.boosts};

  return {battle, own, opponent};
}

function assertFrozenOwnStats(entry, own) {
  if (own.maxhp !== entry.own.max_hp) {
    fail(
      `${entry.fixture_id}: max HP drifted: expected ${entry.own.max_hp}, got ${own.maxhp}`
    );
  }
  for (const [stat, expected] of Object.entries(entry.own.expected_stats)) {
    const actual = own.baseStoredStats[stat];
    if (actual !== expected) {
      fail(
        `${entry.fixture_id}: ${stat} drifted: expected ${expected}, got ${actual}`
      );
    }
  }
}

function publicHpRange(percent, maxhp) {
  if (percent === 100) return [maxhp, maxhp];
  const lower = Math.floor(((percent - 1) * maxhp) / 100) + 1;
  const upper = Math.floor((percent * maxhp) / 100);
  return [lower, upper];
}

function deterministicRandom(roll) {
  return (from, to) => {
    if (to === undefined) {
      return Math.min(roll, from - 1);
    }
    return from + Math.min(roll, to - from - 1);
  };
}

function damageRolls(entry, opponentItem, sourceSide, moveName) {
  const values = [];
  for (let roll = 0; roll < 16; roll++) {
    const {battle, own, opponent} = buildBattle(entry, opponentItem);
    battle.random = deterministicRandom(roll);
    const source = sourceSide === "own" ? own : opponent;
    const target = sourceSide === "own" ? opponent : own;
    const move = battle.dex.getActiveMove(moveName);
    move.willCrit = false;
    const damage = battle.actions.getDamage(source, target, move, true);
    if (typeof damage !== "number") {
      fail(
        `${entry.fixture_id}: ${moveName} produced non-numeric damage ${String(damage)}`
      );
    }
    values.push(damage);
  }
  return [...new Set(values)].sort((left, right) => left - right);
}

function simulateProtect(entry, opponentItem) {
  const {battle, own, opponent} = buildBattle(entry, opponentItem);
  assertFrozenOwnStats(entry, own);

  battle.makeChoices(
    "move protect",
    `move ${entry.opponent.locked_move.toLowerCase().replace(/[^a-z0-9]/g, "")}`
  );

  const ownDamageEvents = battle.log.filter(
    line => line.startsWith("|-damage|p1a:")
  );
  const opponentMoveEvents = battle.log.filter(
    line => line.startsWith("|move|p2a:")
  );
  const protectActivations = battle.log.filter(
    line => line.includes("|-activate|p1a:") && line.includes("move: Protect")
  );

  return {
    own_hp_after: own.hp,
    own_status_after: own.status || null,
    opponent_hp_after: opponent.hp,
    opponent_move_events: opponentMoveEvents,
    own_damage_events: ownDamageEvents,
    protect_activation_count: protectActivations.length,
    blocked_without_damage: ownDamageEvents.length === 0,
  };
}

function moveSummary(entry, opponentItem) {
  const {own, opponent} = buildBattle(entry, opponentItem);
  assertFrozenOwnStats(entry, own);

  const [hpLower, hpUpper] = publicHpRange(
    entry.opponent.public_hp_percent,
    opponent.maxhp
  );

  const ownMoves = [];
  for (const moveName of entry.own.moves) {
    const move = own.battle.dex.moves.get(moveName);
    if (move.category === "Status") continue;
    const rolls = damageRolls(entry, opponentItem, "own", moveName);
    ownMoves.push({
      move: move.name,
      accuracy: move.accuracy,
      damage_min: rolls[0],
      damage_max: rolls[rolls.length - 1],
      opponent_public_hp_lower: hpLower,
      opponent_public_hp_upper: hpUpper,
      conditional_on_hit_guaranteed_ko: rolls[0] >= hpUpper,
      conditional_on_hit_possible_ko: rolls[rolls.length - 1] >= hpLower,
    });
  }

  const opponentMove = opponent.battle.dex.moves.get(
    entry.opponent.locked_move
  );
  const opponentRolls = damageRolls(
    entry,
    opponentItem,
    "opponent",
    entry.opponent.locked_move
  );

  return {
    item: opponentItem,
    own_speed: own.getActionSpeed(),
    opponent_speed: opponent.getActionSpeed(),
    opponent_move: {
      move: opponentMove.name,
      accuracy: opponentMove.accuracy,
      contact: Boolean(opponentMove.flags?.contact),
      protectable: Boolean(opponentMove.flags?.protect),
      damage_min: opponentRolls[0],
      damage_max: opponentRolls[opponentRolls.length - 1],
      conditional_on_hit_guaranteed_ko_at_current_hp:
        opponentRolls[0] >= entry.own.current_hp,
      conditional_on_hit_possible_ko_at_current_hp:
        opponentRolls[opponentRolls.length - 1] >= entry.own.current_hp,
    },
    own_moves: ownMoves,
    protect: simulateProtect(entry, opponentItem),
  };
}

const results = [];
for (const entry of fixtures.cases) {
  results.push({
    fixture_id: entry.fixture_id,
    turn: entry.turn,
    own_species: entry.own.species,
    opponent_species: entry.opponent.species,
    worlds: entry.opponent.item_worlds.map(item => moveSummary(entry, item)),
  });
}

process.stdout.write(
  JSON.stringify(
    {
      schema: "azelficoast.protect-speed-fork-mechanics",
      schema_version: 1,
      source_schema: fixtures.schema,
      source_artifact: fixtures.source_artifact,
      showdown_commit: fixtures.showdown_commit,
      cases: results,
    },
    null,
    2
  ) + "\n"
);
