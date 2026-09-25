#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(2);
}

const [showdownRoot, candidatesPath, fixturesPath, roundsText] = process.argv.slice(2);
if (!showdownRoot || !candidatesPath || !fixturesPath) {
  fail(
    "usage: screen_public_belief_speed_forks.cjs SHOWDOWN_ROOT CANDIDATES FIXTURES [ROUNDS]"
  );
}

const rounds = roundsText ? Number(roundsText) : 512;
if (!Number.isSafeInteger(rounds) || rounds < 1 || rounds > 65536) {
  fail("ROUNDS must be an integer from 1 through 65536");
}

const common = require(path.join(showdownRoot, "test", "common.js"));
const {Teams} = require(path.join(showdownRoot, "dist", "sim", "teams"));
const randomSets = require(
  path.join(showdownRoot, "data", "random-battles", "gen9", "sets.json")
);

const candidatesDocument = JSON.parse(fs.readFileSync(candidatesPath, "utf8"));
const fixtures = new Map(
  fs.readFileSync(fixturesPath, "utf8")
    .split(/\n/)
    .filter(Boolean)
    .map(line => JSON.parse(line))
    .map(fixture => [fixture.fixture_id, fixture])
);

const EVS = {hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85};
const IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};

function toID(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function generatorSpecies(generator, requested) {
  const species = generator.dex.species.get(requested);
  const options = [
    species.id,
    typeof species.battleOnly === "string" ? toID(species.battleOnly) : "",
    typeof species.baseSpecies === "string" ? toID(species.baseSpecies) : "",
  ].filter(Boolean);
  for (const option of [...new Set(options)]) {
    if (randomSets[option]) return option;
  }
  fail(`no random-battle set for ${requested}`);
}

function compatibleVariants(candidate, fixture, rounds = 512) {
  const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);
  const species = generatorSpecies(generator, candidate.generator_species);
  const observed = new Set(candidate.revealed_moves.map(toID));
  const allowedItems = new Set(Object.keys(candidate.item_counts));
  const variants = new Map();

  for (let seed = 0; seed < rounds; seed++) {
    generator.setSeed([seed, seed, seed, seed]);
    const set = generator.randomSet(species, {}, Boolean(candidate.is_lead), false);
    if (
      toID(set.species || candidate.opponent_species) !==
      toID(fixture.state.opponent_active.species)
    ) continue;
    if (Number(set.level) !== Number(fixture.state.opponent_active.level)) continue;
    const moves = [...set.moves].map(toID);
    if (![...observed].every(move => moves.includes(move))) continue;
    if (!allowedItems.has(set.item || "")) continue;

    const variant = {
      species,
      ability: set.ability || "",
      item: set.item || "",
      level: set.level,
      moves: [...set.moves],
      role: set.role || "",
      teraType: set.teraType || null,
    };
    const key = JSON.stringify(variant);
    variants.set(key, (variants.get(key) || 0) + 1);
  }

  return [...variants.entries()]
    .map(([raw, count]) => ({...JSON.parse(raw), count}))
    .sort((a, b) =>
      a.item.localeCompare(b.item) ||
      a.ability.localeCompare(b.ability) ||
      a.role.localeCompare(b.role)
    );
}

function pokemonSet(snapshot, overrides = {}) {
  return {
    species: overrides.species || snapshot.species,
    level: overrides.level || snapshot.level,
    ability: overrides.ability ?? snapshot.ability ?? "",
    item: overrides.item ?? snapshot.item ?? "",
    moves: overrides.moves || snapshot.moves,
    nature: "Serious",
    evs: EVS,
    ivs: IVS,
  };
}

function buildBattle(fixture, variant) {
  const state = fixture.state;
  const ownSnapshot = state.active;
  const opponentSnapshot = state.opponent_active;
  const battle = common.createBattle(
    {preview: false, seed: [1, 2, 3, 4]},
    [
      [pokemonSet(ownSnapshot)],
      [pokemonSet(opponentSnapshot, variant)],
    ]
  );
  const own = battle.p1.active[0];
  const opponent = battle.p2.active[0];

  // The trace is authoritative for our exact known stats. Generator-style
  // EV/nature defaults are only scaffolding needed to instantiate Showdown.
  const stored = Object.fromEntries(
    ["atk", "def", "spa", "spd", "spe"].map(stat => [
      stat,
      Number(ownSnapshot.stats[stat]),
    ])
  );
  own.baseStoredStats = {hp: Number(ownSnapshot.max_hp), ...stored};
  own.storedStats = {...stored};
  own.baseMaxhp = Number(ownSnapshot.max_hp);
  own.maxhp = Number(ownSnapshot.max_hp);
  own.hp = Number(ownSnapshot.current_hp);
  own.boosts = {...ownSnapshot.boosts};
  if (ownSnapshot.status && ownSnapshot.status !== "FNT") {
    own.status = toID(ownSnapshot.status);
  }
  opponent.boosts = {...opponentSnapshot.boosts};
  if (opponentSnapshot.status && opponentSnapshot.status !== "FNT") {
    opponent.status = toID(opponentSnapshot.status);
  }

  return {battle, own, opponent};
}

function assertOwnStats(fixture, own) {
  const expected = fixture.state.active.stats;
  for (const stat of ["atk", "def", "spa", "spd", "spe"]) {
    if (expected[stat] == null) continue;
    if (own.baseStoredStats[stat] !== expected[stat]) {
      fail(
        `${fixture.fixture_id}: own ${stat} drift: expected ${expected[stat]}, got ${own.baseStoredStats[stat]}`
      );
    }
  }
  if (own.maxhp !== fixture.state.active.max_hp) {
    fail(
      `${fixture.fixture_id}: own max HP drift: expected ${fixture.state.active.max_hp}, got ${own.maxhp}`
    );
  }
}

function publicHpRange(percent, maxhp) {
  if (percent >= 100) return [maxhp, maxhp];
  const lower = Math.floor(((percent - 1) * maxhp) / 100) + 1;
  const upper = Math.floor((percent * maxhp) / 100);
  return [Math.max(1, lower), Math.max(1, upper)];
}

function deterministicRandom(roll) {
  return (from, to) => {
    if (to === undefined) return Math.min(roll, from - 1);
    return from + Math.min(roll, to - from - 1);
  };
}

function damageRolls(fixture, variant, sourceSide, moveName) {
  const values = [];
  for (let roll = 0; roll < 16; roll++) {
    const {battle, own, opponent} = buildBattle(fixture, variant);
    assertOwnStats(fixture, own);
    battle.random = deterministicRandom(roll);
    const source = sourceSide === "own" ? own : opponent;
    const target = sourceSide === "own" ? opponent : own;
    const move = battle.dex.getActiveMove(moveName);
    move.willCrit = false;
    move.critRatio = 0;
    const result = battle.actions.getDamage(source, target, move, true);
    if (typeof result !== "number") {
      battle.destroy();
      return null;
    }
    values.push(result);
    battle.destroy();
  }
  return values;
}

function moveMetrics(fixture, variant, moveName) {
  const {battle, own, opponent} = buildBattle(fixture, variant);
  assertOwnStats(fixture, own);
  const move = battle.dex.moves.get(moveName);
  const ownSpeed = own.getActionSpeed();
  const opponentSpeed = opponent.getActionSpeed();
  const [hpLower, hpUpper] = publicHpRange(
    fixture.state.opponent_active.current_hp,
    opponent.maxhp
  );
  battle.destroy();

  if (move.category === "Status") {
    return {
      move: move.id,
      category: move.category,
      accuracy: move.accuracy,
      own_speed: ownSpeed,
      opponent_speed: opponentSpeed,
    };
  }

  const rolls = damageRolls(fixture, variant, "own", moveName);
  if (!rolls) return null;
  return {
    move: move.id,
    category: move.category,
    accuracy: move.accuracy,
    own_speed: ownSpeed,
    opponent_speed: opponentSpeed,
    damage_min: Math.min(...rolls),
    damage_max: Math.max(...rolls),
    public_hp_lower: hpLower,
    public_hp_upper: hpUpper,
    guaranteed_ko_on_hit: Math.min(...rolls) >= hpUpper,
    possible_ko_on_hit: Math.max(...rolls) >= hpLower,
    guaranteed_preemptive_ko:
      ownSpeed > opponentSpeed &&
      (move.accuracy === true || move.accuracy === 100) &&
      Math.min(...rolls) >= hpUpper,
  };
}

function incomingMetrics(fixture, variant, lockedMove) {
  const {battle, own, opponent} = buildBattle(fixture, variant);
  assertOwnStats(fixture, own);
  const move = battle.dex.moves.get(lockedMove);
  const ownSpeed = own.getActionSpeed();
  const opponentSpeed = opponent.getActionSpeed();
  battle.destroy();

  const rolls = damageRolls(fixture, variant, "opponent", lockedMove);
  if (!rolls) {
    return {
      move: toID(lockedMove),
      accuracy: move.accuracy,
      own_speed: ownSpeed,
      opponent_speed: opponentSpeed,
      non_damage: true,
    };
  }
  const hp = fixture.state.active.current_hp;
  const koRolls = rolls.filter(value => value >= hp).length;
  return {
    move: move.id,
    accuracy: move.accuracy,
    own_speed: ownSpeed,
    opponent_speed: opponentSpeed,
    damage_min: Math.min(...rolls),
    damage_max: Math.max(...rolls),
    active_hp: hp,
    ko_rolls: koRolls,
    guaranteed_preempted_faint:
      opponentSpeed > ownSpeed &&
      (move.accuracy === true || move.accuracy === 100) &&
      Math.min(...rolls) >= hp,
    possible_preempted_faint:
      opponentSpeed > ownSpeed &&
      Math.max(...rolls) >= hp,
  };
}

const cases = [];
for (const candidate of candidatesDocument.candidates) {
  const fixture = fixtures.get(candidate.fixture_id);
  if (!fixture) fail(`missing fixture ${candidate.fixture_id}`);

  const variants = compatibleVariants(candidate, fixture, rounds);
  const worlds = [];
  for (const variant of variants) {
    const incoming = incomingMetrics(fixture, variant, candidate.locked_move);
    const moves = fixture.state.available_moves
      .map(move => moveMetrics(fixture, variant, move))
      .filter(Boolean);
    worlds.push({
      ...variant,
      incoming,
      moves,
    });
  }

  const byItem = {};
  for (const item of Object.keys(candidate.item_counts)) {
    const itemWorlds = worlds.filter(world => world.item === item);
    byItem[item] = {
      variant_count: itemWorlds.length,
      sample_count: itemWorlds.reduce((sum, world) => sum + world.count, 0),
      abilities: [...new Set(itemWorlds.map(world => world.ability))].sort(),
      incoming_guaranteed_preempted_faint:
        itemWorlds.length > 0 &&
        itemWorlds.every(world => world.incoming.guaranteed_preempted_faint),
      incoming_possible_preempted_faint:
        itemWorlds.some(world => world.incoming.possible_preempted_faint),
      guaranteed_preemptive_ko_moves: [
        ...new Set(
          itemWorlds.flatMap(world =>
            world.moves
              .filter(move => move.guaranteed_preemptive_ko)
              .map(move => move.move)
          )
        ),
      ].sort(),
      worlds: itemWorlds,
    };
  }

  const powerItems = Object.keys(candidate.item_counts).filter(
    item => item !== "Choice Scarf"
  );
  const powerHasGuaranteedPreemptiveKo = powerItems.some(
    item => byItem[item]?.guaranteed_preemptive_ko_moves?.length
  );
  const scarfGuaranteesPreemptedFaint =
    Boolean(byItem["Choice Scarf"]?.incoming_guaranteed_preempted_faint);

  cases.push({
    fixture_id: candidate.fixture_id,
    turn: fixture.state.turn,
    active_species: fixture.state.active.species,
    active_hp: fixture.state.active.current_hp,
    active_max_hp: fixture.state.active.max_hp,
    opponent_species: candidate.opponent_species,
    opponent_public_hp_percent: fixture.state.opponent_active.current_hp,
    locked_move: candidate.locked_move,
    item_counts: candidate.item_counts,
    by_item: byItem,
    strict_execution_fork:
      powerHasGuaranteedPreemptiveKo && scarfGuaranteesPreemptedFaint,
  });
}

process.stdout.write(JSON.stringify({
  schema: "azelficoast.public-belief-speed-fork-mechanics",
  schema_version: 1,
  showdown_commit: candidatesDocument.candidates[0]?.showdown_commit || null,
  case_count: cases.length,
  strict_execution_fork_count: cases.filter(c => c.strict_execution_fork).length,
  rounds,
  cases,
}, null, 2) + "\n");
