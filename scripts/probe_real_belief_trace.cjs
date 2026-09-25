#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const {execFileSync} = require("node:child_process");

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(2);
}

const argv = process.argv.slice(2);
const showdownRoot = argv[0];
const fixturePath = argv[1];
let benchPriorPath = null;
for (let i = 2; i < argv.length; i++) {
  if (argv[i] === "--bench-prior") {
    benchPriorPath = argv[++i];
    if (!benchPriorPath) fail("--bench-prior requires a JSON path");
  } else {
    fail("unknown argument: " + argv[i]);
  }
}
if (!showdownRoot || !fixturePath) {
  fail(
    "usage: probe_real_belief_trace.cjs SHOWDOWN_ROOT SOURCE_FIXTURE_JSON " +
    "[--bench-prior CONDITIONAL_TEAM_PRIOR_JSON]"
  );
}

const SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc";
const GENERATOR_ROUNDS = 2048;

function environmentInteger(name, fallback, {min = 0} = {}) {
  const raw = process.env[name];
  if (raw == null || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < min) {
    fail(`${name} must be an integer >= ${min}, got ${JSON.stringify(raw)}`);
  }
  return value;
}

const ROOT_CHANCE_SAMPLES = environmentInteger(
  "AZELFICOAST_ROOT_CHANCE_SAMPLES",
  8,
  {min: 1}
);
const CONTINUATION_CHANCE_SAMPLES = environmentInteger(
  "AZELFICOAST_CONTINUATION_CHANCE_SAMPLES",
  8,
  {min: 1}
);
const CONTINUATION_DECISION_HORIZONS = environmentInteger(
  "AZELFICOAST_CONTINUATION_DECISION_HORIZONS",
  1,
  {min: 1}
);
if (![1, 2].includes(CONTINUATION_DECISION_HORIZONS)) {
  fail(
    "AZELFICOAST_CONTINUATION_DECISION_HORIZONS must be 1 or 2, got " +
    CONTINUATION_DECISION_HORIZONS
  );
}
const CHANCE_SEED_FAMILY = environmentInteger(
  "AZELFICOAST_CHANCE_SEED_FAMILY",
  0
);
const MARGINALIZED_VARIANT_FIELD = "opponent.active.generator_variant_remainder";

const DEPENDENCY_CANDIDATES = [
  "opponent.active.item",
  "opponent.active.ability",
  "opponent.active.evs",
  "opponent.active.ivs",
  "opponent.active.exact_hp",
];
const BENCH_FACTOR_FIELD = "opponent.bench.species";

const actualCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();
if (actualCommit !== SHOWDOWN_COMMIT) {
  fail(`expected Showdown ${SHOWDOWN_COMMIT}, got ${actualCommit}`);
}

const source = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
if (source.schema !== "azelficoast.real-belief-source-fixture" || source.schema_version !== 1) {
  fail("unexpected source fixture schema");
}
if (source.showdown_commit !== SHOWDOWN_COMMIT) {
  fail("source fixture is bound to a different Showdown revision");
}
const fixture = source.fixture || {
  fixture_id: source.fixture_id,
  state: source.state,
  protocol_prefix: source.protocol_prefix,
  control_decisions: source.control_decisions || [],
};
if (!fixture || fixture.fixture_id !== source.fixture_id) fail("fixture identity mismatch");

function publicOpponentSpecies() {
  return [...new Set(
    Object.values(fixture.state.opponent_team || {})
      .map(view => toID(view && view.species))
      .filter(Boolean)
  )].sort();
}

function loadBenchPrior() {
  if (!benchPriorPath) return null;
  const document = JSON.parse(fs.readFileSync(benchPriorPath, "utf8"));
  if (
    document.schema !== "azelficoast.conditional-team-prior-evaluation" ||
    document.schema_version !== 1
  ) {
    fail("unexpected conditional team prior schema");
  }
  if (document.showdown_commit !== SHOWDOWN_COMMIT) {
    fail("conditional team prior is bound to a different Showdown revision");
  }
  const real = document.real_staraptor_fixture;
  if (!real || !Array.isArray(real.species_prior) || !real.species_prior.length) {
    fail("conditional team prior does not expose full real fixture species_prior");
  }
  const expectedKnown = publicOpponentSpecies();
  const priorKnown = [...real.known_species].map(toID).sort();
  if (JSON.stringify(priorKnown) !== JSON.stringify(expectedKnown)) {
    fail(
      "conditional team prior known species do not match this public fixture: " +
      JSON.stringify({priorKnown, expectedKnown})
    );
  }

  const distribution = real.species_prior.map(row => ({
    value: toID(row.id),
    weight: Number(row.probability),
  }));
  if (distribution.some(row => !row.value || !(row.weight > 0))) {
    fail("conditional team prior contains invalid species probabilities");
  }
  if (new Set(distribution.map(row => row.value)).size !== distribution.length) {
    fail("conditional team prior contains duplicate species");
  }
  const total = distribution.reduce((sum, row) => sum + row.weight, 0);
  if (Math.abs(total - 1) > 1e-9) {
    fail("conditional team prior probabilities sum to " + total);
  }
  if (Number(real.support_count) !== distribution.length) {
    fail("conditional team prior support count does not match species_prior");
  }
  return {
    distribution,
    evidenceSha256: String(document.evidence_sha256 || ""),
    selectedLambda: Number(document.selected_lambda),
  };
}

const common = require(path.join(showdownRoot, "test", "common.js"));
const {Battle, extractChannelMessages} = require(path.join(showdownRoot, "dist", "sim", "battle.js"));
const {Teams} = require(path.join(showdownRoot, "dist", "sim", "teams.js"));
const randomSets = require(
  path.join(showdownRoot, "data", "random-battles", "gen9", "sets.json")
);

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])]));
  }
  return value;
}

function sha256(value) {
  return crypto.createHash("sha256").update(JSON.stringify(stable(value))).digest("hex");
}

function toID(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function protocolSides() {
  const ownName = toID(fixture.state.player);
  const opponentName = toID(fixture.state.opponent);
  let own = null;
  let opponent = null;

  for (const batch of fixture.protocol_prefix || []) {
    for (const message of batch) {
      if (
        message[0] !== "" ||
        message[1] !== "player" ||
        !["p1", "p2"].includes(message[2])
      ) {
        continue;
      }
      const name = toID(message[3]);
      if (name === ownName) own = message[2];
      if (name === opponentName) opponent = message[2];
    }
  }

  // Older frozen fixtures can omit player names while still retaining
  // enough public battle protocol to determine orientation. Resolve only from
  // observed switch/drag species matching the current public actives.
  if (!own || !opponent) {
    const ownSpecies = toID(fixture.state.active && fixture.state.active.species);
    const opponentSpecies = toID(
      fixture.state.opponent_active && fixture.state.opponent_active.species
    );
    let ownFromSpecies = null;
    let opponentFromSpecies = null;
    if (ownSpecies && opponentSpecies && ownSpecies !== opponentSpecies) {
      for (const batch of fixture.protocol_prefix || []) {
        for (const message of batch) {
          if (
            message[0] !== "" ||
            !["switch", "drag"].includes(message[1]) ||
            typeof message[2] !== "string"
          ) {
            continue;
          }
          const side = message[2].startsWith("p1") ? "p1"
            : message[2].startsWith("p2") ? "p2"
            : null;
          if (!side) continue;
          const species = toID(String(message[3] || "").split(",", 1)[0]);
          if (species === ownSpecies) ownFromSpecies = side;
          if (species === opponentSpecies) opponentFromSpecies = side;
        }
      }
    }
    if (!own && ownFromSpecies) own = ownFromSpecies;
    if (!opponent && opponentFromSpecies) opponent = opponentFromSpecies;
  }

  if (!own && opponent) own = opponent === "p1" ? "p2" : "p1";
  if (!opponent && own) opponent = own === "p1" ? "p2" : "p1";
  if (!own || !opponent || own === opponent) {
    return {own: null, opponent: null};
  }
  return {own, opponent};
}

const PROTOCOL_SIDES = protocolSides();

function requireProtocolOpponentSide() {
  if (!PROTOCOL_SIDES.opponent) {
    fail("could not resolve player/opponent protocol sides");
  }
  return PROTOCOL_SIDES.opponent;
}

function seed(index, salt) {
  const base = index + 1 + salt * 257 + CHANCE_SEED_FAMILY * 4099;
  return [
    base & 0xffff,
    (base * 17 + 11) & 0xffff,
    (base * 97 + 23) & 0xffff,
    (base * 193 + 47) & 0xffff,
  ];
}

function observedOpponentMoves() {
  if (Array.isArray(source.observed_opponent_moves)) {
    return [...new Set(source.observed_opponent_moves.map(toID))].sort();
  }
  const species = toID(fixture.state.opponent_active.species);
  const moves = new Set();
  let active = "";
  for (const batch of fixture.protocol_prefix) {
    for (const message of batch) {
      if (message[0] !== "" || message.length < 2) continue;
      if (["switch", "drag"].includes(message[1]) && String(message[2] || "").startsWith(requireProtocolOpponentSide())) {
        active = toID(String(message[3] || "").split(",", 1)[0]);
      }
      if (message[1] === "move" && String(message[2] || "").startsWith(requireProtocolOpponentSide()) && active === species) {
        moves.add(toID(message[3]));
      }
    }
  }
  return [...moves].sort();
}

function lastOpponentMove() {
  if (source.opponent_response_move) return toID(source.opponent_response_move);
  let last = null;
  for (const batch of fixture.protocol_prefix) {
    for (const message of batch) {
      if (message[0] === "" && message[1] === "move" && String(message[2] || "").startsWith(requireProtocolOpponentSide())) {
        last = String(message[3]);
      }
    }
  }
  if (!last) fail("fixture contains no opponent move to use as bounded response policy");
  return toID(last);
}

function resolveGeneratorSpecies(requested) {
  const dexSpecies = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0])
    .dex.species.get(requested);
  const candidates = [
    dexSpecies.id,
    typeof dexSpecies.battleOnly === "string" ? toID(dexSpecies.battleOnly) : "",
    typeof dexSpecies.baseSpecies === "string" ? toID(dexSpecies.baseSpecies) : "",
  ].filter(Boolean);
  for (const candidate of [...new Set(candidates)]) {
    if (randomSets[candidate]) return candidate;
  }
  fail(`no randbats set data for ${requested}`);
}

function generatorVariants() {
  const requested = fixture.state.opponent_active.species;
  const species = resolveGeneratorSpecies(requested);
  const observed = new Set(observedOpponentMoves());
  const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);
  const variants = new Map();
  const itemCounts = new Map();
  let matched = 0;
  for (let i = 0; i < GENERATOR_ROUNDS; i++) {
    generator.setSeed([i, i, i, i]);
    const set = generator.randomSet(
      species,
      {},
      source.opponent_is_lead === true,
      false
    );
    if (
      toID(set.species || requested) !==
      toID(fixture.state.opponent_active.species)
    ) continue;
    if (Number(set.level) !== Number(fixture.state.opponent_active.level)) continue;
    const publicAbility = toID(fixture.state.opponent_active.ability || "");
    if (publicAbility && toID(set.ability) !== publicAbility) continue;

    const moves = [...set.moves].map(toID).sort();
    if (![...observed].every(move => moves.includes(move))) continue;
    const plausibleItems = Array.isArray(source.plausible_items)
      ? new Set(source.plausible_items)
      : null;
    if (plausibleItems && !plausibleItems.has(set.item)) continue;

    matched++;
    itemCounts.set(set.item, (itemCounts.get(set.item) || 0) + 1);
    const semantic = {
      species: toID(fixture.state.opponent_active.species),
      ability: set.ability,
      item: set.item,
      level: set.level,
      moves,
      evs: {...set.evs},
      ivs: {...set.ivs},
      teraType: set.teraType,
    };
    const key = JSON.stringify(stable(semantic));
    const prior = variants.get(key) || {set: semantic, count: 0};
    prior.count++;
    variants.set(key, prior);
  }
  if (!matched) fail("generator sweep produced no Choice worlds compatible with public moves");
  return {
    matched,
    itemCounts: Object.fromEntries(
      [...itemCounts.entries()].sort(([left], [right]) => left.localeCompare(right))
    ),
    variants: [...variants.values()],
  };
}

function executionClasses(variants) {
  const classes = new Map();
  for (const entry of variants) {
    const execution = {
      species: entry.set.species,
      ability: entry.set.ability,
      item: entry.set.item,
      level: entry.set.level,
      evs: entry.set.evs,
      ivs: entry.set.ivs,
    };
    const key = JSON.stringify(stable(execution));
    const remainder = stable({
      moves: entry.set.moves,
      teraType: entry.set.teraType,
    });
    const existing = classes.get(key);
    if (!existing) {
      classes.set(key, {
        set: entry.set,
        count: entry.count,
        generator_variant_count: 1,
        marginalized_remainders: [{value: remainder, count: entry.count}],
      });
      continue;
    }
    existing.count += entry.count;
    existing.generator_variant_count += 1;
    const remainderKey = JSON.stringify(remainder);
    const prior = existing.marginalized_remainders.find(
      candidate => JSON.stringify(candidate.value) === remainderKey
    );
    if (prior) prior.count += entry.count;
    else existing.marginalized_remainders.push({value: remainder, count: entry.count});
  }
  return [...classes.values()];
}

function ownActiveTeraType() {
  if (source.own_active_tera_type) return String(source.own_active_tera_type);

  let inferred = null;
  for (const batch of fixture.protocol_prefix || []) {
    for (const message of batch) {
      if (
        message[0] !== "" ||
        message[1] !== "request" ||
        typeof message[2] !== "string"
      ) {
        continue;
      }
      try {
        const request = JSON.parse(message[2]);
        const tera = request.active?.[0]?.canTerastallize;
        if (tera) inferred = String(tera);
      } catch (_error) {
        // Non-request protocol text is irrelevant to this inference.
      }
    }
  }
  if (!inferred) {
    fail("source fixture must expose the active player's Tera type in a request or override");
  }
  return inferred;
}

function opponentTeamSize() {
  let size = null;
  for (const batch of fixture.protocol_prefix || []) {
    for (const message of batch) {
      if (
        message[0] === "" &&
        message[1] === "teamsize" &&
        message[2] === requireProtocolOpponentSide() &&
        Number.isInteger(Number(message[3]))
      ) {
        size = Number(message[3]);
      }
    }
  }
  return size;
}

function opponentBenchSpecies() {
  if (source.opponent_bench_species) return String(source.opponent_bench_species);

  let active = null;
  const seen = [];
  const fainted = new Set();
  for (const batch of fixture.protocol_prefix || []) {
    for (const message of batch) {
      if (message[0] !== "" || message.length < 2) continue;
      if (
        ["switch", "drag"].includes(message[1]) &&
        String(message[2] || "").startsWith(requireProtocolOpponentSide())
      ) {
        active = String(message[3] || "").split(",", 1)[0];
        if (active && !seen.some(species => toID(species) === toID(active))) {
          seen.push(active);
        }
      }
      if (
        message[1] === "faint" &&
        String(message[2] || "").startsWith(requireProtocolOpponentSide()) &&
        active
      ) {
        fainted.add(toID(active));
      }
    }
  }

  const current = toID(fixture.state.opponent_active.species);
  const candidate = [...seen].reverse().find(
    species => toID(species) !== current && !fainted.has(toID(species))
  );
  if (candidate) return candidate;

  const publicTeam = Object.values(fixture.state.opponent_team || {});
  const knownSpecies = new Set(
    publicTeam
      .map(view => toID(view && view.species))
      .filter(Boolean)
  );
  const teamSize = opponentTeamSize();

  if (
    teamSize !== null &&
    knownSpecies.size >= teamSize &&
    publicTeam.every(
      view =>
        toID(view && view.species) === current ||
        Boolean(view && view.fainted)
    )
  ) {
    return null;
  }

  fail(
    "source fixture must expose one surviving public opponent bench species, " +
    "prove the public bench is exhausted, or provide an override"
  );
}

const OWN_ACTIVE_TERA_TYPE = ownActiveTeraType();
const BENCH_PRIOR = loadBenchPrior();
const OPPONENT_BENCH_SPECIES = BENCH_PRIOR
  ? BENCH_PRIOR.distribution[0].value
  : opponentBenchSpecies();

function ownSet(view, {active = false} = {}) {
  const set = {
    species: view.species,
    level: view.level,
    ability: view.ability,
    item: view.item || "",
    moves: view.moves,
    nature: "Serious",
    evs: {hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85},
    ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
  };
  if (active) set.teraType = OWN_ACTIVE_TERA_TYPE;
  return set;
}

function opponentSet(world) {
  return {
    species: world.variant.species,
    level: world.variant.level,
    ability: world.variant.ability,
    item: world.variant.item,
    moves: world.variant.moves,
    teraType: world.variant.teraType,
    nature: "Serious",
    evs: {...world.variant.evs},
    ivs: {...world.variant.ivs},
  };
}

function opponentBenchSet(speciesName = OPPONENT_BENCH_SPECIES) {
  const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);
  generator.setSeed([0, 0, 0, 0]);

  const requested = generator.dex.species.get(speciesName);
  if (!requested.exists) fail(`unknown opponent bench species ${speciesName}`);

  if (generator.randomSets[requested.id]) {
    return generator.randomSet(requested, {}, false, false);
  }

  const base = generator.dex.species.get(requested.baseSpecies);
  if (!base.exists || !generator.randomSets[base.id]) {
    fail(`no random-set data for opponent bench ${speciesName}`);
  }

  const sameBattleData =
    JSON.stringify(requested.baseStats) === JSON.stringify(base.baseStats) &&
    JSON.stringify(requested.types) === JSON.stringify(base.types) &&
    JSON.stringify(requested.abilities) === JSON.stringify(base.abilities);
  if (!sameBattleData) {
    fail(
      `opponent bench forme ${speciesName} lacks exact random-set data and is not cosmetic`
    );
  }

  const set = generator.randomSet(base, {}, false, false);
  set.species = requested.name;
  return set;
}

const ownState = fixture.state.team;
const activeId = toID(fixture.state.active.species);
const ownViews = Object.values(ownState);
const ownOrdered = [
  ownViews.find(view => toID(view.species) === activeId),
  ...ownViews.filter(view => toID(view.species) !== activeId),
];
if (!ownOrdered[0]) fail("could not place current player active first");
const ownTeam = ownOrdered.map((view, index) => ownSet(view, {active: index === 0}));
const benchSet = OPPONENT_BENCH_SPECIES ? opponentBenchSet() : null;

function applyRecordedOwnStats(battle) {
  for (const view of ownOrdered) {
    const pokemon = battle.p1.pokemon.find(
      candidate => toID(candidate.species.name) === toID(view.species)
    );
    if (!pokemon) fail(`missing reconstructed own Pokemon ${view.species}`);

    // The decision trace is authoritative for our own exact stats. Reconstruct
    // those directly instead of guessing the randbats EV/nature spread.
    const stored = Object.fromEntries(
      ["atk", "def", "spa", "spd", "spe"].map(stat => [stat, Number(view.stats[stat])])
    );
    pokemon.baseStoredStats = {hp: Number(view.max_hp), ...stored};
    pokemon.storedStats = {...stored};
    pokemon.baseMaxhp = Number(view.max_hp);
    pokemon.maxhp = Number(view.max_hp);
    pokemon.hp = Number(view.current_hp);
  }
}

function publicPercent(hp, maxhp) {
  if (hp <= 0) return 0;
  let percentage = Math.ceil(100 * hp / maxhp);
  if (percentage === 100 && hp < maxhp) percentage = 99;
  return percentage;
}

function hpSupportForVariant(variant) {
  const provisional = {variant, exactHp: 1};
  const battle = buildBattle(provisional);
  const maxhp = battle.p2.active[0].maxhp;
  battle.destroy();
  const observed = Number(fixture.state.opponent_active.current_hp);
  const support = [];
  for (let hp = 1; hp <= maxhp; hp++) {
    if (publicPercent(hp, maxhp) === observed) support.push(hp);
  }
  if (!support.length) fail(`no exact HP is compatible with public ${observed}/100`);
  return {maxhp, support};
}

function applyFixtureState(battle, world) {
  battle.turn = Number(fixture.state.turn);
  for (const view of ownOrdered) {
    const pokemon = battle.p1.pokemon.find(candidate => toID(candidate.species.name) === toID(view.species));
    pokemon.hp = Number(view.current_hp);
    pokemon.fainted = Boolean(view.fainted);
    pokemon.status = view.fainted ? "" : (view.status ? toID(view.status) : "");
    pokemon.boosts = {...view.boosts};
  }
  const opponent = battle.p2.active[0];
  opponent.hp = Number(world.exactHp);
  opponent.boosts = {...fixture.state.opponent_active.boosts};
  const opponentStatus = fixture.state.opponent_active.status;
  opponent.status =
    opponentStatus && opponentStatus !== "FNT" ? toID(opponentStatus) : "";

  for (const condition of Object.keys(fixture.state.side_conditions || {})) {
    battle.p1.addSideCondition(toID(condition), "debug");
  }
  for (const condition of Object.keys(fixture.state.opponent_side_conditions || {})) {
    battle.p2.addSideCondition(toID(condition), "debug");
  }
}

function buildBattle(world, benchOverride = benchSet) {
  const battle = common.createBattle(
    {preview: false, seed: [1, 2, 3, 4]},
    [
      ownTeam,
      benchOverride
        ? [opponentSet(world), benchOverride]
        : [opponentSet(world)],
    ]
  );
  applyRecordedOwnStats(battle);
  applyFixtureState(battle, world);
  const tera = battle.p1.active[0].canTerastallize;
  if (tera !== OWN_ACTIVE_TERA_TYPE) {
    fail(
      `expected ${fixture.state.active.species} Tera ${OWN_ACTIVE_TERA_TYPE}, got ${String(tera)}`
    );
  }
  return battle;
}

function cloneBattle(snapshot, chanceSeed) {
  const battle = Battle.fromJSON(snapshot);
  battle.restart(() => {});
  battle.prng.setSeed(chanceSeed.join(","));
  return battle;
}

function rootChoice(action) {
  if (!action.startsWith("/choose ")) fail(`unexpected root action ${action}`);
  return action.slice("/choose ".length);
}

function legalP1Continuations(battle) {
  const request = battle.p1.activeRequest;
  if (!request || request.wait) return [];
  const choices = [];
  const requestSwitches = () => {
    const active = battle.p1.active[0];
    const revivalBlessing = Boolean(
      active &&
      battle.p1.slotConditions[active.position]?.revivalblessing
    );
    for (const [index, pokemon] of request.side.pokemon.entries()) {
      if (pokemon.active) continue;
      const fainted = String(pokemon.condition).endsWith(" fnt");
      if (revivalBlessing ? !fainted : fainted) continue;
      choices.push(`switch ${index + 1}`);
    }
  };
  if (request.forceSwitch) {
    requestSwitches();
    return choices.sort();
  }
  if (request.active) {
    const activeRequest = request.active[0];
    for (const move of activeRequest.moves || []) {
      if (!move.disabled) choices.push(`move ${move.id}`);
    }
    if (!activeRequest.trapped) requestSwitches();
  }
  return [...new Set(choices)].sort();
}

function opponentChoice(battle, hiddenReads = null) {
  const request = battle.p2.activeRequest;
  if (!request || request.wait) return "";
  if (request.forceSwitch) {
    if (hiddenReads && BENCH_PRIOR) hiddenReads.add(BENCH_FACTOR_FIELD);
    const target = battle.p2.pokemon.find(pokemon => pokemon.hp && !pokemon.active);
    return target ? `switch ${target.position + 1}` : "";
  }
  if (request.active) {
    const moves = request.active[0].moves || [];
    const locked = lastOpponentMove();
    if (moves.some(move => move.id === locked && !move.disabled)) return `move ${locked}`;
    if (hiddenReads) hiddenReads.add(MARGINALIZED_VARIANT_FIELD);
    fail(
      `bounded opponent response ${locked} became unavailable; hidden move fallback would be required`
    );
  }
  return "";
}

function observation(battle, logStart) {
  const lines = extractChannelMessages(battle.log.slice(logStart).join("\n"), [1])[1]
    // Server wall-clock transport metadata is public but not battle semantics.
    // Keeping it would make equivalent simulated worlds differ by execution time.
    .filter(line => !line.startsWith("|t:|"));
  const request = battle.p1.activeRequest ? JSON.parse(JSON.stringify(battle.p1.activeRequest)) : null;
  return {protocol: lines, request};
}

function stateSummary(battle, world) {
  function pokemonSummary(pokemon, {includeItem = true} = {}) {
    return {
      species: pokemon.species.id,
      hp: pokemon.hp,
      maxhp: pokemon.maxhp,
      status: pokemon.status || null,
      boosts: stable(pokemon.boosts),
      ...(includeItem ? {item: pokemon.item || null} : {}),
      active: pokemon.active,
      fainted: pokemon.fainted,
      terastallized: pokemon.terastallized || null,
      volatiles: Object.keys(pokemon.volatiles).sort(),
    };
  }
  return {
    turn: battle.turn,
    request_state: battle.requestState,
    ended: battle.ended,
    winner: battle.winner || null,
    p1: battle.p1.pokemon.map(pokemonSummary),
    // Unchanged hidden opponent identity belongs to the world, not the transition
    // delta. Carrying it here would make every action spuriously item-dependent.
    p2_active: battle.p2.active[0]
      ? {
          ...pokemonSummary(battle.p2.active[0], {includeItem: false}),
          // Untouched hidden HP remains carried by the hidden world itself.
          // Only expose it in transition evidence when this action changed it.
          hp:
            battle.p2.active[0].hp === Number(world.exactHp)
              ? "<unchanged>"
              : battle.p2.active[0].hp,
        }
      : null,
    weather: battle.field.weather || null,
    pseudo_weather: Object.keys(battle.field.pseudoWeather).sort(),
    p1_side_conditions: Object.keys(battle.p1.sideConditions).sort(),
    p2_side_conditions: Object.keys(battle.p2.sideConditions).sort(),
    p1_slot_conditions: battle.p1.slotConditions.map(slot => Object.keys(slot).sort()),
    p2_slot_conditions: battle.p2.slotConditions.map(slot => Object.keys(slot).sort()),
  };
}

function utility(battle) {
  const ownMaterial = battle.p1.pokemon.reduce(
    (sum, pokemon) => sum + (pokemon.maxhp ? pokemon.hp / pokemon.maxhp : 0),
    0
  );
  const opponent = battle.p2.active[0];
  const opponentActive = opponent && opponent.maxhp ? opponent.hp / opponent.maxhp : 0;
  return ownMaterial - opponentActive;
}

function leafContinuationValues(snapshot, hiddenReads, seedSalt) {
  const probe = Battle.fromJSON(snapshot);
  probe.restart(() => {});
  const choices = legalP1Continuations(probe);
  if (!choices.length) {
    const value = utility(probe);
    probe.destroy();
    return {terminal_utility: value};
  }
  probe.destroy();

  const values = {};
  for (const choice of choices) {
    let sum = 0;
    for (let i = 0; i < CONTINUATION_CHANCE_SAMPLES; i++) {
      const battle = cloneBattle(
        snapshot,
        seed(i, seedSalt + sha256(choice).charCodeAt(0))
      );
      const foe = opponentChoice(battle, hiddenReads);
      battle.makeChoices(choice, foe);
      sum += utility(battle);
      battle.destroy();
    }
    values[choice] = sum / CONTINUATION_CHANCE_SAMPLES;
  }
  return {continuations: values};
}

function continuationValues(rootSnapshot) {
  const hiddenReads = new Set();
  const probe = Battle.fromJSON(rootSnapshot);
  probe.restart(() => {});
  const choices = legalP1Continuations(probe);
  if (!choices.length) {
    const value = utility(probe);
    probe.destroy();
    return {
      terminal_utility: value,
      hidden_reads: [],
    };
  }
  probe.destroy();

  if (CONTINUATION_DECISION_HORIZONS === 1) {
    const values = {};
    for (const choice of choices) {
      let sum = 0;
      for (let i = 0; i < CONTINUATION_CHANCE_SAMPLES; i++) {
        const battle = cloneBattle(
          rootSnapshot,
          seed(i, 10_000 + sha256(choice).charCodeAt(0))
        );
        const foe = opponentChoice(battle, hiddenReads);
        battle.makeChoices(choice, foe);
        sum += utility(battle);
        battle.destroy();
      }
      values[choice] = sum / CONTINUATION_CHANCE_SAMPLES;
    }
    return {
      continuations: values,
      hidden_reads: [...hiddenReads].sort(),
    };
  }

  const continuationTransitions = {};
  for (const choice of choices) {
    const outcomes = [];
    const firstSalt = sha256(choice).charCodeAt(0);
    for (let i = 0; i < CONTINUATION_CHANCE_SAMPLES; i++) {
      const battle = cloneBattle(
        rootSnapshot,
        seed(i, 10_000 + firstSalt)
      );
      const logStart = battle.log.length;
      const foe = opponentChoice(battle, hiddenReads);
      battle.makeChoices(choice, foe);
      const nextObservation = observation(battle, logStart);
      const nextSnapshot = JSON.stringify(battle);
      const leaf = leafContinuationValues(
        nextSnapshot,
        hiddenReads,
        20_000 + firstSalt * 257 + i * 17
      );
      outcomes.push({
        probability: 1 / CONTINUATION_CHANCE_SAMPLES,
        observation: nextObservation,
        ...leaf,
      });
      battle.destroy();
    }
    continuationTransitions[choice] = outcomes;
  }

  return {
    continuation_transitions: continuationTransitions,
    hidden_reads: [...hiddenReads].sort(),
  };
}

function declaredReads(_action) {
  // Pokémon Showdown is an external oracle rather than an instrumented lowering.
  // Declare the whole hidden adapter boundary conservatively; the analyzer then
  // derives the empirically required subset from exact mechanics outcomes.
  return [...DEPENDENCY_CANDIDATES];
}

function factoredBenchAudit(worlds, legalActions, transitions) {
  if (!BENCH_PRIOR) return null;

  const rootSurvivalByAction = Object.fromEntries(
    legalActions.map(action => [
      action,
      transitions
        .filter(transition => transition.action === action)
        .every(transition =>
          transition.outcomes.every(outcome => {
            const active = outcome.successor && outcome.successor.p2_active;
            return Boolean(active && !active.fainted && active.hp !== 0);
          })
        ),
    ])
  );

  const continuationReadByAction = Object.fromEntries(
    legalActions.map(action => [
      action,
      transitions
        .filter(transition => transition.action === action)
        .some(transition =>
          transition.outcomes.some(outcome =>
            Array.isArray(outcome.hidden_reads) &&
            outcome.hidden_reads.includes(BENCH_FACTOR_FIELD)
          )
        ),
    ])
  );

  const auditWorlds = [];
  const seenVariants = new Set();
  for (const world of worlds) {
    const key = JSON.stringify(stable(world.variant));
    if (seenVariants.has(key)) continue;
    seenVariants.add(key);
    auditWorlds.push(world);
  }

  function immediateSignature(world, action, benchSpecies) {
    const battle = buildBattle(world, opponentBenchSet(benchSpecies));
    battle.prng.setSeed(
      seed(0, 50_000 + legalActions.indexOf(action)).join(",")
    );
    const logStart = battle.log.length;
    battle.makeChoices(rootChoice(action), `move ${lastOpponentMove()}`);
    const signature = sha256({
      observation: observation(battle, logStart),
      successor: stateSummary(battle, world),
    });
    battle.destroy();
    return signature;
  }

  const representative = OPPONENT_BENCH_SPECIES;
  const baselineByAction = Object.fromEntries(
    legalActions.map(action => [
      action,
      auditWorlds.map(world =>
        immediateSignature(world, action, representative)
      ),
    ])
  );

  const immediateEquivalenceByAction = Object.fromEntries(
    legalActions.map(action => [action, true])
  );
  const firstDivergenceByAction = {};

  for (const row of BENCH_PRIOR.distribution) {
    for (const action of legalActions) {
      if (!immediateEquivalenceByAction[action]) continue;
      const signatures = auditWorlds.map(world =>
        immediateSignature(world, action, row.value)
      );
      const baseline = baselineByAction[action];
      const divergentIndex = signatures.findIndex(
        (signature, index) => signature !== baseline[index]
      );
      if (divergentIndex >= 0) {
        immediateEquivalenceByAction[action] = false;
        firstDivergenceByAction[action] = {
          species: row.value,
          active_variant_index: divergentIndex,
          baseline_hash: baseline[divergentIndex],
          candidate_hash: signatures[divergentIndex],
        };
      }
    }
  }

  const unreadActions = legalActions.filter(
    action =>
      !continuationReadByAction[action] &&
      immediateEquivalenceByAction[action]
  );

  return {
    field: BENCH_FACTOR_FIELD,
    distribution: BENCH_PRIOR.distribution,
    unread_actions: unreadActions,
    evidence: {
      kind: "inactive-bench-bounded-horizon",
      prior_evidence_sha256: BENCH_PRIOR.evidenceSha256,
      selected_lambda: BENCH_PRIOR.selectedLambda,
      representative_species: representative,
      audited_support_count: BENCH_PRIOR.distribution.length,
      audited_active_variant_count: auditWorlds.length,
      root_survival_by_action: rootSurvivalByAction,
      continuation_read_by_action: continuationReadByAction,
      immediate_equivalence_by_action: immediateEquivalenceByAction,
      first_divergence_by_action: firstDivergenceByAction,
      immediate_audit_chance_samples: 1,
      dynamic_read_chance_samples: CONTINUATION_CHANCE_SAMPLES,
      root_survival_chance_samples: ROOT_CHANCE_SAMPLES,
      continuation_rule:
        "a bench-species read is recorded only when opponentChoice actually selects a forced switch during the bounded continuation",
    },
  };
}

const {matched, itemCounts, variants} = generatorVariants();
const executionVariants = executionClasses(variants);
if (
  source.expected_generator_rounds != null &&
  Number(source.expected_generator_rounds) !== GENERATOR_ROUNDS
) {
  fail(
    `expected generator rounds ${source.expected_generator_rounds}, probe uses ${GENERATOR_ROUNDS}`
  );
}
if (source.expected_item_counts != null) {
  const expectedCounts = stable(source.expected_item_counts);
  const observedCounts = stable(itemCounts);
  if (JSON.stringify(expectedCounts) !== JSON.stringify(observedCounts)) {
    fail(
      `exact hidden-world prior drift: expected ${JSON.stringify(expectedCounts)}, got ${JSON.stringify(observedCounts)}`
    );
  }
}

const worldById = new Map();
for (const entry of executionVariants) {
  const {maxhp, support} = hpSupportForVariant(entry.set);
  for (const exactHp of support) {
    const hidden = {
      "opponent.active.item": entry.set.item,
      "opponent.active.ability": entry.set.ability,
      "opponent.active.evs": entry.set.evs,
      "opponent.active.ivs": entry.set.ivs,
      "opponent.active.exact_hp": exactHp,
    };
    const worldId = sha256(hidden);
    const weight = (entry.count / matched) / support.length;
    const candidate = {
      world_id: worldId,
      weight,
      hidden,
      variant: entry.set,
      exactHp,
      opponent_max_hp: maxhp,
      generator_count: entry.count,
      generator_variant_count: entry.generator_variant_count,
      marginalized_remainders: entry.marginalized_remainders,
    };
    const existing = worldById.get(worldId);
    if (!existing) {
      worldById.set(worldId, candidate);
      continue;
    }

    const existingMechanics = stable({
      variant: existing.variant,
      exactHp: existing.exactHp,
      opponent_max_hp: existing.opponent_max_hp,
    });
    const candidateMechanics = stable({
      variant: candidate.variant,
      exactHp: candidate.exactHp,
      opponent_max_hp: candidate.opponent_max_hp,
    });
    if (JSON.stringify(existingMechanics) !== JSON.stringify(candidateMechanics)) {
      fail(`hidden world identity collision across distinct mechanics: ${worldId}`);
    }
    existing.weight += weight;
    existing.generator_count += entry.count;
    existing.generator_variant_count += entry.generator_variant_count;
    existing.marginalized_remainders.push(...entry.marginalized_remainders);
  }
}
const worlds = [...worldById.values()];
if (worlds.length < 2) fail("real trace did not reconstruct multiple hidden worlds");

const legalActions = fixture.state.legal_actions.map(String);
const transitions = [];
for (const world of worlds) {
  const base = buildBattle(world);
  const baseSnapshot = JSON.stringify(base);
  base.destroy();

  for (const action of legalActions) {
    const outcomes = [];
    for (let i = 0; i < ROOT_CHANCE_SAMPLES; i++) {
      const battle = cloneBattle(
        baseSnapshot,
        seed(i, 1_000 + legalActions.indexOf(action))
      );
      const logStart = battle.log.length;
      battle.makeChoices(rootChoice(action), `move ${lastOpponentMove()}`);
      const rootObservation = observation(battle, logStart);
      const successor = stateSummary(battle, world);
      const rootSnapshot = JSON.stringify(battle);
      const continuation = continuationValues(rootSnapshot);
      outcomes.push({
        probability: 1 / ROOT_CHANCE_SAMPLES,
        observation: rootObservation,
        successor,
        ...continuation,
      });
      battle.destroy();
    }
    transitions.push({world_id: world.world_id, action, outcomes});
  }
}

const benchFactor = factoredBenchAudit(worlds, legalActions, transitions);
const factoredHidden = benchFactor
  ? {
      [benchFactor.field]: {
        distribution: benchFactor.distribution,
        unread_actions: benchFactor.unread_actions,
        evidence: benchFactor.evidence,
      },
    }
  : {};

const marginalizedHidden = {
  [MARGINALIZED_VARIANT_FIELD]: {
    fields: ["opponent.active.moves", "opponent.active.tera_type"],
    generator_variant_count: variants.length,
    execution_variant_count: executionVariants.length,
    all_root_actions: legalActions,
    proof: {
      response_policy: `repeat observed ${lastOpponentMove()}`,
      hidden_fallback: "fail-closed",
      opponent_terastallized: false,
    },
  },
};

const declared = Object.fromEntries(
  legalActions.map(action => [action, declaredReads(action)])
);
const outputWorlds = worlds.map(world => ({
  world_id: world.world_id,
  weight: world.weight,
  hidden: world.hidden,
  provenance: {
    generator_count: world.generator_count,
    generator_rounds: GENERATOR_ROUNDS,
    generator_variant_count: world.generator_variant_count,
    marginalized_variant_digest: sha256(
      world.marginalized_remainders
        .map(entry => stable(entry))
        .sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)))
    ),
    opponent_max_hp: world.opponent_max_hp,
    hp_prior: "uniform-within-public-percentage-bucket",
  },
}));

process.stdout.write(JSON.stringify({
  schema: "azelficoast.real-belief-transition-oracle",
  schema_version: 1,
  source_fixture_id: fixture.fixture_id,
  source_artifact: source.source_artifact,
  showdown_commit: actualCommit,
  mechanics: {
    engine: "pokemon-showdown",
    format: "gen9customgame state reconstruction",
    root_chance_samples: ROOT_CHANCE_SAMPLES,
    continuation_chance_samples: CONTINUATION_CHANCE_SAMPLES,
    continuation_decision_horizons: CONTINUATION_DECISION_HORIZONS,
    chance_seed_family: CHANCE_SEED_FAMILY,
    opponent_response: `repeat observed ${lastOpponentMove()}`,
    continuation_scope:
      CONTINUATION_DECISION_HORIZONS === 1
        ? "all non-Tera player choices at the next decision"
        : "all non-Tera player choices at the next two public decisions",
    utility: "sum own team HP fractions minus opposing active HP fraction",
  },
  reconstruction: {
    generator_rounds: GENERATOR_ROUNDS,
    generator_matches: matched,
    generator_variant_count: variants.length,
    execution_variant_count: executionVariants.length,
    marginalized_generator_variant_fields: [
      "opponent.active.moves",
      "opponent.active.tera_type",
    ],
    marginalized_variant_rule:
      "sum prior mass across variants sharing species/ability/item/level/EVs/IVs; fail closed if the fixed public response would require any hidden fallback move",
    observed_opponent_moves: observedOpponentMoves(),
    hidden_world_count: outputWorlds.length,
    own_active_tera_type: OWN_ACTIVE_TERA_TYPE,
    opponent_bench_species: OPPONENT_BENCH_SPECIES,
    bench_species_mode: BENCH_PRIOR ? "factored-prior" : "concrete",
    bench_factor_support_count: BENCH_PRIOR
      ? BENCH_PRIOR.distribution.length
      : 0,
    bench_factor_unread_action_count: benchFactor
      ? benchFactor.unread_actions.length
      : 0,
    declared_read_mode: "conservative-external-oracle-boundary",
  },
  factored_hidden: factoredHidden,
  marginalized_hidden: marginalizedHidden,
  dependency_candidates: DEPENDENCY_CANDIDATES,
  declared_reads: declared,
  worlds: outputWorlds,
  legal_actions: legalActions,
  transitions,
}, null, 2) + "\n");
