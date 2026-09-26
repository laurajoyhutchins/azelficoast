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
let posteriorOnly = false;
let transitionProgramOnly = false;
let historicalShowdownCommit = null;
let generatorCacheDir = null;
for (let i = 2; i < argv.length; i++) {
  if (argv[i] === "--bench-prior") {
    benchPriorPath = argv[++i];
    if (!benchPriorPath) fail("--bench-prior requires a JSON path");
  } else if (argv[i] === "--posterior-only") {
    posteriorOnly = true;
  } else if (argv[i] === "--transition-program-only") {
    transitionProgramOnly = true;
  } else if (argv[i] === "--historical-showdown-commit") {
    historicalShowdownCommit = argv[++i];
    if (!historicalShowdownCommit) {
      fail("--historical-showdown-commit requires a 40-hex commit");
    }
  } else if (argv[i] === "--generator-cache-dir") {
    generatorCacheDir = argv[++i];
    if (!generatorCacheDir) fail("--generator-cache-dir requires a path");
  } else {
    fail("unknown argument: " + argv[i]);
  }
}
if (!showdownRoot || !fixturePath) {
  fail(
    "usage: probe_real_belief_trace.cjs SHOWDOWN_ROOT SOURCE_FIXTURE_JSON " +
    "[--bench-prior CONDITIONAL_TEAM_PRIOR_JSON] " +
    "[--posterior-only | --transition-program-only] " +
    "[--historical-showdown-commit COMMIT] " +
    "[--generator-cache-dir PATH]"
  );
}
if (posteriorOnly && transitionProgramOnly) {
  fail("--posterior-only and --transition-program-only are mutually exclusive");
}
if (historicalShowdownCommit && !posteriorOnly) {
  fail("--historical-showdown-commit is allowed only with --posterior-only");
}
if (generatorCacheDir && !posteriorOnly) {
  fail("--generator-cache-dir is allowed only with --posterior-only");
}
if (
  historicalShowdownCommit &&
  !/^[0-9a-f]{40}$/.test(historicalShowdownCommit)
) {
  fail("--historical-showdown-commit must be a 40-hex commit");
}

const PINNED_SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc";
const SHOWDOWN_COMMIT = historicalShowdownCommit || PINNED_SHOWDOWN_COMMIT;
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
const DEPENDENCY_CANDIDATES = [
  "opponent.active.item",
  "opponent.active.ability",
  "opponent.active.moves",
  "opponent.active.evs",
  "opponent.active.ivs",
  "opponent.active.exact_hp",
  "opponent.active.tera_type",
];
const BENCH_FACTOR_FIELD = "opponent.bench.species";
const PUBLIC_ROOT_DEPENDENCY_SCHEMA = 1;
const PUBLIC_BATTLE_CANDIDATES = [
  "turn",
  "requestState",
  "midTurn",
  "lastDamage",
  "lastMoveLine",
  "lastSuccessfulMoveThisTurn",
  "quickClawRoll",
];
const PUBLIC_POKEMON_CANDIDATES = [
  "hp",
  "status",
  "boosts",
  "fainted",
  "usedItemThisTurn",
  "ateBerry",
  "itemKnockedOff",
  "trapped",
  "maybeTrapped",
  "maybeDisabled",
  "maybeLocked",
  "transformed",
  "switchFlag",
  "forceSwitchFlag",
  "draggedIn",
  "newlySwitched",
  "beingCalledBack",
  "moveThisTurn",
  "statsRaisedThisTurn",
  "statsLoweredThisTurn",
  "hurtThisTurn",
  "lastDamage",
  "timesAttacked",
  "isActive",
  "activeTurns",
  "activeMoveActions",
  "previouslySwitchedIn",
  "truantTurn",
  "bondTriggered",
  "heroMessageDisplayed",
  "swordBoost",
  "shieldBoost",
  "syrupTriggered",
  "isStarted",
  "duringMove",
  "speed",
  "canTerastallize",
];

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
const {State} = require(path.join(showdownRoot, "dist", "sim", "state.js"));
const {Teams} = require(path.join(showdownRoot, "dist", "sim", "teams.js"));
const {
  applySuccessorDelta,
  successorDelta,
} = require(
  path.join(path.dirname(process.argv[1]), "transition_successor_delta.cjs")
);
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

function sha256PythonCanonical(value) {
  const encoded = JSON.stringify(stable(value)).replace(
    /[^\x00-\x7f]/g,
    character =>
      "\\u" + character.charCodeAt(0).toString(16).padStart(4, "0")
  );
  return crypto.createHash("sha256").update(encoded).digest("hex");
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

  const opponentSide = requireProtocolOpponentSide();
  const currentSpecies = toID(
    fixture.state.opponent_active && fixture.state.opponent_active.species
  );
  let activeSpecies = "";
  let last = null;
  for (const batch of fixture.protocol_prefix) {
    for (const message of batch) {
      if (message[0] !== "" || message.length < 2) continue;
      const actor = String(message[2] || "");
      if (!actor.startsWith(opponentSide)) continue;

      if (
        ["switch", "drag", "replace"].includes(message[1]) &&
        message.length >= 4
      ) {
        activeSpecies = toID(String(message[3] || "").split(",", 1)[0]);
        continue;
      }
      if (message[1] !== "move" || message.length < 4) continue;

      let moveSpecies = activeSpecies;
      if (!moveSpecies && actor.includes(":")) {
        moveSpecies = toID(actor.split(":", 2)[1]);
      }
      if (moveSpecies === currentSpecies) last = String(message[3]);
    }
  }
  return last ? toID(last) : null;
}

function normalizedOpponentPolicy() {
  const configured = source.opponent_policy;
  if (configured != null) {
    if (!configured || typeof configured !== "object" || Array.isArray(configured)) {
      fail("opponent_policy must be an object");
    }
    const kind = String(configured.kind || "");
    if (kind === "uniform-legal-moves") {
      if (configured.voluntary_switches === true) {
        fail("uniform-legal-moves does not yet model voluntary switches");
      }
      return {kind, voluntary_switches: false};
    }
    if (kind === "strategy-mixture") {
      if (configured.weighting !== "equal-active-strategies") {
        fail("strategy-mixture requires equal-active-strategies weighting");
      }
      if (
        configured.voluntary_switches != null &&
        typeof configured.voluntary_switches !== "boolean"
      ) {
        fail("strategy-mixture voluntary_switches must be boolean");
      }
      if (!Array.isArray(configured.strategies) || !configured.strategies.length) {
        fail("strategy-mixture requires at least one strategy");
      }
      const allowed = new Set([
        "simple-heuristics",
        "dirty-tricks",
        "max-damage",
        "repeat-observed-move",
        "uniform-legal-moves",
      ]);
      const seen = new Set();
      const strategies = configured.strategies.map(raw => {
        if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
          fail("strategy-mixture strategies must be objects");
        }
        const strategyKind = String(raw.kind || "");
        if (!allowed.has(strategyKind)) {
          fail("unsupported opponent strategy " + strategyKind);
        }
        if (seen.has(strategyKind)) {
          fail("strategy-mixture contains duplicate strategy " + strategyKind);
        }
        seen.add(strategyKind);
        if (strategyKind === "repeat-observed-move") {
          const move = toID(raw.move || source.opponent_response_move);
          if (!move) fail("repeat-observed-move requires a public move");
          return {kind: strategyKind, move};
        }
        return {kind: strategyKind};
      });
      return {
        kind,
        weighting: "equal-active-strategies",
        strategies,
        voluntary_switches: configured.voluntary_switches === true,
      };
    }
    fail("unsupported opponent_policy kind " + kind);
  }

  const strategies = [
    {kind: "simple-heuristics"},
    {kind: "dirty-tricks"},
    {kind: "max-damage"},
    {kind: "uniform-legal-moves"},
  ];
  const observed = lastOpponentMove();
  if (observed) {
    strategies.push({kind: "repeat-observed-move", move: observed});
  }
  return {
    kind: "strategy-mixture",
    weighting: "equal-active-strategies",
    strategies,
    voluntary_switches: true,
  };
}

const OPPONENT_POLICY = normalizedOpponentPolicy();

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

const GENERATOR_CACHE_SCHEMA =
  "azelficoast.randombattle-generator-population-cache";
const GENERATOR_CACHE_SCHEMA_VERSION = 1;
const GENERATOR_CACHE_EVENT_PREFIX = "azelficoast-generator-cache:";

function generatorPopulationMaterial(species) {
  return {
    schema: GENERATOR_CACHE_SCHEMA,
    schema_version: GENERATOR_CACHE_SCHEMA_VERSION,
    showdown_commit: actualCommit,
    format: "gen9randombattle",
    species,
    opponent_is_lead: source.opponent_is_lead === true,
    generator_rounds: GENERATOR_ROUNDS,
  };
}

function validateGeneratorPopulation(document, material) {
  if (
    !document ||
    document.schema !== GENERATOR_CACHE_SCHEMA ||
    document.schema_version !== GENERATOR_CACHE_SCHEMA_VERSION
  ) {
    fail("generator cache has an unexpected schema");
  }
  if (
    JSON.stringify(stable(document.material)) !==
    JSON.stringify(stable(material))
  ) {
    fail("generator cache identity does not match requested population");
  }
  if (!Array.isArray(document.variants) || !document.variants.length) {
    fail("generator cache contains no variants");
  }
  if (document.variants_sha256 !== sha256(document.variants)) {
    fail("generator cache variant digest mismatch");
  }
  const sampled = document.variants.reduce((sum, entry) => {
    const count = Number(entry && entry.count);
    if (!entry || !entry.set || !Number.isSafeInteger(count) || count <= 0) {
      fail("generator cache contains an invalid variant");
    }
    return sum + count;
  }, 0);
  if (sampled !== GENERATOR_ROUNDS) {
    fail(
      `generator cache sample count ${sampled} does not match ${GENERATOR_ROUNDS}`
    );
  }
  return document.variants;
}

function generateGeneratorPopulation(species, requested) {
  const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);
  const variants = new Map();
  for (let i = 0; i < GENERATOR_ROUNDS; i++) {
    generator.setSeed([i, i, i, i]);
    const set = generator.randomSet(
      species,
      {},
      source.opponent_is_lead === true,
      false
    );
    const semantic = {
      species: toID(set.species || requested),
      ability: set.ability,
      item: set.item,
      level: set.level,
      moves: [...set.moves].map(toID).sort(),
      evs: {...set.evs},
      ivs: {...set.ivs},
      teraType: set.teraType,
    };
    const key = JSON.stringify(stable(semantic));
    const prior = variants.get(key) || {set: semantic, count: 0};
    prior.count++;
    variants.set(key, prior);
  }
  return [...variants.values()];
}

function generatorPopulation(species, requested) {
  if (!generatorCacheDir) {
    return generateGeneratorPopulation(species, requested);
  }

  const material = generatorPopulationMaterial(species);
  const cacheKey = sha256(material);
  const cachePath = path.join(generatorCacheDir, cacheKey + ".json");
  fs.mkdirSync(generatorCacheDir, {recursive: true});

  if (fs.existsSync(cachePath)) {
    let document;
    try {
      document = JSON.parse(fs.readFileSync(cachePath, "utf8"));
    } catch (error) {
      fail(`cannot read generator cache ${cachePath}: ${error}`);
    }
    const variants = validateGeneratorPopulation(document, material);
    process.stderr.write(
      GENERATOR_CACHE_EVENT_PREFIX + "hit:" + cacheKey + "\n"
    );
    return variants;
  }

  const variants = generateGeneratorPopulation(species, requested);
  const document = {
    schema: GENERATOR_CACHE_SCHEMA,
    schema_version: GENERATOR_CACHE_SCHEMA_VERSION,
    material,
    variants,
    variants_sha256: sha256(variants),
  };
  const temporary = cachePath + "." + process.pid + ".tmp";
  fs.writeFileSync(temporary, JSON.stringify(stable(document)) + "\n", "utf8");
  try {
    fs.renameSync(temporary, cachePath);
  } catch (error) {
    if (!fs.existsSync(cachePath)) throw error;
    fs.unlinkSync(temporary);
    let winner;
    try {
      winner = JSON.parse(fs.readFileSync(cachePath, "utf8"));
    } catch (readError) {
      fail(`cannot read concurrent generator cache ${cachePath}: ${readError}`);
    }
    validateGeneratorPopulation(winner, material);
  }
  process.stderr.write(
    GENERATOR_CACHE_EVENT_PREFIX + "miss:" + cacheKey + "\n"
  );
  return variants;
}

function generatorVariants() {
  const requested = fixture.state.opponent_active.species;
  const species = resolveGeneratorSpecies(requested);
  const observed = new Set(observedOpponentMoves());
  const population = generatorPopulation(species, requested);
  const variants = new Map();
  const itemCounts = new Map();
  const publicAbility = toID(fixture.state.opponent_active.ability || "");
  const plausibleItemIds = Array.isArray(source.plausible_items)
    ? new Set(source.plausible_items.map(toID))
    : source.known_opponent_item
      ? new Set([toID(source.known_opponent_item)])
      : null;
  let matched = 0;

  for (const entry of population) {
    const set = entry.set;
    const count = Number(entry.count);
    if (
      toID(set.species || requested) !==
      toID(fixture.state.opponent_active.species)
    ) continue;
    if (Number(set.level) !== Number(fixture.state.opponent_active.level)) continue;
    if (publicAbility && toID(set.ability) !== publicAbility) continue;

    const moves = [...set.moves].map(toID).sort();
    if (![...observed].every(move => moves.includes(move))) continue;
    if (plausibleItemIds && !plausibleItemIds.has(toID(set.item))) continue;

    matched += count;
    itemCounts.set(set.item, (itemCounts.get(set.item) || 0) + count);
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
    prior.count += count;
    variants.set(key, prior);
  }
  if (!matched) fail("generator sweep produced no hidden worlds compatible with public evidence");
  return {
    matched,
    itemCounts: Object.fromEntries(
      [...itemCounts.entries()].sort(([left], [right]) => left.localeCompare(right))
    ),
    variants: [...variants.values()],
  };
}

function mechanicsProjectionVariantCount(variants) {
  const keys = new Set();
  for (const entry of variants) {
    keys.add(JSON.stringify(stable({
      species: entry.set.species,
      ability: entry.set.ability,
      item: entry.set.item,
      level: entry.set.level,
      ...(["uniform-legal-moves", "strategy-mixture"].includes(OPPONENT_POLICY.kind)
        ? {moves: entry.set.moves}
        : {}),
      evs: entry.set.evs,
      ivs: entry.set.ivs,
    })));
  }
  return keys.size;
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

function publicOpponentView(speciesName) {
  const species = toID(speciesName);
  return Object.values(fixture.state.opponent_team || {}).find(
    view => toID(view && view.species) === species
  ) || null;
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

function exactMaxHpForVariant(variant) {
  const species = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0])
    .dex.species.get(variant.species);
  if (!species.exists) {
    fail(`cannot resolve HP species ${variant.species}`);
  }

  const level = Number(variant.level);
  const iv = Number(variant.ivs && variant.ivs.hp);
  const ev = Number(variant.evs && variant.evs.hp);
  if (
    !Number.isSafeInteger(level) ||
    level <= 0 ||
    !Number.isSafeInteger(iv) ||
    iv < 0 ||
    iv > 31 ||
    !Number.isSafeInteger(ev) ||
    ev < 0 ||
    ev > 252
  ) {
    fail(
      "invalid HP mechanics input " +
      JSON.stringify({species: variant.species, level, iv, ev})
    );
  }

  if (species.maxHP) return Number(species.maxHP);

  // Pokémon Showdown sim/battle.ts::statModify for HP in Gen 9:
  // floor(floor(2 * base + IV + floor(EV / 4) + 100) * level / 100 + 10)
  const inner =
    2 * Number(species.baseStats.hp) +
    iv +
    Math.floor(ev / 4) +
    100;
  return Math.floor(Math.floor(inner) * level / 100 + 10);
}

function hpSupportForVariant(variant) {
  const maxhp = exactMaxHpForVariant(variant);
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

  for (const bench of battle.p2.pokemon.filter(pokemon => !pokemon.active)) {
    const view = publicOpponentView(bench.species.name);
    if (!view) continue;
    const hpFraction = Number(view.hp_fraction);
    if (Number.isFinite(hpFraction) && Math.abs(hpFraction - 1) <= 1e-12) {
      bench.hp = bench.maxhp;
    }
    const status = view.status;
    bench.status =
      status && status !== "FNT" ? toID(status) : "";
  }

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

function instrumentOpponentHiddenReads(battle) {
  const reads = new Set();
  const pokemon = battle.p2.active[0];
  if (!pokemon) {
    return {
      reads: () => [],
      restore: () => {},
    };
  }

  const restorers = [];
  const mark = (...fields) => {
    for (const field of fields) reads.add(field);
  };

  function trackDataProperty(object, key, fields) {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    if (
      !descriptor ||
      !Object.prototype.hasOwnProperty.call(descriptor, "value") ||
      descriptor.configurable === false
    ) {
      mark(...fields);
      return;
    }

    let value = descriptor.value;
    Object.defineProperty(object, key, {
      configurable: descriptor.configurable,
      enumerable: descriptor.enumerable,
      get() {
        mark(...fields);
        return value;
      },
      set(next) {
        if (descriptor.writable === false) {
          throw new TypeError("cannot write instrumented read-only property " + key);
        }
        value = next;
      },
    });
    restorers.push(() => {
      Object.defineProperty(object, key, {
        ...descriptor,
        value,
      });
    });
  }

  function trackStatObject(key) {
    const descriptor = Object.getOwnPropertyDescriptor(pokemon, key);
    if (
      !descriptor ||
      !Object.prototype.hasOwnProperty.call(descriptor, "value") ||
      descriptor.configurable === false ||
      !descriptor.value ||
      typeof descriptor.value !== "object"
    ) {
      mark("opponent.active.evs", "opponent.active.ivs");
      return;
    }
    const target = descriptor.value;
    const proxy = new Proxy(target, {
      get(object, property, receiver) {
        if (
          typeof property === "string" &&
          ["atk", "def", "spa", "spd", "spe"].includes(property)
        ) {
          mark("opponent.active.evs", "opponent.active.ivs");
        }
        return Reflect.get(object, property, receiver);
      },
      set(object, property, value, receiver) {
        return Reflect.set(object, property, value, receiver);
      },
    });
    pokemon[key] = proxy;
    restorers.push(() => {
      pokemon[key] = target;
    });
  }

  trackDataProperty(pokemon, "item", ["opponent.active.item"]);
  trackDataProperty(pokemon, "ability", ["opponent.active.ability"]);
  trackDataProperty(pokemon, "hp", ["opponent.active.exact_hp"]);
  trackDataProperty(
    pokemon,
    "teraType",
    ["opponent.active.tera_type"]
  );
  trackDataProperty(
    pokemon,
    "canTerastallize",
    ["opponent.active.tera_type"]
  );
  trackDataProperty(
    pokemon,
    "maxhp",
    ["opponent.active.evs", "opponent.active.ivs"]
  );
  trackDataProperty(
    pokemon,
    "baseMaxhp",
    ["opponent.active.evs", "opponent.active.ivs"]
  );
  trackStatObject("storedStats");
  trackStatObject("baseStoredStats");

  let restored = false;
  return {
    reads() {
      return [...reads].sort();
    },
    restore() {
      if (restored) return;
      restored = true;
      for (const restore of restorers.reverse()) restore();
    },
  };
}

function instrumentPublicRootReads(battle) {
  const reads = new Set();
  const restorers = [];
  let complete = true;

  function trackDataProperty(object, key, field) {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    if (
      !descriptor ||
      !Object.prototype.hasOwnProperty.call(descriptor, "value") ||
      descriptor.configurable === false
    ) {
      complete = false;
      return;
    }

    let value = descriptor.value;
    Object.defineProperty(object, key, {
      configurable: descriptor.configurable,
      enumerable: descriptor.enumerable,
      get() {
        reads.add(field);
        return value;
      },
      set(next) {
        if (descriptor.writable === false) {
          throw new TypeError("cannot write instrumented read-only property " + field);
        }
        value = next;
      },
    });
    restorers.push(() => {
      Object.defineProperty(object, key, {
        ...descriptor,
        value,
      });
    });
  }

  for (const key of PUBLIC_BATTLE_CANDIDATES) {
    trackDataProperty(battle, key, "battle." + key);
  }
  for (const [sideIndex, side] of battle.sides.entries()) {
    const sideId = "p" + String(sideIndex + 1);
    for (const [pokemonIndex, pokemon] of side.pokemon.entries()) {
      for (const key of PUBLIC_POKEMON_CANDIDATES) {
        trackDataProperty(
          pokemon,
          key,
          sideId + ".pokemon." + String(pokemonIndex) + "." + key
        );
      }
    }
  }

  let restored = false;
  return {
    reads() {
      return [...reads].sort();
    },
    complete() {
      return complete;
    },
    restore() {
      if (restored) return;
      restored = true;
      for (const restore of restorers.reverse()) restore();
    },
  };
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

function uniformMoveDistribution(legalMoves, mode) {
  const probability = 1 / legalMoves.length;
  return legalMoves.map(move => ({
    choice: `move ${move}`,
    probability,
    mode,
  }));
}

function exactTieDistribution(scored, mode) {
  if (!scored.length) return [];
  const best = Math.max(...scored.map(row => row.score));
  const choices = scored
    .filter(row => Math.abs(row.score - best) <= 1e-12)
    .map(row => row.move)
    .sort();
  return uniformMoveDistribution(choices, mode);
}

function expectedHitMultiplier(move) {
  if (!Array.isArray(move.multihit)) return 1;
  if (move.multihit.length !== 2) return 1;
  const low = Number(move.multihit[0]);
  const high = Number(move.multihit[1]);
  if (!Number.isFinite(low) || !Number.isFinite(high)) return 1;
  // This is a policy heuristic, not a mechanics oracle. Exact turn execution remains
  // Showdown-owned. The mean only ranks candidate attacks for the opponent prior.
  return (low + high) / 2;
}

function moveDamageHeuristic(battle, moveId) {
  const attacker = battle.p2.active[0];
  const defender = battle.p1.active[0];
  if (!attacker || !defender) return 0;
  const move = battle.dex.moves.get(moveId);
  if (!move.exists || move.category === "Status" || Number(move.basePower) <= 0) {
    return 0;
  }
  const rawAccuracy = move.accuracy === true ? 1 : Number(move.accuracy) / 100;
  const accuracy = Number.isFinite(rawAccuracy) ? rawAccuracy : 1;
  const stab = attacker.getTypes().includes(move.type) ? 1.5 : 1;
  const immune = battle.dex.getImmunity(move, defender);
  if (!immune) return 0;
  const effectiveness = 2 ** battle.dex.getEffectiveness(move, defender);
  const attackStat =
    move.category === "Physical"
      ? Number(attacker.storedStats.atk)
      : Number(attacker.storedStats.spa);
  const defenseStat =
    move.category === "Physical"
      ? Number(defender.storedStats.def)
      : Number(defender.storedStats.spd);
  const statRatio =
    Number.isFinite(attackStat) &&
    Number.isFinite(defenseStat) &&
    defenseStat > 0
      ? attackStat / defenseStat
      : 1;
  return (
    Number(move.basePower) *
    Math.max(0, accuracy) *
    expectedHitMultiplier(move) *
    stab *
    effectiveness *
    statRatio
  );
}

function maybeTeraDamageDistribution(battle, distribution) {
  const attacker = battle.p2.active[0];
  const defender = battle.p1.active[0];
  const teraType = attacker && attacker.canTerastallize;
  if (!attacker || !defender || !teraType) return distribution;

  const threatTypes = [...new Set(
    (defender.moveSlots || [])
      .map(slot => battle.dex.moves.get(slot.id || slot.move))
      .filter(move => move.exists && move.category !== "Status")
      .map(move => move.type)
  )];
  if (!threatTypes.length) return distribution;
  const currentWorst = Math.max(
    ...threatTypes.map(type =>
      battle.dex.getImmunity(type, attacker)
        ? 2 ** battle.dex.getEffectiveness(type, attacker)
        : 0
    )
  );
  const teraWorst = Math.max(
    ...threatTypes.map(type =>
      battle.dex.getImmunity(type, String(teraType))
        ? 2 ** battle.dex.getEffectiveness(type, String(teraType))
        : 0
    )
  );

  return distribution.flatMap(row => {
    const moveId = String(row.choice).replace(/^move\s+/, "");
    const move = battle.dex.moves.get(moveId);
    const offensive = move.exists && toID(move.type) === toID(teraType);
    const defensive = teraWorst + 1e-12 < currentWorst;
    if (!offensive && !defensive) return [row];
    return [
      {...row, probability: row.probability * 0.5},
      {
        choice: row.choice + " terastallize",
        probability: row.probability * 0.5,
        mode: row.mode + "-tera",
      },
    ];
  });
}

function maxDamageDistribution(battle, legalMoves) {
  const scored = legalMoves.map(move => ({
    move,
    score: moveDamageHeuristic(battle, move),
  }));
  const best = Math.max(...scored.map(row => row.score));
  if (!(best > 0)) return uniformMoveDistribution(legalMoves, "max-damage-fallback");
  return maybeTeraDamageDistribution(
    battle,
    exactTieDistribution(scored, "max-damage")
  );
}

function matchupPressure(battle, attacker, defender) {
  const offensive = Math.max(
    ...attacker.getTypes().map(type =>
      battle.dex.getImmunity(type, defender)
        ? 2 ** battle.dex.getEffectiveness(type, defender)
        : 0
    )
  );
  const defensive = Math.max(
    ...defender.getTypes().map(type =>
      battle.dex.getImmunity(type, attacker)
        ? 2 ** battle.dex.getEffectiveness(type, attacker)
        : 0
    )
  );
  const attackerSpeed = Number(attacker.species?.baseStats?.spe || 0);
  const defenderSpeed = Number(defender.species?.baseStats?.spe || 0);
  const speed = attackerSpeed > defenderSpeed ? 0.1 : attackerSpeed < defenderSpeed ? -0.1 : 0;
  const publicView = publicOpponentView(attacker.species?.name);
  const publicHp = Number(publicView && publicView.hp_fraction);
  const hp = Number.isFinite(publicHp)
    ? publicHp
    : attacker.maxhp > 0
      ? attacker.hp / attacker.maxhp
      : 0;
  return offensive - defensive + speed + 0.4 * hp;
}

function legalOpponentSwitches(battle) {
  const request = battle.p2.activeRequest;
  if (
    !OPPONENT_POLICY.voluntary_switches ||
    !request ||
    !request.active ||
    request.active[0]?.trapped
  ) {
    return [];
  }
  return battle.p2.pokemon
    .filter(pokemon => {
      if (pokemon.hp <= 0 || pokemon.active) return false;
      const view = publicOpponentView(pokemon.species.name);
      if (!view || view.fainted) return false;
      const hpFraction = Number(view.hp_fraction);
      // Full health has one exact public interpretation. Damaged bench HP is still
      // percentage-censored and needs its own posterior before it can be switched in
      // without inventing an exact value.
      return Number.isFinite(hpFraction) && Math.abs(hpFraction - 1) <= 1e-12;
    })
    .map(pokemon => ({
      pokemon,
      choice: `switch ${pokemon.position + 1}`,
    }))
    .sort((left, right) => left.choice.localeCompare(right.choice));
}

function voluntarySwitchDistribution(battle, legalSwitches, mode, {urgent = false} = {}) {
  if (!legalSwitches.length) return null;
  const active = battle.p2.active[0];
  const defender = battle.p1.active[0];
  if (!active || !defender) return null;

  const activeScore = matchupPressure(battle, active, defender);
  const activeBoosts = active.boosts || {};
  const deeplyDropped =
    Number(activeBoosts.def || 0) <= -3 ||
    Number(activeBoosts.spd || 0) <= -3 ||
    Number(activeBoosts.atk || 0) <= -3 ||
    Number(activeBoosts.spa || 0) <= -3;

  const scored = legalSwitches.map(row => ({
    ...row,
    score: matchupPressure(battle, row.pokemon, defender),
  }));
  const best = Math.max(...scored.map(row => row.score));
  const shouldSwitch =
    urgent ||
    deeplyDropped ||
    (activeScore < -0.75 && best >= activeScore + 0.75);
  if (!shouldSwitch) return null;

  const choices = scored
    .filter(row => Math.abs(row.score - best) <= 1e-12)
    .map(row => row.choice)
    .sort();
  const probability = 1 / choices.length;
  return choices.map(choice => ({choice, probability, mode}));
}

const HAZARD_MOVES = new Map([
  ["spikes", "spikes"],
  ["stealthrock", "stealthrock"],
  ["stickyweb", "stickyweb"],
  ["toxicspikes", "toxicspikes"],
]);
const HAZARD_REMOVAL_MOVES = new Set(["defog", "rapidspin"]);

function simpleHeuristicsDistribution(battle, legalMoves, legalSwitches) {
  const attacker = battle.p2.active[0];
  const defender = battle.p1.active[0];
  if (!attacker || !defender) {
    return uniformMoveDistribution(legalMoves, "simple-heuristics-fallback");
  }

  const voluntarySwitch = voluntarySwitchDistribution(
    battle,
    legalSwitches,
    "simple-heuristics-switch"
  );
  if (voluntarySwitch) return voluntarySwitch;

  const moveObjects = legalMoves.map(move => battle.dex.moves.get(move));

  const hazards = moveObjects
    .filter(move => {
      const condition = HAZARD_MOVES.get(move.id);
      return condition && !battle.p1.sideConditions[condition];
    })
    .map(move => move.id)
    .sort();
  if (hazards.length) {
    return uniformMoveDistribution(hazards, "simple-heuristics-hazard");
  }

  if (Object.keys(battle.p2.sideConditions).length) {
    const removals = moveObjects
      .filter(move => HAZARD_REMOVAL_MOVES.has(move.id))
      .map(move => move.id)
      .sort();
    if (removals.length) {
      return uniformMoveDistribution(removals, "simple-heuristics-hazard-removal");
    }
  }

  const hpFraction = attacker.maxhp > 0 ? attacker.hp / attacker.maxhp : 0;
  if (hpFraction >= 0.8) {
    const setup = moveObjects
      .filter(move => {
        if (move.target !== "self" || !move.boosts) return false;
        const positive = Object.entries(move.boosts)
          .filter(([, value]) => Number(value) > 0);
        if (!positive.length) return false;
        const total = positive.reduce((sum, [, value]) => sum + Number(value), 0);
        const room = positive.some(
          ([stat]) => Number(attacker.boosts[stat] || 0) < 4
        );
        return total >= 2 && room;
      })
      .map(move => move.id)
      .sort();
    if (setup.length) {
      return uniformMoveDistribution(setup, "simple-heuristics-setup");
    }
  }

  if (hpFraction <= 0.5) {
    const recovery = moveObjects
      .filter(move => Array.isArray(move.heal) && Number(move.heal[0]) > 0)
      .map(move => move.id)
      .sort();
    if (recovery.length) {
      return uniformMoveDistribution(recovery, "simple-heuristics-recovery");
    }
  }

  return maxDamageDistribution(battle, legalMoves).map(row => ({
    ...row,
    mode: "simple-heuristics-damage",
  }));
}

const DIRTY_ANTI_SETUP_MOVES = new Set([
  "clearsmog",
  "encore",
  "haze",
  "roar",
  "taunt",
  "topsyturvy",
  "whirlwind",
]);
const DIRTY_DENIAL_MOVES = new Set([
  "disable",
  "encore",
  "knockoff",
  "switcheroo",
  "taunt",
  "torment",
  "trick",
]);
const DIRTY_CHIP_MOVES = new Set([
  "firespin",
  "infestation",
  "leechseed",
  "magmastorm",
  "saltcure",
  "sandtomb",
  "whirlpool",
]);
const DIRTY_STALL_MOVES = new Set([
  "banefulbunker",
  "burningbulwark",
  "detect",
  "kingsshield",
  "protect",
  "silktrap",
  "substitute",
]);

function moveCanInflictMajorStatus(move) {
  if (move.status) return true;
  if (move.secondary && move.secondary.status) return true;
  if (Array.isArray(move.secondaries)) {
    return move.secondaries.some(secondary => secondary && secondary.status);
  }
  return false;
}

function dirtyTricksDistribution(battle, legalMoves, legalSwitches) {
  const attacker = battle.p2.active[0];
  const defender = battle.p1.active[0];
  if (!attacker || !defender) {
    return uniformMoveDistribution(legalMoves, "dirty-tricks-fallback");
  }

  const moves = legalMoves.map(move => battle.dex.moves.get(move));
  const choose = (candidates, mode) => {
    const ids = candidates.map(move => move.id).filter(Boolean).sort();
    return ids.length ? uniformMoveDistribution(ids, mode) : null;
  };

  const positiveBoosts = Object.values(defender.boosts || {})
    .reduce((sum, value) => sum + Math.max(0, Number(value) || 0), 0);
  if (positiveBoosts >= 2) {
    const antiSetup = choose(
      moves.filter(move => DIRTY_ANTI_SETUP_MOVES.has(move.id)),
      "dirty-tricks-anti-setup"
    );
    if (antiSetup) return antiSetup;
  }

  const activeHpFraction = attacker.maxhp > 0 ? attacker.hp / attacker.maxhp : 0;
  const voluntarySwitch = voluntarySwitchDistribution(
    battle,
    legalSwitches,
    "dirty-tricks-switch",
    {urgent: activeHpFraction <= 0.25}
  );
  if (voluntarySwitch) return voluntarySwitch;

  const defenderHpFraction = defender.maxhp > 0 ? defender.hp / defender.maxhp : 1;
  if (defenderHpFraction <= 0.35) {
    const priority = moves.filter(
      move => Number(move.priority) > 0 && moveDamageHeuristic(battle, move.id) > 0
    );
    if (priority.length) {
      return maxDamageDistribution(
        battle,
        priority.map(move => move.id)
      ).map(row => ({...row, mode: "dirty-tricks-priority-cleanup"}));
    }
  }

  if (!defender.status) {
    const status = choose(
      moves.filter(move => moveCanInflictMajorStatus(move)),
      "dirty-tricks-status"
    );
    if (status) return status;
  }

  const denial = choose(
    moves.filter(move => DIRTY_DENIAL_MOVES.has(move.id)),
    "dirty-tricks-denial"
  );
  if (denial) return denial;

  const chip = choose(
    moves.filter(move => DIRTY_CHIP_MOVES.has(move.id)),
    "dirty-tricks-chip"
  );
  if (chip) return chip;

  const hasResidualPressure = Boolean(
    defender.status ||
    defender.volatiles?.leechseed ||
    defender.volatiles?.saltcure ||
    defender.volatiles?.partiallytrapped
  );
  if (hasResidualPressure) {
    const stall = choose(
      moves.filter(move => DIRTY_STALL_MOVES.has(move.id)),
      "dirty-tricks-stall"
    );
    if (stall) return stall;
  }

  const hazards = choose(
    moves.filter(move => {
      const condition = HAZARD_MOVES.get(move.id);
      return condition && !battle.p1.sideConditions[condition];
    }),
    "dirty-tricks-hazard"
  );
  if (hazards) return hazards;

  return maxDamageDistribution(battle, legalMoves).map(row => ({
    ...row,
    mode: "dirty-tricks-damage",
  }));
}

function repeatObservedDistribution(strategy, legalMoves) {
  const move = toID(strategy.move);
  if (!move || !legalMoves.includes(move)) return null;
  return [{
    choice: `move ${move}`,
    probability: 1,
    mode: "repeat-observed-move",
  }];
}

function strategyDistribution(battle, legalMoves, legalSwitches, strategy) {
  if (strategy.kind === "uniform-legal-moves") {
    return uniformMoveDistribution(legalMoves, "uniform-legal-moves");
  }
  if (strategy.kind === "dirty-tricks") {
    return dirtyTricksDistribution(battle, legalMoves, legalSwitches);
  }
  if (strategy.kind === "max-damage") {
    return maxDamageDistribution(battle, legalMoves);
  }
  if (strategy.kind === "simple-heuristics") {
    return simpleHeuristicsDistribution(battle, legalMoves, legalSwitches);
  }
  if (strategy.kind === "repeat-observed-move") {
    return repeatObservedDistribution(strategy, legalMoves);
  }
  fail("unsupported normalized opponent strategy " + strategy.kind);
}

function equalStrategyMixture(battle, legalMoves, legalSwitches) {
  const active = [];
  for (const strategy of OPPONENT_POLICY.strategies) {
    const distribution = strategyDistribution(
      battle,
      legalMoves,
      legalSwitches,
      strategy
    );
    if (distribution && distribution.length) {
      active.push({strategy, distribution});
    }
  }
  if (!active.length) {
    return uniformMoveDistribution(legalMoves, "strategy-mixture-fallback");
  }

  const mass = new Map();
  const modes = new Map();
  const strategyWeight = 1 / active.length;
  for (const {strategy, distribution} of active) {
    for (const row of distribution) {
      mass.set(
        row.choice,
        (mass.get(row.choice) || 0) + strategyWeight * row.probability
      );
      const priorModes = modes.get(row.choice) || new Set();
      priorModes.add(strategy.kind);
      modes.set(row.choice, priorModes);
    }
  }

  return [...mass.entries()]
    .map(([choice, probability]) => ({
      choice,
      probability,
      mode: "strategy-mixture:" + [...modes.get(choice)].sort().join("+"),
    }))
    .sort((left, right) => left.choice.localeCompare(right.choice));
}

function opponentActionDistribution(battle, hiddenReads = null) {
  const request = battle.p2.activeRequest;
  if (!request || request.wait) {
    return [{choice: "", probability: 1, mode: "wait"}];
  }
  if (request.forceSwitch) {
    if (hiddenReads && BENCH_PRIOR) hiddenReads.add(BENCH_FACTOR_FIELD);
    const switches = battle.p2.pokemon
      .filter(pokemon => pokemon.hp && !pokemon.active)
      .map(pokemon => `switch ${pokemon.position + 1}`)
      .sort();
    if (!switches.length) {
      return [{choice: "", probability: 1, mode: "forced-switch-unavailable"}];
    }
    const probability = 1 / switches.length;
    return switches.map(choice => ({
      choice,
      probability,
      mode: "uniform-forced-switch",
    }));
  }
  if (!request.active) {
    return [{choice: "", probability: 1, mode: "no-action"}];
  }

  const legalMoves = [...new Set(
    (request.active[0].moves || [])
      .filter(move => !move.disabled)
      .map(move => toID(move.id))
      .filter(Boolean)
  )].sort();
  if (!legalMoves.length) fail("opponent policy found no legal move choices");
  const legalSwitches = legalOpponentSwitches(battle);

  if (hiddenReads) {
    hiddenReads.add("opponent.active.moves");
    if (OPPONENT_POLICY.kind === "strategy-mixture") {
      hiddenReads.add("opponent.active.evs");
      hiddenReads.add("opponent.active.ivs");
      hiddenReads.add("opponent.active.exact_hp");
      hiddenReads.add("opponent.active.tera_type");
      if (legalSwitches.length && BENCH_PRIOR) {
        hiddenReads.add(BENCH_FACTOR_FIELD);
      }
    }
  }

  if (OPPONENT_POLICY.kind === "uniform-legal-moves") {
    return uniformMoveDistribution(legalMoves, "uniform-legal-moves");
  }
  if (OPPONENT_POLICY.kind === "strategy-mixture") {
    return equalStrategyMixture(battle, legalMoves, legalSwitches);
  }
  fail("unsupported normalized opponent policy " + OPPONENT_POLICY.kind);
}

function opponentDistributionForSnapshot(
  snapshot,
  hiddenReads = null,
  publicReads = null,
  publicTraceState = null
) {
  const probe = Battle.fromJSON(snapshot);
  probe.restart(() => {});
  const publicTrace = publicReads ? instrumentPublicRootReads(probe) : null;
  try {
    return opponentActionDistribution(probe, hiddenReads);
  } finally {
    if (publicTrace) {
      for (const field of publicTrace.reads()) publicReads.add(field);
      if (publicTraceState && !publicTrace.complete()) {
        publicTraceState.complete = false;
      }
      publicTrace.restore();
    }
    probe.destroy();
  }
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
          maxhp:
            battle.p2.active[0].maxhp === Number(world.opponent_max_hp)
              ? "<unchanged>"
              : battle.p2.active[0].maxhp,
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

function expectedContinuationValue(snapshot, choice, hiddenReads, seedSalt) {
  const responses = opponentDistributionForSnapshot(snapshot, hiddenReads);
  let total = 0;
  for (const [responseIndex, response] of responses.entries()) {
    let responseSum = 0;
    for (let i = 0; i < CONTINUATION_CHANCE_SAMPLES; i++) {
      const battle = cloneBattle(
        snapshot,
        seed(i, seedSalt + sha256(choice).charCodeAt(0) + responseIndex * 4099)
      );
      battle.makeChoices(choice, response.choice);
      responseSum += utility(battle);
      battle.destroy();
    }
    total += response.probability * (responseSum / CONTINUATION_CHANCE_SAMPLES);
  }
  return total;
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
    values[choice] = expectedContinuationValue(snapshot, choice, hiddenReads, seedSalt);
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
    return {terminal_utility: value, hidden_reads: []};
  }
  probe.destroy();

  if (CONTINUATION_DECISION_HORIZONS === 1) {
    const values = {};
    for (const choice of choices) {
      values[choice] = expectedContinuationValue(rootSnapshot, choice, hiddenReads, 10_000);
    }
    return {continuations: values, hidden_reads: [...hiddenReads].sort()};
  }

  const continuationTransitions = {};
  for (const choice of choices) {
    const outcomes = [];
    const firstSalt = sha256(choice).charCodeAt(0);
    const responses = opponentDistributionForSnapshot(rootSnapshot, hiddenReads);
    for (const [responseIndex, response] of responses.entries()) {
      for (let i = 0; i < CONTINUATION_CHANCE_SAMPLES; i++) {
        const battle = cloneBattle(
          rootSnapshot,
          seed(i, 10_000 + firstSalt + responseIndex * 4099)
        );
        const logStart = battle.log.length;
        battle.makeChoices(choice, response.choice);
        const nextObservation = observation(battle, logStart);
        const nextSnapshot = JSON.stringify(battle);
        const leaf = leafContinuationValues(
          nextSnapshot,
          hiddenReads,
          20_000 + firstSalt * 257 + responseIndex * 4099 + i * 17
        );
        outcomes.push({
          probability: response.probability / CONTINUATION_CHANCE_SAMPLES,
          observation: nextObservation,
          opponent_action: response.choice,
          opponent_policy_mode: response.mode,
          ...leaf,
        });
        battle.destroy();
      }
    }
    continuationTransitions[choice] = outcomes;
  }

  return {
    continuation_transitions: continuationTransitions,
    hidden_reads: [...hiddenReads].sort(),
  };
}

function declaredReads(action) {
  const reads = new Set();
  for (const transition of transitions) {
    if (transition.action !== action) continue;
    for (const outcome of transition.outcomes) {
      for (const field of outcome.transition_reads || []) reads.add(field);
    }
  }
  return [...reads].sort();
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
    const base = buildBattle(world, opponentBenchSet(benchSpecies));
    const snapshot = JSON.stringify(base);
    base.destroy();
    const responses = opponentDistributionForSnapshot(snapshot);
    const outcomes = [];
    for (const [responseIndex, response] of responses.entries()) {
      const battle = cloneBattle(
        snapshot,
        seed(0, 50_000 + legalActions.indexOf(action) + responseIndex * 4099)
      );
      const logStart = battle.log.length;
      battle.makeChoices(rootChoice(action), response.choice);
      outcomes.push({
        probability: response.probability,
        observation: observation(battle, logStart),
        successor: stateSummary(battle, world),
      });
      battle.destroy();
    }
    outcomes.sort((left, right) =>
      JSON.stringify(stable(left)).localeCompare(JSON.stringify(stable(right)))
    );
    return sha256(outcomes);
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
        "a bench-species read is recorded only when the opponent policy actually selects a forced switch during the bounded continuation",
    },
  };
}

const {matched, itemCounts, variants} = generatorVariants();
const mechanicsProjectionCount = mechanicsProjectionVariantCount(variants);
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
for (const entry of variants) {
  const {maxhp, support} = hpSupportForVariant(entry.set);
  for (const exactHp of support) {
    const hidden = {
      "opponent.active.item": entry.set.item,
      "opponent.active.ability": entry.set.ability,
      "opponent.active.moves": entry.set.moves,
      "opponent.active.tera_type": entry.set.teraType,
      "opponent.active.evs": entry.set.evs,
      "opponent.active.ivs": entry.set.ivs,
      "opponent.active.exact_hp": exactHp,
    };
    const worldId = sha256(hidden);
    if (worldById.has(worldId)) {
      fail(`duplicate semantic hidden-world identity: ${worldId}`);
    }
    worldById.set(worldId, {
      world_id: worldId,
      weight: (entry.count / matched) / support.length,
      hidden,
      variant: entry.set,
      exactHp,
      opponent_max_hp: maxhp,
      generator_count: entry.count,
    });
  }
}
const worlds = [...worldById.values()];
if (worlds.length < 2) fail("real trace did not reconstruct multiple hidden worlds");

const legalActions = fixture.state.legal_actions.map(String);
const outputWorlds = worlds.map(world => ({
  world_id: world.world_id,
  weight: world.weight,
  hidden: world.hidden,
  provenance: {
    generator_count: world.generator_count,
    generator_rounds: GENERATOR_ROUNDS,
    generator_variant_count: 1,
    opponent_max_hp: world.opponent_max_hp,
    hp_prior: "uniform-within-public-percentage-bucket",
  },
}));

if (posteriorOnly) {
  process.stdout.write(JSON.stringify({
    schema: "azelficoast.live-belief-posterior",
    schema_version: 1,
    source_fixture_id: fixture.fixture_id,
    showdown_commit: actualCommit,
    conditioned_on_public_history: true,
    realized_hidden_state_revealed: false,
    reconstruction: {
      generator_rounds: GENERATOR_ROUNDS,
      generator_matches: matched,
      generator_variant_count: variants.length,
      mechanics_projection_variant_count: mechanicsProjectionCount,
      mechanics_projection_fields:
        OPPONENT_POLICY.kind === "uniform-legal-moves"
          ? ["opponent.active.tera_type"]
          : ["opponent.active.moves", "opponent.active.tera_type"],
      mechanics_projection_scope:
        "execution optimization only; semantic posterior support retains every generator variant",
      observed_opponent_moves: observedOpponentMoves(),
      known_opponent_item: source.known_opponent_item || null,
      opponent_policy: OPPONENT_POLICY,
      hidden_world_count: outputWorlds.length,
      own_active_tera_type: OWN_ACTIVE_TERA_TYPE,
      opponent_bench_species: OPPONENT_BENCH_SPECIES,
    },
    legal_actions: legalActions,
    worlds: outputWorlds,
  }, null, 2) + "\n");
  process.exit(0);
}
function rootSnapshotForWorld(world) {
  const base = buildBattle(world);
  const snapshot = JSON.stringify(base);
  base.destroy();
  return snapshot;
}

function immediateWholeTurn(world, action, baseSnapshot = null) {
  const rootSnapshot =
    baseSnapshot == null ? rootSnapshotForWorld(world) : baseSnapshot;

  const outcomes = [];
  const reads = new Set();
  const publicReads = new Set();
  const publicTraceState = {complete: true};
  const policyReads = new Set();
  const responses = opponentDistributionForSnapshot(
    rootSnapshot,
    policyReads,
    publicReads,
    publicTraceState
  );
  for (const field of policyReads) reads.add(field);

  for (const [responseIndex, response] of responses.entries()) {
    for (let i = 0; i < ROOT_CHANCE_SAMPLES; i++) {
      const battle = cloneBattle(
        rootSnapshot,
        seed(i, 1_000 + legalActions.indexOf(action) + responseIndex * 4099)
      );
      const logStart = battle.log.length;
      const hiddenTrace = instrumentOpponentHiddenReads(battle);
      const publicTrace = instrumentPublicRootReads(battle);
      let outcome;
      try {
        let transitionReads;
        let observationRecord;
        let legalActionRecord;
        try {
          try {
            battle.makeChoices(rootChoice(action), response.choice);
          } finally {
            hiddenTrace.restore();
          }
          transitionReads = [
            ...new Set([...policyReads, ...hiddenTrace.reads()]),
          ].sort();
          for (const field of transitionReads) reads.add(field);
          observationRecord = observation(battle, logStart);
          legalActionRecord = battle.ended
            ? ["<terminal>"]
            : battle.p1.activeRequest?.wait
              ? ["<wait>"]
              : legalP1Continuations(battle);
          for (const field of publicTrace.reads()) publicReads.add(field);
          if (!publicTrace.complete()) publicTraceState.complete = false;
        } finally {
          publicTrace.restore();
        }

        // Successor rendering is intentionally outside the mechanics read trace.
        // A projected cache entry stores this as a delta against the current root
        // and rehydrates unchanged public leaves from the next root.
        outcome = {
          probability: response.probability / ROOT_CHANCE_SAMPLES,
          observation: observationRecord,
          successor: stateSummary(battle, world),
          legal_actions: legalActionRecord,
          transition_reads: transitionReads,
          opponent_action: response.choice,
          opponent_policy_mode: response.mode,
        };
      } finally {
        battle.destroy();
      }
      outcomes.push(outcome);
    }
  }
  return {
    outcomes,
    opponent_action_branch_count: responses.length,
    showdown_turn_executions: outcomes.length,
    read_fields: [...reads].sort(),
    public_read_fields: [...publicReads].sort(),
    public_trace_complete: publicTraceState.complete,
    semantic_hash: semanticHashForOutcomes(outcomes),
  };
}

function projectionKey(world, fields) {
  return fields.map(field => JSON.stringify(stable(world.hidden[field])));
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function semanticHashForOutcomes(outcomes) {
  const semantics = outcomes
    .map(outcome => ({
      probability: outcome.probability,
      observation: outcome.observation,
      successor: outcome.successor,
      legal_actions: outcome.legal_actions,
    }))
    .sort((left, right) =>
      JSON.stringify(stable(left)).localeCompare(JSON.stringify(stable(right)))
    );
  return sha256PythonCanonical(semantics);
}

function publicRootMaterial(snapshot, world) {
  const prepared = Battle.fromJSON(snapshot);
  prepared.restart(() => {});
  let state;
  let successor;
  try {
    state = JSON.parse(JSON.stringify(prepared));
    state.__azelficoast_active_requests = prepared.sides.map(side =>
      side.activeRequest == null ? side.activeRequest : cloneJson(side.activeRequest)
    );
    successor = stateSummary(prepared, world);
  } finally {
    prepared.destroy();
  }

  State.normalize(state);
  const exactHash = sha256({
    schema: PUBLIC_ROOT_DEPENDENCY_SCHEMA,
    state,
  });
  const masked = cloneJson(state);
  const values = {};
  let complete = true;
  const markerValue = {"__azelficoast_public_dependency__": true};

  function mask(object, key, field) {
    if (
      !object ||
      typeof object !== "object" ||
      !Object.prototype.hasOwnProperty.call(object, key)
    ) {
      complete = false;
      return;
    }
    values[field] = cloneJson(object[key]);
    object[key] = markerValue;
  }

  for (const key of PUBLIC_BATTLE_CANDIDATES) {
    mask(masked, key, "battle." + key);
  }
  if (!Array.isArray(masked.sides)) {
    complete = false;
  } else {
    for (let sideIndex = 0; sideIndex < masked.sides.length; sideIndex++) {
      const side = masked.sides[sideIndex];
      const sideId = "p" + String(sideIndex + 1);
      if (!side || !Array.isArray(side.pokemon)) {
        complete = false;
        continue;
      }
      for (let pokemonIndex = 0; pokemonIndex < side.pokemon.length; pokemonIndex++) {
        const pokemon = side.pokemon[pokemonIndex];
        for (const key of PUBLIC_POKEMON_CANDIDATES) {
          mask(
            pokemon,
            key,
            sideId + ".pokemon." + String(pokemonIndex) + "." + key
          );
        }
      }
    }
  }

  return {
    complete,
    exact_hash: exactHash,
    static_hash: sha256({
      schema: PUBLIC_ROOT_DEPENDENCY_SCHEMA,
      state: masked,
    }),
    values,
    successor,
  };
}

function publicReadProjection(material, fields) {
  const projection = [];
  for (const field of fields) {
    if (!Object.prototype.hasOwnProperty.call(material.values, field)) {
      return null;
    }
    projection.push([field, material.values[field]]);
  }
  return projection;
}

function counterfactualWorld(baseWorld, donorWorld, field) {
  const world = cloneJson(baseWorld);
  world.hidden[field] = cloneJson(donorWorld.hidden[field]);

  if (field === "opponent.active.item") {
    world.variant.item = donorWorld.variant.item;
  } else if (field === "opponent.active.ability") {
    world.variant.ability = donorWorld.variant.ability;
  } else if (field === "opponent.active.moves") {
    world.variant.moves = cloneJson(donorWorld.variant.moves);
  } else if (field === "opponent.active.evs") {
    world.variant.evs = cloneJson(donorWorld.variant.evs);
  } else if (field === "opponent.active.ivs") {
    world.variant.ivs = cloneJson(donorWorld.variant.ivs);
  } else if (field === "opponent.active.exact_hp") {
    world.exactHp = donorWorld.exactHp;
  } else {
    fail("cannot intervene on unknown hidden transition field " + field);
  }

  if (
    field === "opponent.active.evs" ||
    field === "opponent.active.ivs"
  ) {
    const battle = buildBattle(world);
    world.opponent_max_hp = battle.p2.active[0].maxhp;
    battle.destroy();
    if (world.exactHp > world.opponent_max_hp) {
      world.exactHp = world.opponent_max_hp;
      world.hidden["opponent.active.exact_hp"] = world.exactHp;
    }
  }

  world.world_id = "counterfactual-" + sha256({
    base_world_id: baseWorld.world_id,
    donor_world_id: donorWorld.world_id,
    field,
    hidden: world.hidden,
  });
  return world;
}

function executionWorldMaterial(world) {
  return {
    hidden: world.hidden,
    variant: {
      species: world.variant.species,
      ability: world.variant.ability,
      item: world.variant.item,
      level: world.variant.level,
      moves: world.variant.moves,
      evs: world.variant.evs,
      ivs: world.variant.ivs,
      teraType: world.variant.teraType,
    },
    exact_hp: world.exactHp,
    opponent_max_hp: world.opponent_max_hp,
  };
}

function projectionWorldStaticMaterial(world) {
  return {
    hidden: Object.fromEntries(
      Object.entries(world.hidden).filter(
        ([field]) => field !== "opponent.active.exact_hp"
      )
    ),
    variant: {
      species: world.variant.species,
      ability: world.variant.ability,
      item: world.variant.item,
      level: world.variant.level,
      moves: world.variant.moves,
      evs: world.variant.evs,
      ivs: world.variant.ivs,
      teraType: world.variant.teraType,
    },
    opponent_max_hp: world.opponent_max_hp,
  };
}

function hiddenReadProjection(world, fields) {
  const projection = [];
  for (const field of fields) {
    if (!Object.prototype.hasOwnProperty.call(world.hidden, field)) {
      return null;
    }
    projection.push([field, world.hidden[field]]);
  }
  return projection;
}

function executionWorldKey(world, action, rootMaterial) {
  return action + "\u0000" + sha256({
    showdown_commit: actualCommit,
    normalized_root_sha256: rootMaterial.exact_hash,
    world: executionWorldMaterial(world),
    opponent_policy: OPPONENT_POLICY,
    root_chance_samples: ROOT_CHANCE_SAMPLES,
    chance_seed_family: CHANCE_SEED_FAMILY,
    action_index: legalActions.indexOf(action),
  });
}

function executionProjectionBaseKey(world, action, rootMaterial) {
  return action + "\u0000" + sha256({
    showdown_commit: actualCommit,
    public_dependency_schema: PUBLIC_ROOT_DEPENDENCY_SCHEMA,
    static_root_sha256: rootMaterial.static_hash,
    world_static: projectionWorldStaticMaterial(world),
    opponent_policy: OPPONENT_POLICY,
    root_chance_samples: ROOT_CHANCE_SAMPLES,
    chance_seed_family: CHANCE_SEED_FAMILY,
    action_index: legalActions.indexOf(action),
  });
}

function sharedTransitionExecutionCache() {
  const cache = globalThis.__azelficoastTransitionExecutionCache;
  if (
    cache &&
    typeof cache.get === "function" &&
    typeof cache.set === "function" &&
    typeof cache.delete === "function" &&
    typeof cache.keys === "function"
  ) {
    return cache;
  }
  return null;
}

function sharedTransitionProjectionCache() {
  const cache = globalThis.__azelficoastTransitionProjectionCache;
  if (
    cache &&
    typeof cache.get === "function" &&
    typeof cache.set === "function" &&
    typeof cache.delete === "function" &&
    typeof cache.keys === "function" &&
    typeof cache.values === "function"
  ) {
    return cache;
  }
  return null;
}

function sharedTransitionExecutionCacheLimit() {
  const raw = globalThis.__azelficoastTransitionExecutionCacheMaxEntries;
  return Number.isSafeInteger(raw) && raw > 0 ? raw : 0;
}

function sharedTransitionProjectionCacheLimit() {
  const raw = globalThis.__azelficoastTransitionProjectionCacheMaxEntries;
  return Number.isSafeInteger(raw) && raw > 0 ? raw : 0;
}

function sharedCacheGet(cache, key) {
  const value = cache.get(key);
  if (value === undefined) return undefined;
  cache.delete(key);
  cache.set(key, value);
  return cloneJson(value);
}

function sharedCacheSet(cache, key, value) {
  cache.delete(key);
  cache.set(key, cloneJson(value));
  const limit = sharedTransitionExecutionCacheLimit();
  while (limit > 0 && cache.size > limit) {
    const oldest = cache.keys().next().value;
    cache.delete(oldest);
  }
}

function deltaEncodeExecution(rootSuccessor, execution) {
  return {
    ...cloneJson(execution),
    semantic_hash: undefined,
    outcomes: execution.outcomes.map(outcome => {
      const encoded = cloneJson(outcome);
      encoded.successor_delta = successorDelta(rootSuccessor, outcome.successor);
      delete encoded.successor;
      return encoded;
    }),
    successor_delta_schema: 1,
  };
}

function deltaRehydrateExecution(rootSuccessor, encoded) {
  if (
    !encoded ||
    encoded.successor_delta_schema !== 1 ||
    !Array.isArray(encoded.outcomes)
  ) {
    return undefined;
  }
  const execution = cloneJson(encoded);
  execution.outcomes = encoded.outcomes.map(outcome => {
    if (!Array.isArray(outcome.successor_delta)) {
      throw new Error("projected execution is missing successor delta");
    }
    const hydrated = cloneJson(outcome);
    hydrated.successor = applySuccessorDelta(
      rootSuccessor,
      hydrated.successor_delta
    );
    delete hydrated.successor_delta;
    return hydrated;
  });
  delete execution.successor_delta_schema;
  delete execution.semantic_hash;
  execution.semantic_hash = semanticHashForOutcomes(execution.outcomes);
  return execution;
}

function sharedProjectionCacheGet(cache, baseKey, material, world) {
  for (const [key, entry] of [...cache.entries()].reverse()) {
    if (!entry || entry.base_key !== baseKey) continue;
    const projection = publicReadProjection(material, entry.read_fields || []);
    if (projection === null) continue;
    if (sha256(projection) !== entry.projection_sha256) continue;
    const hiddenProjection = hiddenReadProjection(
      world,
      entry.hidden_read_fields || []
    );
    if (hiddenProjection === null) continue;
    if (sha256(hiddenProjection) !== entry.hidden_projection_sha256) continue;
    const execution = deltaRehydrateExecution(
      material.successor,
      entry.delta_execution
    );
    if (execution === undefined) continue;
    cache.delete(key);
    cache.set(key, entry);
    return execution;
  }
  return undefined;
}

function sharedProjectionCacheSet(cache, baseKey, material, world, execution) {
  if (
    !execution ||
    execution.public_trace_complete !== true ||
    !Array.isArray(execution.public_read_fields)
  ) {
    return false;
  }
  const projection = publicReadProjection(material, execution.public_read_fields);
  if (projection === null) return false;
  const hiddenProjection = hiddenReadProjection(world, execution.read_fields || []);
  if (hiddenProjection === null) return false;
  const entry = {
    base_key: baseKey,
    read_fields: [...execution.public_read_fields],
    projection_sha256: sha256(projection),
    hidden_read_fields: [...(execution.read_fields || [])],
    hidden_projection_sha256: sha256(hiddenProjection),
    delta_execution: deltaEncodeExecution(material.successor, execution),
  };
  const key = sha256({
    base_key: baseKey,
    read_fields: entry.read_fields,
    projection_sha256: entry.projection_sha256,
    hidden_read_fields: entry.hidden_read_fields,
    hidden_projection_sha256: entry.hidden_projection_sha256,
  });
  cache.delete(key);
  cache.set(key, entry);
  const limit = sharedTransitionProjectionCacheLimit();
  while (limit > 0 && cache.size > limit) {
    const oldest = cache.keys().next().value;
    cache.delete(oldest);
  }
  return true;
}

function compileLazyWholeTurnPrograms() {
  const orderedWorlds = [...worlds].sort((left, right) =>
    left.world_id.localeCompare(right.world_id)
  );
  const executionCache = new Map();
  const rootSnapshotCache = new Map();
  const sharedExecutionCache = sharedTransitionExecutionCache();
  const sharedProjectionCache = sharedTransitionProjectionCache();
  let exactExecutionCacheHits = 0;
  let projectedExecutionCacheHits = 0;
  let sharedExecutionCacheMisses = 0;
  let publicTraceIncompleteExecutions = 0;
  let rootSnapshotBuilds = 0;
  let publicRootMaterialBuilds = 0;
  const programs = [];

  function rootRecord(world) {
    const key = sha256(executionWorldMaterial(world));
    let record = rootSnapshotCache.get(key);
    if (record === undefined) {
      const snapshot = rootSnapshotForWorld(world);
      const material = publicRootMaterial(snapshot, world);
      record = {snapshot, material};
      rootSnapshotCache.set(key, record);
      rootSnapshotBuilds++;
      publicRootMaterialBuilds++;
    }
    return record;
  }

  function executeWorld(world, action, role) {
    const root = rootRecord(world);
    const key = executionWorldKey(world, action, root.material);
    const projectionBaseKey = executionProjectionBaseKey(
      world,
      action,
      root.material
    );
    let record = executionCache.get(key);
    if (!record) {
      let execution;
      let reused = false;
      let reuseKind = null;

      if (sharedExecutionCache !== null) {
        execution = sharedCacheGet(sharedExecutionCache, key);
        if (execution !== undefined) {
          reused = true;
          reuseKind = "exact";
          exactExecutionCacheHits++;
        }
      }
      if (
        execution === undefined &&
        root.material.complete &&
        sharedProjectionCache !== null
      ) {
        execution = sharedProjectionCacheGet(
          sharedProjectionCache,
          projectionBaseKey,
          root.material,
          world
        );
        if (execution !== undefined) {
          reused = true;
          reuseKind = "public-projection";
          projectedExecutionCacheHits++;
          if (sharedExecutionCache !== null) {
            sharedCacheSet(sharedExecutionCache, key, execution);
          }
        }
      }

      if (execution === undefined) {
        sharedExecutionCacheMisses++;
        execution = immediateWholeTurn(world, action, root.snapshot);
        if (execution.public_trace_complete !== true) {
          publicTraceIncompleteExecutions++;
        }
        if (sharedExecutionCache !== null) {
          sharedCacheSet(sharedExecutionCache, key, execution);
        }
        if (
          root.material.complete &&
          sharedProjectionCache !== null
        ) {
          sharedProjectionCacheSet(
            sharedProjectionCache,
            projectionBaseKey,
            root.material,
            world,
            execution
          );
        }
      }

      record = {
        execution,
        reused,
        reuse_kind: reuseKind,
        roles: new Set(),
        synthetic: String(world.world_id).startsWith("counterfactual-"),
      };
      executionCache.set(key, record);
    }
    record.roles.add(role);
    return record.execution;
  }

  for (const action of legalActions) {
    const representative = orderedWorlds[0];
    const baseline = executeWorld(representative, action, "causal-baseline");
    const observedFields = new Set(baseline.read_fields);
    const pendingFields = [...observedFields];
    const causalFields = new Set();
    const probes = [];

    for (let cursor = 0; cursor < pendingFields.length; cursor++) {
      const field = pendingFields[cursor];
      const donorsByValue = new Map();
      for (const donor of orderedWorlds) {
        const valueKey = JSON.stringify(stable(donor.hidden[field]));
        if (!donorsByValue.has(valueKey)) donorsByValue.set(valueKey, donor);
      }

      const baselineValue = JSON.stringify(stable(representative.hidden[field]));
      if (donorsByValue.size <= 1) continue;

      for (const [valueKey, donor] of donorsByValue) {
        if (valueKey === baselineValue) continue;
        const counterfactual = counterfactualWorld(representative, donor, field);
        const execution = executeWorld(counterfactual, action, "causal-probe");
        for (const discovered of execution.read_fields) {
          if (!observedFields.has(discovered)) {
            observedFields.add(discovered);
            pendingFields.push(discovered);
          }
        }
        const changed = execution.semantic_hash !== baseline.semantic_hash;
        probes.push({
          field,
          donor_world_id: donor.world_id,
          semantic_changed: changed,
          baseline_semantic_hash: baseline.semantic_hash,
          candidate_semantic_hash: execution.semantic_hash,
        });
        if (changed) {
          causalFields.add(field);
          break;
        }
      }
    }

    // Causal interventions can miss interactions between fields or boundaries
    // where a one-unit change alters a sampled whole-turn result. Any varying
    // field observed by Showdown is therefore retained as a conservative
    // partition key; causalFields remains the narrower intervention evidence.
    const varyingObservedFields = [...observedFields].filter((field) => {
      const values = new Set(
        orderedWorlds.map((world) => JSON.stringify(stable(world.hidden[field])))
      );
      return values.size > 1;
    });
    const dependencyFields = [...new Set([...causalFields, ...varyingObservedFields])].sort();
    const groups = new Map();
    for (const world of orderedWorlds) {
      const key = JSON.stringify(projectionKey(world, dependencyFields));
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(world);
    }

    const classes = [];
    for (const [key, members] of [...groups.entries()].sort(([left], [right]) =>
      left.localeCompare(right)
    )) {
      const representativeWorld = [...members].sort((left, right) =>
        left.world_id.localeCompare(right.world_id)
      )[0];
      const execution = executeWorld(
        representativeWorld,
        action,
        "class-representative"
      );
      const memberWorldIds = members.map(world => world.world_id).sort();
      const classId = "transition-class-" + sha256({
        action,
        causal_fields: [...causalFields].sort(),
        key,
        semantic_hash: execution.semantic_hash,
      }).slice(0, 24);
      classes.push({
        class_id: classId,
        read_fields: [...observedFields].sort(),
        causal_fields: [...causalFields].sort(),
        projection_key: projectionKey(representativeWorld, dependencyFields),
        representative_world_id: representativeWorld.world_id,
        member_world_ids: memberWorldIds,
        semantic_hash: execution.semantic_hash,
        outcomes: execution.outcomes,
      });
    }

    const partitionKeyHash = sha256({
      action,
      partition_method: "counterfactual-causal-refinement",
      fields: dependencyFields,
      classes: classes.map(row => ({
        class_id: row.class_id,
        members: row.member_world_ids,
        semantic_hash: row.semantic_hash,
      })),
    });
    const effectSignature = "sha256:" + sha256({
      showdown_commit: actualCommit,
      source_fixture_id: fixture.fixture_id,
      opponent_policy: OPPONENT_POLICY,
      action,
      dependency_fields: dependencyFields,
      partition_key_hash: partitionKeyHash,
    });

    programs.push({
      action,
      effect_signature: effectSignature,
      dependency_fields: dependencyFields,
      observed_read_fields: [...observedFields].sort(),
      partition_method: "counterfactual-causal-refinement",
      representative_world_count: classes.length,
      worlds_in: worlds.length,
      classes_out: classes.length,
      world_reduction: worlds.length - classes.length,
      reduction_fraction: 1 - classes.length / worlds.length,
      partition_key_hash: partitionKeyHash,
      causal_probe_count: probes.length,
      causal_probes: probes,
      classes,
    });
  }

  const cacheRows = [...executionCache.values()];
  const uniqueExecutions = cacheRows.length;
  const causalProbeExecutions = cacheRows.filter(
    row => row.roles.has("causal-probe")
  ).length;
  const classRepresentativeExecutions = cacheRows.filter(
    row => row.roles.has("class-representative")
  ).length;
  const showdownTurnExecutions = cacheRows.reduce(
    (sum, row) => sum + row.execution.showdown_turn_executions,
    0
  );
  const freshShowdownTurnExecutions = cacheRows.reduce(
    (sum, row) =>
      sum + (row.reused ? 0 : row.execution.showdown_turn_executions),
    0
  );
  const reusedShowdownTurnExecutions =
    showdownTurnExecutions - freshShowdownTurnExecutions;
  const opponentActionBranches = cacheRows.reduce(
    (sum, row) => sum + row.execution.opponent_action_branch_count,
    0
  );
  const publicReadFields = [
    ...new Set(
      cacheRows.flatMap(row =>
        Array.isArray(row.execution.public_read_fields)
          ? row.execution.public_read_fields
          : []
      )
    ),
  ].sort();
  const exhaustiveWorldActionProduct = worlds.length * legalActions.length;

  return {
    schema: "azelficoast.core.transition-program-set",
    schema_version: 1,
    source_fixture_id: fixture.fixture_id,
    showdown_commit: actualCommit,
    world_ids: worlds.map(world => world.world_id).sort(),
    legal_actions: legalActions,
    dependency_candidates: DEPENDENCY_CANDIDATES,
    programs,
    producer: {
      strategy: "counterfactual-causal-refinement",
      opponent_policy: OPPONENT_POLICY,
      root_chance_samples: ROOT_CHANCE_SAMPLES,
      unique_world_action_executions: uniqueExecutions,
      causal_probe_executions: causalProbeExecutions,
      class_representative_executions: classRepresentativeExecutions,
      root_snapshot_builds: rootSnapshotBuilds,
      saved_root_snapshot_builds: uniqueExecutions - rootSnapshotBuilds,
      public_root_dependency_schema: PUBLIC_ROOT_DEPENDENCY_SCHEMA,
      public_successor_delta_schema: 1,
      public_root_material_builds: publicRootMaterialBuilds,
      public_read_fields: publicReadFields,
      public_trace_incomplete_executions: publicTraceIncompleteExecutions,
      exact_transition_execution_cache_hits: exactExecutionCacheHits,
      public_projection_cache_hits: projectedExecutionCacheHits,
      transition_delta_rehydrations: projectedExecutionCacheHits,
      transition_execution_cache_hits:
        exactExecutionCacheHits + projectedExecutionCacheHits,
      transition_execution_cache_misses: sharedExecutionCacheMisses,
      opponent_action_branches: opponentActionBranches,
      showdown_turn_executions: showdownTurnExecutions,
      fresh_showdown_turn_executions: freshShowdownTurnExecutions,
      reused_showdown_turn_executions: reusedShowdownTurnExecutions,
      exhaustive_world_action_product: exhaustiveWorldActionProduct,
      saved_world_action_executions:
        exhaustiveWorldActionProduct - uniqueExecutions,
      execution_reduction_fraction:
        1 - uniqueExecutions / exhaustiveWorldActionProduct,
    },
    claim:
      "Candidate dependency fields are retained only when a one-field " +
      "counterfactual intervention changes the complete pinned-Showdown turn " +
      "semantics. Final classes are independently verifiable against the direct oracle.",
    non_claim:
      "One-factor interventions do not prove absence of higher-order field " +
      "interactions or generalization beyond this support. The direct oracle verifier " +
      "must reject any causally proposed class that merges different turn semantics.",
  };
}

if (transitionProgramOnly) {
  process.stdout.write(
    JSON.stringify(compileLazyWholeTurnPrograms(), null, 2) + "\n"
  );
  process.exit(0);
}


const transitions = [];
for (const world of worlds) {
  const base = buildBattle(world);
  const baseSnapshot = JSON.stringify(base);
  base.destroy();

  for (const action of legalActions) {
    const outcomes = [];
    const policyReads = new Set();
    const responses = opponentDistributionForSnapshot(baseSnapshot, policyReads);
    for (const [responseIndex, response] of responses.entries()) {
      for (let i = 0; i < ROOT_CHANCE_SAMPLES; i++) {
        const battle = cloneBattle(
          baseSnapshot,
          seed(i, 1_000 + legalActions.indexOf(action) + responseIndex * 4099)
        );
        const logStart = battle.log.length;
        const readTrace = instrumentOpponentHiddenReads(battle);
        try {
          battle.makeChoices(rootChoice(action), response.choice);
        } finally {
          readTrace.restore();
        }
        const transitionReads = [...new Set([...policyReads, ...readTrace.reads()])].sort();
        const rootObservation = observation(battle, logStart);
        const successor = stateSummary(battle, world);
        const rootSnapshot = JSON.stringify(battle);
        const continuation = continuationValues(rootSnapshot);
        outcomes.push({
          probability: response.probability / ROOT_CHANCE_SAMPLES,
          observation: rootObservation,
          successor,
          transition_reads: transitionReads,
          opponent_action: response.choice,
          opponent_policy_mode: response.mode,
          ...continuation,
        });
        battle.destroy();
      }
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

const declared = Object.fromEntries(
  legalActions.map(action => [action, declaredReads(action)])
);
process.stdout.write(JSON.stringify({
  schema: "azelficoast.core.transition-oracle",
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
    opponent_response: (
      OPPONENT_POLICY.kind === "strategy-mixture"
        ? "equal mixture of simple heuristics, dirty tricks, max damage, uniform legal moves, and public move persistence when available"
        : "uniform legal moves"
    ),
    opponent_policy: OPPONENT_POLICY,
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
    mechanics_projection_variant_count: mechanicsProjectionCount,
    mechanics_projection_fields:
      OPPONENT_POLICY.kind === "uniform-legal-moves"
        ? ["opponent.active.tera_type"]
        : ["opponent.active.moves", "opponent.active.tera_type"],
    mechanics_projection_rule:
      "projection is used only to estimate mechanics-equivalent execution shapes; semantic posterior worlds retain moves and Tera type",
    observed_opponent_moves: observedOpponentMoves(),
    opponent_policy: OPPONENT_POLICY,
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
    declared_read_mode: "instrumented-showdown-hidden-state-boundary",
  },
  factored_hidden: factoredHidden,
  dependency_candidates: DEPENDENCY_CANDIDATES,
  declared_reads: declared,
  worlds: outputWorlds,
  legal_actions: legalActions,
  transitions,
}, null, 2) + "\n");
