#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const {execFileSync} = require("node:child_process");

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(2);
}

function parseArgs(argv) {
  const args = [...argv];
  const showdownRoot = args.shift();
  if (!showdownRoot) fail("missing SHOWDOWN_ROOT");

  const options = {
    source: null,
    controlSpecies: null,
    particles: 20000,
    alpha: 0.85,
  };
  while (args.length) {
    const flag = args.shift();
    const value = args.shift();
    if (value === undefined) fail(`missing value for ${flag}`);
    if (flag === "--source") options.source = value;
    else if (flag === "--control-species") options.controlSpecies = value;
    else if (flag === "--particles") options.particles = Number(value);
    else if (flag === "--alpha") options.alpha = Number(value);
    else fail(`unknown argument ${flag}`);
  }
  if (!!options.source === !!options.controlSpecies) {
    fail("choose exactly one of --source or --control-species");
  }
  if (!Number.isInteger(options.particles) || options.particles < 1) {
    fail("--particles must be a positive integer");
  }
  if (!(options.alpha >= 0 && options.alpha < 1)) {
    fail("--alpha must satisfy 0 <= alpha < 1");
  }
  return {showdownRoot, ...options};
}

const {
  showdownRoot,
  source: sourcePath,
  controlSpecies,
  particles,
  alpha,
} = parseArgs(process.argv.slice(2));

const SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc";
const actualCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();
if (actualCommit !== SHOWDOWN_COMMIT) {
  fail(`expected Showdown ${SHOWDOWN_COMMIT}, got ${actualCommit}`);
}

const {Teams} = require(path.join(showdownRoot, "dist", "sim", "teams.js"));

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map(key => [key, stable(value[key])])
    );
  }
  return value;
}

function sha256(value) {
  return crypto
    .createHash("sha256")
    .update(JSON.stringify(stable(value)))
    .digest("hex");
}

function toID(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function seed(index) {
  const n = index + 1;
  return [
    n & 0xffff,
    (n * 17 + 11) & 0xffff,
    (n * 97 + 23) & 0xffff,
    (n * 193 + 47) & 0xffff,
  ];
}

function proposalRng(index) {
  const digest = crypto
    .createHash("sha256")
    .update(`azelficoast-team-proposal:${index}`)
    .digest();
  let state = digest.readUInt32LE(0) || 1;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fixtureConditioning(source, dex) {
  const fixture = source.fixture || source;
  if (!fixture || !fixture.state || !fixture.protocol_prefix) {
    fail("source must contain an exact decision fixture");
  }

  const views = Object.values(fixture.state.opponent_team || {});
  if (!views.length) fail("fixture exposes no public opponent species");

  const required = new Map();
  for (const view of views) {
    const species = dex.species.get(view.species);
    if (!species.exists) fail(`unknown public species ${view.species}`);
    required.set(species.id, {
      species_id: species.id,
      base_species: species.baseSpecies,
      level: Number(view.level),
      moves: [...new Set((view.moves || []).map(toID))].sort(),
      ability: view.ability ? toID(view.ability) : null,
      item: view.item ? toID(view.item) : null,
      tera_type: view.tera_type ? toID(view.tera_type) : null,
    });
  }

  let lead = null;
  for (const batch of fixture.protocol_prefix) {
    for (const message of batch) {
      if (
        message[0] === "" &&
        ["switch", "drag"].includes(message[1]) &&
        String(message[2] || "").startsWith("p2")
      ) {
        lead = toID(String(message[3] || "").split(",", 1)[0]);
        break;
      }
    }
    if (lead) break;
  }
  if (!lead) fail("public history does not expose the opponent lead");

  const active = toID(fixture.state.opponent_active.species);
  const plausibleItems = new Set(
    (source.plausible_items || []).map(toID)
  );

  return {
    fixture_id: fixture.fixture_id,
    required,
    lead,
    active,
    plausibleItems,
    public_evidence: true,
  };
}

function speciesOnlyConditioning(speciesName, dex) {
  const species = dex.species.get(speciesName);
  if (!species.exists) fail(`unknown control species ${speciesName}`);
  return {
    fixture_id: null,
    required: new Map([
      [
        species.id,
        {
          species_id: species.id,
          base_species: species.baseSpecies,
          level: null,
          moves: [],
          ability: null,
          item: null,
          tera_type: null,
        },
      ],
    ]),
    lead: null,
    active: null,
    plausibleItems: new Set(),
    public_evidence: false,
  };
}

const sourceDocument = sourcePath
  ? JSON.parse(fs.readFileSync(sourcePath, "utf8"))
  : null;

const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);
const dex = generator.dex;
const conditioning = controlSpecies
  ? speciesOnlyConditioning(controlSpecies, dex)
  : fixtureConditioning(sourceDocument, dex);

const requiredBaseSpecies = new Set(
  [...conditioning.required.values()].map(view => view.base_species)
);

let proposalState = null;
const originalGetPokemonPool = generator.getPokemonPool.bind(generator);
const originalSampleNoReplace = generator.sampleNoReplace.bind(generator);
const originalRandomSet = generator.randomSet.bind(generator);

generator.getPokemonPool = function (...args) {
  const result = originalGetPokemonPool(...args);
  if (proposalState) proposalState.basePool = result[1];
  return result;
};

generator.sampleNoReplace = function (list) {
  const state = proposalState;
  if (
    !state ||
    list !== state.basePool ||
    !state.remainingBaseSpecies.size
  ) {
    return originalSampleNoReplace(list);
  }

  const requiredIndices = [];
  for (let i = 0; i < list.length; i++) {
    if (state.remainingBaseSpecies.has(list[i])) requiredIndices.push(i);
  }
  if (!requiredIndices.length) {
    state.requiredSpeciesExhausted = true;
    return originalSampleNoReplace(list);
  }

  const slotsLeft = generator.maxTeamSize - state.acceptedCount;
  let dynamicAlpha =
    1 - Math.pow(1 - state.alpha, state.remainingBaseSpecies.size);
  if (state.remainingBaseSpecies.size >= slotsLeft) {
    dynamicAlpha = Math.max(dynamicAlpha, 0.8);
  }

  const chooseRequired = state.proposalRandom() < dynamicAlpha;
  const index = chooseRequired
    ? requiredIndices[
        Math.floor(state.proposalRandom() * requiredIndices.length)
      ]
    : Math.floor(state.proposalRandom() * list.length);

  const isRequiredIndex = requiredIndices.includes(index);
  const targetProbability = 1 / list.length;
  const proposalProbability =
    (1 - dynamicAlpha) / list.length +
    (isRequiredIndex ? dynamicAlpha / requiredIndices.length : 0);

  if (!(proposalProbability > 0)) fail("proposal assigned zero mass to sampled draw");
  state.logWeight += Math.log(targetProbability / proposalProbability);
  state.biasedDrawCount++;

  // Preserve the target generator's PRNG call count. The proposal index itself
  // comes from an independent deterministic stream.
  generator.random(list.length);
  return generator.fastPop(list, index);
};

generator.randomSet = function (species, ...args) {
  const set = originalRandomSet(species, ...args);
  const state = proposalState;
  if (state) {
    state.acceptedCount++;
    state.remainingBaseSpecies.delete(species.baseSpecies);
  }
  return set;
};

function setMatches(view, set) {
  if (toID(set.species) !== view.species_id) return false;
  if (view.level !== null && Number(set.level) !== view.level) return false;

  const moves = new Set((set.moves || []).map(toID));
  if (!view.moves.every(move => moves.has(move))) return false;

  if (view.ability && toID(set.ability) !== view.ability) return false;
  if (view.item && toID(set.item) !== view.item) return false;
  if (view.tera_type && toID(set.teraType) !== view.tera_type) return false;
  return true;
}

function teamMatches(team) {
  const bySpecies = new Map(team.map(set => [toID(set.species), set]));
  for (const view of conditioning.required.values()) {
    const set = bySpecies.get(view.species_id);
    if (!set || !setMatches(view, set)) return false;
  }

  if (conditioning.lead && toID(team[0]?.species) !== conditioning.lead) {
    return false;
  }

  if (conditioning.active && conditioning.plausibleItems.size) {
    const active = bySpecies.get(conditioning.active);
    if (!active || !conditioning.plausibleItems.has(toID(active.item))) {
      return false;
    }
  }
  return true;
}

function semanticSet(set) {
  return {
    species: toID(set.species),
    level: Number(set.level),
    ability: set.ability,
    item: set.item,
    moves: [...set.moves].map(toID).sort(),
    tera_type: set.teraType,
    role: set.role,
  };
}

function logAdd(logA, logB) {
  if (logA === -Infinity) return logB;
  if (logB === -Infinity) return logA;
  const high = Math.max(logA, logB);
  return high + Math.log(Math.exp(logA - high) + Math.exp(logB - high));
}

function logSum(values) {
  let total = -Infinity;
  for (const value of values) total = logAdd(total, value);
  return total;
}

const eventParticles = [];
const allLogWeights = [];
let generationErrors = 0;
let requiredSpeciesExhausted = 0;

for (let i = 0; i < particles; i++) {
  const state = {
    alpha,
    basePool: null,
    remainingBaseSpecies: new Set(requiredBaseSpecies),
    acceptedCount: 0,
    biasedDrawCount: 0,
    requiredSpeciesExhausted: false,
    logWeight: 0,
    proposalRandom: proposalRng(i),
  };
  proposalState = state;
  generator.setSeed(seed(i));

  let team;
  try {
    team = generator.getTeam();
  } catch (_error) {
    generationErrors++;
    allLogWeights.push(state.logWeight);
    continue;
  } finally {
    proposalState = null;
  }

  allLogWeights.push(state.logWeight);
  if (state.requiredSpeciesExhausted) requiredSpeciesExhausted++;
  if (!teamMatches(team)) continue;

  const knownIds = new Set(conditioning.required.keys());
  const unknown = team.filter(set => !knownIds.has(toID(set.species)));
  eventParticles.push({
    logWeight: state.logWeight,
    unknown: unknown.map(semanticSet),
    team: team.map(semanticSet),
    requiredPositions: Object.fromEntries(
      [...conditioning.required.keys()].map(speciesId => [
        speciesId,
        team.findIndex(set => toID(set.species) === speciesId),
      ])
    ),
    biasedDrawCount: state.biasedDrawCount,
  });
}

const logTotalWeight = logSum(allLogWeights);
const meanImportanceWeight = Math.exp(logTotalWeight - Math.log(particles));

const eventLogWeights = eventParticles.map(row => row.logWeight);
const logEventWeight = logSum(eventLogWeights);
const estimatedEventProbability =
  eventParticles.length === 0
    ? 0
    : Math.exp(logEventWeight - Math.log(particles));

const speciesLogMass = new Map();
const setLogMass = new Map();
const positionLogMass = new Map();
for (const particle of eventParticles) {
  for (const [speciesId, position] of Object.entries(particle.requiredPositions)) {
    const key = `${speciesId}:${position}`;
    positionLogMass.set(
      key,
      logAdd(positionLogMass.get(key) ?? -Infinity, particle.logWeight)
    );
  }

  for (const set of particle.unknown) {
    const speciesKey = set.species;
    speciesLogMass.set(
      speciesKey,
      logAdd(speciesLogMass.get(speciesKey) ?? -Infinity, particle.logWeight)
    );

    const setKey = JSON.stringify(stable(set));
    setLogMass.set(
      setKey,
      logAdd(setLogMass.get(setKey) ?? -Infinity, particle.logWeight)
    );
  }
}

function normalizedRows(logMass, parseKey) {
  if (logEventWeight === -Infinity) return [];
  return [...logMass.entries()]
    .map(([key, mass]) => ({
      ...parseKey(key),
      probability: Math.exp(mass - logEventWeight),
    }))
    .sort(
      (left, right) =>
        right.probability - left.probability ||
        JSON.stringify(left).localeCompare(JSON.stringify(right))
    );
}

let effectiveSampleSize = 0;
if (eventParticles.length) {
  const maxLog = Math.max(...eventLogWeights);
  const scaled = eventLogWeights.map(value => Math.exp(value - maxLog));
  const sum = scaled.reduce((a, b) => a + b, 0);
  const sumSquares = scaled.reduce((a, b) => a + b * b, 0);
  effectiveSampleSize = (sum * sum) / sumSquares;
}

const positionPosterior = normalizedRows(
  positionLogMass,
  key => {
    const split = key.lastIndexOf(":");
    return {
      species_id: key.slice(0, split),
      position: Number(key.slice(split + 1)),
    };
  }
);
const speciesPosterior = normalizedRows(
  speciesLogMass,
  key => ({species_id: key})
);
const setPosterior = normalizedRows(
  setLogMass,
  key => ({set: JSON.parse(key)})
);

const evidence = {
  schema: "azelficoast.importance-sampled-team-belief",
  schema_version: 1,
  showdown_commit: SHOWDOWN_COMMIT,
  mode: controlSpecies ? "species-only-control" : "public-fixture",
  fixture_id: conditioning.fixture_id,
  control_species: controlSpecies ? toID(controlSpecies) : null,
  particles,
  proposal_alpha: alpha,
  required_species: [...conditioning.required.keys()].sort(),
  event_particle_count: eventParticles.length,
  generation_error_count: generationErrors,
  required_species_exhausted_count: requiredSpeciesExhausted,
  mean_importance_weight: meanImportanceWeight,
  estimated_event_probability: estimatedEventProbability,
  effective_sample_size: effectiveSampleSize,
  required_position_posterior: positionPosterior,
  species_posterior: speciesPosterior,
  set_posterior_top_50: setPosterior.slice(0, 50),
  caveats: [
    "finite-deterministic-seed-family",
    "importance-bias-applies-only-to-base-species-draws",
    "all-non-species-generator-randomness-remains-unmodified",
    "public-fixture-mode-conditions-on-recorded-level-moves-ability-item-tera-and-lead",
  ],
};

process.stdout.write(
  JSON.stringify(
    {
      ...evidence,
      evidence_sha256: sha256(evidence),
    },
    null,
    2
  ) + "\n"
);
