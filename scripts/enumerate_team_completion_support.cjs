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

const [showdownRoot, fixturePath] = process.argv.slice(2);
if (!showdownRoot || !fixturePath) {
  fail("usage: enumerate_team_completion_support.cjs SHOWDOWN_ROOT FIXTURE_JSON");
}

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
const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
if (!fixture || !fixture.state || !fixture.protocol_prefix) {
  fail("fixture JSON must contain state and protocol_prefix");
}

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

function publicOpponentTeamSize() {
  let size = null;
  for (const batch of fixture.protocol_prefix) {
    for (const message of batch) {
      if (
        message[0] === "" &&
        message[1] === "teamsize" &&
        message[2] === "p2"
      ) {
        size = Number(message[3]);
      }
    }
  }
  return size;
}

const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);
const dex = generator.dex;
const randomSetSpecies = Object.keys(generator.randomSets);

const publicTeam = Object.values(fixture.state.opponent_team || {});
const knownSpecies = [];
const seenIds = new Set();
for (const view of publicTeam) {
  const species = dex.species.get(view.species);
  if (!species.exists) fail(`unknown public species ${view.species}`);
  if (seenIds.has(species.id)) continue;
  seenIds.add(species.id);
  knownSpecies.push(species);
}

const teamSize = publicOpponentTeamSize();
if (!Number.isInteger(teamSize) || teamSize < 1) {
  fail("public protocol did not expose opponent team size");
}
if (knownSpecies.length >= teamSize) {
  fail(
    `expected an unrevealed team slot, got ${knownSpecies.length}/${teamSize} known species`
  );
}
if (knownSpecies.length !== teamSize - 1) {
  fail(
    `this bounded completion experiment requires exactly one unknown slot, got ${knownSpecies.length}/${teamSize}`
  );
}

const typeCount = {};
const typeWeaknesses = {};
const typeDoubleWeaknesses = {};
let freezeDryWeaknesses = 0;
let level100Count = 0;
const knownBaseSpecies = new Set();

function weakToFreezeDry(species) {
  return (
    dex.getEffectiveness("Ice", species) > 0 ||
    (dex.getEffectiveness("Ice", species) > -2 && species.types.includes("Water"))
  );
}

function increment(object, key) {
  object[key] = (object[key] || 0) + 1;
}

for (const species of knownSpecies) {
  knownBaseSpecies.add(species.baseSpecies);

  for (const type of species.types) increment(typeCount, type);

  for (const type of dex.types.names()) {
    const effectiveness = dex.getEffectiveness(type, species);
    if (effectiveness > 0) increment(typeWeaknesses, type);
    if (effectiveness > 1) increment(typeDoubleWeaknesses, type);
  }

  if (weakToFreezeDry(species)) freezeDryWeaknesses++;
  if (generator.getLevel(species, false) === 100) level100Count++;
}

const knownCompatibilitySets = knownSpecies.map(species => ({
  species: species.name,
  speciesId: species.id,
}));

const rejectionCounts = {};
function reject(reason) {
  rejectionCounts[reason] = (rejectionCounts[reason] || 0) + 1;
  return false;
}

function speciesLevelCompatible(species) {
  if (knownBaseSpecies.has(species.baseSpecies)) {
    return reject("species-clause");
  }

  for (const type of species.types) {
    if ((typeCount[type] || 0) >= 2) {
      return reject("type-cap");
    }
  }

  for (const type of dex.types.names()) {
    const effectiveness = dex.getEffectiveness(type, species);
    if (effectiveness > 0 && (typeWeaknesses[type] || 0) >= 3) {
      return reject("weakness-cap");
    }
    if (effectiveness > 1 && (typeDoubleWeaknesses[type] || 0) >= 1) {
      return reject("double-weakness-cap");
    }
  }

  if (
    dex.getEffectiveness("Fire", species) === 0 &&
    Object.values(species.abilities).some(
      ability => ability === "Dry Skin" || ability === "Fluffy"
    ) &&
    (typeWeaknesses.Fire || 0) >= 3
  ) {
    return reject("possible-ability-fire-weakness-cap");
  }

  if (weakToFreezeDry(species) && freezeDryWeaknesses >= 4) {
    return reject("freeze-dry-cap");
  }

  if (generator.getLevel(species, false) === 100 && level100Count >= 1) {
    return reject("level-100-cap");
  }

  if (!generator.getPokemonCompatibility(species, knownCompatibilitySets, false)) {
    return reject("species-incompatibility");
  }

  return true;
}

const support = [];
for (const speciesId of randomSetSpecies) {
  const species = dex.species.get(speciesId);
  if (!species.exists) continue;
  if (!speciesLevelCompatible(species)) continue;

  const caveats = [];
  if (species.baseSpecies === "Zoroark") {
    caveats.push("generation-order-can-exclude-illusion-from-last-slot");
  }
  if (["ogerpon", "ogerponhearthflame", "terapagos"].includes(species.id)) {
    caveats.push("tera-blast-role-conflict-requires-hidden-set-state");
  }

  support.push({
    species_id: species.id,
    species: species.name,
    base_species: species.baseSpecies,
    level: generator.getLevel(species, false),
    types: [...species.types],
    caveats,
  });
}

support.sort((left, right) => left.species_id.localeCompare(right.species_id));

const evidence = {
  schema: "azelficoast.opponent-team-completion-support",
  schema_version: 1,
  showdown_commit: SHOWDOWN_COMMIT,
  fixture_id: fixture.fixture_id,
  public_team_size: teamSize,
  known_species: knownSpecies.map(species => species.id).sort(),
  unknown_slots: teamSize - knownSpecies.length,
  random_set_species_count: randomSetSpecies.length,
  support_count: support.length,
  rejection_counts: Object.fromEntries(
    Object.entries(rejectionCounts).sort(([a], [b]) => a.localeCompare(b))
  ),
  support,
  caveats: [
    "support-is-not-a-probability-distribution",
    "known-hidden-set-abilities-can-strengthen-the-fire-weakness-cap",
    "teamDetails-dependent-set-role-constraints-are-not-certified-here",
    "generation-order-constraints-are-reported-but-not-used-to-prune-support",
  ],
};

process.stdout.write(
  JSON.stringify(
    {
      ...evidence,
      support_sha256: sha256(evidence),
    },
    null,
    2
  ) + "\n"
);
