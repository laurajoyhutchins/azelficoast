#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const {execFileSync} = require("node:child_process");

const PINNED_SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc";
const SCHEMA = "azelficoast.showdown-vocabulary";
const SCHEMA_VERSION = 1;

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(2);
}

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])]));
  }
  return value;
}

function digest(value) {
  return "sha256:" + crypto.createHash("sha256")
    .update(JSON.stringify(stable(value)))
    .digest("hex");
}

function toID(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function canonicalNumeric(entries, label) {
  const rows = entries
    .filter(entry => entry && entry.exists && entry.isNonstandard === null && Number.isSafeInteger(entry.num) && entry.num > 0)
    .map(entry => ({id: entry.id, num: entry.num}))
    .sort((left, right) => left.id.localeCompare(right.id));
  const numbers = new Map();
  for (const row of rows) {
    const prior = numbers.get(row.num);
    if (prior && prior !== row.id) {
      fail(`${label} native num ${row.num} aliases ${prior} and ${row.id}`);
    }
    numbers.set(row.num, row.id);
  }
  return rows;
}

function speciesRows(species) {
  const groups = new Map();
  for (const entry of species) {
    if (!entry || !entry.exists || entry.isNonstandard !== null || !Number.isSafeInteger(entry.num) || entry.num <= 0) continue;
    const rows = groups.get(entry.num) || [];
    rows.push(entry);
    groups.set(entry.num, rows);
  }

  const output = [];
  for (const num of [...groups.keys()].sort((a, b) => a - b)) {
    const group = groups.get(num).sort((left, right) => left.id.localeCompare(right.id));
    const baseIndex = group.findIndex(entry => entry.id === toID(entry.baseSpecies));
    if (baseIndex > 0) {
      const [base] = group.splice(baseIndex, 1);
      group.unshift(base);
    }
    for (let formeIndex = 0; formeIndex < group.length; formeIndex++) {
      const entry = group[formeIndex];
      output.push({
        id: entry.id,
        num,
        forme_index: formeIndex,
        base_species_id: toID(entry.baseSpecies),
        forme: toID(entry.forme),
      });
    }
  }
  output.sort((left, right) => left.id.localeCompare(right.id));
  return output;
}

function derivedRows(ids) {
  return [...new Set(ids.map(toID).filter(Boolean))]
    .sort()
    .map((id, index) => ({id, index: index + 1}));
}

function collectRoles(value, output) {
  if (Array.isArray(value)) {
    for (const child of value) collectRoles(child, output);
    return;
  }
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) {
    if (key === "role" && typeof child === "string") output.add(child);
    collectRoles(child, output);
  }
}

const showdownRoot = process.argv[2];
const outputPath = process.argv[3] || null;
if (!showdownRoot) {
  fail("usage: export_showdown_vocabulary.cjs SHOWDOWN_ROOT [OUTPUT_JSON]");
}

const actualCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();
if (actualCommit !== PINNED_SHOWDOWN_COMMIT) {
  fail(`expected Showdown ${PINNED_SHOWDOWN_COMMIT}, got ${actualCommit}`);
}

const {Dex} = require(path.join(showdownRoot, "dist", "sim", "dex.js"));
const dex = Dex.forGen(9);
const randomSets = require(
  path.join(showdownRoot, "data", "random-battles", "gen9", "sets.json")
);

const roles = new Set();
collectRoles(randomSets, roles);

const material = {
  schema: SCHEMA,
  schema_version: SCHEMA_VERSION,
  showdown_commit: actualCommit,
  generation: 9,
  identity: {
    species: "showdown BasicEffect.num + revision-bound forme_index",
    moves: "showdown BasicEffect.num",
    items: "showdown BasicEffect.num",
    abilities: "showdown BasicEffect.num",
    derived_tables: "1-based sorted canonical Showdown IDs; 0 reserved for padding",
  },
  species: speciesRows(dex.species.all()),
  moves: canonicalNumeric(dex.moves.all(), "move"),
  items: canonicalNumeric(dex.items.all(), "item"),
  abilities: canonicalNumeric(dex.abilities.all(), "ability"),
  types: derivedRows(dex.types.all().map(entry => entry.id)),
  natures: derivedRows(dex.natures.all().map(entry => entry.id)),
  roles: derivedRows([...roles]),
};
const document = {...material, vocabulary_sha256: digest(material)};
const payload = JSON.stringify(stable(document)) + "\n";

if (outputPath) {
  fs.mkdirSync(path.dirname(outputPath), {recursive: true});
  fs.writeFileSync(outputPath, payload, "utf8");
} else {
  process.stdout.write(payload);
}
