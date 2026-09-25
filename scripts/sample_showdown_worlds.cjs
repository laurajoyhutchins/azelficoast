#!/usr/bin/env node
"use strict";

const path = require("node:path");
const {execFileSync} = require("node:child_process");

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(2);
}

const [
  showdownRoot,
  species,
  observedMovesCsv,
  roundsText,
  isLeadText,
  seedOffsetText,
  publicLevelText,
] = process.argv.slice(2);
if (!showdownRoot || !species || !observedMovesCsv) {
  fail(
    "usage: sample_showdown_worlds.cjs SHOWDOWN_ROOT SPECIES OBSERVED_MOVES [ROUNDS] [IS_LEAD] [SEED_OFFSET] [PUBLIC_LEVEL]"
  );
}

const rounds = roundsText ? Number(roundsText) : 65536;
if (!Number.isInteger(rounds) || rounds < 1 || rounds > 65536) {
  fail("ROUNDS must be an integer from 1 through 65536");
}

const isLead = isLeadText === "true";
if (isLeadText !== undefined && !["true", "false"].includes(isLeadText)) {
  fail("IS_LEAD must be true or false");
}

const seedOffset = seedOffsetText ? Number(seedOffsetText) : 0;
if (
  !Number.isInteger(seedOffset) ||
  seedOffset < 0 ||
  seedOffset >= 65536 ||
  seedOffset + rounds > 65536
) {
  fail("SEED_OFFSET must define a non-wrapping seed window within 0..65535");
}

const publicLevel =
  publicLevelText === undefined || publicLevelText === ""
    ? null
    : Number(publicLevelText);
if (
  publicLevel !== null &&
  (!Number.isSafeInteger(publicLevel) || publicLevel < 1)
) {
  fail("PUBLIC_LEVEL must be a positive integer");
}

const observedMoves = new Set(
  observedMovesCsv
    .split(",")
    .map(move => move.trim().toLowerCase().replace(/[^a-z0-9]/g, ""))
    .filter(Boolean)
);

const showdownCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();

const {Teams} = require(path.join(showdownRoot, "dist", "sim", "teams"));
const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);
const randomSets = require(
  path.join(showdownRoot, "data", "random-battles", "gen9", "sets.json")
);

function toID(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function resolveGeneratorSpecies(requested) {
  const dexSpecies = generator.dex.species.get(requested);
  const candidates = [
    dexSpecies.id,
    typeof dexSpecies.battleOnly === "string" ? toID(dexSpecies.battleOnly) : "",
    typeof dexSpecies.baseSpecies === "string" ? toID(dexSpecies.baseSpecies) : "",
  ].filter(Boolean);

  for (const candidate of [...new Set(candidates)]) {
    if (randomSets[candidate]) return candidate;
  }

  fail(
    `no Gen 9 randbats generator set for ${requested} (tried ${candidates.join(", ")})`
  );
}

const generatorSpecies = resolveGeneratorSpecies(species);

const itemCounts = new Map();
const variants = new Map();
let matched = 0;

for (let index = 0; index < rounds; index++) {
  const seed = seedOffset + index;
  generator.setSeed([seed, seed, seed, seed]);
  const set = generator.randomSet(generatorSpecies, {}, isLead, false);
  if (publicLevel !== null && Number(set.level) !== publicLevel) {
    continue;
  }
  const moves = [...set.moves].sort();

  if (![...observedMoves].every(move => moves.includes(move))) {
    continue;
  }

  matched++;
  const item = set.item || "";
  itemCounts.set(item, (itemCounts.get(item) || 0) + 1);

  const variant = {
    ability: set.ability,
    item,
    level: set.level,
    moves,
    role: set.role,
    teraType: set.teraType,
  };
  const key = JSON.stringify(variant);
  variants.set(key, (variants.get(key) || 0) + 1);
}

if (!matched) {
  fail("seed sweep produced no sets compatible with the observed moves");
}

const sortedItemCounts = Object.fromEntries(
  [...itemCounts.entries()].sort(([left], [right]) => left.localeCompare(right))
);
const weightedItems = Object.fromEntries(
  Object.entries(sortedItemCounts).map(([item, count]) => [
    item,
    count / matched,
  ])
);
const sortedVariants = [...variants.entries()]
  .map(([variant, count]) => ({...JSON.parse(variant), count, weight: count / matched}))
  .sort((left, right) => {
    if (left.item !== right.item) return left.item.localeCompare(right.item);
    if (left.role !== right.role) return left.role.localeCompare(right.role);
    return left.moves.join(",").localeCompare(right.moves.join(","));
  });

process.stdout.write(
  JSON.stringify(
    {
      schema: "azelficoast.showdown-world-sample",
      schema_version: 1,
      species: generatorSpecies,
      requested_species: species,
      observed_moves: [...observedMoves].sort(),
      showdown_commit: showdownCommit,
      seed_family: "[offset+i,offset+i,offset+i,offset+i]",
      seed_offset: seedOffset,
      generator_context: {
        format: "gen9randombattle",
        teamDetails: {},
        isLead,
        isDoubles: false,
        publicLevel,
      },
      rounds,
      matched,
      item_counts: sortedItemCounts,
      item_weights: weightedItems,
      variants: sortedVariants,
    },
    null,
    2
  ) + "\n"
);
