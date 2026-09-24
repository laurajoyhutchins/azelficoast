#!/usr/bin/env node
"use strict";

const path = require("node:path");

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(2);
}

const [showdownRoot, species, observedMovesCsv, roundsText] = process.argv.slice(2);
if (!showdownRoot || !species || !observedMovesCsv) {
  fail(
    "usage: sample_showdown_worlds.cjs SHOWDOWN_ROOT SPECIES OBSERVED_MOVES [ROUNDS]"
  );
}

const rounds = roundsText ? Number(roundsText) : 65536;
if (!Number.isInteger(rounds) || rounds < 1 || rounds > 65536) {
  fail("ROUNDS must be an integer from 1 through 65536");
}

const observedMoves = new Set(
  observedMovesCsv
    .split(",")
    .map(move => move.trim().toLowerCase().replace(/[^a-z0-9]/g, ""))
    .filter(Boolean)
);

const {Teams} = require(path.join(showdownRoot, "dist", "sim", "teams"));
const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);

const itemCounts = new Map();
const variants = new Map();
let matched = 0;

for (let seed = 0; seed < rounds; seed++) {
  generator.setSeed([seed, seed, seed, seed]);
  const set = generator.randomSet(species, {}, false, false);
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
      species,
      observed_moves: [...observedMoves].sort(),
      seed_family: "[i,i,i,i]",
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
