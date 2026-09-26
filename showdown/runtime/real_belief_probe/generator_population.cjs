"use strict";

const fs = require("node:fs");
const path = require("node:path");

function createGeneratorPopulationSource({
  Teams,
  randomSets,
  source,
  fixture,
  observedOpponentMoves,
  generatorCacheDir,
  actualCommit,
  generatorRounds,
  opponentPolicy,
  stable,
  sha256,
  toID,
  fail,
}) {
  const GENERATOR_ROUNDS = generatorRounds;
  const OPPONENT_POLICY = opponentPolicy;

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


  return {generatorVariants, mechanicsProjectionVariantCount};
}

module.exports = {createGeneratorPopulationSource};
