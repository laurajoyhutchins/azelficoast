#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const path = require("node:path");
const {execFileSync} = require("node:child_process");

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(2);
}

const args = process.argv.slice(2);
const showdownRoot = args[0];
const trainTeamCount = Number(args[1] || "80000");
const validationTeamCount = Number(args[2] || "10000");
const testTeamCount = Number(args[3] || "10000");

if (!showdownRoot) {
  fail("usage: evaluate_conditional_team_prior.cjs SHOWDOWN_ROOT [TRAIN_TEAMS] [VALIDATION_TEAMS] [TEST_TEAMS]");
}
/** @type {Array<[string, number]>} */
const teamCounts = [
  ["TRAIN_TEAMS", trainTeamCount],
  ["VALIDATION_TEAMS", validationTeamCount],
  ["TEST_TEAMS", testTeamCount],
];
for (const [label, count] of teamCounts) {
  if (!Number.isInteger(count) || count < 1) {
    fail(label + " must be a positive integer");
  }
}

const SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc";
const actualCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();
if (actualCommit !== SHOWDOWN_COMMIT) {
  fail("expected Showdown " + SHOWDOWN_COMMIT + ", got " + actualCommit);
}

const {Teams} = require(path.join(showdownRoot, "dist", "sim", "teams.js"));
const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);
const dex = generator.dex;

function toID(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
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

function seed(index) {
  const n = index + 1;
  return [
    n & 0xffff,
    (n * 17 + 11) & 0xffff,
    (n * 97 + 23) & 0xffff,
    (n * 193 + 47) & 0xffff,
  ];
}

function generatedTeam(index) {
  generator.setSeed(seed(index));
  const team = generator.getTeam();
  const species = team.map(set => canonicalTeamSpecies(set.species));
  if (species.length !== 6 || new Set(species).size !== 6) {
    fail("unexpected generated team at seed " + index + ": " + JSON.stringify(species));
  }
  return species;
}

const candidateIds = [...new Set(Object.keys(generator.randomSets).map(toID))].sort();
const candidateSet = new Set(candidateIds);
const candidateCount = candidateIds.length;

const candidateIdsByBaseSpecies = new Map();
for (const candidateId of candidateIds) {
  const species = dex.species.get(candidateId);
  const ids = candidateIdsByBaseSpecies.get(species.baseSpecies) || [];
  ids.push(candidateId);
  candidateIdsByBaseSpecies.set(species.baseSpecies, ids);
}

function canonicalTeamSpecies(value) {
  const id = toID(value);
  if (candidateSet.has(id)) return id;

  const emitted = dex.species.get(id);
  if (!emitted.exists) fail("unknown emitted species " + value);
  const candidates = candidateIdsByBaseSpecies.get(emitted.baseSpecies) || [];
  if (candidates.length !== 1) {
    fail(
      "cannot uniquely map emitted species " + id +
      " to a random-set family: " + JSON.stringify(candidates)
    );
  }
  return candidates[0];
}

const featureById = new Map();
for (const speciesId of candidateIds) {
  const species = dex.species.get(speciesId);
  if (!species.exists) fail("missing species data for " + speciesId);
  const weak = [];
  const doubleWeak = [];
  for (const type of dex.types.names()) {
    const effectiveness = dex.getEffectiveness(type, species);
    if (effectiveness > 0) weak.push(type);
    if (effectiveness > 1) doubleWeak.push(type);
  }
  featureById.set(speciesId, {
    species,
    baseSpecies: species.baseSpecies,
    types: [...species.types],
    weak,
    doubleWeak,
    freezeDryWeak:
      dex.getEffectiveness("Ice", species) > 0 ||
      (dex.getEffectiveness("Ice", species) > -2 && species.types.includes("Water")),
    level: generator.getLevel(species, false),
  });
}

function pairKey(a, b) {
  return a < b ? a + "|" + b : b + "|" + a;
}

const speciesCounts = new Map(candidateIds.map(id => [id, 0]));
const pairCounts = new Map();

function increment(map, key, by = 1) {
  map.set(key, (map.get(key) || 0) + by);
}

function observeTrainingTeam(species) {
  for (const id of species) {
    if (!candidateSet.has(id)) fail("generated species absent from randbats pool: " + id);
    increment(speciesCounts, id);
  }
  for (let i = 0; i < species.length; i++) {
    for (let j = i + 1; j < species.length; j++) {
      increment(pairCounts, pairKey(species[i], species[j]));
    }
  }
}

for (let i = 0; i < trainTeamCount; i++) {
  observeTrainingTeam(generatedTeam(i));
}

function contextSummary(known) {
  const baseSpecies = new Set();
  const typeCount = new Map();
  const weaknessCount = new Map();
  const doubleWeaknessCount = new Map();
  let freezeDryWeaknesses = 0;
  let level100Count = 0;

  for (const id of known) {
    const feature = featureById.get(id);
    if (!feature) fail("missing context feature for " + id);
    baseSpecies.add(feature.baseSpecies);
    for (const type of feature.types) increment(typeCount, type);
    for (const type of feature.weak) increment(weaknessCount, type);
    for (const type of feature.doubleWeak) increment(doubleWeaknessCount, type);
    if (feature.freezeDryWeak) freezeDryWeaknesses++;
    if (feature.level === 100) level100Count++;
  }

  const compatibilitySets = known.map(id => ({
    species: featureById.get(id).species.name,
    speciesId: id,
  }));

  return {
    knownSet: new Set(known),
    baseSpecies,
    typeCount,
    weaknessCount,
    doubleWeaknessCount,
    freezeDryWeaknesses,
    level100Count,
    compatibilitySets,
  };
}

function compatibleCandidate(candidateId, summary) {
  if (summary.knownSet.has(candidateId)) return false;
  const feature = featureById.get(candidateId);
  if (summary.baseSpecies.has(feature.baseSpecies)) return false;
  return true;
}

function supportFor(known) {
  const summary = contextSummary(known);
  return candidateIds.filter(id => compatibleCandidate(id, summary));
}

function rejectionReasons(candidateId, known) {
  const summary = contextSummary(known);
  const reasons = [];
  const feature = featureById.get(candidateId);
  if (!feature) return ["missing-candidate-feature"];
  if (summary.knownSet.has(candidateId)) reasons.push("exact-species-present");
  if (summary.baseSpecies.has(feature.baseSpecies)) reasons.push("base-species-present");
  return reasons;
}

const PRIOR_SMOOTHING = 1;
const PAIR_SMOOTHING = 0.5;
const lambdaGrid = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2];

function logPrior(id) {
  return Math.log((speciesCounts.get(id) || 0) + PRIOR_SMOOTHING);
}

function logLift(candidate, known) {
  const pair = pairCounts.get(pairKey(candidate, known)) || 0;
  const candidateCountSeen = speciesCounts.get(candidate) || 0;
  const knownCountSeen = speciesCounts.get(known) || 0;
  const expected =
    trainTeamCount > 0
      ? (candidateCountSeen * knownCountSeen) / trainTeamCount
      : 0;
  return Math.log((pair + PAIR_SMOOTHING) / (expected + PAIR_SMOOTHING));
}

function scoreCandidate(candidate, known, lambda) {
  let score = logPrior(candidate);
  if (lambda) {
    for (const observed of known) {
      score += lambda * logLift(candidate, observed);
    }
  }
  return score;
}

function scoringRows(known, support) {
  return support.map(id => ({
    id,
    prior: logPrior(id),
    association: known.reduce(
      (sum, observed) => sum + logLift(id, observed),
      0
    ),
  }));
}

function probabilitiesFromRows(rows, lambda) {
  const scored = rows.map(row => ({
    id: row.id,
    score: row.prior + lambda * row.association,
  }));
  const maxScore = Math.max(...scored.map(row => row.score));
  let total = 0;
  for (const row of scored) {
    row.weight = Math.exp(row.score - maxScore);
    total += row.weight;
  }
  return scored
    .map(row => ({id: row.id, probability: row.weight / total}))
    .sort(
      (left, right) =>
        right.probability - left.probability || left.id.localeCompare(right.id)
    );
}

function probabilities(known, support, lambda) {
  return probabilitiesFromRows(scoringRows(known, support), lambda);
}

function blankMetrics() {
  return {
    examples: 0,
    supportMisses: 0,
    supportMissExamples: [],
    nll: 0,
    reciprocalRank: 0,
    top1: 0,
    top5: 0,
    top20: 0,
    supportSize: 0,
  };
}

function observePrediction(metrics, target, distribution) {
  metrics.examples++;
  metrics.supportSize += distribution.length;
  const index = distribution.findIndex(row => row.id === target);
  if (index < 0) {
    metrics.supportMisses++;
    metrics.nll += 50;
    return;
  }
  const probability = Math.max(distribution[index].probability, Number.MIN_VALUE);
  const rank = index + 1;
  metrics.nll -= Math.log(probability);
  metrics.reciprocalRank += 1 / rank;
  if (rank <= 1) metrics.top1++;
  if (rank <= 5) metrics.top5++;
  if (rank <= 20) metrics.top20++;
}

function finalize(metrics) {
  const n = metrics.examples;
  return {
    examples: n,
    support_misses: metrics.supportMisses,
    support_miss_examples: metrics.supportMissExamples,
    mean_nll: metrics.nll / n,
    mean_reciprocal_rank: metrics.reciprocalRank / n,
    top1_recall: metrics.top1 / n,
    top5_recall: metrics.top5 / n,
    top20_recall: metrics.top20 / n,
    mean_support_size: metrics.supportSize / n,
  };
}

function uniformDistribution(support) {
  const p = 1 / support.length;
  return support.map(id => ({id, probability: p}));
}

function exampleForTeam(team, globalIndex) {
  const targetIndex = globalIndex % team.length;
  const target = team[targetIndex];
  const known = team.filter((_, index) => index !== targetIndex).sort();
  return {target, known};
}

function evaluateRange(start, count, lambdas) {
  const byLambda = new Map(lambdas.map(lambda => [lambda, blankMetrics()]));
  const uniform = blankMetrics();

  for (let offset = 0; offset < count; offset++) {
    const globalIndex = start + offset;
    const team = generatedTeam(globalIndex);
    const {target, known} = exampleForTeam(team, globalIndex);
    const support = supportFor(known);

    if (!support.includes(target)) {
      const miss = {
        global_index: globalIndex,
        target,
        known,
        reasons: rejectionReasons(target, known),
      };
      uniform.examples++;
      uniform.supportMisses++;
      if (uniform.supportMissExamples.length < 20) {
        uniform.supportMissExamples.push(miss);
      }
      for (const metrics of byLambda.values()) {
        metrics.examples++;
        metrics.supportMisses++;
        if (metrics.supportMissExamples.length < 20) {
          metrics.supportMissExamples.push(miss);
        }
      }
      continue;
    }

    observePrediction(uniform, target, uniformDistribution(support));
    const rows = scoringRows(known, support);
    for (const lambda of lambdas) {
      observePrediction(
        byLambda.get(lambda),
        target,
        probabilitiesFromRows(rows, lambda)
      );
    }
  }

  return {
    uniform: finalize(uniform),
    byLambda: Object.fromEntries(
      [...byLambda.entries()].map(([lambda, metrics]) => [
        String(lambda),
        finalize(metrics),
      ])
    ),
  };
}

const validationStart = trainTeamCount;
const testStart = trainTeamCount + validationTeamCount;
const validation = evaluateRange(validationStart, validationTeamCount, lambdaGrid);

const selectedLambda = lambdaGrid
  .map(lambda => ({
    lambda,
    nll: validation.byLambda[String(lambda)].mean_nll,
  }))
  .sort((a, b) => a.nll - b.nll || a.lambda - b.lambda)[0].lambda;

const test = evaluateRange(testStart, testTeamCount, [0, selectedLambda]);

const realKnownSpecies = [
  "camerupt",
  "clawitzer",
  "hitmonchan",
  "ironthorns",
  "staraptor",
].sort();
const realSupport = supportFor(realKnownSpecies);
const realPrior = probabilities(realKnownSpecies, realSupport, selectedLambda);

const evidence = {
  schema: "azelficoast.conditional-team-prior-evaluation",
  schema_version: 1,
  showdown_commit: SHOWDOWN_COMMIT,
  train_team_count: trainTeamCount,
  validation_team_count: validationTeamCount,
  test_team_count: testTeamCount,
  candidate_species_count: candidateCount,
  prior_smoothing: PRIOR_SMOOTHING,
  pair_smoothing: PAIR_SMOOTHING,
  lambda_grid: lambdaGrid,
  selected_lambda: selectedLambda,
  validation,
  test: {
    uniform_compatible: test.uniform,
    unigram_compatible: test.byLambda["0"],
    pairwise_compatible: test.byLambda[String(selectedLambda)],
  },
  real_staraptor_fixture: {
    known_species: realKnownSpecies,
    support_count: realSupport.length,
    top_20: realPrior.slice(0, 20),
    species_prior: realPrior,
  },
  caveats: [
    "model-is-a-learned-team-composition-prior-not-an-exact-generator-posterior",
    "emitted-species-not-having-their-own-random-set-key-are-canonicalized-to-a-unique-base-species-set-family",
    "conditioning-uses-team-species-only-not-revealed-moves-items-abilities-or-tera",
    "candidate-support-is-a-species-clause-safe-overapproximation-because-set-families-can-emit-forms-with-different-typing",
    "pairwise-model-ignores-higher-order-team-correlations",
    "lambda-is-selected-only-on-the-validation-seed-range",
    "test-seed-range-is-not-used-for-model-selection",
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
