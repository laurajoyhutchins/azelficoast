#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const {execFileSync} = require("node:child_process");

function fail(message) {
  process.stderr.write(String(message) + "\n");
  process.exit(2);
}

const args = process.argv.slice(2);
const showdownRoot = args.shift();
const fixturePath = args.shift();
let targetParticles = 256;
let minimumParticles = 32;
let maxRounds = 262144;
let seedFamily = 0;

while (args.length) {
  const flag = args.shift();
  const value = args.shift();
  if (value == null) fail(flag + " requires a value");
  if (flag === "--target-particles") targetParticles = Number(value);
  else if (flag === "--minimum-particles") minimumParticles = Number(value);
  else if (flag === "--max-rounds") maxRounds = Number(value);
  else if (flag === "--seed-family") seedFamily = Number(value);
  else fail("unknown argument: " + flag);
}
if (!showdownRoot || !fixturePath) {
  fail(
    "usage: sample_joint_random_battle_posterior.cjs SHOWDOWN_ROOT FIXTURE_JSON " +
    "[--target-particles N] [--minimum-particles N] [--max-rounds N] [--seed-family N]"
  );
}
for (const [name, value, minimum] of [
  ["target-particles", targetParticles, 1],
  ["minimum-particles", minimumParticles, 1],
  ["max-rounds", maxRounds, 1],
  ["seed-family", seedFamily, 0],
]) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    fail(name + " must be an integer >= " + minimum);
  }
}
if (minimumParticles > targetParticles) {
  fail("minimum-particles may not exceed target-particles");
}

const {PINNED_SHOWDOWN_COMMIT: SHOWDOWN_COMMIT} = require("../../shared/revision.cjs");
const actualCommit = execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();
if (actualCommit !== SHOWDOWN_COMMIT) {
  fail("expected Showdown " + SHOWDOWN_COMMIT + ", got " + actualCommit);
}

const inputDocument = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
const fixture =
  inputDocument && inputDocument.fixture && inputDocument.fixture.state
    ? inputDocument.fixture
    : inputDocument;
if (!fixture || !fixture.state || !Array.isArray(fixture.protocol_prefix)) {
  fail("fixture must contain state and protocol_prefix");
}
if (!fixture.fixture_id) fail("fixture lacks fixture_id");

const {Teams} = require(path.join(showdownRoot, "dist", "sim", "teams.js"));
const generator = Teams.getGenerator("gen9randombattle", [0, 0, 0, 0]);
const dex = generator.dex;

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map(key => [key, stable(value[key])])
    );
  }
  return value;
}

function canonical(value) {
  return JSON.stringify(stable(value));
}

function sha256(value) {
  return crypto.createHash("sha256").update(canonical(value)).digest("hex");
}

function toID(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function seed(index) {
  const digest = crypto
    .createHash("sha256")
    .update("azelficoast-joint-randbats:" + seedFamily + ":" + index)
    .digest();
  return [0, 2, 4, 6].map(offset => digest.readUInt16BE(offset));
}

function actorSide(actor) {
  const text = String(actor || "");
  if (text.startsWith("p1")) return "p1";
  if (text.startsWith("p2")) return "p2";
  return null;
}

function actorSlot(actor) {
  return String(actor || "").split(":", 1)[0];
}

function speciesFromDetails(details) {
  return String(details || "").split(",", 1)[0].trim();
}

function levelFromDetails(details) {
  for (const token of String(details || "").split(",").slice(1)) {
    const trimmed = token.trim();
    if (/^L\d+$/.test(trimmed)) return Number(trimmed.slice(1));
  }
  return null;
}

function genderFromDetails(details) {
  for (const token of String(details || "").split(",").slice(1)) {
    const trimmed = token.trim();
    if (trimmed === "M" || trimmed === "F" || trimmed === "N") return trimmed;
  }
  return null;
}

function resolveSides() {
  const ownName = toID(fixture.state.player);
  const opponentName = toID(fixture.state.opponent);
  let own = null;
  let opponent = null;
  for (const batch of fixture.protocol_prefix) {
    for (const message of batch) {
      if (
        Array.isArray(message) &&
        message[0] === "" &&
        message[1] === "player" &&
        ["p1", "p2"].includes(message[2])
      ) {
        const name = toID(message[3]);
        if (name && name === ownName) own = message[2];
        if (name && name === opponentName) opponent = message[2];
      }
    }
  }
  const sideFromTeamKeys = values => {
    const sides = new Set();
    for (const key of Object.keys(values || {})) {
      if (String(key).startsWith("p1")) sides.add("p1");
      if (String(key).startsWith("p2")) sides.add("p2");
    }
    return sides.size === 1 ? [...sides][0] : null;
  };
  if (!own) own = sideFromTeamKeys(fixture.state.team);
  if (!opponent) opponent = sideFromTeamKeys(fixture.state.opponent_team);

  if (!opponent) {
    const publicOpponentSpecies = new Set(
      Object.values(fixture.state.opponent_team || {})
        .map(view => toID(view && view.species))
        .filter(Boolean)
    );
    const observedSides = new Set();
    for (const batch of fixture.protocol_prefix) {
      for (const message of batch) {
        if (
          Array.isArray(message) &&
          message[0] === "" &&
          ["switch", "drag", "detailschange"].includes(message[1])
        ) {
          const side = actorSide(message[2]);
          const species = toID(speciesFromDetails(message[3]));
          if (side && publicOpponentSpecies.has(species)) observedSides.add(side);
        }
      }
    }
    if (observedSides.size === 1) opponent = [...observedSides][0];
  }

  if (!own && opponent) own = opponent === "p1" ? "p2" : "p1";
  if (!opponent && own) opponent = own === "p1" ? "p2" : "p1";
  if (!opponent) fail("cannot resolve opponent protocol side from public history");
  return {own, opponent};
}

const SIDES = resolveSides();

function blankEvidence(species) {
  return {
    species: toID(species),
    levels: new Set(),
    genders: new Set(),
    moves: new Set(),
    abilities: new Set(),
    items: new Set(),
    tera_types: new Set(),
    was_lead: false,
    transformed: false,
    identity_ambiguous: false,
    item_history_ambiguous: false,
    ability_history_ambiguous: false,
    public_views: 0,
  };
}

const evidenceBySpecies = new Map();
function evidenceFor(species) {
  const id = toID(species);
  if (!id) return null;
  if (!evidenceBySpecies.has(id)) evidenceBySpecies.set(id, blankEvidence(id));
  return evidenceBySpecies.get(id);
}

const activeBySlot = new Map();
let opponentLead = null;
let opponentTeamSize = null;
let firstTurnStarted = false;

for (const batch of fixture.protocol_prefix) {
  for (const message of batch) {
    if (!Array.isArray(message) || message[0] !== "" || message.length < 2) continue;
    const kind = message[1];

    if (kind === "turn") {
      firstTurnStarted = true;
      continue;
    }

    if (
      kind === "teamsize" &&
      message[2] === SIDES.opponent &&
      Number.isSafeInteger(Number(message[3]))
    ) {
      opponentTeamSize = Number(message[3]);
      continue;
    }

    if (kind === "replace" && actorSide(message[2]) === SIDES.opponent) {
      const current = activeBySlot.get(actorSlot(message[2]));
      if (current) evidenceFor(current).identity_ambiguous = true;
      const replacement = speciesFromDetails(message[3]);
      if (replacement) evidenceFor(replacement).identity_ambiguous = true;
      continue;
    }

    if (
      ["switch", "drag", "detailschange"].includes(kind) &&
      actorSide(message[2]) === SIDES.opponent
    ) {
      const species = speciesFromDetails(message[3]);
      const row = evidenceFor(species);
      if (!row) continue;
      const slot = actorSlot(message[2]);
      activeBySlot.set(slot, row.species);
      if (
        !opponentLead &&
        !firstTurnStarted &&
        ["switch", "drag"].includes(kind)
      ) {
        opponentLead = row.species;
        row.was_lead = true;
      }
      const level = levelFromDetails(message[3]);
      const gender = genderFromDetails(message[3]);
      if (level) row.levels.add(level);
      if (gender) row.genders.add(gender);
      continue;
    }

    if (actorSide(message[2]) !== SIDES.opponent) continue;
    const species = activeBySlot.get(actorSlot(message[2]));
    const row = evidenceFor(species);
    if (!row) continue;

    const extras = message.slice(3).map(String);
    const sourceOwnerField = extras.find(field => field.startsWith("[of] "));
    const sourceOwnerActor = sourceOwnerField
      ? sourceOwnerField.slice("[of] ".length)
      : message[2];
    const sourceOwnerSpecies =
      actorSide(sourceOwnerActor) === SIDES.opponent
        ? activeBySlot.get(actorSlot(sourceOwnerActor))
        : null;
    const sourceOwner = sourceOwnerSpecies
      ? evidenceFor(sourceOwnerSpecies)
      : null;
    const sourceItemField = extras.find(field => field.startsWith("[from] item: "));
    if (
      sourceOwner &&
      sourceItemField &&
      !sourceOwner.item_history_ambiguous
    ) {
      sourceOwner.items.add(
        toID(sourceItemField.slice("[from] item: ".length))
      );
    }
    const sourceAbilityField = extras.find(
      field => field.startsWith("[from] ability: ")
    );
    if (
      sourceOwner &&
      sourceAbilityField &&
      !sourceOwner.ability_history_ambiguous
    ) {
      sourceOwner.abilities.add(
        toID(sourceAbilityField.slice("[from] ability: ".length))
      );
    }

    if (kind === "move" && message[3]) {
      row.moves.add(toID(message[3]));
      continue;
    }
    if (kind === "-terastallize" && message[3]) {
      row.tera_types.add(toID(message[3]));
      continue;
    }
    if (kind === "-transform") {
      row.transformed = true;
      continue;
    }
    if (kind === "-ability" && message[3]) {
      const extras = message.slice(4).map(String);
      const changedByMove = extras.some(field => field.startsWith("[from] move:"));
      if (changedByMove) row.ability_history_ambiguous = true;
      else if (!row.ability_history_ambiguous) row.abilities.add(toID(message[3]));
      continue;
    }
    if (kind === "-item" && message[3]) {
      const extras = message.slice(4).map(String);
      const friskReveal = extras.some(field => field.includes("ability: Frisk"));
      if (friskReveal && !row.item_history_ambiguous) {
        row.items.add(toID(message[3]));
      } else {
        row.item_history_ambiguous = true;
      }
      continue;
    }
    if (kind === "-enditem" && message[3]) {
      if (!row.item_history_ambiguous && row.items.size === 0) {
        row.items.add(toID(message[3]));
      }
      continue;
    }
  }
}

for (const view of Object.values(fixture.state.opponent_team || {})) {
  if (!view || typeof view !== "object") continue;
  const row = evidenceFor(view.species);
  if (!row) continue;
  row.public_views++;
  if (view.transformed) row.transformed = true;
  if (Number.isSafeInteger(Number(view.level)) && Number(view.level) > 0) {
    row.levels.add(Number(view.level));
  }
  for (const move of view.moves || []) row.moves.add(toID(move));
  if (view.ability && !row.ability_history_ambiguous) {
    row.abilities.add(toID(view.ability));
  }
  if (view.item && !row.item_history_ambiguous) {
    row.items.add(toID(view.item));
  }
  if (view.tera_type) row.tera_types.add(toID(view.tera_type));
}

for (const row of evidenceBySpecies.values()) {
  if (row.transformed) {
    fail("opponent transform is not yet supported by joint posterior conditioning");
  }
  if (row.identity_ambiguous) {
    fail("opponent Illusion/replace identity is not yet supported by joint posterior conditioning");
  }
  for (const [name, values] of [
    ["level", row.levels],
    ["gender", row.genders],
    ["ability", row.abilities],
    ["item", row.items],
    ["tera type", row.tera_types],
  ]) {
    if (values.size > 1) {
      fail("conflicting public " + name + " evidence for " + row.species);
    }
  }
}

const evidenceRows = [...evidenceBySpecies.values()]
  .map(row => ({
    species: row.species,
    level: row.levels.size ? [...row.levels][0] : null,
    gender: row.genders.size ? [...row.genders][0] : null,
    moves: [...row.moves].sort(),
    ability: row.abilities.size ? [...row.abilities][0] : null,
    item: row.items.size ? [...row.items][0] : null,
    tera_type: row.tera_types.size ? [...row.tera_types][0] : null,
    was_lead: row.was_lead,
    item_history_ambiguous: row.item_history_ambiguous,
    ability_history_ambiguous: row.ability_history_ambiguous,
    public_views: row.public_views,
  }))
  .sort((a, b) => a.species.localeCompare(b.species));

if (!evidenceRows.length) fail("public history exposes no opponent species");
if (opponentTeamSize !== null && evidenceRows.length > opponentTeamSize) {
  fail("public species count exceeds public team size");
}

function canonicalSet(set, index) {
  return {
    species: toID(set.species),
    level: Number(set.level),
    gender: set.gender || null,
    ability: toID(set.ability),
    item: toID(set.item),
    moves: [...set.moves].map(toID).sort(),
    tera_type: toID(set.teraType),
    role: set.role || null,
    nature: set.nature || null,
    evs: stable(set.evs || {}),
    ivs: stable(set.ivs || {}),
    generator_slot: index,
  };
}

function compatibleSet(set, evidence) {
  if (set.species !== evidence.species) return false;
  if (evidence.level !== null && set.level !== evidence.level) return false;
  if (evidence.gender !== null && set.gender !== evidence.gender) return false;
  if (evidence.ability !== null && set.ability !== evidence.ability) return false;
  if (evidence.item !== null && set.item !== evidence.item) return false;
  if (evidence.tera_type !== null && set.tera_type !== evidence.tera_type) return false;
  if (!evidence.moves.every(move => set.moves.includes(move))) return false;
  return true;
}

function compatibleTeam(team) {
  if (opponentTeamSize !== null && team.length !== opponentTeamSize) return false;
  const bySpecies = new Map(team.map(set => [set.species, set]));
  for (const evidence of evidenceRows) {
    const set = bySpecies.get(evidence.species);
    if (!set || !compatibleSet(set, evidence)) return false;
  }
  if (opponentLead && (!team[0] || team[0].species !== opponentLead)) return false;
  return true;
}

function semanticParticle(team) {
  const lead = team[0]?.species || null;
  const sets = team
    .map(set => ({
      species: set.species,
      level: set.level,
      gender: set.gender,
      ability: set.ability,
      item: set.item,
      moves: set.moves,
      tera_type: set.tera_type,
      role: set.role,
      nature: set.nature,
      evs: set.evs,
      ivs: set.ivs,
      was_lead: set.species === lead,
    }))
    .sort((a, b) => a.species.localeCompare(b.species));
  return {team: sets};
}

const randomSetSpecies = Object.keys(generator.randomSets)
  .map(id => dex.species.get(id))
  .filter(species => species.exists);

function increment(object, key) {
  object[key] = (object[key] || 0) + 1;
}

function weakToFreezeDry(species) {
  return (
    dex.getEffectiveness("Ice", species) > 0 ||
    (dex.getEffectiveness("Ice", species) > -2 && species.types.includes("Water"))
  );
}

function blankSpeciesContext() {
  return {
    species: [],
    baseSpecies: new Set(),
    typeCount: {},
    typeWeaknesses: {},
    typeDoubleWeaknesses: {},
    freezeDryWeaknesses: 0,
    level100Count: 0,
  };
}

function addSpeciesContext(context, species) {
  context.species.push({speciesId: species.id});
  context.baseSpecies.add(species.baseSpecies);
  for (const type of species.types) increment(context.typeCount, type);
  for (const type of dex.types.names()) {
    const effectiveness = dex.getEffectiveness(type, species);
    if (effectiveness > 0) increment(context.typeWeaknesses, type);
    if (effectiveness > 1) increment(context.typeDoubleWeaknesses, type);
  }
  if (weakToFreezeDry(species)) context.freezeDryWeaknesses++;
  if (generator.getLevel(species, false) === 100) context.level100Count++;
}

function speciesCompatibleWithContext(species, context, finalSlot) {
  if (context.baseSpecies.has(species.baseSpecies)) return false;
  if (species.baseSpecies === "Zoroark" && finalSlot) return false;

  for (const type of species.types) {
    if ((context.typeCount[type] || 0) >= 2) return false;
  }
  for (const type of dex.types.names()) {
    const effectiveness = dex.getEffectiveness(type, species);
    if (effectiveness > 0 && (context.typeWeaknesses[type] || 0) >= 3) {
      return false;
    }
    if (
      effectiveness > 1 &&
      (context.typeDoubleWeaknesses[type] || 0) >= 1
    ) {
      return false;
    }
  }
  if (
    dex.getEffectiveness("Fire", species) === 0 &&
    Object.values(species.abilities).some(
      ability => ability === "Dry Skin" || ability === "Fluffy"
    ) &&
    (context.typeWeaknesses.Fire || 0) >= 3
  ) {
    return false;
  }
  if (weakToFreezeDry(species) && context.freezeDryWeaknesses >= 4) {
    return false;
  }
  if (generator.getLevel(species, false) === 100 && context.level100Count >= 1) {
    return false;
  }
  if (!generator.getPokemonCompatibility(species, context.species, false)) {
    return false;
  }
  return true;
}

function generatorSpeciesForEvidence(evidence) {
  const requested = dex.species.get(evidence.species);
  const ids = [
    requested.id,
    typeof requested.battleOnly === "string" ? toID(requested.battleOnly) : "",
    typeof requested.baseSpecies === "string" ? toID(requested.baseSpecies) : "",
  ].filter(Boolean);
  for (const id of [...new Set(ids)]) {
    if (generator.randomSets[id]) return dex.species.get(id);
  }
  return null;
}

function updateTeamDetails(teamDetails, set, species) {
  if (set.ability === "Drizzle" || set.moves.includes("raindance")) {
    teamDetails.rain = 1;
  }
  if (
    set.ability === "Drought" ||
    set.ability === "Orichalcum Pulse" ||
    set.moves.includes("sunnyday")
  ) {
    teamDetails.sun = 1;
  }
  if (set.ability === "Sand Stream") teamDetails.sand = 1;
  if (
    set.ability === "Snow Warning" ||
    set.moves.includes("snowscape") ||
    set.moves.includes("chillyreception")
  ) {
    teamDetails.snow = 1;
  }
  if (set.moves.includes("healbell")) teamDetails.statusCure = 1;
  if (set.moves.includes("spikes") || set.moves.includes("ceaselessedge")) {
    teamDetails.spikes = (teamDetails.spikes || 0) + 1;
  }
  if (set.moves.includes("toxicspikes") || set.ability === "Toxic Debris") {
    teamDetails.toxicSpikes = 1;
  }
  if (set.moves.includes("stealthrock") || set.moves.includes("stoneaxe")) {
    teamDetails.stealthRock = 1;
  }
  if (set.moves.includes("stickyweb")) teamDetails.stickyWeb = 1;
  if (set.moves.includes("defog")) teamDetails.defog = 1;
  if (set.moves.includes("rapidspin") || set.moves.includes("mortalspin")) {
    teamDetails.rapidSpin = 1;
  }
  if (
    set.moves.includes("auroraveil") ||
    (set.moves.includes("reflect") && set.moves.includes("lightscreen"))
  ) {
    teamDetails.screens = 1;
  }
  if (
    set.role === "Tera Blast user" ||
    ["ogerpon", "ogerponhearthflame", "terapagos"].includes(species.id)
  ) {
    teamDetails.teraBlast = 1;
  }
}

function shuffled(values) {
  const output = [...values];
  for (let index = output.length - 1; index > 0; index--) {
    const swap = generator.random(index + 1);
    [output[index], output[swap]] = [output[swap], output[index]];
  }
  return output;
}

function sampleCompatibleKnownSet(species, evidence, teamDetails, isLead) {
  for (let retry = 0; retry < 512; retry++) {
    const raw = generator.randomSet(species, teamDetails, isLead, false);
    const candidate = canonicalSet(raw, 0);
    if (compatibleSet(candidate, evidence)) return raw;
  }
  return null;
}

function conditionedCompletionParticle(index) {
  generator.setSeed(seed(index));

  const known = [];
  const speciesContext = blankSpeciesContext();
  for (const evidence of evidenceRows) {
    const species = generatorSpeciesForEvidence(evidence);
    if (!species) return null;
    if (
      !speciesCompatibleWithContext(
        species,
        speciesContext,
        false
      )
    ) {
      return null;
    }
    addSpeciesContext(speciesContext, species);
    known.push({species, evidence});
  }

  const teamSize = opponentTeamSize || 6;
  const unknown = [];
  while (known.length + unknown.length < teamSize) {
    const finalSlot = known.length + unknown.length === teamSize - 1;
    const support = randomSetSpecies.filter(species =>
      speciesCompatibleWithContext(species, speciesContext, finalSlot)
    );
    if (!support.length) return null;
    const species = generator.sample(support);
    addSpeciesContext(speciesContext, species);
    unknown.push({species, evidence: null});
  }

  const leadEntry = opponentLead
    ? known.find(entry => entry.evidence.species === opponentLead)
    : null;
  if (opponentLead && !leadEntry) return null;

  const generationOrder = [
    ...(leadEntry ? [leadEntry] : []),
    ...shuffled(
      [...known, ...unknown].filter(entry => entry !== leadEntry)
    ),
  ];

  const teamDetails = {};
  const rawTeam = [];
  for (const entry of generationOrder) {
    const isLead = leadEntry
      ? entry === leadEntry
      : rawTeam.length === 0;
    let set;
    if (entry.evidence) {
      set = sampleCompatibleKnownSet(
        entry.species,
        entry.evidence,
        teamDetails,
        isLead
      );
      if (!set) return null;
    } else {
      set = generator.randomSet(entry.species, teamDetails, isLead, false);
    }
    const actualSpecies = dex.species.get(set.species);
    if (!actualSpecies.exists) return null;
    rawTeam.push(set);
    if (rawTeam.length < teamSize) {
      updateTeamDetails(teamDetails, set, actualSpecies);
    }
  }

  const canonicalTeam = rawTeam.map(canonicalSet);
  if (!compatibleTeam(canonicalTeam)) return null;

  // The proposal is allowed to reorder non-leads internally, but the public
  // lead remains semantic. Put it first before semanticParticle() records it.
  if (opponentLead) {
    canonicalTeam.sort((left, right) => {
      if (left.species === opponentLead) return -1;
      if (right.species === opponentLead) return 1;
      return 0;
    });
  }
  return canonicalTeam;
}

const particleCounts = new Map();
let accepted = 0;
let attempted = 0;
let generationErrors = 0;
const proposalMode =
  evidenceRows.length <= 1
    ? "generator_rejection"
    : "conditioned_completion";

for (let index = 0; index < maxRounds && accepted < targetParticles; index++) {
  attempted++;
  try {
    let team;
    if (proposalMode === "generator_rejection") {
      generator.setSeed(seed(index));
      team = generator.getTeam().map(canonicalSet);
      if (!compatibleTeam(team)) continue;
    } else {
      team = conditionedCompletionParticle(index);
      if (!team) continue;
    }
    accepted++;
    const hidden = semanticParticle(team);
    const key = canonical(hidden);
    const prior = particleCounts.get(key) || {hidden, count: 0};
    prior.count++;
    particleCounts.set(key, prior);
  } catch (_error) {
    generationErrors++;
  }
}

const particles = [...particleCounts.values()]
  .map(entry => ({
    world_id: sha256(entry.hidden),
    weight: entry.count / accepted,
    sample_count: entry.count,
    hidden: entry.hidden,
  }))
  .sort((a, b) => a.world_id.localeCompare(b.world_id));

const acceptanceRate = attempted ? accepted / attempted : 0;
const effectiveSampleSize = particles.length
  ? 1 / particles.reduce((sum, particle) => sum + particle.weight ** 2, 0)
  : 0;

const publicEvidence = {
  fixture_id: fixture.fixture_id,
  opponent_side: SIDES.opponent,
  opponent_team_size: opponentTeamSize,
  opponent_lead: opponentLead,
  revealed: evidenceRows,
  protocol_prefix_sha256: sha256(fixture.protocol_prefix),
  public_state_sha256: sha256(fixture.state),
};

const output = {
  schema: "azelficoast.joint-random-battle-posterior",
  schema_version: 1,
  source_fixture_id: fixture.fixture_id,
  showdown_commit: actualCommit,
  conditioned_on_public_history: true,
  realized_hidden_state_revealed: false,
  construction: {
    kind:
      proposalMode === "generator_rejection"
        ? "full-team-generator-rejection-particles"
        : "full-team-conditioned-completion-particles",
    generator: "Pokemon Showdown gen9randombattle",
    seed_family: seedFamily,
    seed_rule: "sha256(azelficoast-joint-randbats:<family>:<index>) first four uint16",
    attempted_team_count: attempted,
    accepted_team_count: accepted,
    unique_particle_count: particles.length,
    target_particles: targetParticles,
    minimum_particles: minimumParticles,
    max_rounds: maxRounds,
    generation_error_count: generationErrors,
    acceptance_rate: acceptanceRate,
    effective_sample_size: effectiveSampleSize,
    preserves_joint_team_set_correlations: true,
    posterior_treatment:
      proposalMode === "generator_rejection"
        ? "generator_faithful_joint_empirical"
        : "practical_joint_completion",
    proposal_mode: proposalMode,
    proposal_caveats:
      proposalMode === "generator_rejection"
        ? []
        : [
            "species completion is sampled from Showdown-compatible support rather than the exact conditional randomTeam probability",
            "set generation uses Showdown randomSet with shared teamDetails and rejects particles contradicting public set evidence",
          ],
    particle_atomicity: [
      "species",
      "moves",
      "item",
      "ability",
      "tera_type",
      "level",
      "gender",
      "role",
      "nature",
      "evs",
      "ivs",
      "unrevealed_teammates",
    ],
    conditioning_scope: [
      "revealed species",
      "public team size",
      "observed lead",
      "revealed level",
      "revealed gender",
      "observed moves",
      "unmutated revealed ability, including [from] ability protocol evidence",
      "unmutated or safely revealed item, including [from] item protocol evidence",
      "revealed Tera type",
    ],
    dynamic_battle_evidence_not_yet_likelihood_weighted: [
      "damage rolls",
      "speed-order observations",
      "status/secondary-effect feasibility",
    ],
  },
  public_evidence: publicEvidence,
  public_evidence_sha256: sha256(publicEvidence),
  support_status: accepted >= minimumParticles ? "sufficient" : "insufficient",
  worlds: particles,
};

output.posterior_sha256 = sha256({
  schema: output.schema,
  schema_version: output.schema_version,
  source_fixture_id: output.source_fixture_id,
  showdown_commit: output.showdown_commit,
  public_evidence_sha256: output.public_evidence_sha256,
  construction: output.construction,
  worlds: output.worlds,
});

process.stdout.write(JSON.stringify(output, null, 2) + "\n");
if (accepted < minimumParticles) process.exitCode = 3;
