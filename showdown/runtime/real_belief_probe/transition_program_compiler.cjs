"use strict";

const {performance} = require("node:perf_hooks");
const {
  applySuccessorDelta,
  successorDelta,
} = require("../transition_successor_delta.cjs");

function createTransitionProgramCompiler({
  Battle,
  State,
  actualCommit,
  sourceFixtureId,
  publicRootDependencySchema,
  publicBattleCandidates,
  publicPokemonCandidates,
  opponentPolicy,
  rootChanceSamples,
  chanceSeedFamily,
  dependencyCandidates,
  worlds,
  legalActions,
  stable,
  sha256,
  sha256PythonCanonical,
  fail,
  buildBattle,
  opponentDistributionForSnapshot,
  cloneBattle,
  seed,
  instrumentOpponentHiddenReads,
  instrumentPublicRootReads,
  rootChoice,
  legalP1Continuations,
  observation,
  stateSummary,
}) {
  const PUBLIC_ROOT_DEPENDENCY_SCHEMA = publicRootDependencySchema;
  const PUBLIC_BATTLE_CANDIDATES = publicBattleCandidates;
  const PUBLIC_POKEMON_CANDIDATES = publicPokemonCandidates;
  const OPPONENT_POLICY = opponentPolicy;
  const ROOT_CHANCE_SAMPLES = rootChanceSamples;
  const CHANCE_SEED_FAMILY = chanceSeedFamily;
  const DEPENDENCY_CANDIDATES = dependencyCandidates;

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

function compileLazyWholeTurnPrograms(
  cacheMode = globalThis.__azelficoastTransitionCacheMode || "projection"
) {
  if (!["fresh", "exact", "projection", "projection-first"].includes(cacheMode)) {
    throw new Error("unknown transition cache mode: " + cacheMode);
  }
  const cacheProbeOrder =
    cacheMode === "projection-first"
      ? ["projection", "exact"]
      : cacheMode === "projection"
        ? ["exact", "projection"]
        : cacheMode === "exact"
          ? ["exact"]
          : [];
  const orderedWorlds = [...worlds].sort((left, right) =>
    left.world_id.localeCompare(right.world_id)
  );
  const executionCache = new Map();
  const rootSnapshotCache = new Map();
  const sharedExecutionCache = sharedTransitionExecutionCache();
  const sharedProjectionCache = sharedTransitionProjectionCache();
  let exactExecutionCacheHits = 0;
  let exactExecutionCacheMisses = 0;
  let projectedExecutionCacheHits = 0;
  let projectedExecutionCacheMisses = 0;
  let sharedExecutionCacheMisses = 0;
  let exactCacheHitMs = 0;
  let exactCacheMissMs = 0;
  let projectionCacheHitMs = 0;
  let projectionCacheMissMs = 0;
  let freshExecutionMs = 0;
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

      for (const cacheKind of cacheProbeOrder) {
        if (execution !== undefined) break;

        if (cacheKind === "exact" && sharedExecutionCache !== null) {
          const started = performance.now();
          execution = sharedCacheGet(sharedExecutionCache, key);
          const elapsed = performance.now() - started;
          if (execution !== undefined) {
            reused = true;
            reuseKind = "exact";
            exactExecutionCacheHits++;
            exactCacheHitMs += elapsed;
          } else {
            exactExecutionCacheMisses++;
            exactCacheMissMs += elapsed;
          }
          continue;
        }

        if (
          cacheKind === "projection" &&
          root.material.complete &&
          sharedProjectionCache !== null
        ) {
          const started = performance.now();
          execution = sharedProjectionCacheGet(
            sharedProjectionCache,
            projectionBaseKey,
            root.material,
            world
          );
          const elapsed = performance.now() - started;
          if (execution !== undefined) {
            reused = true;
            reuseKind = "public-projection";
            projectedExecutionCacheHits++;
            projectionCacheHitMs += elapsed;
            if (sharedExecutionCache !== null) {
              sharedCacheSet(sharedExecutionCache, key, execution);
            }
          } else {
            projectedExecutionCacheMisses++;
            projectionCacheMissMs += elapsed;
          }
        }
      }

      if (execution === undefined) {
        sharedExecutionCacheMisses++;
        const started = performance.now();
        execution = immediateWholeTurn(world, action, root.snapshot);
        freshExecutionMs += performance.now() - started;
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
      source_fixture_id: sourceFixtureId,
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
    source_fixture_id: sourceFixtureId,
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
      transition_cache_mode: cacheMode,
      cache_probe_order: cacheProbeOrder,
      physical_cost_observations: {
        exact_cache_hit_count: exactExecutionCacheHits,
        exact_cache_hit_total_ms: exactCacheHitMs,
        exact_cache_miss_count: exactExecutionCacheMisses,
        exact_cache_miss_total_ms: exactCacheMissMs,
        projection_cache_hit_count: projectedExecutionCacheHits,
        projection_cache_hit_total_ms: projectionCacheHitMs,
        projection_cache_miss_count: projectedExecutionCacheMisses,
        projection_cache_miss_total_ms: projectionCacheMissMs,
        fresh_execution_count: sharedExecutionCacheMisses,
        fresh_execution_total_ms: freshExecutionMs,
        route_execution_count: uniqueExecutions,
        route_total_ms:
          exactCacheHitMs +
          exactCacheMissMs +
          projectionCacheHitMs +
          projectionCacheMissMs +
          freshExecutionMs,
      },
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

  return {compileLazyWholeTurnPrograms};
}

module.exports = {createTransitionProgramCompiler};
