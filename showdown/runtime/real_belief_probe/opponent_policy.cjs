"use strict";

function createOpponentPolicyEngine({
  Battle,
  benchFactorField,
  benchPrior,
  fail,
  instrumentPublicRootReads,
  opponentPolicy,
  publicOpponentView,
  toID,
}) {
  const BENCH_FACTOR_FIELD = benchFactorField;
  const BENCH_PRIOR = benchPrior;
  const OPPONENT_POLICY = opponentPolicy;

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


  return {opponentDistributionForSnapshot};
}

module.exports = {createOpponentPolicyEngine};
