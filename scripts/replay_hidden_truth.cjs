#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const {execFileSync} = require("node:child_process");

const PINNED_SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc";

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

function hiddenFor(battle, perspective) {
  const ownIndex = Number(perspective.slice(1)) - 1;
  const opponentIndex = ownIndex === 0 ? 1 : 0;
  const opponent = battle.sides[opponentIndex];
  const active = opponent && opponent.active && opponent.active[0];
  if (!active || !active.set) {
    throw new Error(`${perspective}: opponent active truth is unavailable`);
  }
  const set = active.set;
  const hidden = {
    "opponent.active.item": set.item || "",
    "opponent.active.ability": set.ability || "",
    "opponent.active.moves": [...(set.moves || [])].map(toID).sort(),
    "opponent.active.tera_type": set.teraType || "",
    "opponent.active.evs": stable(set.evs || {}),
    "opponent.active.ivs": stable(set.ivs || {}),
    "opponent.active.exact_hp": Number(active.hp),
  };
  return {
    opponent_species: toID(active.species && active.species.id ? active.species.id : set.species),
    opponent_hidden_world_id: sha256(hidden),
    opponent_hidden: hidden,
  };
}

async function main() {
  const showdownRoot = process.argv[2];
  const inputPath = process.argv[3];
  const replayId = process.argv[4] || null;
  if (!showdownRoot || !inputPath) {
    throw new Error(
      "usage: replay_hidden_truth.cjs SHOWDOWN_ROOT INPUTLOG [REPLAY_ID]"
    );
  }

  const actualCommit = execFileSync(
    "git",
    ["-C", showdownRoot, "rev-parse", "HEAD"],
    {encoding: "utf8"}
  ).trim();
  if (actualCommit !== PINNED_SHOWDOWN_COMMIT) {
    throw new Error(
      `expected Showdown ${PINNED_SHOWDOWN_COMMIT}, got ${actualCommit}`
    );
  }

  const battleStreams = require(path.join(
    showdownRoot,
    "dist",
    "sim",
    "battle-stream.js"
  ));
  const BattleStream = battleStreams.BattleStream;
  if (typeof BattleStream !== "function") {
    throw new Error("pinned Showdown does not expose BattleStream");
  }

  const inputlog = fs.readFileSync(inputPath, "utf8").replace(/\r/g, "");
  if (
    !inputlog.includes(">start ") ||
    !inputlog.includes(">player p1 ") ||
    !inputlog.includes(">player p2 ")
  ) {
    throw new Error("inputlog lacks required battle initialization");
  }
  if (/(^|\n)>eval\s/.test(inputlog)) {
    throw new Error("refusing to execute replay inputlog containing >eval");
  }

  const stream = new BattleStream({keepAlive: true});
  const truths = {p1: [], p2: []};
  const capturedChoices = {p1: [], p2: []};

  for (const line of inputlog.split("\n")) {
    if (!line) continue;
    const choiceMatch = line.match(/^>(p[12])\s+(.+)$/);
    if (choiceMatch) {
      const side = choiceMatch[1];
      const choice = choiceMatch[2].trim();
      if (choice === "undo") {
        const captured = capturedChoices[side].pop();
        if (captured === true) truths[side].pop();
        await stream.write(line);
        continue;
      }

      const actionKind = choice.split(/\s+/, 1)[0].toLowerCase();
      const capturesDecision =
        actionKind === "move" || actionKind === "switch";
      if (capturesDecision) {
        if (!stream.battle) {
          throw new Error(`${side}: choice occurred before battle initialization`);
        }
        truths[side].push({
          decision_index: truths[side].length,
          ...hiddenFor(stream.battle, side),
        });
      }
      capturedChoices[side].push(capturesDecision);
    }
    await stream.write(line);
  }
  await stream.writeEnd();

  process.stdout.write(JSON.stringify({
    schema: "azelficoast.public-replay-hidden-truth",
    schema_version: 1,
    replay_id: replayId,
    showdown_commit: actualCommit,
    truth_scope:
      "post-hoc omniscient validation only; never supplied to policy, posterior, search, or training inputs",
    sides: truths,
  }));
}

main().catch(error => {
  process.stderr.write(String(error && error.stack ? error.stack : error) + "\n");
  process.exit(2);
});
