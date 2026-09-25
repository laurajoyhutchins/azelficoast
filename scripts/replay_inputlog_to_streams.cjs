#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");

async function main() {
  const showdownRoot = process.argv[2];
  const inputPath = process.argv[3];
  if (!showdownRoot || !inputPath) {
    throw new Error("usage: replay_inputlog_to_streams.cjs SHOWDOWN_ROOT INPUTLOG");
  }

  const battleStreams = require(path.join(
    showdownRoot,
    "dist",
    "sim",
    "battle-stream.js"
  ));
  const BattleStream = battleStreams.BattleStream;
  const getPlayerStreams = battleStreams.getPlayerStreams;
  if (typeof BattleStream !== "function" || typeof getPlayerStreams !== "function") {
    throw new Error("pinned Showdown does not expose BattleStream/getPlayerStreams");
  }

  const inputlog = fs.readFileSync(inputPath, "utf8").replace(/\r/g, "");
  if (!inputlog.includes(">start ") || !inputlog.includes(">player p1 ") || !inputlog.includes(">player p2 ")) {
    throw new Error("inputlog lacks required battle initialization");
  }
  if (/(^|\n)>eval\s/.test(inputlog)) {
    throw new Error("refusing to execute replay inputlog containing >eval");
  }

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);

  async function collect(playerStream) {
    const chunks = [];
    for await (const chunk of playerStream) {
      chunks.push(String(chunk));
    }
    return chunks;
  }

  const p1Read = collect(streams.p1);
  const p2Read = collect(streams.p2);
  await streams.omniscient.write(inputlog);
  await streams.omniscient.writeEnd();

  const [p1, p2] = await Promise.all([p1Read, p2Read]);
  process.stdout.write(JSON.stringify({
    schema: "azelficoast.showdown-player-streams",
    schema_version: 1,
    p1,
    p2,
  }));
}

main().catch(error => {
  process.stderr.write(String(error && error.stack ? error.stack : error) + "\n");
  process.exit(2);
});
