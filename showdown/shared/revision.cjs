"use strict";

const path = require("node:path");

const revision = require(path.join(__dirname, "..", "revision.json"));
if (
  !revision ||
  revision.schema !== "azelficoast.showdown-revision" ||
  revision.schema_version !== 1 ||
  typeof revision.commit !== "string" ||
  !/^[0-9a-f]{40}$/.test(revision.commit)
) {
  throw new Error("invalid repository Showdown revision authority");
}

module.exports = {PINNED_SHOWDOWN_COMMIT: revision.commit};
