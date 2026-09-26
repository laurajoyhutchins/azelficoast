#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const readline = require("node:readline");
const {runProbe} = require("./probe_real_belief_trace.cjs");

const showdownRoot = process.argv[2];
if (!showdownRoot) {
  process.stderr.write("usage: probe_real_belief_worker.cjs SHOWDOWN_ROOT\n");
  process.exit(2);
}

const generatorCacheDir = fs.mkdtempSync(
  path.join(os.tmpdir(), "azelficoast-showdown-probe-")
);

function environmentInteger(name, fallback, {min = 1} = {}) {
  const raw = process.env[name];
  if (raw == null || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < min) {
    throw new Error(`${name} must be an integer >= ${min}`);
  }
  return value;
}

const maxSessions = environmentInteger(
  "AZELFICOAST_SHOWDOWN_PROBE_SESSIONS",
  4
);
const maxTransitionExecutions = environmentInteger(
  "AZELFICOAST_SHOWDOWN_TRANSITION_CACHE_ENTRIES",
  512
);
const maxProjectionExecutions = environmentInteger(
  "AZELFICOAST_SHOWDOWN_PUBLIC_PROJECTION_CACHE_ENTRIES",
  512
);
const sessions = new Map();
const transitionExecutionCache = new Map();
const transitionProjectionCache = new Map();

globalThis.__azelficoastTransitionExecutionCache = transitionExecutionCache;
globalThis.__azelficoastTransitionProjectionCache = transitionProjectionCache;
globalThis.__azelficoastTransitionExecutionCacheMaxEntries = maxTransitionExecutions;
globalThis.__azelficoastTransitionProjectionCacheMaxEntries = maxProjectionExecutions;

function emit(response) {
  process.stdout.write(JSON.stringify(response) + "\n");
}

function createSession(sessionKey, source) {
  const result = runProbe(
    [
      showdownRoot,
      `azelficoast://fixture/${sessionKey}.json`,
      "--posterior-only",
      "--generator-cache-dir",
      generatorCacheDir,
    ],
    source
  );
  if (
    !result ||
    typeof result !== "object" ||
    !result.document ||
    typeof result.document !== "object" ||
    typeof result.compileTransitionProgram !== "function"
  ) {
    throw new Error("posterior probe did not return a resumable session");
  }

  sessions.delete(sessionKey);
  sessions.set(sessionKey, {
    compileTransitionProgram: result.compileTransitionProgram,
  });
  while (sessions.size > maxSessions) {
    const oldest = sessions.keys().next().value;
    sessions.delete(oldest);
  }
  return result.document;
}

function compileTransitionProgram(sessionKey, cacheMode) {
  if (!["fresh", "exact", "projection", "projection-first"].includes(cacheMode)) {
    throw new Error("unknown transition cache mode");
  }
  const session = sessions.get(sessionKey);
  if (!session) {
    throw new Error("posterior session is unavailable; reconstruct before exact search");
  }
  try {
    return session.compileTransitionProgram(cacheMode);
  } finally {
    sessions.delete(sessionKey);
  }
}

function releaseSession(sessionKey) {
  return sessions.delete(sessionKey);
}

async function handle(request) {
  if (!request || typeof request !== "object") {
    throw new Error("request must be a JSON object");
  }
  const id = request.id;
  const op = request.op;
  const session = request.session;
  if (!Number.isSafeInteger(id) || id < 0) {
    throw new Error("request id must be a nonnegative integer");
  }
  if (typeof session !== "string" || !/^[0-9a-f]{64}$/.test(session)) {
    throw new Error("session must be a sha256 hex digest");
  }

  if (op === "posterior") {
    if (!request.source || typeof request.source !== "object") {
      throw new Error("posterior request requires source");
    }
    return {id, ok: true, document: createSession(session, request.source)};
  }
  if (op === "transition_program") {
    const cacheMode =
      typeof request.cache_mode === "string" ? request.cache_mode : "projection";
    return {
      id,
      ok: true,
      document: compileTransitionProgram(session, cacheMode),
    };
  }
  if (op === "release") {
    return {id, ok: true, released: releaseSession(session)};
  }
  throw new Error("unknown probe operation");
}

const input = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

input.on("line", async line => {
  if (!line.trim()) return;
  let request;
  try {
    request = JSON.parse(line);
    emit(await handle(request));
  } catch (error) {
    emit({
      id:
        request && Number.isSafeInteger(request.id)
          ? request.id
          : null,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

process.on("exit", () => {
  try {
    fs.rmSync(generatorCacheDir, {recursive: true, force: true});
  } catch {}
});
