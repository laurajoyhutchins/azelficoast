#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const readline = require("node:readline");
const vm = require("node:vm");
const childProcess = require("node:child_process");

const showdownRoot = process.argv[2];
if (!showdownRoot) {
  process.stderr.write("usage: probe_real_belief_worker.cjs SHOWDOWN_ROOT\n");
  process.exit(2);
}

const probePath = path.join(__dirname, "probe_real_belief_trace.cjs");
const probeSource = fs.readFileSync(probePath, "utf8");
const probeScript = new vm.Script(probeSource, {filename: probePath});
const transitionScript = new vm.Script(
  "JSON.stringify(compileLazyWholeTurnPrograms())",
  {filename: "azelficoast-transition-program.vm.js"}
);
const generatorCacheDir = fs.mkdtempSync(
  path.join(os.tmpdir(), "azelficoast-showdown-probe-")
);
const showdownCommit = childProcess.execFileSync(
  "git",
  ["-C", showdownRoot, "rev-parse", "HEAD"],
  {encoding: "utf8"}
).trim();

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

class ProbeExit extends Error {
  constructor(code) {
    super(`probe exited with status ${code}`);
    this.code = Number(code);
  }
}

function emit(response) {
  process.stdout.write(JSON.stringify(response) + "\n");
}

function makeFs(source, virtualFixture) {
  const proxy = Object.create(fs);
  proxy.readFileSync = function readFileSync(file, options) {
    if (String(file) === virtualFixture) {
      const encoded = JSON.stringify(source);
      if (typeof options === "string") return encoded;
      if (typeof options === "object" && options && options.encoding) return encoded;
      return Buffer.from(encoded, "utf8");
    }
    return fs.readFileSync(file, options);
  };
  return proxy;
}

function makeChildProcess() {
  const proxy = Object.create(childProcess);
  proxy.execFileSync = function execFileSync(command, args, options) {
    if (
      command === "git" &&
      Array.isArray(args) &&
      args.length === 4 &&
      args[0] === "-C" &&
      args[1] === showdownRoot &&
      args[2] === "rev-parse" &&
      args[3] === "HEAD"
    ) {
      if (options && options.encoding) return showdownCommit + "\n";
      return Buffer.from(showdownCommit + "\n", "utf8");
    }
    return childProcess.execFileSync(command, args, options);
  };
  return proxy;
}

function createSession(sessionKey, source) {
  let stdout = "";
  let stderr = "";
  const virtualFixture = `azelficoast://fixture/${sessionKey}.json`;
  const fakeProcess = {
    pid: process.pid,
    argv: [
      process.execPath,
      probePath,
      showdownRoot,
      virtualFixture,
      "--posterior-only",
      "--generator-cache-dir",
      generatorCacheDir,
    ],
    env: process.env,
    stdout: {
      write(value) {
        stdout += String(value);
        return true;
      },
    },
    stderr: {
      write(value) {
        stderr += String(value);
        return true;
      },
    },
    exit(code = 0) {
      throw new ProbeExit(code);
    },
  };
  const virtualFs = makeFs(source, virtualFixture);
  const virtualChildProcess = makeChildProcess();
  function sharedRequire(id) {
    if (id === "node:fs" || id === "fs") return virtualFs;
    if (id === "node:child_process" || id === "child_process") {
      return virtualChildProcess;
    }
    return require(id);
  }

  const context = vm.createContext({
    process: fakeProcess,
    require: sharedRequire,
    __azelficoastTransitionExecutionCache: transitionExecutionCache,
    __azelficoastTransitionProjectionCache: transitionProjectionCache,
    __azelficoastTransitionExecutionCacheMaxEntries: maxTransitionExecutions,
    __azelficoastTransitionProjectionCacheMaxEntries: maxProjectionExecutions,
  });

  try {
    probeScript.runInContext(context);
    throw new Error("posterior probe returned without its expected exit boundary");
  } catch (error) {
    if (!(error instanceof ProbeExit) || error.code !== 0) {
      const detail = stderr.trim();
      throw new Error(
        detail || (error instanceof Error ? error.message : String(error))
      );
    }
  }

  let document;
  try {
    document = JSON.parse(stdout);
  } catch (error) {
    throw new Error(
      "posterior probe returned invalid JSON: " +
      (error instanceof Error ? error.message : String(error))
    );
  }

  sessions.delete(sessionKey);
  sessions.set(sessionKey, {context});
  while (sessions.size > maxSessions) {
    const oldest = sessions.keys().next().value;
    sessions.delete(oldest);
  }
  return document;
}

function compileTransitionProgram(sessionKey) {
  const session = sessions.get(sessionKey);
  if (!session) {
    throw new Error("posterior session is unavailable; reconstruct before exact search");
  }
  let encoded;
  try {
    encoded = transitionScript.runInContext(session.context);
  } finally {
    sessions.delete(sessionKey);
  }
  if (typeof encoded !== "string") {
    throw new Error("transition program probe returned a non-string payload");
  }
  return JSON.parse(encoded);
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
    return {id, ok: true, document: compileTransitionProgram(session)};
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
