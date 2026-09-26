"use strict";

const {once} = require("node:events");

function normalizeToJSON(value) {
  if (value !== null && typeof value === "object" &&
      typeof value.toJSON === "function") {
    return value.toJSON();
  }
  return value;
}

function* jsonTokens(rawValue, depth, arraySlot, ancestors, normalized = false) {
  const value = normalized ? rawValue : normalizeToJSON(rawValue);

  if (value === null || typeof value === "string" ||
      typeof value === "number" || typeof value === "boolean") {
    const encoded = JSON.stringify(value);
    yield encoded === undefined ? "null" : encoded;
    return;
  }
  if (typeof value === "undefined" || typeof value === "function" ||
      typeof value === "symbol") {
    if (arraySlot) yield "null";
    return;
  }
  if (typeof value === "bigint") {
    throw new TypeError("Do not know how to serialize a BigInt");
  }
  if (typeof value !== "object") {
    return;
  }
  if (ancestors.has(value)) {
    throw new TypeError("Converting circular structure to JSON");
  }

  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      if (value.length === 0) {
        yield "[]";
        return;
      }
      yield "[\n";
      for (let index = 0; index < value.length; index += 1) {
        if (index > 0) yield ",\n";
        yield "  ".repeat(depth + 1);
        yield* jsonTokens(value[index], depth + 1, true, ancestors);
      }
      yield "\n";
      yield "  ".repeat(depth);
      yield "]";
      return;
    }

    const entries = [];
    for (const key of Object.keys(value)) {
      const entryValue = normalizeToJSON(value[key]);
      if (entryValue === undefined || typeof entryValue === "function" ||
          typeof entryValue === "symbol") {
        continue;
      }
      entries.push([key, entryValue]);
    }
    if (entries.length === 0) {
      yield "{}";
      return;
    }
    yield "{\n";
    for (let index = 0; index < entries.length; index += 1) {
      if (index > 0) yield ",\n";
      const [key, entryValue] = entries[index];
      yield "  ".repeat(depth + 1);
      yield JSON.stringify(key);
      yield ": ";
      yield* jsonTokens(entryValue, depth + 1, false, ancestors, true);
    }
    yield "\n";
    yield "  ".repeat(depth);
    yield "}";
  } finally {
    ancestors.delete(value);
  }
}

async function writeJsonStream(
  value,
  writable = process.stdout,
  {chunkBytes = 64 * 1024} = {},
) {
  if (!Number.isInteger(chunkBytes) || chunkBytes < 1) {
    throw new RangeError("chunkBytes must be a positive integer");
  }

  let chunks = [];
  let bufferedBytes = 0;
  const flush = async () => {
    if (chunks.length === 0) return;
    const output = chunks.join("");
    chunks = [];
    bufferedBytes = 0;
    if (!writable.write(output)) await once(writable, "drain");
  };

  for (const token of jsonTokens(value, 0, false, new WeakSet())) {
    const tokenBytes = Buffer.byteLength(token, "utf8");
    if (chunks.length > 0 && bufferedBytes + tokenBytes > chunkBytes) {
      await flush();
    }
    chunks.push(token);
    bufferedBytes += tokenBytes;
    if (bufferedBytes >= chunkBytes) await flush();
  }
  chunks.push("\n");
  await flush();
}

module.exports = {writeJsonStream};
