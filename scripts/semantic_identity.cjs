"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map(key => [key, stable(value[key])])
    );
  }
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new TypeError("semantic JSON numbers must be finite");
  }
  return value;
}

function canonicalPythonJson(value) {
  return JSON.stringify(stable(value)).replace(
    /[^\x00-\x7f]/g,
    character =>
      "\\u" + character.charCodeAt(0).toString(16).padStart(4, "0")
  );
}

function sha256PythonCanonical(value) {
  return crypto
    .createHash("sha256")
    .update(canonicalPythonJson(value))
    .digest("hex");
}

if (require.main === module) {
  const source = process.argv[2]
    ? fs.readFileSync(process.argv[2], "utf8")
    : fs.readFileSync(0, "utf8");
  process.stdout.write(sha256PythonCanonical(JSON.parse(source)) + "\n");
}

module.exports = {canonicalPythonJson, sha256PythonCanonical};
