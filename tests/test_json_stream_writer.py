from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_json_stream_writer_preserves_values_and_honors_backpressure() -> None:
    program = r"""
const assert = require("node:assert/strict");
const {Writable} = require("node:stream");
const {writeJsonStream} = require("./scripts/json_stream_writer.cjs");

const expected = {
  schema: "azelficoast.test",
  rows: Array.from({length: 4096}, (_, index) => ({
    index,
    payload: "x".repeat(256),
    unicode: "é🐾",
  })),
  omitted: undefined,
  arrayValues: [undefined, Number.POSITIVE_INFINITY],
};
const normalized = JSON.parse(JSON.stringify(expected));
const chunks = [];
const sink = new Writable({
  highWaterMark: 1024,
  write(chunk, _encoding, callback) {
    chunks.push(Buffer.from(chunk));
    setImmediate(callback);
  },
});

writeJsonStream(expected, sink, {chunkBytes: 512})
  .then(() => {
    sink.end(() => {
      const actual = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      assert.deepEqual(actual, normalized);
      assert.ok(chunks.length > 10, "writer should flush multiple bounded chunks");
      assert.ok(
        chunks.every(chunk => chunk.length <= 512),
        "writer should keep output chunks within the requested bound",
      );
      process.stdout.write("ok");
    });
  })
  .catch(error => {
    process.stderr.write(String(error.stack || error));
    process.exitCode = 1;
  });
"""
    result = subprocess.run(
        ["node", "-e", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "ok"
