"use strict";

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map(key => [key, stable(value[key])])
    );
  }
  return value;
}

function equivalent(left, right) {
  return JSON.stringify(stable(left)) === JSON.stringify(stable(right));
}

function successorDelta(before, after, path = [], operations = []) {
  if (equivalent(before, after)) return operations;

  if (
    before &&
    after &&
    typeof before === "object" &&
    typeof after === "object" &&
    !Array.isArray(before) &&
    !Array.isArray(after)
  ) {
    const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort();
    for (const key of keys) {
      if (!Object.prototype.hasOwnProperty.call(after, key)) {
        operations.push({op: "delete", path: [...path, key]});
      } else if (!Object.prototype.hasOwnProperty.call(before, key)) {
        operations.push({
          op: "set",
          path: [...path, key],
          value: cloneJson(after[key]),
        });
      } else {
        successorDelta(before[key], after[key], [...path, key], operations);
      }
    }
    return operations;
  }

  if (
    Array.isArray(before) &&
    Array.isArray(after) &&
    before.length === after.length
  ) {
    for (let index = 0; index < before.length; index++) {
      successorDelta(before[index], after[index], [...path, index], operations);
    }
    return operations;
  }

  operations.push({op: "set", path: [...path], value: cloneJson(after)});
  return operations;
}

function applySuccessorDelta(root, operations) {
  let result = cloneJson(root);

  function parentFor(path) {
    let parent = result;
    for (const segment of path.slice(0, -1)) {
      if (
        parent == null ||
        typeof parent !== "object" ||
        !Object.prototype.hasOwnProperty.call(parent, segment)
      ) {
        return null;
      }
      parent = parent[segment];
    }
    return parent;
  }

  for (const operation of operations) {
    if (!operation || !Array.isArray(operation.path)) {
      throw new Error("invalid successor delta operation");
    }
    if (operation.path.length === 0) {
      if (operation.op !== "set") {
        throw new Error("root successor delta must be a set");
      }
      result = cloneJson(operation.value);
      continue;
    }
    const parent = parentFor(operation.path);
    if (parent == null) {
      throw new Error("successor delta path is unavailable");
    }
    const key = operation.path[operation.path.length - 1];
    if (operation.op === "set") {
      parent[key] = cloneJson(operation.value);
    } else if (operation.op === "delete") {
      if (Array.isArray(parent)) {
        throw new Error("successor delta cannot delete an array element");
      }
      delete parent[key];
    } else {
      throw new Error("unknown successor delta operation");
    }
  }
  return result;
}

module.exports = {applySuccessorDelta, successorDelta};
