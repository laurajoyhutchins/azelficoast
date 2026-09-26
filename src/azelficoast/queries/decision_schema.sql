CREATE TABLE hidden_worlds (
    world_id INTEGER PRIMARY KEY,
    weight REAL NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0, 1))
);

CREATE TABLE legal_actions (
    action_id INTEGER PRIMARY KEY
);

CREATE TABLE transitions (
    world_id INTEGER NOT NULL,
    action_id INTEGER NOT NULL,
    successor_id INTEGER NOT NULL
);

CREATE TABLE evaluations (
    successor_id INTEGER PRIMARY KEY,
    value REAL NOT NULL
);

CREATE VIEW active_worlds AS
SELECT world_id, weight
FROM hidden_worlds
WHERE active = 1 AND weight > 0;

CREATE VIEW action_value_terms AS
SELECT
    t.action_id AS action_id,
    w.weight AS weight,
    e.value AS value
FROM active_worlds AS w
JOIN transitions AS t
  ON t.world_id = w.world_id
JOIN legal_actions AS a
  ON a.action_id = t.action_id
JOIN evaluations AS e
  ON e.successor_id = t.successor_id;


CREATE VIEW action_statistics AS
SELECT
    action_id,
    SUM(weight) AS posterior_mass,
    SUM(weight * value) AS expected_value,
    MIN(value) AS worst_value,
    MAX(value) AS best_value
FROM action_value_terms
GROUP BY action_id;
