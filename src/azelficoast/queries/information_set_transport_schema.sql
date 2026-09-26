CREATE TABLE worlds (
    world_index INTEGER PRIMARY KEY,
    weight REAL NOT NULL CHECK (weight > 0)
);

CREATE TABLE leaves (
    leaf_index INTEGER PRIMARY KEY
);

CREATE TABLE edges (
    leaf_index INTEGER NOT NULL REFERENCES leaves(leaf_index),
    world_index INTEGER NOT NULL REFERENCES worlds(world_index),
    chance REAL NOT NULL CHECK (chance > 0)
);
