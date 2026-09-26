WITH normalized_worlds AS (
    SELECT
        world_index,
        weight / SUM(weight) OVER () AS weight
    FROM worlds
),
edge_mass AS (
    SELECT
        e.leaf_index,
        e.world_index,
        w.weight * e.chance AS mass
    FROM edges AS e
    JOIN normalized_worlds AS w
      ON w.world_index = e.world_index
),
leaf_world_mass AS (
    SELECT
        leaf_index,
        world_index,
        SUM(mass) AS mass
    FROM edge_mass
    GROUP BY leaf_index, world_index
),
leaf_mass AS (
    SELECT
        leaf_index,
        SUM(mass) AS mass
    FROM leaf_world_mass
    GROUP BY leaf_index
)
SELECT
    l.leaf_index,
    w.world_index,
    COALESCE(m.mass, 0.0) AS leaf_world_mass,
    lm.mass AS leaf_mass,
    CASE
        WHEN m.mass IS NULL THEN 0.0
        ELSE m.mass / lm.mass
    END AS conditional_weight,
    w.weight AS normalized_world_weight
FROM leaves AS l
CROSS JOIN normalized_worlds AS w
JOIN leaf_mass AS lm
  ON lm.leaf_index = l.leaf_index
LEFT JOIN leaf_world_mass AS m
  ON m.leaf_index = l.leaf_index
 AND m.world_index = w.world_index
ORDER BY l.leaf_index ASC, w.world_index ASC;
