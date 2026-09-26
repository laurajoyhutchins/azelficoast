WITH policy AS (
    SELECT
        action_id,
        worst_value AS score
    FROM action_statistics
)
SELECT
    action_id,
    score
FROM policy
ORDER BY score DESC, action_id ASC
