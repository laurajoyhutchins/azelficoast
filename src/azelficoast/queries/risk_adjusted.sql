WITH scored AS (
    SELECT
        action_id,
        expected_value - :risk_aversion * (expected_value - worst_value) AS score
    FROM action_statistics
)
SELECT
    action_id,
    score
FROM scored
