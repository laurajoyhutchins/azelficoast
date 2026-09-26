SELECT
    action_id,
    SUM(weight * value) AS expected_value
FROM action_value_terms
GROUP BY action_id
ORDER BY expected_value DESC, action_id ASC
LIMIT 1
