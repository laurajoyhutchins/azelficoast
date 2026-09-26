SELECT
    action_id,
    posterior_mass,
    expected_value,
    worst_value,
    best_value
FROM action_statistics
ORDER BY action_id ASC
