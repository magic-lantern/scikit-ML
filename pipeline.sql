

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.45a8018b-2074-42a4-b3a3-fb60ed35ac0f"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07")
)
SELECT year_end, month_end, bad_outcome, count(1) as rec_count
FROM (
SELECT year(visit_end_date) as year_end, month(visit_end_date) as month_end, bad_outcome
FROM data_encoded_and_outcomes
)
GROUP BY year_end, month_end, bad_outcome
order by year_end, month_end, bad_outcome

