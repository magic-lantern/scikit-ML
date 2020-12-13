

@transform_pandas(
    Output(rid="ri.vector.main.execute.ddeb99b5-9f4a-4582-8037-9d4c0c0f43a6"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07")
)
SELECT *
FROM data_encoded_and_outcomes

