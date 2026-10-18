from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")


def validate_dataframe(frame: Any, batch_name: str) -> None:
    try:
        import great_expectations as gx
    except ImportError as exc:
        raise RuntimeError(
            "Great Expectations is required for the pipeline quality gate; "
            "install requirements-quality.txt"
        ) from exc

    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name="creditcard_validation_source")
    asset = source.add_dataframe_asset(name=f"creditcard_chunk_{batch_name}")
    batch_definition = asset.add_batch_definition_whole_dataframe(name="full_chunk")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": frame})
    suite = gx.ExpectationSuite(name=f"creditcard_quality_{batch_name}")

    for column in ("Time", *(f"V{i}" for i in range(1, 29)), "Amount", "Class"):
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="Class", value_set=[0, 1])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="Time", min_value=0)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="Amount", min_value=0)
    )

    result = batch.validate(suite)
    if not result.success:
        raise ValueError(f"Great Expectations checks failed for dataset chunk {batch_name}")
