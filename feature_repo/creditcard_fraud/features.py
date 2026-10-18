from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64
from feast.value_type import ValueType

transaction_row = Entity(
    name="dataset_row_id",
    join_keys=["dataset_row_id"],
    value_type=ValueType.INT64,
    description="Surrogate row key for the public benchmark dataset; not a card/customer ID.",
)

source = FileSource(
    name="creditcard_offline_features",
    path="data/creditcard_features.parquet",
    timestamp_field="event_timestamp",
)

creditcard_features = FeatureView(
    name="creditcard_transaction",
    entities=[transaction_row],
    ttl=None,
    schema=[
        Field(name="Time", dtype=Float64),
        *[Field(name=f"V{i}", dtype=Float64) for i in range(1, 29)],
        Field(name="Amount", dtype=Float64),
    ],
    source=source,
    online=False,
    description=(
        "Offline benchmark features matching creditcard_model.joblib. The dataset has "
        "no durable card/customer key for a meaningful online serving lookup."
    ),
)
