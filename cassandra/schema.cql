CREATE KEYSPACE IF NOT EXISTS fraud_platform
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

CREATE TABLE IF NOT EXISTS fraud_platform.scored_events_by_day (
    event_date date,
    event_timestamp timestamp,
    event_id text,
    transaction_id text,
    decision text,
    fraud_probability double,
    threshold double,
    model_version text,
    PRIMARY KEY ((event_date), event_timestamp, event_id)
) WITH CLUSTERING ORDER BY (event_timestamp DESC, event_id ASC);
