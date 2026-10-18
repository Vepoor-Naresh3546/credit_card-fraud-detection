from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fraud_platform.config import ROOT, settings
from fraud_platform.model import FraudModel

FEATURE_COLUMNS = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
DATASET_COLUMNS = (*FEATURE_COLUMNS, "Class")


def validate_dataset_columns(columns: list[str]) -> None:
    missing = sorted(set(DATASET_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    extra = sorted(set(columns) - set(DATASET_COLUMNS))
    if extra:
        raise ValueError(f"Dataset contains unexpected columns: {extra}")


def _spark_session(app_name: str) -> Any:
    from pyspark.sql import SparkSession

    return SparkSession.builder.appName(app_name).getOrCreate()


def validate_and_prepare(input_csv: Path, output_parquet: Path) -> dict[str, int]:
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

    spark = _spark_session("creditcard-dataset-validation")
    try:
        headers = spark.read.option("header", "true").csv(str(input_csv)).columns
        validate_dataset_columns(headers)
        schema = StructType(
            [
                *[
                    StructField(name, DoubleType(), nullable=True)
                    for name in FEATURE_COLUMNS
                ],
                StructField("Class", IntegerType(), nullable=True),
                StructField("_corrupt_record", StringType(), nullable=True),
            ]
        )
        frame = (
            spark.read.option("header", "true")
            .option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .schema(schema)
            .csv(str(input_csv))
        )
        feature_invalid = F.lit(False)
        for column in FEATURE_COLUMNS:
            feature_invalid = feature_invalid | F.col(column).isNull() | F.isnan(F.col(column))
        valid_condition = (
            F.col("_corrupt_record").isNull()
            & (~feature_invalid)
            & F.col("Class").isin(0, 1)
            & (F.col("Time") >= 0)
            & (F.col("Amount") >= 0)
        )
        valid_condition = F.coalesce(valid_condition, F.lit(False))
        valid = frame.filter(valid_condition).select(*FEATURE_COLUMNS, "Class")
        invalid = frame.filter(~valid_condition)
        # Keep rejected rows visible for inspection rather than silently discarding them.
        invalid_path = output_parquet.with_name(f"{output_parquet.name}_rejected")
        invalid.write.mode("overwrite").parquet(str(invalid_path))
        valid.write.mode("overwrite").parquet(str(output_parquet))
        counts = valid.agg(
            F.count("*").alias("rows"),
            F.sum("Class").alias("frauds"),
        ).first()
        if counts["rows"] == 0:
            raise ValueError("Dataset validation produced no valid rows")
        return {"rows": int(counts["rows"]), "frauds": int(counts["frauds"] or 0)}
    finally:
        spark.stop()


def aggregate_prepared(input_parquet: Path, output_path: Path) -> dict[str, Any]:
    from pyspark.sql import functions as F

    spark = _spark_session("creditcard-fraud-analytics")
    try:
        frame = spark.read.parquet(str(input_parquet))
        summary_row = frame.agg(
            F.count("*").alias("transactions"),
            F.sum("Class").alias("fraud_transactions"),
            F.avg("Class").alias("fraud_rate"),
            F.avg("Amount").alias("mean_amount"),
        ).first()
        summary = {
            "transactions": int(summary_row["transactions"]),
            "fraud_transactions": int(summary_row["fraud_transactions"] or 0),
            "fraud_rate": float(summary_row["fraud_rate"] or 0.0),
            "mean_amount": float(summary_row["mean_amount"] or 0.0),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        spark.stop()


def batch_score(input_parquet: Path, output_csv: Path, chunk_size: int = 2000) -> int:
    """Score Spark input in driver-side chunks; this is not distributed inference."""
    from pyspark.sql import functions as F

    model = FraudModel.load(settings.model_path, settings.metadata_path, settings.threshold)
    spark = _spark_session("creditcard-driver-batch-scoring")
    count = 0
    try:
        raw_frame = spark.read.parquet(str(input_parquet))
        validate_dataset_columns(raw_frame.columns)
        frame = raw_frame.withColumn("dataset_row_index", F.monotonically_increasing_id()).select(
            "dataset_row_index", *model.feature_columns
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as output_file:
            writer = _csv_writer(output_file)
            writer.writeheader()
            buffer: list[dict[str, Any]] = []
            for row in frame.toLocalIterator():
                item = row.asDict(recursive=True)
                buffer.append(item)
                if len(buffer) >= chunk_size:
                    count += _write_score_chunk(writer, buffer, model)
                    buffer.clear()
            if buffer:
                count += _write_score_chunk(writer, buffer, model)
        return count
    finally:
        spark.stop()


def _csv_writer(output_file: Any) -> csv.DictWriter:
    fieldnames = [
        "dataset_row_index",
        "fraud_probability",
        "decision",
        "threshold",
        "model_version",
    ]
    return csv.DictWriter(output_file, fieldnames=fieldnames)


def _write_score_chunk(
    writer: csv.DictWriter,
    rows: list[dict[str, Any]],
    model: FraudModel,
) -> int:
    scores = model.predict(
        [{feature: float(row[feature]) for feature in model.feature_columns} for row in rows]
    )
    for offset, score in enumerate(scores):
        writer.writerow(
            {
                "dataset_row_index": int(rows[offset]["dataset_row_index"]),
                **score,
                "threshold": model.threshold,
                "model_version": model.model_version,
            }
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit-card dataset Spark jobs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", type=Path, default=ROOT / "data" / "creditcard.csv")
    validate.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "creditcard")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, default=ROOT / "reports" / "spark_summary.json")
    score = subparsers.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, default=ROOT / "reports" / "batch_scores.csv")
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate_and_prepare(args.input, args.output), indent=2))
    elif args.command == "aggregate":
        print(json.dumps(aggregate_prepared(args.input, args.output), indent=2))
    else:
        print(f"Scored {batch_score(args.input, args.output)} rows to {args.output}")


if __name__ == "__main__":
    main()
