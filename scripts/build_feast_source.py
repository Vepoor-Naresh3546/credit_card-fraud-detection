from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def build_source(dataset: Path, destination: Path) -> int:
    frame = pd.read_csv(dataset)
    missing = set(FEATURE_COLUMNS + ["Class"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame[FEATURE_COLUMNS + ["Class"]].isna().any().any():
        raise ValueError("Dataset has null feature or target values")
    frame = frame.loc[:, FEATURE_COLUMNS + ["Class"]].copy()
    frame.insert(0, "dataset_row_id", range(1, len(frame) + 1))
    # Feast needs an event timestamp; this is a dataset-relative ordering key, not a
    # claim about the transactions' real-world calendar timestamps.
    frame["event_timestamp"] = pd.to_datetime(frame["Time"], unit="s", utc=True)
    frame = frame.drop(columns=["Class"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Feast offline source from Kaggle data")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "creditcard.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "feature_repo" / "data" / "creditcard_features.parquet",
    )
    args = parser.parse_args()
    print(f"Wrote {build_source(args.input, args.output)} dataset rows to {args.output}")


if __name__ == "__main__":
    main()
