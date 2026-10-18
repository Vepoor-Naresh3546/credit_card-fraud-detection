from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_sample(source: Path, destination: Path, legitimate: int, fraud: int) -> int:
    if legitimate < 1 or fraud < 1:
        raise ValueError("Sample must include at least one row from each class")
    frame = pd.read_csv(source)
    if "Class" not in frame.columns or frame["Class"].nunique() != 2:
        raise ValueError("Expected source dataset with both Class labels")
    if frame.isna().any().any():
        raise ValueError("Source dataset contains missing values")
    parts = [
        frame.loc[frame["Class"] == 0].head(legitimate),
        frame.loc[frame["Class"] == 1].head(fraud),
    ]
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) != legitimate + fraud:
        raise ValueError("Source dataset has too few rows for the requested sample")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination, index=False)
    return len(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small real-row sample for Spark checks")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "creditcard.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "sample_creditcard.csv")
    parser.add_argument("--legitimate", type=int, default=40)
    parser.add_argument("--fraud", type=int, default=10)
    args = parser.parse_args()
    print(f"Wrote {build_sample(args.input, args.output, args.legitimate, args.fraud)} rows")


if __name__ == "__main__":
    main()
