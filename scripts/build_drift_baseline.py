from __future__ import annotations

from fraud_platform.config import ROOT, settings
from fraud_platform.drift import build_drift_baseline
from fraud_platform.model import FraudModel


if __name__ == "__main__":
    model = FraudModel.load(
        settings.model_path,
        settings.metadata_path,
        settings.threshold,
    )
    output = ROOT / "models" / "creditcard_feature_baseline.json"
    baseline = build_drift_baseline(
        ROOT / "data" / "creditcard.csv",
        model,
        output,
    )
    print(
        f"Wrote {len(baseline['feature_bins'])} feature baselines from "
        f"{baseline['training_rows']} training rows to {output}"
    )
