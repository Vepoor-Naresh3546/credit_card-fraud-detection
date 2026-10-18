"""Generate clearly synthetic transaction data for the first end-to-end demo.

These generated labels are rule-driven and are NOT evidence of real-world fraud
performance. Replace this data with a documented public dataset for evaluation.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "transactions.csv"


def main(n_rows: int = 30_000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    amount = np.clip(rng.lognormal(mean=4.0, sigma=1.0, size=n_rows), 1, 5000)
    hour = rng.integers(0, 24, size=n_rows)
    transactions_last_1h = np.clip(rng.poisson(lam=1.4, size=n_rows), 0, 12)
    avg_amount_24h = np.clip(rng.lognormal(mean=3.7, sigma=0.75, size=n_rows), 1, 2000)
    amount_to_avg_ratio = amount / (avg_amount_24h + 1.0)
    is_international = rng.binomial(1, 0.12, size=n_rows)
    new_device = rng.binomial(1, 0.10, size=n_rows)
    merchant_risk_score = rng.beta(a=2.0, b=5.0, size=n_rows)

    # Synthetic risk rule: intentionally creates an imbalanced target for practice.
    risk_score = (
        -5.0
        + 0.75 * np.log1p(amount / 50.0)
        + 0.55 * np.maximum(transactions_last_1h - 2, 0)
        + 1.05 * is_international
        + 1.15 * new_device
        + 2.0 * merchant_risk_score
        + 0.60 * np.maximum(amount_to_avg_ratio - 2.0, 0)
        + 0.15 * ((hour <= 4) | (hour >= 23))
    )
    fraud_probability = 1.0 / (1.0 + np.exp(-risk_score))
    is_fraud = rng.binomial(1, fraud_probability)

    df = pd.DataFrame({
        "transaction_id": [f"TX{i:07d}" for i in range(1, n_rows + 1)],
        "amount": np.round(amount, 2),
        "hour": hour.astype(int),
        "transactions_last_1h": transactions_last_1h.astype(int),
        "avg_amount_24h": np.round(avg_amount_24h, 2),
        "amount_to_avg_ratio": np.round(amount_to_avg_ratio, 4),
        "is_international": is_international.astype(int),
        "new_device": new_device.astype(int),
        "merchant_risk_score": np.round(merchant_risk_score, 4),
        "is_fraud": is_fraud.astype(int),
    })
    df.to_csv(OUTPUT, index=False)
    fraud_rate = df["is_fraud"].mean() * 100
    print(f"Saved {len(df):,} synthetic rows to: {OUTPUT}")
    print(f"Synthetic fraud rate: {fraud_rate:.2f}%")
    print("Reminder: rule-generated data is for demonstrating the pipeline only.")


if __name__ == "__main__":
    main()
