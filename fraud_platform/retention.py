from __future__ import annotations

import os

from fraud_platform.db.database import purge_expired_prediction_audits


def main() -> None:
    retention_days = int(os.getenv("PREDICTION_AUDIT_RETENTION_DAYS", "90"))
    deleted = purge_expired_prediction_audits(retention_days)
    print(
        f"Deleted {deleted} prediction audit rows older than {retention_days} days."
    )


if __name__ == "__main__":
    main()
