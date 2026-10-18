from __future__ import annotations

from typing import Any

import bentoml
from pydantic import ValidationError

from fraud_platform.api import CreditCardTransaction, MODEL


@bentoml.service(name="fraud-card-service")
class FraudCardService:
    @bentoml.api
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            transaction = CreditCardTransaction.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("Invalid transaction request.") from exc
        row = {
            feature: getattr(transaction, feature)
            for feature in MODEL.feature_columns
        }
        result = MODEL.predict([row])[0]
        return {
            "transaction_id": transaction.transaction_id,
            **result,
            "threshold": MODEL.threshold,
            "model_version": MODEL.model_version,
        }
