from __future__ import annotations

import pytest

from desktop_app import InputValidationError, load_feature_columns, parse_feature_values
from run_app import load_launcher_config


@pytest.fixture
def feature_columns() -> tuple[str, ...]:
    return load_feature_columns()


def valid_values(feature_columns: tuple[str, ...]) -> dict[str, str]:
    return {
        feature: (
            "406"
            if feature == "Time"
            else "12.5"
            if feature == "Amount"
            else "0.1"
        )
        for feature in feature_columns
    }


def test_feature_values_follow_real_model_metadata_order(
    feature_columns: tuple[str, ...],
) -> None:
    parsed = parse_feature_values(valid_values(feature_columns), feature_columns)

    assert tuple(parsed) == feature_columns
    assert len(parsed) == 30
    assert parsed["Time"] == 406
    assert parsed["Amount"] == 12.5


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "not-a-number"])
def test_feature_values_reject_nonfinite_or_nonnumeric_data(
    feature_columns: tuple[str, ...], value: str
) -> None:
    values = valid_values(feature_columns)
    values["V1"] = value

    with pytest.raises(InputValidationError, match="V1"):
        parse_feature_values(values, feature_columns)


def test_feature_values_reject_missing_features(
    feature_columns: tuple[str, ...],
) -> None:
    values = valid_values(feature_columns)
    del values["V28"]

    with pytest.raises(InputValidationError, match="V28"):
        parse_feature_values(values, feature_columns)


@pytest.mark.parametrize(
    ("feature", "value", "message"),
    [
        ("Time", "-1", "Time must be zero or greater"),
        ("Amount", "-0.01", "Amount must be between"),
        ("Amount", "1000000.01", "Amount must be between"),
    ],
)
def test_feature_values_enforce_api_numeric_bounds(
    feature_columns: tuple[str, ...], feature: str, value: str, message: str
) -> None:
    values = valid_values(feature_columns)
    values[feature] = value

    with pytest.raises(InputValidationError, match=message):
        parse_feature_values(values, feature_columns)


def test_launcher_configuration_defaults_to_local_api() -> None:
    config = load_launcher_config({})

    assert config.host == "127.0.0.1"
    assert config.port == 8001
    assert config.base_url == "http://127.0.0.1:8001"
    assert config.startup_timeout == 90
    assert config.api_key == ""


def test_launcher_configuration_reads_port_timeout_and_key() -> None:
    config = load_launcher_config(
        {
            "FRAUD_DESKTOP_PORT": "8123",
            "FRAUD_DESKTOP_STARTUP_TIMEOUT": "45",
            "FRAUD_API_KEY": "runtime-test-key",
        }
    )

    assert config.port == 8123
    assert config.startup_timeout == 45
    assert config.api_key == "runtime-test-key"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"FRAUD_DESKTOP_PORT": "not-a-port"}, "FRAUD_DESKTOP_PORT"),
        ({"FRAUD_DESKTOP_PORT": "0"}, "FRAUD_DESKTOP_PORT"),
        ({"FRAUD_DESKTOP_PORT": "65536"}, "FRAUD_DESKTOP_PORT"),
        ({"FRAUD_DESKTOP_STARTUP_TIMEOUT": "0"}, "FRAUD_DESKTOP_STARTUP_TIMEOUT"),
        ({"FRAUD_DESKTOP_STARTUP_TIMEOUT": "nan"}, "FRAUD_DESKTOP_STARTUP_TIMEOUT"),
        ({"FRAUD_DESKTOP_STARTUP_TIMEOUT": "301"}, "FRAUD_DESKTOP_STARTUP_TIMEOUT"),
    ],
)
def test_launcher_configuration_rejects_invalid_values(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_launcher_config(environment)
