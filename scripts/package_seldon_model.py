from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from fraud_platform.config import ROOT, settings


def package_model(destination: Path) -> Path:
    if not settings.model_path.is_file() or not settings.metadata_path.is_file():
        raise FileNotFoundError("Both active model and metadata artifacts are required.")
    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(settings.model_path, destination / "creditcard_model.joblib")
    shutil.copy2(settings.metadata_path, destination / "creditcard_model_metadata.json")
    model_settings = {
        "name": "creditcard-fraud",
        "implementation": "mlserver_sklearn.SKLearnModel",
        "parameters": {
            "uri": "./creditcard_model.joblib",
            "version": metadata["model_version"],
        },
    }
    (destination / "model-settings.json").write_text(
        json.dumps(model_settings, indent=2),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the active pipeline and metadata for Seldon Core 2 MLServer."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "deployment" / "seldon" / "creditcard-fraud",
    )
    args = parser.parse_args()
    print(f"Packaged active model to {package_model(args.output)}")


if __name__ == "__main__":
    main()
