"""Start/reuse the local FastAPI service, then open the native desktop client."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
from typing import Mapping, Any

from desktop_app import (
    ApiConnectionError,
    DesktopApp,
    get_readiness,
    load_feature_columns,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8001
DEFAULT_STARTUP_TIMEOUT = 90.0
POLL_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True)
class LauncherConfig:
    host: str
    port: int
    startup_timeout: float
    api_key: str = field(repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class BackendSession:
    process: subprocess.Popen[bytes] | None
    api_key: str = field(repr=False)
    readiness: dict[str, Any]


class LauncherError(RuntimeError):
    """A useful, displayable launcher failure."""


def load_launcher_config(environ: Mapping[str, str] | None = None) -> LauncherConfig:
    """Load launcher-only settings and validate their supported ranges."""
    source = os.environ if environ is None else environ
    port_value = source.get("FRAUD_DESKTOP_PORT", str(DEFAULT_PORT))
    try:
        port = int(port_value)
    except ValueError as exc:
        raise ValueError("FRAUD_DESKTOP_PORT must be a number from 1 to 65535.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("FRAUD_DESKTOP_PORT must be a number from 1 to 65535.")

    timeout_value = source.get(
        "FRAUD_DESKTOP_STARTUP_TIMEOUT", str(DEFAULT_STARTUP_TIMEOUT)
    )
    try:
        startup_timeout = float(timeout_value)
    except ValueError as exc:
        raise ValueError(
            "FRAUD_DESKTOP_STARTUP_TIMEOUT must be a number between 1 and 300."
        ) from exc
    if not math.isfinite(startup_timeout) or not 1 <= startup_timeout <= 300:
        raise ValueError(
            "FRAUD_DESKTOP_STARTUP_TIMEOUT must be a number between 1 and 300."
        )
    return LauncherConfig(
        host="127.0.0.1",
        port=port,
        startup_timeout=startup_timeout,
        api_key=source.get("FRAUD_API_KEY", "").strip(),
    )


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _validate_backend_contract(
    readiness: Mapping[str, Any],
    feature_columns: tuple[str, ...],
    model_version: str,
) -> dict[str, Any]:
    if readiness.get("feature_count") != len(feature_columns):
        raise LauncherError(
            "A ready API is using a different number of model features. "
            "Close that service or select another FRAUD_DESKTOP_PORT."
        )
    if readiness.get("model_version") != model_version:
        raise LauncherError(
            "A ready API is using a different model version. "
            "Close that service or select another FRAUD_DESKTOP_PORT."
        )
    return dict(readiness)


def _wait_for_readiness(
    config: LauncherConfig,
    process: subprocess.Popen[bytes] | None,
) -> dict[str, Any]:
    deadline = time.monotonic() + config.startup_timeout
    latest_error = "The API did not become ready before the startup timeout."
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise LauncherError(
                "The FastAPI backend stopped during startup "
                f"(exit code {process.returncode}). Check that dependencies and "
                "the trained model files are present, and that DATABASE_URL is usable."
            )
        try:
            return get_readiness(config.base_url, timeout=1.0)
        except ApiConnectionError as exc:
            latest_error = str(exc)
            time.sleep(POLL_INTERVAL_SECONDS)
    if process is None:
        raise LauncherError(
            f"Port {config.port} is already in use, but that service did not become "
            f"ready ({latest_error}). Choose another FRAUD_DESKTOP_PORT or check "
            "the existing service."
        )
    raise LauncherError(
        f"The FastAPI backend did not become ready within "
        f"{config.startup_timeout:g} seconds ({latest_error}). Check DATABASE_URL, "
        "audit database availability, and the local model artifacts. If the database "
        "has not been initialized, run `python -m alembic upgrade head` first."
    )


def start_or_connect(
    config: LauncherConfig,
    feature_columns: tuple[str, ...],
    model_version: str,
) -> BackendSession:
    """Reuse a ready matching API, otherwise launch exactly one local backend."""
    try:
        readiness = get_readiness(config.base_url, timeout=1.0)
    except ApiConnectionError:
        readiness = None
    if readiness is not None:
        return BackendSession(
            process=None,
            api_key=config.api_key,
            readiness=_validate_backend_contract(
                readiness, feature_columns, model_version
            ),
        )

    if _port_is_open(config.host, config.port):
        readiness = _wait_for_readiness(config, process=None)
        return BackendSession(
            process=None,
            api_key=config.api_key,
            readiness=_validate_backend_contract(
                readiness, feature_columns, model_version
            ),
        )

    api_key = config.api_key or secrets.token_urlsafe(32)
    backend_environment = os.environ.copy()
    backend_environment["FRAUD_API_KEY"] = api_key
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app_real:app",
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=backend_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except OSError as exc:
        raise LauncherError(
            "Could not start the FastAPI backend. Verify that this Python environment "
            "has Uvicorn installed."
        ) from exc

    session: BackendSession | None = None
    try:
        readiness = _wait_for_readiness(config, process)
        session = BackendSession(
            process=process,
            api_key=api_key,
            readiness=_validate_backend_contract(
                readiness, feature_columns, model_version
            ),
        )
        return session
    finally:
        if session is None:
            stop_backend(process)


def stop_backend(process: subprocess.Popen[bytes] | None) -> None:
    """Stop only the backend process created by this launcher."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _load_model_version() -> str:
    metadata_path = ROOT / "models" / "creditcard_model_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LauncherError(
            f"Unable to read the trained model metadata at {metadata_path}."
        ) from exc
    if not isinstance(metadata, dict):
        raise LauncherError("The trained model metadata must contain a JSON object.")
    version = metadata.get("model_version")
    if not isinstance(version, str) or not version:
        raise LauncherError("The model metadata does not declare a model_version.")
    return version


def main() -> int:
    root: tk.Tk | None = None
    backend: BackendSession | None = None
    try:
        config = load_launcher_config()
        feature_columns = load_feature_columns()
        model_version = _load_model_version()
        root = tk.Tk()
        root.withdraw()
        backend = start_or_connect(config, feature_columns, model_version)
        api_key = backend.api_key
        if not api_key:
            api_key = simpledialog.askstring(
                "Connect to existing fraud API",
                "Enter the API key for the already-running local service:",
                show="*",
                parent=root,
            ) or ""
            if not api_key:
                raise LauncherError(
                    "An API key is required to send predictions to the existing service."
                )
        app = DesktopApp(
            root=root,
            base_url=config.base_url,
            api_key=api_key,
            feature_columns=feature_columns,
            model_version=str(backend.readiness["model_version"]),
        )
        root.deiconify()
        root.mainloop()
        return 0
    except (RuntimeError, ValueError) as exc:
        if root is None:
            try:
                root = tk.Tk()
                root.withdraw()
            except tk.TclError:
                print(f"Unable to start the desktop application: {exc}", file=sys.stderr)
                return 1
        messagebox.showerror("Credit Card Fraud Detection", str(exc), parent=root)
        return 1
    except tk.TclError as exc:
        print(f"Unable to open the Tkinter desktop window: {exc}", file=sys.stderr)
        return 1
    finally:
        if backend is not None:
            stop_backend(backend.process)
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
