"""Native Tkinter client for the existing Credit Card Fraud Detection API."""

from __future__ import annotations

import json
import math
import queue
import threading
from http.client import HTTPException as HTTPProtocolError
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
import tkinter as tk
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = ROOT / "models" / "creditcard_model_metadata.json"
REQUEST_TIMEOUT_SECONDS = 15.0
COLORS = {
    "background": "#f3f6fb",
    "navy": "#14233b",
    "muted": "#66758a",
    "blue": "#2864dc",
    "green": "#16865a",
    "red": "#c13f4d",
    "white": "#ffffff",
    "line": "#dce4ef",
    "pale_blue": "#eaf1ff",
}


class InputValidationError(ValueError):
    """A user-entered model feature is missing or invalid."""


class ApiConnectionError(RuntimeError):
    """The API is unavailable or did not return a valid response."""


def load_feature_columns(metadata_path: Path = DEFAULT_METADATA_PATH) -> tuple[str, ...]:
    """Read the feature order used by the active real-data model."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read model metadata at {metadata_path}.") from exc

    if not isinstance(metadata, dict):
        raise RuntimeError("Model metadata must contain a JSON object.")
    columns = metadata.get("feature_columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise RuntimeError("Model metadata must declare unique feature_columns.")
    return tuple(columns)


def parse_feature_values(
    values: Mapping[str, str], feature_columns: Sequence[str]
) -> dict[str, float]:
    """Convert all required form values to finite floats in model feature order."""
    expected = set(feature_columns)
    actual = set(values)
    missing = [name for name in feature_columns if name not in actual]
    if missing:
        raise InputValidationError(f"Missing required feature: {missing[0]}.")
    unexpected = sorted(actual - expected)
    if unexpected:
        raise InputValidationError(f"Unexpected feature: {unexpected[0]}.")

    parsed: dict[str, float] = {}
    for feature in feature_columns:
        raw_value = values[feature].strip()
        if not raw_value:
            raise InputValidationError(f"{feature} is required.")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise InputValidationError(f"{feature} must be a number.") from exc
        if not math.isfinite(value):
            raise InputValidationError(f"{feature} must be a finite number.")
        if feature == "Time" and value < 0:
            raise InputValidationError("Time must be zero or greater.")
        if feature == "Amount" and not 0 <= value <= 1_000_000:
            raise InputValidationError("Amount must be between 0 and 1,000,000.")
        parsed[feature] = value
    return parsed


def get_readiness(base_url: str, timeout: float = 2.0) -> dict[str, Any]:
    """Return the JSON readiness response or raise a user-safe connection error."""
    request = Request(f"{base_url.rstrip('/')}/ready", method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 503:
            raise ApiConnectionError(
                "The API is running, but its database is not ready."
            ) from exc
        raise ApiConnectionError(f"The API readiness check returned HTTP {exc.code}.") from exc
    except TimeoutError as exc:
        raise ApiConnectionError("The API readiness check timed out.") from exc
    except (URLError, OSError, HTTPProtocolError) as exc:
        raise ApiConnectionError("The API is not reachable yet.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiConnectionError("The API returned an invalid readiness response.") from exc
    if not isinstance(result, dict) or result.get("status") != "ready":
        raise ApiConnectionError("The API did not report a ready status.")
    return result


def _api_error_message(status_code: int, response_body: bytes) -> str:
    if status_code == 401:
        return (
            "The API rejected its key. Check that FRAUD_API_KEY matches the running "
            "backend, or restart the local backend with the key configured."
        )
    if status_code == 422:
        try:
            details = json.loads(response_body.decode("utf-8")).get("detail")
            if isinstance(details, list):
                messages = [
                    str(item.get("msg", "Invalid value"))
                    for item in details
                    if isinstance(item, dict)
                ]
                if messages:
                    return "The API rejected the input: " + "; ".join(messages)
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass
        return "The API rejected the transaction values (HTTP 422)."
    if status_code == 429:
        return "The API request limit was reached. Wait a minute, then try again."
    if status_code == 503:
        return (
            "The API cannot complete predictions. Check API-key configuration and "
            "database/audit storage availability."
        )
    return f"The API could not complete the prediction (HTTP {status_code})."


def request_prediction(
    base_url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Send the API's documented JSON contract without exposing the API key."""
    request = Request(
        f"{base_url.rstrip('/')}/predict",
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ApiConnectionError(_api_error_message(exc.code, exc.read())) from exc
    except TimeoutError as exc:
        raise ApiConnectionError(
            "The prediction request timed out. Check the backend and try again."
        ) from exc
    except (URLError, OSError, HTTPProtocolError) as exc:
        raise ApiConnectionError(
            "Could not reach the prediction API. Check that the backend is available "
            "and try again."
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiConnectionError("The API returned an invalid prediction response.") from exc
    if not isinstance(result, dict) or not {
        "decision",
        "fraud_probability",
        "threshold",
        "model_version",
        "request_id",
    }.issubset(result):
        raise ApiConnectionError("The API response is missing prediction result fields.")
    return result


class DesktopApp:
    """Responsive Tkinter interface for single-transaction fraud scoring."""

    def __init__(
        self,
        root: tk.Tk,
        base_url: str,
        api_key: str,
        feature_columns: Sequence[str],
        model_version: str,
    ) -> None:
        self.root = root
        self.base_url = base_url
        self.api_key = api_key
        self.feature_columns = tuple(feature_columns)
        self._messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._busy = False
        self._closed = False
        self.variables = {name: tk.StringVar() for name in self.feature_columns}

        root.title("Credit Card Fraud Detection System")
        root.geometry("1180x830")
        root.minsize(1000, 690)
        root.configure(background=COLORS["background"])
        self._configure_styles()
        self._build_layout(model_version)
        self._set_backend_status(True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(200, self._drain_messages)
        root.after(10000, self._refresh_backend_status)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "TFrame",
            background=COLORS["background"],
        )
        style.configure(
            "Card.TFrame",
            background=COLORS["white"],
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TLabel",
            background=COLORS["background"],
            foreground=COLORS["navy"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Card.TLabel",
            background=COLORS["white"],
            foreground=COLORS["navy"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.Card.TLabel",
            background=COLORS["white"],
            foreground=COLORS["muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Title.TLabel",
            background=COLORS["background"],
            foreground=COLORS["navy"],
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Section.Card.TLabel",
            background=COLORS["white"],
            foreground=COLORS["navy"],
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "TEntry",
            padding=(8, 7),
            fieldbackground="#fbfcfe",
            bordercolor=COLORS["line"],
        )
        style.configure(
            "Primary.TButton",
            background=COLORS["blue"],
            foreground=COLORS["white"],
            font=("Segoe UI", 10, "bold"),
            padding=(14, 10),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#1e52bd"), ("disabled", "#91a8da")],
            foreground=[("disabled", "#f4f6fa")],
        )
        style.configure(
            "Secondary.TButton",
            background="#eef2f8",
            foreground=COLORS["navy"],
            padding=(11, 9),
        )
        style.map("Secondary.TButton", background=[("active", "#e0e7f1")])
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#e8edf5",
            background=COLORS["blue"],
            borderwidth=0,
        )

    def _build_layout(self, model_version: str) -> None:
        shell = ttk.Frame(self.root, padding=(26, 20, 26, 22))
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x", pady=(0, 18))
        title_column = ttk.Frame(header)
        title_column.pack(side="left", fill="x", expand=True)
        ttk.Label(
            title_column,
            text="Credit Card Fraud Detection System",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            title_column,
            text="Analyze a transaction with the existing credit-card model",
            foreground=COLORS["muted"],
        ).pack(anchor="w", pady=(4, 0))
        self.backend_badge = tk.Label(
            header,
            text="●  CHECKING API",
            bg="#fff4da",
            fg="#8a6410",
            font=("Segoe UI", 9, "bold"),
            padx=13,
            pady=8,
        )
        self.backend_badge.pack(side="right", anchor="n", padx=(12, 0))

        columns = ttk.Frame(shell)
        columns.pack(fill="both", expand=True)
        columns.columnconfigure(0, weight=6, uniform="main")
        columns.columnconfigure(1, weight=4, uniform="main")
        columns.rowconfigure(0, weight=1)
        form_card = ttk.Frame(columns, style="Card.TFrame", padding=18)
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        result_card = ttk.Frame(columns, style="Card.TFrame", padding=20)
        result_card.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        self._build_form(form_card)
        self._build_result(result_card, model_version)

    def _build_form(self, card: ttk.Frame) -> None:
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Transaction details", style="Section.Card.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            card,
            text="Enter the model features. All values are required.",
            style="Muted.Card.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))
        self.sample_notice = tk.Label(
            card,
            text="",
            bg=COLORS["pale_blue"],
            fg="#234b99",
            anchor="w",
            font=("Segoe UI", 9),
            padx=9,
            pady=7,
        )
        self.sample_notice.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.sample_notice.grid_remove()

        actions = ttk.Frame(card, style="Card.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        self.predict_button = ttk.Button(
            actions,
            text="Predict Transaction",
            style="Primary.TButton",
            command=self.predict,
        )
        self.predict_button.pack(side="left")
        self.clear_button = ttk.Button(
            actions,
            text="Clear Form",
            style="Secondary.TButton",
            command=self.clear_form,
        )
        self.clear_button.pack(side="left", padx=(8, 0))
        self.sample_button = ttk.Button(
            actions,
            text="Example / Test Values",
            style="Secondary.TButton",
            command=self.fill_example,
        )
        self.sample_button.pack(side="left", padx=(8, 0))

        form_body = ttk.Frame(card, style="Card.TFrame")
        form_body.grid(row=4, column=0, sticky="nsew")
        card.rowconfigure(4, weight=1)
        form_body.rowconfigure(0, weight=1)
        form_body.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            form_body,
            background=COLORS["white"],
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(form_body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.fields_frame = ttk.Frame(self.canvas, style="Card.TFrame")
        self.fields_window = self.canvas.create_window(
            (0, 0), window=self.fields_frame, anchor="nw"
        )
        self.fields_frame.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.fields_window, width=event.width),
        )
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self._create_fields()

    def _create_fields(self) -> None:
        context_features = [name for name in ("Time", "Amount") if name in self.feature_columns]
        pca_features = [name for name in self.feature_columns if name not in context_features]
        row = 0
        if context_features:
            ttk.Label(
                self.fields_frame,
                text="Transaction",
                style="Section.Card.TLabel",
            ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 8))
            row += 1
            row = self._create_field_rows(context_features, row)
        if pca_features:
            ttk.Label(
                self.fields_frame,
                text="PCA features",
                style="Section.Card.TLabel",
            ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(16, 8))
            row += 1
            self._create_field_rows(pca_features, row)

    def _create_field_rows(self, features: Sequence[str], start_row: int) -> int:
        for index, feature in enumerate(features):
            row = start_row + index // 2
            column = (index % 2) * 2
            cell = ttk.Frame(self.fields_frame, style="Card.TFrame")
            cell.grid(row=row, column=column, sticky="ew", padx=(0, 12), pady=(0, 10))
            self.fields_frame.columnconfigure(column, weight=1, uniform="feature")
            ttk.Label(cell, text=feature, style="Card.TLabel").pack(anchor="w", pady=(0, 4))
            entry = ttk.Entry(cell, textvariable=self.variables[feature], width=17)
            entry.pack(fill="x")
        return start_row + (len(features) + 1) // 2

    def _build_result(self, card: ttk.Frame, model_version: str) -> None:
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Prediction result", style="Section.Card.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            card,
            text="Results from the currently active model",
            style="Muted.Card.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 18))
        self.decision_label = tk.Label(
            card,
            text="Ready for analysis",
            background="#f3f6fb",
            foreground=COLORS["muted"],
            font=("Segoe UI", 16, "bold"),
            padx=15,
            pady=18,
            anchor="w",
        )
        self.decision_label.grid(row=2, column=0, sticky="ew")
        ttk.Label(card, text="Fraud score", style="Muted.Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=(22, 5)
        )
        self.score_label = ttk.Label(
            card,
            text="—",
            style="Section.Card.TLabel",
            font=("Segoe UI", 24, "bold"),
        )
        self.score_label.grid(row=4, column=0, sticky="w")
        self.score_bar = ttk.Progressbar(
            card, orient="horizontal", mode="determinate", maximum=100
        )
        self.score_bar.grid(row=5, column=0, sticky="ew", pady=(10, 7))
        self.threshold_label = ttk.Label(
            card, text="Threshold: —", style="Muted.Card.TLabel"
        )
        self.threshold_label.grid(row=6, column=0, sticky="w")

        self.progress = ttk.Progressbar(card, mode="indeterminate")
        self.progress.grid(row=7, column=0, sticky="ew", pady=(24, 8))
        self.progress.grid_remove()
        self.status_label = ttk.Label(
            card,
            text="Request status: idle",
            style="Muted.Card.TLabel",
            wraplength=350,
        )
        self.status_label.grid(row=8, column=0, sticky="w", pady=(3, 0))
        self.model_label = ttk.Label(
            card,
            text=f"Model version: {model_version}",
            style="Muted.Card.TLabel",
        )
        self.model_label.grid(row=9, column=0, sticky="w", pady=(9, 0))
        ttk.Separator(card).grid(row=10, column=0, sticky="ew", pady=(21, 14))
        ttk.Label(
            card,
            text=(
                "A higher score indicates a stronger model signal at its configured "
                "threshold. It is not proof of fraud, and this score is not guaranteed "
                "to be calibrated."
            ),
            style="Muted.Card.TLabel",
            wraplength=350,
            justify="left",
        ).grid(row=11, column=0, sticky="w")
        ttk.Label(
            card,
            text="Review transactions using your normal verification process.",
            style="Muted.Card.TLabel",
            wraplength=350,
            justify="left",
        ).grid(row=12, column=0, sticky="w", pady=(12, 0))

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _set_backend_status(self, available: bool, detail: str = "") -> None:
        if available:
            self.backend_badge.configure(
                text="●  API AVAILABLE",
                bg="#e5f6ee",
                fg=COLORS["green"],
            )
        else:
            self.backend_badge.configure(
                text="●  API UNAVAILABLE",
                bg="#fdebed",
                fg=COLORS["red"],
            )
        if detail and self.status_label.cget("text") == "Request status: idle":
            self.status_label.configure(text=f"Request status: {detail}")

    def _refresh_backend_status(self) -> None:
        if self._closed:
            return

        def check() -> None:
            try:
                result = get_readiness(self.base_url)
                self._messages.put(("backend", result))
            except ApiConnectionError as exc:
                self._messages.put(("backend_error", str(exc)))

        threading.Thread(target=check, daemon=True, name="fraud-api-status").start()
        self.root.after(10000, self._refresh_backend_status)

    def _drain_messages(self) -> None:
        if self._closed:
            return
        try:
            while True:
                kind, result = self._messages.get_nowait()
                if kind == "prediction":
                    self._show_prediction(result)
                elif kind == "prediction_error":
                    self._show_prediction_error(result)
                elif kind == "backend":
                    self._set_backend_status(True)
                elif kind == "backend_error":
                    self._set_backend_status(False, result)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_messages)

    def predict(self) -> None:
        if self._busy:
            return
        raw_values = {name: variable.get() for name, variable in self.variables.items()}
        try:
            features = parse_feature_values(raw_values, self.feature_columns)
        except InputValidationError as exc:
            messagebox.showerror("Check transaction values", str(exc), parent=self.root)
            return

        payload = {"transaction_id": f"desktop-{uuid4().hex}", **features}
        self._busy = True
        self.predict_button.configure(state="disabled")
        self.clear_button.configure(state="disabled")
        self.sample_button.configure(state="disabled")
        self.progress.grid()
        self.progress.start(10)
        self.status_label.configure(text="Request status: sending transaction to API…")
        self.decision_label.configure(
            text="Analyzing transaction…",
            background="#f3f6fb",
            foreground=COLORS["muted"],
        )
        self.score_label.configure(text="—")
        self.score_bar["value"] = 0
        self.threshold_label.configure(text="Threshold: —")

        def submit() -> None:
            try:
                result = request_prediction(self.base_url, self.api_key, payload)
                self._messages.put(("prediction", result))
            except ApiConnectionError as exc:
                self._messages.put(("prediction_error", str(exc)))

        threading.Thread(target=submit, daemon=True, name="fraud-prediction").start()

    def _finish_request(self) -> None:
        self._busy = False
        self.predict_button.configure(state="normal")
        self.clear_button.configure(state="normal")
        self.sample_button.configure(state="normal")
        self.progress.stop()
        self.progress.grid_remove()

    def _show_prediction(self, result: Mapping[str, Any]) -> None:
        self._finish_request()
        is_fraud = result["decision"] == "fraud"
        score = float(result["fraud_probability"])
        threshold = float(result["threshold"])
        self.decision_label.configure(
            text=(
                "Higher fraud-risk signal"
                if is_fraud
                else "Lower fraud-risk signal"
            ),
            background="#fff0f0" if is_fraud else "#e9f7ef",
            foreground=COLORS["red"] if is_fraud else COLORS["green"],
        )
        self.score_label.configure(text=f"{score:.2%}")
        self.score_bar["value"] = max(0, min(100, score * 100))
        self.threshold_label.configure(text=f"Decision threshold: {threshold:.2%}")
        self.model_label.configure(text=f"Model version: {result['model_version']}")
        self.status_label.configure(
            text=f"Request status: complete · ID {result['request_id']}"
        )
        self._set_backend_status(True)

    def _show_prediction_error(self, error: str) -> None:
        self._finish_request()
        self._set_backend_status(False)
        self.decision_label.configure(
            text="Prediction unavailable",
            background="#fdebed",
            foreground=COLORS["red"],
        )
        self.status_label.configure(text=f"Request status: failed · {error}")
        messagebox.showerror("Prediction failed", error, parent=self.root)

    def clear_form(self) -> None:
        if self._busy:
            return
        for variable in self.variables.values():
            variable.set("")
        self.sample_notice.grid_remove()
        self.decision_label.configure(
            text="Ready for analysis",
            background="#f3f6fb",
            foreground=COLORS["muted"],
        )
        self.score_label.configure(text="—")
        self.score_bar["value"] = 0
        self.threshold_label.configure(text="Threshold: —")
        self.status_label.configure(text="Request status: idle")

    def fill_example(self) -> None:
        if self._busy:
            return
        sample = {
            feature: (
                406.0
                if feature == "Time"
                else 12.5
                if feature == "Amount"
                else 0.1
            )
            for feature in self.feature_columns
        }
        for feature, value in sample.items():
            self.variables[feature].set(str(value))
        self.sample_notice.configure(
            text="SAMPLE DATA · illustrative values only; not a real transaction"
        )
        self.sample_notice.grid()

    def close(self) -> None:
        self._closed = True
        self.root.destroy()
