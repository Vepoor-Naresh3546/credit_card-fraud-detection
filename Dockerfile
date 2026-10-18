FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt requirements-streaming.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ARG INSTALL_STREAMING=false
RUN if [ "$INSTALL_STREAMING" = "true" ]; then pip install --no-cache-dir -r requirements-streaming.txt; fi

COPY app_real.py train_real.py ./
COPY fraud_platform ./fraud_platform
COPY alembic.ini ./
COPY alembic ./alembic
COPY models/creditcard_model.joblib models/creditcard_model_metadata.json ./models/
COPY models/creditcard_feature_baseline.json ./models/

RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p /app/data /app/mlflow && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app_real:app --host 0.0.0.0 --port ${PORT:-8000}"]
