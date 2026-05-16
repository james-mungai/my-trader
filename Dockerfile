FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]" \
    && mkdir -p /app/data \
    && chown -R app:app /app

USER app

EXPOSE 8090

CMD ["uvicorn", "futures_lab.api:app", "--host", "0.0.0.0", "--port", "8090"]
