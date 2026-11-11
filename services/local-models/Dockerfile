# local-models service image (CPU). Separate dependency environment per ADR 001:
# transformers 4.57.x / sentence-transformers 5.7.x for Giga remote code.
# Weights are NOT baked into the image: a HF cache volume is mounted read-only.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS runtime

RUN groupadd -r lmuser && useradd -r -g lmuser -m lmuser
WORKDIR /srv
COPY --from=builder --chown=lmuser:lmuser /srv/.venv /srv/.venv
COPY --chown=lmuser:lmuser app_lm /srv/app_lm

ENV PATH="/srv/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER lmuser
EXPOSE 8081
CMD ["uvicorn", "app_lm.server:app", "--host", "0.0.0.0", "--port", "8081"]
