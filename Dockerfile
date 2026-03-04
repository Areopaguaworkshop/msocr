FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
# Copy package source before sync so editable install can resolve project files.
COPY msocr/ ./msocr/
RUN pip install uv && uv sync --frozen
ENV PATH="/app/.venv/bin:${PATH}"

COPY models/ ./models/
COPY source_registry/ ./source_registry/

EXPOSE 8000

CMD ["uv", "run", "msocr", "api", "--host", "0.0.0.0", "--port", "8000"]
