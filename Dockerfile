FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir uvicorn fastapi pydantic && \
    pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "2024"]