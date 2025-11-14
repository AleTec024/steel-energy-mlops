# api/Dockerfile

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 🔹 Usar solo las dependencias de la API
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# 🔹 Copiar código
COPY api ./api
COPY src ./src

ENV PYTHONPATH=/app

EXPOSE 8000

WORKDIR /app/api

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]