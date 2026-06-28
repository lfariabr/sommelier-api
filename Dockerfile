# FastAPI service image (Render / any container host).
FROM python:3.11.9-slim

WORKDIR /app

# Install deps first for layer caching; pins come from constraints.txt.
COPY constraints.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Serving needs the ml core + committed artifacts + the API. (No data/ — inference
# never reads the CSVs; the joblib artifacts are self-contained.)
COPY ml/ ./ml/
COPY api/ ./api/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
