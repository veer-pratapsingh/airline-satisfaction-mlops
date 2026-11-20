# Pipeline image: contains the code + all Python deps so the GitLab runner can
# execute any pipeline stage (data / train / infer) with `python <script>.py`.
FROM python:3.11-slim

# System libs needed by lightgbm (libgomp) and psycopg2 at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first so layer caches across code changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the project (respecting .dockerignore).
COPY . .

# Default command just lists the pipeline entrypoint; CI overrides this per job.
CMD ["python", "run_pipeline.py"]
