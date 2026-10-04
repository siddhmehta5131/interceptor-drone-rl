FROM python:3.12-slim

# Avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Keep stdout/stderr unbuffered
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (ffmpeg is needed for imageio-ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY stage-runtime/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The application code will be bind-mounted via docker-compose
# We set PYTHONPATH to include /app (which will map to interceptor-training)
# and /app/stage-runtime (which will map to stage-runtime)
ENV PYTHONPATH=/app/interceptor-training:/app/stage-runtime

# Default command
CMD ["python", "-m", "stage_runtime", "--all", "--out", "/app/stage-runtime/out"]
