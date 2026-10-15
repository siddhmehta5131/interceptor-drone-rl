# stage-runtime

A standalone package to run and render the interceptor drone curriculum stages.

## Usage

You can run the runtime directly using python, or via Docker.

```bash
# Run locally (requires interceptor-training dependencies)
python -m stage_runtime --all --out ./out

# Or render a specific stage
python -m stage_runtime --stage 1 --format png --segments 1 --out ./out
```

## Docker

A `Dockerfile` and `docker-compose.yml` are provided to run the environment isolated, ensuring all dependencies (including `ffmpeg`) are installed.

```bash
# Build the image
docker compose build

# Run the smoke test inside the container
docker compose run --rm stage-runtime python scripts/stage_smoke_test.py

# Render stages 1, 3, and 8 to mp4
docker compose run --rm stage-runtime python -m stage_runtime --stages 1,3,8 --out /app/stage-runtime/out
```
