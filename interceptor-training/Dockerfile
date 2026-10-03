# Multi-stage RL Training Pipeline for the Autonomous Interceptor Drone.
#
# Base image: PyTorch 2.7.0 (CUDA 12.8, cuDNN 9 runtime) — supports sm_120
# (RTX 50 series) GPUs. Fallback if the image is unavailable: switch base to
# `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` or install torch from the
# cu128 index (`pip install torch --index-url https://download.pytorch.org/whl/cu128`).
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# System dependencies (git for pip VCS installs and repo tooling).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# opencv-python needs GUI libs (libxcb) missing here; headless is enough for training.
RUN pip uninstall -y opencv-python && pip install --no-cache-dir --default-timeout=120 --retries 10 opencv-python-headless

# Application source, entrypoint scripts and shipped configs.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY run.py ./run.py

# Host-mounted volume: checkpoints/, tb_logs/, curriculum_state/, results/.
# The host also maps its local configs directory onto /data/configs.
VOLUME /data

# `python scripts/train.py` is the CLI wrapper around train_model(); the
# one-file runner is `python run.py`.  There is no longer a single default run
# YAML (it was split into model.yaml + config.yaml, plan A-5), so the default
# invocation just prints the CLI help.
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--help"]