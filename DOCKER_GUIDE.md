# Docker Setup & Running Guide — v6.0
## RL-Based Autonomous Interceptor Drone

**Docker Hub image:** `siddhmehta5131/interceptor-drone-rl:v6.0`  
**No git clone required on the training PC.**  
**All config files live outside the image — change them without rebuilding.**

---

## 0. What changed from the old setup

| Old setup | New setup (v6.0) |
|---|---|
| `default_run.yaml` — one config file | `model.yaml` + `config.yaml` — two files |
| Config baked into image | Config mounted from your PC at runtime |
| Must clone repo on training PC | Just `docker pull` — no git needed |
| `docker compose up` | `docker run` with your own data folder |
| Image built locally from repo | Image pulled directly from Docker Hub |

---

## PART 1 — Clean up everything old

### 1A. Delete old images from Docker Hub

1. Go to **https://hub.docker.com** and log in
2. Click **Repositories**
3. Find `interceptor-drone-rl` (or any old repo from this project)
4. Click on it → **Settings** tab → **Delete Repository** → confirm
5. Repeat for any other old repos

> This removes all old tags (`:latest`, `:v1`, etc.) at once.

### 1B. Remove old images from the training PC

SSH into your training PC, then:

```bash
# See all existing images
docker images

# Remove the old interceptor image(s) by name:
docker rmi siddhmehta5131/interceptor-drone-rl:latest
docker rmi siddhmehta5131/interceptor-drone-rl

# Or nuclear option — remove ALL unused images:
docker image prune -a -f

# Remove any stopped containers from the old setup:
docker container prune -f

# Full cleanup (images + containers + networks + build cache):
docker system prune -a -f
```

Verify everything is gone:
```bash
docker images   # should show nothing or only unrelated images
```

### 1C. Remove old project folder from training PC (if cloned before)

```bash
rm -rf ~/interceptor-drone-rl
rm -rf ~/rl-drone-flight-simulator

# Check it's gone:
ls ~
```

---

## PART 2 — Build and push the new image (do this on YOUR Windows PC)

> **Prerequisite:** Start Docker Desktop on your Windows PC first.

### 2-1. Log in to Docker Hub

```powershell
docker login
# Enter your Docker Hub username: siddhmehta5131
# Enter your password or access token
```

### 2-2. Build the new image

```powershell
cd "C:\Users\ADMIN\Desktop\projects\github_repos\rl-drone-flight-simulator\interceptor-training"
docker build -t siddhmehta5131/interceptor-drone-rl:v6.0 .
```

Takes 10-30 minutes on first build (downloads the ~9 GB PyTorch base image).

### 2-3. Verify the image

```powershell
# Check it exists
docker images

# Run smoke test (no GPU needed)
docker run --rm siddhmehta5131/interceptor-drone-rl:v6.0 --smoke
# Expected: 12 passed, 0 failed
```

### 2-4. Push to Docker Hub

```powershell
docker push siddhmehta5131/interceptor-drone-rl:v6.0

# Also tag as latest for easy pulling:
docker tag siddhmehta5131/interceptor-drone-rl:v6.0 siddhmehta5131/interceptor-drone-rl:latest
docker push siddhmehta5131/interceptor-drone-rl:latest
```

Confirm at: **https://hub.docker.com/r/siddhmehta5131/interceptor-drone-rl/tags**

---

## PART 3 — Set up the training PC (Ubuntu + NVIDIA GPU)

> No git clone. No project files. Just Docker.

### 3-1. One-time Ubuntu setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl

# Install NVIDIA driver (then REBOOT)
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, verify GPU:
```bash
nvidia-smi
# Must show your GPU name and driver version >= 570
```

### 3-2. Install Docker Engine

```bash
curl -fsSL https://get.docker.com | sh
```

Allow your user to run Docker without sudo (then **log out and back in**):
```bash
sudo usermod -aG docker $USER
newgrp docker
```

Verify:
```bash
docker run hello-world
# Must print "Hello from Docker!"
```

### 3-3. Install NVIDIA Container Toolkit (GPU inside containers)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU reaches containers:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
# Must print the same GPU table as before
```

---

## PART 4 — Pull the image and prepare your data folder

### 4-1. Pull the image from Docker Hub

```bash
# No git clone, no repo — just pull:
docker pull siddhmehta5131/interceptor-drone-rl:v6.0
```

Verify:
```bash
docker images
# Should show: siddhmehta5131/interceptor-drone-rl   v6.0   ...
```

### 4-2. Create your data folder

```bash
mkdir -p ~/training/data/configs
mkdir -p ~/training/data/checkpoints
mkdir -p ~/training/data/results
mkdir -p ~/training/data/tb_logs
mkdir -p ~/training/data/curriculum_state
mkdir -p ~/training/data/runs
```

### 4-3. Create `model.yaml` on the training PC

```bash
nano ~/training/data/configs/model.yaml
```

Paste this (edit to your needs):

```yaml
ppo_baseline:
  algo: PPO
  policy: MlpPolicy
  net_arch: {pi: [256, 256, 128], vf: [256, 256, 128]}
  activation: ReLU
  obs_history: {frames: 3, skip: 2}
  hyperparameters:
    n_steps: 4096
    batch_size: 512
    gamma: 0.995
    learning_rate: 3.0e-4
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.005
    n_epochs: 10
    max_grad_norm: 0.5
    vf_coef: 0.5
  privileged_critic:
    - time_remaining
    - facing_error
    - target_true_pos
    - target_true_vel
```

Save: **Ctrl+O** → Enter → **Ctrl+X**

### 4-4. Create `config.yaml` on the training PC

```bash
nano ~/training/data/configs/config.yaml
```

Paste this:

```yaml
global:
  seed: 42
  device: cuda              # use 'cpu' if no GPU
  data_dir: /data
  n_parallel_envs: 8        # lower to 4 if CUDA out of memory
  checkpoint_interval_steps: 50000
  tensorboard: true
  execution_mode: sequential

rollback:
  max_attempts: 2
  retry_budget_scale: 0.5
  threshold_scale: 0.5
  min_steps_scale: 0.25

on_capped: continue         # 'continue' or 'stop'

observation:
  history_frames: 3
  history_skip: 2
  future_samples: 3
  future_skip: 5
  future_source: true
  predictor: const_vel      # 'const_vel' or 'linear_ridge'

target_alt: 5.0

stages: {}
```

Save: **Ctrl+O** → Enter → **Ctrl+X**

---

## PART 5 — Run training

### 5-1. Smoke test first (always — no GPU needed)

```bash
docker run --rm \
  -v ~/training/data:/data \
  siddhmehta5131/interceptor-drone-rl:v6.0 \
  --smoke
```

Expected: `12 passed, 0 failed`

### 5-2. Start real training (foreground — see live logs)

```bash
docker run --gpus all --rm \
  -v ~/training/data:/data \
  siddhmehta5131/interceptor-drone-rl:v6.0 \
  --source /data/configs/model.yaml \
  --config /data/configs/config.yaml \
  --stages 1,2,3,4 \
  --name experiment_1
```

Press **Ctrl+C** to stop. All progress is saved in `~/training/data/`.

### 5-3. Start training in the background (recommended for long runs)

```bash
docker run --gpus all -d \
  --name interceptor_train \
  -v ~/training/data:/data \
  siddhmehta5131/interceptor-drone-rl:v6.0 \
  --source /data/configs/model.yaml \
  --config /data/configs/config.yaml \
  --stages 1,2,3,4 \
  --name experiment_1
```

Watch logs:
```bash
docker logs -f interceptor_train
# Ctrl+C stops watching; does NOT stop training
```

Stop training:
```bash
docker stop interceptor_train
docker rm interceptor_train
```

### 5-4. Resume an interrupted run

Just re-run the exact same command — auto-resumes from last checkpoint:

```bash
docker run --gpus all -d \
  --name interceptor_train \
  -v ~/training/data:/data \
  siddhmehta5131/interceptor-drone-rl:v6.0 \
  --source /data/configs/model.yaml \
  --config /data/configs/config.yaml \
  --stages 1,2,3,4 \
  --name experiment_1
```

### 5-5. Continue to harder stages after first run completes

```bash
docker run --gpus all -d \
  --name interceptor_train_p2 \
  -v ~/training/data:/data \
  siddhmehta5131/interceptor-drone-rl:v6.0 \
  --continue-from /data/runs/experiment_1/results/run_summary.json \
  --stages 5,6,7 \
  --name experiment_2
```

### 5-6. Train only one model (when model.yaml has multiple)

```bash
docker run --gpus all -d \
  --name interceptor_train \
  -v ~/training/data:/data \
  siddhmehta5131/interceptor-drone-rl:v6.0 \
  --source /data/configs/model.yaml \
  --config /data/configs/config.yaml \
  --stages 1,2,3,4 \
  --models ppo_baseline \
  --name experiment_1
```

---

## PART 6 — Monitor with TensorBoard

```bash
docker run --rm -d \
  --name tb \
  -v ~/training/data:/data \
  -p 6006:6006 \
  --entrypoint tensorboard \
  siddhmehta5131/interceptor-drone-rl:v6.0 \
  --logdir /data/tb_logs --host 0.0.0.0 --port 6006
```

Open in browser:
- Same PC: **http://localhost:6006**
- From another PC: **http://\<training-pc-ip\>:6006**

Stop TensorBoard:
```bash
docker stop tb && docker rm tb
```

---

## PART 7 — Evaluate a trained model

```bash
docker run --rm \
  -v ~/training/data:/data \
  --entrypoint python \
  siddhmehta5131/interceptor-drone-rl:v6.0 \
  scripts/evaluate.py \
  --run /data/runs/experiment_1 \
  --model ppo_baseline \
  --stage 3 \
  --episodes 50
```

Results saved to `~/training/data/runs/experiment_1/results/eval_ppo_baseline_stage_3.json`

---

## PART 8 — Where are my results?

Everything is in `~/training/data/` — not inside Docker:

```
~/training/data/
├── configs/
│   ├── model.yaml               <- your model definitions (edit freely)
│   └── config.yaml              <- your training settings (edit freely)
└── runs/
    └── experiment_1/
        ├── checkpoints/
        │   └── ppo_baseline/
        │       ├── stage_1/
        │       │   ├── PPO_50000_steps.zip    <- periodic checkpoint
        │       │   └── PPO_stage_1_final.zip  <- stage-final model
        │       └── stage_2/ ...
        ├── results/
        │   ├── run_summary.json              <- full run summary
        │   ├── ppo_baseline.zip              <- final model
        │   └── ppo_baseline.result.json
        ├── tb_logs/                          <- TensorBoard
        └── curriculum_state/
            └── run_state.json                <- resume state
```

Copy results off the training PC to your Windows PC:
```bash
# Run this on your Windows PC:
scp -r ubuntu@<training-pc-ip>:~/training/data/runs/experiment_1/results ./my_results
```

---

## PART 9 — Change config without rebuilding

Edit directly on the training PC:

```bash
nano ~/training/data/configs/model.yaml
```

Then re-run the container — it picks up the new file immediately. **No rebuild needed.**

---

## PART 10 — Everyday command reference

| Task | Command |
|---|---|
| Pull latest image | `docker pull siddhmehta5131/interceptor-drone-rl:v6.0` |
| Smoke test | `docker run --rm -v ~/training/data:/data siddhmehta5131/interceptor-drone-rl:v6.0 --smoke` |
| Train in foreground | `docker run --gpus all --rm -v ~/training/data:/data siddhmehta5131/interceptor-drone-rl:v6.0 --source /data/configs/model.yaml --config /data/configs/config.yaml --stages 1,2,3,4 --name exp1` |
| Train in background | Same but `-d --name interceptor_train` instead of `--rm` |
| Watch logs | `docker logs -f interceptor_train` |
| Stop training | `docker stop interceptor_train && docker rm interceptor_train` |
| Resume | Re-run same command — auto-resumes |
| Continue next stages | Replace `--source` with `--continue-from /data/runs/exp1/results/run_summary.json` |
| TensorBoard | `docker run -d --name tb -v ~/training/data:/data -p 6006:6006 --entrypoint tensorboard siddhmehta5131/interceptor-drone-rl:v6.0 --logdir /data/tb_logs --host 0.0.0.0 --port 6006` |
| See all containers | `docker ps -a` |
| Free disk space | `docker system prune -f` (safe — does NOT touch `~/training/data/`) |
| Check GPU in container | `docker run --rm --gpus all siddhmehta5131/interceptor-drone-rl:v6.0 nvidia-smi` |

---

## PART 11 — Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: permission denied` | `sudo usermod -aG docker $USER` then log out/in |
| `could not select device driver "nvidia"` | Re-run NVIDIA toolkit section, then `sudo systemctl restart docker` |
| Training shows `device=cpu` unexpectedly | Check `config.yaml` has `device: cuda`; verify `docker run --rm --gpus all siddhmehta5131/interceptor-drone-rl:v6.0 nvidia-smi` |
| `CUDA out of memory` | Lower `n_parallel_envs` in `config.yaml` (try 4 or 2) |
| `Cannot connect to Docker daemon` | `sudo systemctl start docker` |
| No space left on device | `docker system prune -a -f` (safe — your `~/training/data/` is untouched) |
| Port 6006 already in use | Change to `-p 6007:6006` and open `http://localhost:6007` |
| Run didn't resume / started fresh | Make sure `--name` matches exactly the previous run name |
| `run_summary.json: not found` for `--continue-from` | Check path: `ls ~/training/data/runs/<name>/results/` |

---

## PART 12 — CPU-only training (no GPU)

1. Edit `config.yaml` → set `device: cpu` and `n_parallel_envs: 2`
2. Remove `--gpus all` from all `docker run` commands
3. Skip sections 3-3 (NVIDIA toolkit)

---

## Summary — zero to training in 8 commands

```bash
# ── One-time setup on training PC ──────────────────────────────────────────

# 1. NVIDIA driver + reboot
sudo ubuntu-drivers autoinstall && sudo reboot

# 2. Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# 3. NVIDIA container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# ── Every new experiment ────────────────────────────────────────────────────

# 4. Pull image (once)
docker pull siddhmehta5131/interceptor-drone-rl:v6.0

# 5. Create data folder and drop in your configs
mkdir -p ~/training/data/configs
# Create model.yaml and config.yaml in ~/training/data/configs/

# 6. Smoke test
docker run --rm -v ~/training/data:/data siddhmehta5131/interceptor-drone-rl:v6.0 --smoke

# 7. Train (background)
docker run --gpus all -d --name train \
  -v ~/training/data:/data \
  siddhmehta5131/interceptor-drone-rl:v6.0 \
  --source /data/configs/model.yaml \
  --config /data/configs/config.yaml \
  --stages 1,2,3,4 --name exp1

# 8. Watch
docker logs -f train
```

All results are in `~/training/data/runs/`. No git clone ever needed.
