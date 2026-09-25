# Docker Setup & Running Guide
## RL-Based Autonomous Interceptor Drone — Ubuntu + Docker + NVIDIA GPU Training

**Repo:** https://github.com/siddhmehta5131/interceptor-drone-rl
**What this guide covers:** taking your project from a brand-new Ubuntu PC to a fully running GPU training job, step by step. Written for someone who has never used Docker or Ubuntu before.

---

## 0. What you are actually setting up

Your project is a **Docker container** on GitHub. A Docker container is a ready-made
"box" that already contains everything your code needs to run (Python, PyTorch with
CUDA, Stable-Baselines3, all dependencies). You do **not** install PyTorch or the RL
libraries on Ubuntu directly — you just install Docker, and Docker runs the box.

Three pieces of software are needed on Ubuntu:

| Software | What it does |
|---|---|
| **NVIDIA driver** | Lets Ubuntu talk to your GPU (same driver as gaming) |
| **Docker Engine + Compose** | Runs containers, and the compose file wires everything together |
| **NVIDIA Container Toolkit** | Lets containers *see* and use the GPU |

How the compose file (`interceptor-training/docker-compose.yml`) connects things:

- The container image is built from `interceptor-training/Dockerfile`
  (base: `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`, which already has CUDA).
- Your folders are **mounted** (shared) between Ubuntu and the container:
  - `./data` ↔ `/data` — this is where checkpoints, logs and results are saved.
    Anything saved in the container at `/data` appears in `interceptor-training/data/`
    on your PC, so your training results survive even if the container is deleted.
  - `./configs` ↔ `/data/configs` — the run YAML config is shared too, so you can
    edit settings without rebuilding the image.
- The GPU is passed through so PyTorch inside the container runs on CUDA.

---

## 1. Before you start — checklist

- [ ] A PC with an **NVIDIA GPU** (GTX 10-series or newer; RTX 40/50 series ideal)
- [ ] **Ubuntu 22.04 or 24.04 LTS** installed and booting to the desktop
  (or a server where you log in over SSH)
- [ ] Internet connection
- [ ] **At least 40 GB free disk space** (the container image alone is ~10 GB).
      Check with `df -h`
- [ ] Your GitHub username/password — your machine will also ask for a **Personal
      Access Token** on first push/pull of private data. (Cloning a public repo is fine
      without one.)

> **No NVIDIA GPU?** You can still run the code on CPU. See section 12 ("Optional:
> running without a GPU").

---

## 2. Open a terminal

- On the Ubuntu desktop: press **Ctrl + Alt + T** to open a terminal.
- On a server: connect over **SSH** from your other PC:
  ```bash
  ssh siddh@your-server-ip
  ```
- Everything in this guide is typed into this terminal. Commands are shown in
  code blocks — paste them one at a time and press **Enter**. (Paste in the terminal
  is **Ctrl + Shift + V**, not Ctrl+V.)
- Lines starting with `#` are comments — skip them.

---

## 3. Update Ubuntu & install basic tools

First make sure your system is up to date. This asks for your password and can take
a few minutes:

```bash
sudo apt update
sudo apt upgrade -y
```

Install `git` (to download the project) and `curl` (to download installers):

```bash
sudo apt install -y git curl
```

Check they worked:

```bash
git --version
curl --version | head -1
```

---

## 4. Install the NVIDIA driver

The container provides CUDA **12.8**, which needs an NVIDIA driver of at least
**version 570**. The simplest way to install the driver:

```bash
sudo ubuntu-drivers autoinstall
```

Then **restart the PC** (this is important — the driver only loads after a reboot):

```bash
sudo reboot
```

After it comes back, open a terminal and check the GPU is visible:

```bash
nvidia-smi
```

You should see a table listing your GPU (e.g. "NVIDIA GeForce RTX 4070") and a
CUDA version in the top-right corner (it will show a version >= 12.8 or similar;
the number in the top corner is the *driver's* supported CUDA, and anything >= 570
driver is fine because the container brings its own CUDA 12.8).

> **No `nvidia-smi`? Wrong output?**
> Your GPU is either too old, not seated/connected, or the driver didn't install.
> Try `sudo apt install -y nvidia-driver-550` (or `nvidia-driver-570` if available),
> then reboot again and re-check.

---

## 5. Install Docker Engine + Compose

Use Docker's official one-line installer. It installs the Docker Engine, the
`docker compose` plugin and the Buildx builder:

```bash
curl -fsSL https://get.docker.com | sh
```

Verify:

```bash
docker --version
docker compose version
```

> If `docker compose version` says "command not found", the plugin didn't install.
> Install it manually:
> ```bash
> sudo apt update && sudo apt install -y docker-compose-plugin
> ```

### 5a. Give your user permission to use Docker (no `sudo` every time)

Docker commands need admin rights. Instead of typing `sudo` before every command,
add your user to the `docker` group:

```bash
sudo usermod -aG docker $USER
```

**Log out and log back in** (or reboot), or run this once to apply it in the current
terminal:

```bash
newgrp docker
```

Test that Docker works **without** sudo:

```bash
docker run hello-world
```

You should see "Hello from Docker!" — that confirms Docker is working and your user
has permission. (This also downloaded a tiny test image.)

---

## 6. Install the NVIDIA Container Toolkit

This is what lets a container use the GPU. Add NVIDIA's software repository and
install the toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
```

```bash
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

Now tell Docker to use the `nvidia` runtime, and restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify the GPU reaches containers** — pull a tiny CUDA 12.8 test image and run it:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

This must print the same GPU table as step 4. If you instead see an error like
`could not select device driver "nvidia"`, something in this step went wrong —
re-run the two commands above and restart Docker again.

---

## 7. Download your project from GitHub

```bash
git clone https://github.com/siddhmehta5131/interceptor-drone-rl.git
cd interceptor-drone-rl
```

Look at the layout:

```bash
ls -la
```

You should see `README.md`, `interceptor-training/`, `src/`, `hover_env.py`, etc.
The Docker stuff lives inside `interceptor-training/`:

```bash
ls interceptor-training
```

Expected: `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `configs/`,
`scripts/`, `src/`, `data/`.

---

## 8. Recommended first run: the in-container smoke test

Before launching the long real training, run the built-in **smoke test**. It trains
a tiny PPO policy for a few thousand steps across stages 1–3 and verifies the whole
pipeline works (envs, reward, curriculum, checkpointing). If this passes, your GPU +
Docker + repo setup is 100% correct.

```bash
cd interceptor-training
docker compose run --rm interceptor-train --smoke \
  --smoke-config ppo_baseline --smoke-stages 1,2,3 --smoke-steps 1000
```

What happens:

1. First run **builds the image**: downloads the ~9 GB PyTorch base image, then
   installs your dependencies (torch/SB3/tensorboard/etc.). This is the only slow
   part — **10–30 minutes** depending on your internet. Later runs use the cache
   and start in seconds.
2. The container starts `scripts/train.py` with `--smoke` and runs the checks.
3. On success you'll see a summary line with **8 PASS** (or similar) and "smoke OK".

> Building takes a while and prints lots of download bars. That is normal.
> If it appears frozen, wait — Docker is downloading. You can watch progress with
> `docker images` in a second terminal.

If the smoke test passes, **everything is set up correctly**. Move to section 9.

---

## 9. Start the real training

### 9a. Quick look: what the default run does

The default config is `configs/default_run.yaml`, which on Ubuntu lives at
`/data/configs/default_run.yaml` *inside* the container:

- 4 algorithm presets defined: `ppo_baseline`, `sac_baseline`, `td3_baseline`,
  `ppo_5layer_deep`. The default is **`ppo_baseline`**.
- It trains the **8-stage curriculum** (stages 1–7 by default in the run list),
  advancing a stage once its success-rate threshold is met.
- **8 parallel environments**, seed **42**, checkpoint saved every **50,000 steps**,
  TensorBoard logging enabled.
- Each finished stage saves `data/checkpoints/PPO_stage_<N>_final.zip`
  (plus periodic snapshots `PPO_<steps>_steps.zip`) and the config-wide final model
  is saved to `data/results/`.

### 9b. Run it

From inside `interceptor-training/`:

```bash
docker compose up --build
```

- `--build` rebuilds the image first (fast after the first build — it only re-copies
  changed code layers).
- The terminal fills with training logs: per-stage progress, reward components,
  episode stats, curriculum checks ("advance", "rollback", ...).
- **Stop training:** press **Ctrl + C** in this terminal. Everything saved so far is
  safe on your disk in `interceptor-training/data/`.

### 9c. Run it in the background (recommended for long training)

Training can take many hours/days. Instead of keeping the terminal open, run
Docker **detached**:

```bash
docker compose up -d --build
```

This returns immediately and training continues in the background, surviving even
if you close the terminal or log out.

See the logs whenever you want:

```bash
docker compose logs -f
```

(Press **Ctrl + C** to stop *watching* the logs — it does not stop training.)

Stop the background training entirely:

```bash
docker compose down
```

---

## 10. Monitor with TensorBoard

Training metrics are logged to `data/tb_logs/`. View them with TensorBoard.

**Option A — run TensorBoard inside the container** (no extra installs):

```bash
docker compose run --rm --entrypoint tensorboard interceptor-train \
  --logdir /data/tb_logs --host 0.0.0.0 --port 6006
```

**Option B — install TensorBoard on Ubuntu** and point it at your mounted folder:

```bash
pip install tensorboard
tensorboard --logdir interceptor-training/data/tb_logs --host 0.0.0.0 --port 6006
```

Then open a browser:

- On the same PC: http://localhost:6006
- From another PC on the network: http://<ubuntu-ip>:6006

You'll see reward curves and stage-advancement history.

> **Port already in use?** Change `--port 6007` in both places.

---

## 11. Evaluate a trained model

To measure a stage's actual performance (interception kill-rate vs. the required
threshold), run the evaluation script in the container:

```bash
docker compose run --rm --entrypoint python interceptor-train \
  scripts/evaluate.py --config /data/configs/default_run.yaml \
  --config-name ppo_baseline --stage 3 --episodes 50
```

This:

- Auto-picks the best available model for stage 3 (stage-final checkpoint, else the
  latest periodic checkpoint, else the config-wide final model).
- Runs 50 deterministic episodes and prints **kill rate vs. required success rate**,
  mean reward, mean final distance vs. threshold, and time-to-intercept.
- Saves the summary JSON to `data/results/eval_ppo_baseline_stage_3.json`.

---

## 12. Optional: running without a GPU (CPU-only)

If the training PC has no NVIDIA GPU:

1. Edit `interceptor-training/Dockerfile` — change the first line to a CPU image:
   ```dockerfile
   FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
   ```
   ↓ becomes
   ```dockerfile
   FROM pytorch/pytorch:2.7.0-cpu
   ```
2. Remove the GPU settings from `interceptor-training/docker-compose.yml` — delete
   the `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES`
   entries and the whole `deploy:` block.
3. Rebuild: `docker compose up -d --build`

Training will run on CPU — expect it to be much slower. Skip sections 4 and 6 entirely.

---

## 13. Everyday workflow — quick reference

| Task | Command (from `interceptor-training/`) |
|---|---|
| First build + start training (foreground) | `docker compose up --build` |
| Start training in the background | `docker compose up -d --build` |
| Watch training logs | `docker compose logs -f` |
| Stop training | `docker compose down` |
| Fast sanity check (after any change) | `docker compose run --rm interceptor-train --smoke --smoke-config ppo_baseline --smoke-stages 1,2,3 --smoke-steps 1000` |
| Pick a different algorithm preset | Edit `default_run.yaml` → set `config: ppo_5layer_deep` (or `sac_baseline`, `td3_baseline`) |
| Change stage list / evaluate a stage | Edit `default_run.yaml` (`stages:` list) / run `scripts/evaluate.py` with `--stage N` |
| Resume an interrupted run | Just run `docker compose up -d --build` again — the orchestrator detects `data/curriculum_state/run_state.json` and resumes automatically |
| See new code changes after `git pull` | Re-run `docker compose up -d --build` |
| Where are my results? | `interceptor-training/data/checkpoints/` (model `.zip` files), `data/tb_logs/` (TensorBoard), `data/results/` (final model + eval JSON), `data/curriculum_state/run_state.json` |
| View the model files | `ls data/checkpoints` — yes they're on your normal disk, not hidden inside Docker |
| Rebuild from scratch (clean cache) | `docker compose build --no-cache` |

---

## 14. Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `docker: permission denied ... connect to the docker daemon` | Your user isn't active in the `docker` group yet. Re-run `sudo usermod -aG docker $USER`, log out/in, try `newgrp docker`. |
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit not configured. Re-do section 6: `sudo nvidia-ctk runtime configure --runtime=docker` then `sudo systemctl restart docker`. |
| Container starts but training is very slow / logs say `cpu` | GPU not passed through. Check with `docker compose exec interceptor-train nvidia-smi` while running (or run the section-6 verify command). |
| `CUDA out of memory` | Other processes using the GPU, or 8 parallel envs too heavy for a small GPU. Edit `default_run.yaml` → lower `n_parallel_envs` (e.g. 4), or stop the other processes, or `nvidia-smi` first. |
| Build downloads forever / first build takes 30+ min | Normal — the CUDA base image is ~9 GB. Don't interrupt it. |
| `docker compose: command not found` | Compose plugin missing: `sudo apt install -y docker-compose-plugin`. |
| `Cannot connect to the Docker daemon` | Docker daemon not running/started: `sudo systemctl start docker` (and `sudo systemctl enable docker` to auto-start on boot). |
| `nvidia-smi: command not found` (on Ubuntu) | Driver not installed: `sudo ubuntu-drivers autoinstall && sudo reboot`. |
| `No space left` / build fails on disk | `df -h` to check disk. Free space: `docker system prune -f` (removes unused images/networks; **your `data/` folder is safe**). |
| GPU columns in `nvidia-smi` show the top-right CUDA version older than 12.8 | Fine — the container ships its own CUDA 12.8. Only the **driver version** matters (≥ 570). |
| `port is already allocated` (TensorBoard) | Use another port: `--port 6007`. |
| Terminal closed mid-training, is my progress lost? | No. Checkpoints + `run_state.json` are in `data/`. Just start the container again — it resumes. |

---

## 15. Quick recap — the 10 commands that get you from zero to training

```bash
# 1. Update the system
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl

# 2. NVIDIA driver (then REBOOT)
sudo ubuntu-drivers autoinstall
sudo reboot

# 3. Docker Engine + Compose
curl -fsSL https://get.docker.com | sh

# 4. Use Docker without sudo (then log out/in)
sudo usermod -aG docker $USER

# 5. GPU inside containers
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 6. Verify GPU reaches containers
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# 7. Get the project
git clone https://github.com/siddhmehta5131/interceptor-drone-rl.git
cd interceptor-drone-rl/interceptor-training

# 8. Sanity check (first run builds the image — 10-30 min)
docker compose run --rm interceptor-train --smoke --smoke-config ppo_baseline --smoke-stages 1,2,3 --smoke-steps 1000

# 9. Start training in the background
docker compose up -d --build

# 10. Watch it
docker compose logs -f
```

Good luck — and remember: everything your training saves lands in
`interceptor-training/data/` on the Ubuntu machine. Copy that folder anywhere
(another PC, Google Drive, a USB stick) to back up your trained models.

---

*Guide generated for the interceptor-drone-rl repository
(https://github.com/siddhmehta5131/interceptor-drone-rl).*