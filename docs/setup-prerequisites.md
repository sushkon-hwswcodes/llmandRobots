# Setup And Prerequisites

This guide lists the practical steps needed before running the code in this repo,
with an emphasis on the current Robosuite + Franka + local-model workflow.

## Platform

- Linux x86_64
- Python 3.10
- NVIDIA GPU recommended for model inference and faster perception / simulation
- A working OpenGL / EGL / OSMesa stack for MuJoCo rendering

## System packages

Install the common native dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  git \
  curl \
  ffmpeg \
  libegl1 \
  libgl1 \
  libglib2.0-0 \
  libosmesa6 \
  libosmesa6-dev \
  libxrender1 \
  libxext6 \
  libsm6
```

Depending on your machine, you may also need MuJoCo / NVIDIA driver support that is
already covered by your local CUDA / graphics setup.

## Clone the repo

```bash
git clone --recurse-submodules <your-repo-url> cap-x
cd cap-x
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Install uv and create the environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv venv -p 3.10
source .venv/bin/activate
```

## Install Python dependencies

For the Robosuite / Franka simulation workflow used in the current benchmarks:

```bash
uv sync --extra robosuite --extra dev
```

If `uv sync` fails while trying to inspect optional CUDA / cuRobo metadata on a
machine where you only want the Robosuite baseline, use this editable-install
fallback instead:

```bash
source .venv/bin/activate
uv pip install -e '.[robosuite]' pytest pytest-timeout ruff mypy requests pillow
```

Optional extras you may want later:

```bash
uv sync --extra robosuite --extra dev --extra contactgraspnet
uv sync --extra verl
uv sync --extra curobo
```

## Environment variables

Typical headless simulation settings used in this repo:

```bash
export PATH=/root/.local/bin:$PATH
export PYTHONPATH=/path/to/cap-x
export MUJOCO_GL=osmesa
export MANISKILL_ASSET_DIR=/root/.maniskill/data
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Notes:

- Many tests default to `MUJOCO_GL=egl`, but the current local smoke runs used
  `MUJOCO_GL=osmesa`.
- `DISPLAY=:1` is only needed on machines that already rely on an existing X
  display; it is not required for the usual headless OSMesa path.

## LLM backend

The evaluation harness expects an OpenAI-compatible endpoint. For the current local
workflow, that is Ollama.

Install Ollama, pull a model, and start the server:

```bash
ollama serve
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

The server listens on `http://127.0.0.1:11434` by default.

## IK / control server

Most Franka simulation configs auto-launch PyRoKi from the YAML, but you can also
start it manually:

```bash
python -m capx.serving.launch_pyroki_server \
  --robot panda_description \
  --target-link panda_hand \
  --port 8116 \
  --host 127.0.0.1
```

Notes:

- On first startup, PyRoKi may download robot description assets and build its
  collision model, so port `8116` may take a minute or two to open.
- Many evaluation configs auto-launch PyRoKi, but starting it once manually is a
  useful sanity check when bringing up a new machine.

## Sanity checks before running evals

1. Verify Python imports:

```bash
python -c "import capx; print('capx import ok')"
```

2. Verify Ollama is up:

```bash
ollama list
```

3. Run a lightweight test:

```bash
uv run pytest tests/test_environments.py -q
```

This test file now focuses on the currently active Robosuite baseline and skips
optional environment families (for example LIBERO / R1Pro) when they are not
registered in the current install.

## Running the current Robosuite benchmark workflow

Example privileged Panda baseline:

```bash
python3 -u -m capx.envs.launch \
  --config-path env_configs/cube_lifting_qwen/franka_qwen_privileged.yaml \
  --model "ollama/qwen2.5-coder:7b-instruct-q4_K_M"
```

Example hand-configuration smoke test:

```bash
uv run pytest tests/test_hand_configs.py -q
```

Example YCB asset setup for the real-object benchmark path:

```bash
mkdir -p /root/.maniskill/data/assets
cd /root/.maniskill/data/assets
curl -L \
  https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip \
  -o mani_skill2_ycb.zip
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
zip_path = Path('/root/.maniskill/data/assets/mani_skill2_ycb.zip')
with ZipFile(zip_path) as zf:
    zf.extractall(zip_path.parent)
zip_path.unlink()
PY
```

## Current practical prerequisites for reproducing our local runs

Before reproducing the current Panda benchmark work, or the minimal hand-configuration smoke path, make sure all of the
following are true:

1. The repo was cloned with submodules.
2. `.venv` exists and `uv sync --extra robosuite --extra dev` completed.
3. `ollama serve` is running and the target model is already pulled.
4. MuJoCo rendering works with `MUJOCO_GL=osmesa` on this machine.
5. The Franka PyRoKi server can be auto-launched or reached on `127.0.0.1:8116`.
6. You are running commands from the repo root with `PYTHONPATH` set correctly.
7. For the YCB real-object configs, `MANISKILL_ASSET_DIR` points at the downloaded
   ManiSkill asset root and `assets/mani_skill2_ycb/info_pick_v0.json` exists there.

## Current caveat

The reachable vendored Robosuite submodule now supports the active Panda /
shape-generalization / clutter / YCB benchmark path on this machine, but the
YCB real-object configs additionally depend on the downloaded ManiSkill YCB
assets. If those assets are missing, the YCB envs will fail during reset even
though the Robosuite registry itself is healthy.

## Related docs

- [README](../README.md)
- [Project Plan](./plan.md)
- [Project Status](./status.md)
- [Development](./development.md)
