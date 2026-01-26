# Docker Deployment Guide

## Prerequisites

- Docker Engine 24.0+ with Compose V2
- NVIDIA Container Toolkit (for GPU support)
- 16GB+ GPU VRAM (RTX 5080 or equivalent)

### Installing NVIDIA Container Toolkit (Linux)

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Windows with Docker Desktop

Ensure you have:

1. Docker Desktop with WSL2 backend enabled
2. NVIDIA drivers installed on Windows host
3. GPU support enabled in Docker Desktop settings

## Architecture

```
                    Docker Compose Network
    +-----------------------------------------------------+
    |                                                     |
    |  +------------------+      +---------------------+  |
    |  |    gameplay      |      |       ollama        |  |
    |  |  (Python 3.12)   |----->|  (GPU: NVIDIA)      |  |
    |  |                  | :11434|                     |  |
    |  |  - interpreter   |      |  Models:            |  |
    |  |  - reasoner      |      |  - qwen3-vl:8b      |  |
    |  |  - batch_analyze |      |  - qwen3:14b        |  |
    |  +--------+---------+      +---------------------+  |
    |           |                                         |
    +-----------|-----------------------------------------+
                |
    +-----------|-----------------------------------------+
    |   Volumes |                                         |
    |   +-------v-------+  +----------+  +-----------+    |
    |   |    /videos    |  |  /output |  | /profiles |    |
    |   | (input files) |  |  (JSON)  |  | (player)  |    |
    |   +---------------+  +----------+  +-----------+    |
    +-----------------------------------------------------+
```

Two containers:

- **ollama**: GPU-accelerated model serving
- **gameplay**: Python application that runs the analysis pipeline

## Quick Start

### 1. First-Time Setup

```bash
# Create required directories
mkdir -p videos output profiles

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Start Ollama
docker compose up -d ollama

# Wait for healthy status, then pull models (10-15 minutes)
docker compose --profile init up ollama-init
```

### 2. Place Your Video

```bash
# Copy your gameplay video to the videos directory
cp /path/to/your/gameplay.mp4 videos/
```

### 3. Configure Your Analysis

Edit `.env`:

```bash
VIDEO_PATH=/app/videos/your_gameplay.mp4
MAP_NAME=Nuketown
GAME_MODE=Domination
PLAYER_ID=your_name
BATCH_MODE=true
```

### 4. Run Analysis

```bash
docker compose up gameplay
```

For subsequent runs with different videos, update `.env` and run:

```bash
docker compose up gameplay --force-recreate
```

### 5. Stop Services

```bash
docker compose down

# To also remove model volumes (will require re-downloading models):
docker compose down -v
```

## Configuration

### Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama server URL |
| `INTERPRETER_MODEL` | `qwen3-vl:8b-instruct-q4_K_M` | Vision model for video analysis |
| `REASONER_MODEL` | `qwen3:14b-q4_K_M` | Reasoning model for pattern analysis |
| `QUALITY_MODE` | `high` | Processing quality (`high` or `fast`) |
| `VIDEO_PATH` | - | Path to video file (required) |
| `MAP_NAME` | - | Map name (required) |
| `GAME_MODE` | - | Game mode (required) |
| `PLAYER_ID` | `player` | Player identifier for profile tracking |
| `BATCH_MODE` | `true` | Non-interactive batch processing |

### Volume Locations

| Purpose | Container Path | Host Path |
| ------- | -------------- | --------- |
| Input Videos | `/app/videos` | `./videos/` |
| Output JSON | `/app/output` | `./output/` |
| Player Profiles | `/app/profiles` | `./profiles/` |

## GPU Troubleshooting

### Verify NVIDIA Container Toolkit

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Check Ollama GPU Access

```bash
docker compose exec ollama nvidia-smi
```

### View Ollama Logs

```bash
docker compose logs ollama
```

### Common Issues

**"could not select device driver" error:**

- Ensure NVIDIA Container Toolkit is installed
- Restart Docker: `sudo systemctl restart docker`

**Models not loading (OOM):**

- Ensure GPU has 16GB+ VRAM
- Try FAST quality mode: `QUALITY_MODE=fast`

**Container unhealthy:**

- Check logs: `docker compose logs ollama`
- Verify ollama is responding: `docker exec gameplay-ollama ollama list`

## Health Checks

```bash
# Check service status
docker compose ps

# List available models
docker exec gameplay-ollama ollama list

# View application logs
docker compose logs gameplay
```

## Building for Production

```bash
# Build with specific tag
docker compose build --no-cache

# Tag for registry
docker tag gameplay-analysis-system-gameplay:latest your-registry/gameplay-app:v1.0

# Push to registry
docker push your-registry/gameplay-app:v1.0
```
