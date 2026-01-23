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

### Windows with WSL2

Ensure you have:

1. WSL2 with Ubuntu
2. Docker Desktop with WSL2 backend
3. NVIDIA drivers installed on Windows host

## Quick Start

### 1. First-Time Setup (Pull Models)

```bash
# Create required directories
mkdir -p videos output profiles

# Build the application image
docker compose build

# Start Ollama and pull required models (may take 10-15 minutes)
docker compose --profile init up

# Wait until you see "Models ready!" then Ctrl+C
```

### 2. Place Your Video

```bash
# Copy your gameplay video to the videos directory
cp /path/to/your/gameplay.mp4 videos/
```

### 3. Interactive Mode

```bash
# Start services and attach to interactive session
docker compose run --rm gameplay-app
```

Or with custom video/map/mode:

```bash
VIDEO_PATH=/app/videos/your_gameplay.mp4 \
MAP_NAME=Terminal \
GAME_MODE=Hardpoint \
docker compose run --rm gameplay-app
```

### 4. Batch Mode (Non-Interactive)

```bash
# Run analysis without interactive prompts
VIDEO_PATH=/app/videos/your_gameplay.mp4 \
MAP_NAME=Terminal \
GAME_MODE=Hardpoint \
docker compose --profile batch up gameplay-batch
```

### 5. Stop Services

```bash
docker compose down

# To also remove model volumes (will require re-downloading models):
docker compose down -v
```

## Configuration

### Using .env File

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your settings
# Then run without specifying variables
docker compose run --rm gameplay-app
```

### Environment Variables

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama server URL |
| `INTERPRETER_MODEL` | `qwen3-vl:8b-instruct-q4_K_M` | Vision model for video analysis |
| `REASONER_MODEL` | `qwen3:14b-q4_K_M` | Reasoning model for pattern analysis |
| `QUALITY_MODE` | `high` | Processing quality (`high` or `fast`) |
| `VIDEO_PATH` | - | Path to video file (required for batch mode) |
| `MAP_NAME` | - | Map name (required for batch mode) |
| `GAME_MODE` | - | Game mode (required for batch mode) |
| `PLAYER_ID` | `player` | Player identifier for profile tracking |
| `BATCH_MODE` | `false` | Enable non-interactive mode |

## Volume Locations

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

**Connection refused to Ollama:**

- Wait for Ollama health check to pass
- Check logs: `docker compose logs ollama`

## Health Checks

```bash
# Check service status
docker compose ps

# Check Ollama health
curl http://localhost:11434/api/version

# List available models
docker compose exec ollama ollama list

# View application logs
docker compose logs gameplay-app
```

## Building for Production

```bash
# Build with specific tag
docker compose build --no-cache

# Tag for registry
docker tag gameplay-analysis-system_gameplay-app:latest your-registry/gameplay-app:v1.0

# Push to registry
docker push your-registry/gameplay-app:v1.0
```

## Architecture

```
                    Docker Compose Network
    +-----------------------------------------------------+
    |                                                     |
    |  +------------------+      +---------------------+  |
    |  |  gameplay-app    |      |       ollama        |  |
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
