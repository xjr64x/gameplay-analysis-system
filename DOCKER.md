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
    |  |                  |:11434|                     |  |
    |  |  - main.py CLI   |      |  Models:            |  |
    |  |  - interpreter   |      |  - qwen3-vl:8b      |  |
    |  |  - reasoner      |      |  - qwen3:14b        |  |
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
- **gameplay**: Python application running the unified CLI

## Quick Start

### 1. First-Time Setup

```bash
# Create required directories
mkdir -p videos output profiles

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

### 3. Verify System Health

```bash
# Run system checks to ensure everything is working
docker compose run --rm gameplay check --all
```

### 4. Run Analysis

```bash
# Batch mode (non-interactive)
docker compose run --rm gameplay analyze /app/videos/gameplay.mp4 \
  --map Nuketown --mode Domination --batch

# Interactive mode (with chat)
docker compose run --rm -it gameplay analyze /app/videos/gameplay.mp4 \
  --map Nuketown --mode Domination

# Interpreter only (video → narration)
docker compose run --rm gameplay interpret /app/videos/gameplay.mp4 \
  --map Nuketown --mode Domination
```

### 5. Stop Services

```bash
docker compose down

# To also remove model volumes (will require re-downloading models):
docker compose down -v
```

## CLI Commands

The container uses `main.py` as the entrypoint. Available commands:

### analyze

Full pipeline: video → interpreter → reasoner → optional chat

```bash
docker compose run --rm gameplay analyze VIDEO --map MAP --mode MODE [OPTIONS]

Options:
  --batch, -b           Non-interactive mode (no chat)
  --quality, -q         "high" (default) or "fast"
  --fps                 Frames per second (default: 1)
  --overlap             Overlap frames between batches (default: 5)
  --profile, -p         Profile path (default: /app/profiles/player_profile.json)
  --player-id           Player ID for new profiles
  --output-dir, -o      Output directory (default: /app/output)
  --quiet               Reduce verbosity
```

### interpret

Run interpreter only (video → narration JSON)

```bash
docker compose run --rm gameplay interpret VIDEO --map MAP --mode MODE [OPTIONS]

Options:
  --quality, -q         "high" (default) or "fast"
  --print-narration     Print narration to stdout after saving
  --output-dir, -o      Output directory
```

### check

Verify system components

```bash
docker compose run --rm gameplay check [OPTIONS]

Options:
  --all, -a     Run all checks
  --ffmpeg      Check ffmpeg/ffprobe
  --ollama      Check Ollama connection and models
  --deps        Check Python dependencies
  --paths       Check output/profile paths are writable
```

### profile

Manage player profiles

```bash
# Show profile summary
docker compose run --rm gameplay profile show --profile /app/profiles/player.json

# Create new profile
docker compose run --rm gameplay profile create --player-id myname

# Export profile as JSON
docker compose run --rm gameplay profile export --profile /app/profiles/player.json
```

## Configuration

### Environment Variables

All settings can be configured via environment variables or CLI flags. CLI flags take precedence.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama server URL |
| `INTERPRETER_MODEL` | `qwen3-vl:8b-instruct-q4_K_M` | Vision model for video analysis |
| `REASONER_MODEL` | `qwen3:14b-q4_K_M` | Reasoning model for pattern analysis |
| `QUALITY_MODE` | `high` | Processing quality (`high` or `fast`) |
| `OUTPUT_DIR` | `/app/output` | Output directory for analysis JSON |
| `PROFILE_PATH` | `/app/profiles/player_profile.json` | Player profile path |

### Volume Locations

| Purpose | Container Path | Host Path |
| ------- | -------------- | --------- |
| Input Videos | `/app/videos` | `./videos/` |
| Output JSON | `/app/output` | `./output/` |
| Player Profiles | `/app/profiles` | `./profiles/` |

## Examples

### Analyze Multiple Videos

```bash
# First video
docker compose run --rm gameplay analyze /app/videos/match1.mp4 \
  --map Nuketown --mode Domination --batch \
  --profile /app/profiles/player.json

# Second video (same profile, builds history)
docker compose run --rm gameplay analyze /app/videos/match2.mp4 \
  --map Raid --mode Hardpoint --batch \
  --profile /app/profiles/player.json
```

### Fast Quality Mode

```bash
# Use fast mode for quicker processing (lower resolution, more frames per batch)
docker compose run --rm gameplay analyze /app/videos/match.mp4 \
  --map Nuketown --mode Domination --batch --quality fast
```

### Custom Models

```bash
# Override default models via environment
INTERPRETER_MODEL=llava:13b REASONER_MODEL=llama3:8b \
  docker compose run --rm gameplay analyze /app/videos/match.mp4 \
  --map Nuketown --mode Domination --batch

# Or via CLI flags
docker compose run --rm gameplay analyze /app/videos/match.mp4 \
  --map Nuketown --mode Domination --batch \
  --interpreter-model llava:13b --reasoner-model llama3:8b
```

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
- Try FAST quality mode: `--quality fast`

**Container unhealthy:**

- Check logs: `docker compose logs ollama`
- Verify ollama is responding: `docker exec gameplay-ollama ollama list`

**Check failures:**

- Run `docker compose run --rm gameplay check --all` to diagnose
- Check specific components: `--ffmpeg`, `--ollama`, `--deps`, `--paths`

## Health Checks

```bash
# Full system check
docker compose run --rm gameplay check --all

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

## Migrating from Previous Versions

If you were using `batch_analyze.py` or environment-based configuration:

**Old way:**
```bash
VIDEO_PATH=/app/videos/match.mp4 MAP_NAME=Nuketown GAME_MODE=Domination \
  docker compose up gameplay
```

**New way:**
```bash
docker compose run --rm gameplay analyze /app/videos/match.mp4 \
  --map Nuketown --mode Domination --batch
```

The new CLI provides explicit commands and flags, making it clearer what the system will do.