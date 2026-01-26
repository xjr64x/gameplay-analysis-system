"""
Configuration management for the Gameplay Analysis System.

Centralizes environment variable handling and provides defaults.
Supports both local development and containerized deployment.

Local development:
    OLLAMA_HOST defaults to http://127.0.0.1:11434 (local Ollama)

Containerized (Docker):
    Set OLLAMA_HOST=http://ollama:11434 (Docker service DNS)
    This is set automatically in docker-compose.yml
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _get_ollama_host() -> str:
    """
    Get Ollama host URL with smart defaults.
    
    Priority:
    1. OLLAMA_HOST environment variable (if set)
    2. Default based on detected environment
    """
    env_host = os.getenv("OLLAMA_HOST")
    if env_host:
        return env_host
    
    # Check if we're likely running in a container
    # (Docker sets /.dockerenv or we can check for typical container paths)
    in_container = (
        os.path.exists("/.dockerenv") or 
        os.path.exists("/run/.containerenv") or
        os.getenv("DOCKER_CONTAINER") == "true"
    )
    
    if in_container:
        # In container: use Docker service DNS name
        return "http://ollama:11434"
    else:
        # Local development: use localhost
        return "http://127.0.0.1:11434"


@dataclass(frozen=True)
class Config:
    """Immutable configuration loaded from environment variables."""

    # Ollama connection
    ollama_host: str

    # Model names
    interpreter_model: str
    reasoner_model: str

    # Processing settings
    quality_mode: str

    # Paths
    video_dir: Path
    output_dir: Path
    profile_path: Path

    # Runtime mode
    batch_mode: bool

    # Match metadata (for batch mode)
    video_path: Optional[str]
    map_name: Optional[str]
    game_mode: Optional[str]
    player_id: str


def load_config() -> Config:
    """
    Load configuration from environment variables with sensible defaults.

    Environment Variables:
        OLLAMA_HOST: Ollama server URL 
                     Default: http://127.0.0.1:11434 (local) or http://ollama:11434 (container)
        INTERPRETER_MODEL: Vision model name (default: qwen3-vl:8b-instruct-q4_K_M)
        REASONER_MODEL: Reasoning model name (default: qwen3:14b-q4_K_M)
        QUALITY_MODE: Processing quality - "high" or "fast" (default: high)
        VIDEO_DIR: Directory for input videos (default: ./videos)
        OUTPUT_DIR: Directory for output JSON (default: .)
        PROFILE_PATH: Path to player profile JSON (default: player_profile.json)
        BATCH_MODE: "true" for non-interactive mode (default: false)
        VIDEO_PATH: Path to video file for batch mode
        MAP_NAME: Map name for batch mode
        GAME_MODE: Game mode for batch mode
        PLAYER_ID: Player identifier (default: player)

    Returns:
        Config: Immutable configuration object
    """
    return Config(
        ollama_host=_get_ollama_host(),
        interpreter_model=os.getenv(
            "INTERPRETER_MODEL", "qwen3-vl:8b-instruct-q4_K_M"
        ),
        reasoner_model=os.getenv("REASONER_MODEL", "qwen3:14b-q4_K_M"),
        quality_mode=os.getenv("QUALITY_MODE", "high"),
        video_dir=Path(os.getenv("VIDEO_DIR", "./videos")),
        output_dir=Path(os.getenv("OUTPUT_DIR", ".")),
        profile_path=Path(os.getenv("PROFILE_PATH", "player_profile.json")),
        batch_mode=os.getenv("BATCH_MODE", "false").lower() == "true",
        video_path=os.getenv("VIDEO_PATH"),
        map_name=os.getenv("MAP_NAME"),
        game_mode=os.getenv("GAME_MODE"),
        player_id=os.getenv("PLAYER_ID", "player"),
    )


# Global config instance (loaded once at import time)
config = load_config()