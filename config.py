"""
Configuration management for the Gameplay Analysis System.

This module provides configuration dataclasses and utilities.
Configuration values can come from:
1. CLI arguments (highest priority)
2. Environment variables
3. Hardcoded defaults (lowest priority)

The main.py CLI is responsible for constructing configuration and passing
it to modules. Modules should NOT import a global config singleton.

For backwards compatibility during migration, a load_config() function
is still available, but new code should receive config via dependency injection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def get_ollama_host() -> str:
    """
    Get Ollama host URL with smart defaults.
    
    Priority:
    1. OLLAMA_HOST environment variable (if set)
    2. Default based on detected environment
    
    Returns:
        Ollama server URL
    """
    env_host = os.getenv("OLLAMA_HOST")
    if env_host:
        return env_host
    
    # Check if we're likely running in a container
    in_container = (
        os.path.exists("/.dockerenv") or 
        os.path.exists("/run/.containerenv") or
        os.getenv("DOCKER_CONTAINER") == "true"
    )
    
    if in_container:
        return "http://ollama:11434"
    else:
        return "http://127.0.0.1:11434"


def get_default_model(model_type: str) -> str:
    """
    Get default model name for a given type.
    
    Args:
        model_type: Either "interpreter" or "reasoner"
    
    Returns:
        Model name string
    """
    defaults = {
        "interpreter": "qwen3-vl:8b-instruct-q4_K_M",
        "reasoner": "qwen3:14b-q4_K_M",
    }
    
    env_keys = {
        "interpreter": "INTERPRETER_MODEL",
        "reasoner": "REASONER_MODEL",
    }
    
    env_key = env_keys.get(model_type)
    if env_key:
        env_val = os.getenv(env_key)
        if env_val:
            return env_val
    
    return defaults.get(model_type, "")


@dataclass(frozen=True)
class InterpreterConfig:
    """Configuration for the GameplayInterpreter."""
    model: str
    quality_mode: str
    target_fps: int
    overlap_frames: int
    diff_threshold: float
    
    @classmethod
    def from_env(
        cls,
        model: Optional[str] = None,
        quality: Optional[str] = None,
        fps: Optional[int] = None,
        overlap: Optional[int] = None,
        diff_threshold: Optional[float] = None,
    ) -> "InterpreterConfig":
        """Create config with env var fallbacks."""
        return cls(
            model=model or get_default_model("interpreter"),
            quality_mode=quality or os.getenv("QUALITY_MODE", "high"),
            target_fps=fps if fps is not None else 1,
            overlap_frames=overlap if overlap is not None else 5,
            diff_threshold=diff_threshold if diff_threshold is not None else 3.0,
        )


@dataclass(frozen=True)
class ReasonerConfig:
    """Configuration for the GameplayReasoner."""
    model: str
    profile_path: Path
    
    @classmethod
    def from_env(
        cls,
        model: Optional[str] = None,
        profile_path: Optional[str] = None,
    ) -> "ReasonerConfig":
        """Create config with env var fallbacks."""
        return cls(
            model=model or get_default_model("reasoner"),
            profile_path=Path(profile_path or os.getenv("PROFILE_PATH", "player_profile.json")),
        )


# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================
# The following provides backwards compatibility for code that imports
# `from config import config`. New code should avoid using this pattern.

@dataclass(frozen=True)
class LegacyConfig:
    """
    Legacy configuration loaded from environment variables.
    
    DEPRECATED: Use InterpreterConfig/ReasonerConfig with dependency injection.
    This exists only for backwards compatibility during migration.
    """
    ollama_host: str
    interpreter_model: str
    reasoner_model: str
    quality_mode: str
    video_dir: Path
    output_dir: Path
    profile_path: Path
    batch_mode: bool
    video_path: Optional[str]
    map_name: Optional[str]
    game_mode: Optional[str]
    player_id: str


def load_config() -> LegacyConfig:
    """
    Load configuration from environment variables.
    
    DEPRECATED: This function exists for backwards compatibility.
    New code should receive configuration via constructor injection.
    
    Returns:
        LegacyConfig instance
    """
    return LegacyConfig(
        ollama_host=get_ollama_host(),
        interpreter_model=get_default_model("interpreter"),
        reasoner_model=get_default_model("reasoner"),
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


# Global config instance for backwards compatibility
# DEPRECATED: Import and use this pattern is discouraged
config = load_config()