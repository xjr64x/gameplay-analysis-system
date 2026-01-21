"""
Gameplay Interpreter v3
Hybrid cloud/local system with user choice.

Hardware target: RTX 5080 16GB VRAM
Cloud target: Configurable API (OpenRouter, Together, etc.)
"""

import cv2
import subprocess
import tempfile
import os
import json
import base64
from ollama import chat
from PIL import Image
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import date


# ============== CONFIGURATION ==============

@dataclass
class AnalysisOption:
    """Represents one analysis option presented to user."""
    name: str
    resolution: int
    max_frames: int
    diff_threshold: float
    location: str  # "local" or "cloud"
    quality: str  # "best", "good", "economy"
    description: str


def get_options(duration_seconds: float, cloud_available: bool = False) -> list[AnalysisOption]:
    """
    Returns available analysis options based on video duration.
    Sorted by quality (The best first).

    HARDWARE LIMITS (RTX 5080 16GB):
    - 768px: max 45 frames
    - 512px: max 60 frames
    - 448px: max 80 frames
    - 384px: max 100 frames
    """

    options = []

    if duration_seconds <= 90:
        # Short clip - local can handle best quality
        options.append(AnalysisOption(
            name="High Quality Local",
            resolution=768,
            max_frames=45,
            diff_threshold=3.0,
            location="local",
            quality="best",
            description="768px resolution, up to 45 key frames. Best accuracy for short clips.",
        ))
        options.append(AnalysisOption(
            name="Standard Local",
            resolution=512,
            max_frames=50,
            diff_threshold=3.0,
            location="local",
            quality="good",
            description="512px resolution, up to 50 frames. Slightly lower accuracy, more frames.",
        ))

    elif duration_seconds <= 180:  # Up to 3 min
        if cloud_available:
            options.append(AnalysisOption(
                name="Best Quality Cloud",
                resolution=768,
                max_frames=120,
                diff_threshold=4.0,
                location="cloud",
                quality="best",
                description="768px resolution, up to 120 frames. Maximum accuracy and coverage.",
            ))
        options.append(AnalysisOption(
            name="High Quality Local",
            resolution=768,
            max_frames=45,
            diff_threshold=5.0,
            location="local",
            quality="good",
            description="768px resolution, 45 frames. Good accuracy, selective frame sampling.",
        ))
        options.append(AnalysisOption(
            name="Balanced Local",
            resolution=512,
            max_frames=55,
            diff_threshold=4.0,
            location="local",
            quality="economy",
            description="512px resolution, more frames. Trades some accuracy for coverage.",
        ))

    elif duration_seconds <= 360:  # Up to 6 min
        if cloud_available:
            options.append(AnalysisOption(
                name="Best Quality Cloud",
                resolution=768,
                max_frames=180,
                diff_threshold=4.0,
                location="cloud",
                quality="best",
                description="768px resolution, up to 180 frames. Full match coverage at max quality.",
            ))
        options.append(AnalysisOption(
            name="Balanced Local",
            resolution=512,
            max_frames=60,
            diff_threshold=6.0,
            location="local",
            quality="good",
            description="512px resolution, 60 frames. Good balance for longer matches.",
        ))
        options.append(AnalysisOption(
            name="Economy Local",
            resolution=448,
            max_frames=70,
            diff_threshold=5.0,
            location="local",
            quality="economy",
            description="448px resolution, more frames. May miss small HUD details.",
        ))

    elif duration_seconds <= 540:  # Up to 9 min
        if cloud_available:
            options.append(AnalysisOption(
                name="Best Quality Cloud",
                resolution=768,
                max_frames=250,
                diff_threshold=5.0,
                location="cloud",
                quality="best",
                description="768px resolution, up to 250 frames. Full extended match coverage.",
            ))
        options.append(AnalysisOption(
            name="Balanced Local",
            resolution=448,
            max_frames=80,
            diff_threshold=7.0,
            location="local",
            quality="good",
            description="448px resolution, 80 frames. Selective sampling of key moments.",
        ))
        options.append(AnalysisOption(
            name="Economy Local",
            resolution=384,
            max_frames=90,
            diff_threshold=6.0,
            location="local",
            quality="economy",
            description="384px resolution, more frames. Lower accuracy on HUD/killfeed.",
        ))

    else:  # 9+ min (full objective matches)
        if cloud_available:
            options.append(AnalysisOption(
                name="Best Quality Cloud",
                resolution=768,
                max_frames=350,
                diff_threshold=5.0,
                location="cloud",
                quality="best",
                description="768px resolution, up to 350 frames. Complete match analysis.",
            ))
            options.append(AnalysisOption(
                name="Balanced Cloud",
                resolution=512,
                max_frames=400,
                diff_threshold=5.0,
                location="cloud",
                quality="good",
                description="512px resolution, up to 400 frames. Great coverage, good accuracy.",
            ))
        options.append(AnalysisOption(
            name="Economy Local",
            resolution=384,
            max_frames=100,
            diff_threshold=8.0,
            location="local",
            quality="economy",
            description="384px resolution, 100 frames. Best effort locally, may miss details.",
        ))

    return options


def display_options(options: list[AnalysisOption]) -> None:
    """Pretty print available options."""
    print("\n" + "=" * 60)
    print("AVAILABLE ANALYSIS OPTIONS")
    print("=" * 60)

    for i, opt in enumerate(options, 1):
        quality_badge = {"best": "★★★", "good": "★★☆", "economy": "★☆☆"}[opt.quality]
        location_badge = "☁️ " if opt.location == "cloud" else "💻"

        print(f"\n[{i}] {location_badge} {opt.name} {quality_badge}")
        print(f"    Resolution: {opt.resolution}px | Max Frames: {opt.max_frames}")
        print(f"    {opt.description}")

    print("\n" + "=" * 60)


def select_option(options: list[AnalysisOption], auto_select: Optional[str] = None) -> AnalysisOption:
    """
    Let user select an option or auto-select.

    auto_select options: "best", "local_best", "economy", or None for interactive
    """
    if auto_select:
        if auto_select == "best":
            return options[0]
        elif auto_select == "local_best":
            for opt in options:
                if opt.location == "local":
                    return opt
            return options[-1]  # Fallback to last option
        elif auto_select == "economy":
            return options[-1]

    # Interactive selection
    display_options(options)

    while True:
        try:
            choice = input("\nSelect option (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"Please enter 1-{len(options)}")
        except ValueError:
            print("Please enter a number")


# ============== FRAME PROCESSING ==============

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frames(video_path: str, fps: int = 1) -> list:
    """Extract frames from video at specified FPS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = os.path.join(tmpdir, "frame_%06d.jpg")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps}",
            output_pattern,
            "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True)

        frames = []
        for frame in sorted(os.listdir(tmpdir)):
            img = Image.open(os.path.join(tmpdir, frame)).convert("RGB")
            frames.append(img)

    return frames


def pil_to_gray(img: Image.Image):
    """Convert PIL image to grayscale numpy array."""
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def select_frames(frames: list, max_frames: int, diff_threshold: float = 5.0) -> list:
    """Select frames based on visual difference."""
    if not frames:
        return []

    selected = [frames[0]]
    prev_gray = pil_to_gray(frames[0])
    skipped = 0

    for img in frames[1:]:
        gray = pil_to_gray(img)
        diff = cv2.absdiff(prev_gray, gray)
        score = diff.mean()

        if score > diff_threshold:
            selected.append(img)
            prev_gray = gray
        else:
            skipped += 1

    print(f"Frame selection: {len(frames)} raw → {len(selected)} selected ({skipped} stale frames removed)")

    if len(selected) > max_frames:
        step = len(selected) / max_frames
        selected = [selected[int(i * step)] for i in range(max_frames)]
        print(f"Downsampled to {len(selected)} frames")

    return selected


def preprocess_frames(frames: list, max_size: int = 512) -> list:
    """Resize frames to target resolution."""
    processed = []

    for img in frames:
        w, h = img.size
        scale = min(max_size / w, max_size / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
        processed.append(img)

    return processed


# ============== PROMPTS ==============

SYSTEM_PROMPT = """You are a Gameplay Mechanics Analyst. Analyze combat micro-behaviors.
Output valid JSON only. No conversational text."""

ANALYSIS_PROMPT = """Analyze the combat mechanics in these frames.
Identify distinct engagements (Kill or Death).

For each engagement, extract:
1. "result": "Kill" or "Death"
2. "setup": Player's action BEFORE firing (e.g., "Sprinting", "Holding angle").
3. "enemy_state": Enemy awareness (e.g., "Looking away", "Stunned").
4. "mechanics": Aim/Movement description (e.g., "Tracking", "Flick", "Drop-shot").
5. "outcome": The outcome of the event
6. "evidence": STRICTLY textual evidence visible on HUD (e.g., Medals like "Headshot", "Double Kill", or Killfeed text).

Output JSON format:
{
    "engagements": [
        {
            "result": "Kill", 
            "setup": "Player was pre-aiming a doorway while walking.", 
            "enemy_state": "Enemy sprinted through door unaware.", 
            "mechanics": "Minimal crosshair adjustment needed. Player fired immediately upon visual contact.", 
            "outcome": "Instant kill, enemy did not return fire.", 
            "evidence": ["Headshot Medal", "Killfeed confirmation"]
        }
    ]
}

If no combat, return {"engagements": []}."""


# ============== ANALYSIS BACKENDS ==============

def analyze_local(frames: list, model: str = "qwen3-vl:8b-instruct-q4_K_M") -> str:
    """Run analysis using local Ollama."""
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for i, img in enumerate(frames):
            path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
            img.save(path, format="JPEG", quality=85)
            image_paths.append(path)

        print(f"Sending {len(image_paths)} frames to local model...")

        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ANALYSIS_PROMPT, "images": image_paths}
            ],
            # CRITICAL FIX FOR OUTPUT
            options={
                "num_predict": 2048,    # Allow longer responses (prevents cutoff)
                "temperature": 0.1,     # Low temp = more factual/robotic, less creative
                "repeat_penalty": 1.3,  # Prevents repeating
                "repeat_last_n": 256,   # Checks back 256 tokens for repeats
            }
        )

        return response.message.content


def analyze_cloud(frames: list, api_config: dict) -> str:
    """
    Run analysis using cloud API.

    api_config should contain:
    - provider: "openrouter", "together", "anthropic", etc.
    - api_key: your API key
    - model: model identifier
    """
    provider = api_config.get("provider", "openrouter")
    api_key = api_config.get("api_key")
    model = api_config.get("model", "qwen/qwen-2-vl-72b-instruct")

    if not api_key:
        raise ValueError("Cloud API key not configured. Set api_key in config.")

    # Encode frames as base64
    encoded_frames = []
    for img in frames:
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        encoded_frames.append(b64)

    print(f"Sending {len(encoded_frames)} frames to cloud ({provider})...")

    if provider == "openrouter":
        return _call_openrouter(encoded_frames, api_key, model)
    elif provider == "together":
        return _call_together(encoded_frames, api_key, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_openrouter(encoded_frames: list, api_key: str, model: str) -> str:
    """Call OpenRouter API."""
    import requests

    # Build content with images
    content = []
    for b64 in encoded_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": ANALYSIS_PROMPT})

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]
        }
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _call_together(encoded_frames: list, api_key: str, model: str) -> str:
    """Call Together AI API."""
    import requests

    content = []
    for b64 in encoded_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": ANALYSIS_PROMPT})

    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]
        }
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ============== MAIN FUNCTION ==============

def interpret_video(
        video_path: str,
        auto_select: Optional[str] = None,
        cloud_config: Optional[dict] = None,
        local_model: str = "qwen3-vl:8b-instruct-q4_K_M"
) -> dict:
    """
    Main interpretation function with hybrid cloud/local options.

    Args:
        video_path: Path to gameplay video
        auto_select: "best", "local_best", "economy", or None for interactive
        cloud_config: Dict with provider, api_key, model for cloud option
        local_model: Ollama model name for local inference

    Returns:
        Dict with analysis result and metadata
    """
    # Get video info
    duration = get_video_duration(video_path)

    print(f"\n{'=' * 60}")
    print(f"VIDEO: {video_path}")
    print(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")
    print(f"{'=' * 60}")

    # Get available options
    cloud_available = cloud_config is not None and cloud_config.get("api_key")
    options = get_options(duration, cloud_available)

    # Select option
    selected = select_option(options, auto_select)

    print(f"\nSelected: {selected.name}")
    print(f"  Resolution: {selected.resolution}px")
    print(f"  Max frames: {selected.max_frames}")
    print(f"  Location: {selected.location}")

    # Process frames
    print("\nExtracting frames...")
    frames = extract_frames(video_path, fps=1)
    print(f"Extracted {len(frames)} raw frames")

    frames = select_frames(frames, selected.max_frames, selected.diff_threshold)

    print("Preprocessing frames...")
    frames = preprocess_frames(frames, max_size=selected.resolution)

    # Run analysis
    print("\nAnalyzing...")
    if selected.location == "cloud":
        raw_result = analyze_cloud(frames, cloud_config)
    else:
        raw_result = analyze_local(frames, local_model)

    # Parse the JSON from the model
    try:
        vision_data = json.loads(raw_result)
    except json.JSONDecodeError:
        # Fallback if model chats instead of valid JSON
        vision_data = {"raw_output": raw_result}

    # CONSTRUCT THE FINAL PAYLOAD FOR THE REASONER
    # This combines Python's hard facts with the AI's visual observations
    final_payload = {
        # Metadata for the reasoner
        "metadata": {
            "video_path": video_path,
            "duration_seconds": duration,
            "date_processed": str(date.today()),
            "frames_analyzed": len(frames)
        },
        # Parsed vision data for the reasoner
        "vision_data": vision_data,
        # CLI-friendly fields
        "option_used": {
            "name": selected.name,
            "resolution": selected.resolution,
            "max_frames": selected.max_frames,
            "frames_analyzed": len(frames),
            "location": selected.location,
            "quality": selected.quality
        },
        # Raw analysis text for display
        "analysis": raw_result
    }

    return final_payload


# ============== CLI ==============

if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "sample_gameplay.MP4"

    # Example cloud config (uncomment and add your key to use)
    # cloud_config = {
    #     "provider": "openrouter",
    #     "api_key": "your-api-key-here",
    #     "model": "qwen/qwen-2-vl-72b-instruct"
    # }
    cloud_config = None  # Local only for now

    result = interpret_video(
        video_path,
        auto_select=None,  # Interactive mode - set to "best", "local_best", or "economy" for auto
        cloud_config=cloud_config
    )

    print("\n" + "=" * 60)
    print("ANALYSIS RESULT")
    print("=" * 60)
    print(f"Option used: {result['option_used']['name']}")
    print(f"Frames analyzed: {result['option_used']['frames_analyzed']}")
    print(f"Location: {result['option_used']['location']}")
    print("\n" + result["analysis"])