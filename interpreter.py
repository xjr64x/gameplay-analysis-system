"""
Gameplay Video Interpreter - Batched Processing Pipeline

A reusable pipeline for analyzing gameplay videos using vision-language models.
Processes videos in batches with overlap for context continuity, then merges
segments into coherent narration.

Hardware target: RTX 5080 16GB VRAM
Model: qwen3-vl:8b-instruct-q4_K_M via Ollama
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ollama import chat
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

class QualityMode(Enum):
    """Processing quality modes based on tested hardware limits."""
    HIGH = "high"  # 768px, 45 frames/batch
    FAST = "fast"  # 512px, 60 frames/batch


@dataclass(frozen=True)
class QualityConfig:
    """Immutable settings for each quality mode."""
    resolution: int
    frames_per_batch: int
    fallback_frame_reduction: int
    description: str


QUALITY_CONFIGS: dict[QualityMode, QualityConfig] = {
    QualityMode.HIGH: QualityConfig(
        resolution=768,
        frames_per_batch=45,
        fallback_frame_reduction=10,
        description="768px resolution, 45 frames/batch. Best visual detail.",
    ),
    QualityMode.FAST: QualityConfig(
        resolution=512,
        frames_per_batch=60,
        fallback_frame_reduction=15,
        description="512px resolution, 60 frames/batch. More coverage.",
    ),
}

# Context management
MAX_CONTEXT_CHARS = 400
MAX_CONTEXT_HISTORY = 2
DEFAULT_MODEL = "qwen3-vl:8b-instruct-q4_K_M"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MatchMetadata:
    """Required metadata about the match being analyzed."""
    map_name: str          # e.g., "Nuketown", "Raid", "Terminal"
    mode: str              # e.g., "Hardpoint", "Search and Destroy", "Team Deathmatch"

    def __post_init__(self):
        if not self.map_name or not self.map_name.strip():
            raise ValueError("map_name is required")
        if not self.mode or not self.mode.strip():
            raise ValueError("mode is required")
        self.map_name = self.map_name.strip()
        self.mode = self.mode.strip()


@dataclass
class BatchBoundary:
    """Defines a batch's frame boundaries."""
    start_idx: int
    end_idx: int
    overlap_frames: int
    narration_start_idx: int


@dataclass
class BatchResult:
    """Result from processing a single batch."""
    batch_index: int
    start_frame: int
    end_frame: int
    overlap_frames: int
    narration_start_frame: int
    start_timestamp: float
    end_timestamp: float
    narration_start_timestamp: float
    context_summary: str
    narration: str
    frame_count: int
    frames_used: int
    success: bool
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete result from the video analysis pipeline."""
    video_path: str
    match_metadata: MatchMetadata
    duration_seconds: float
    quality_mode: str
    resolution: int
    frames_per_batch: int
    total_frames_extracted: int
    total_frames_used: int
    total_batches: int
    successful_batches: int
    failed_batch_indices: list[int]
    narration: str
    merged_narration: str
    batch_details: list[dict]
    processing_time_seconds: float
    processed_at: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "match": {
                "game": "Call of Duty: Black Ops 6",
                "map": self.match_metadata.map_name,
                "mode": self.match_metadata.mode,
            },
            "metadata": {
                "video_path": self.video_path,
                "duration_seconds": self.duration_seconds,
                "processed_at": self.processed_at,
                "processing_time_seconds": self.processing_time_seconds,
            },
            "processing": {
                "quality_mode": self.quality_mode,
                "resolution": self.resolution,
                "frames_per_batch": self.frames_per_batch,
                "total_frames_extracted": self.total_frames_extracted,
                "total_frames_used": self.total_frames_used,
                "total_batches": self.total_batches,
                "successful_batches": self.successful_batches,
                "failed_batches": self.failed_batch_indices,
            },
            "narration": self.merged_narration or self.narration,
            "segmented_narration": self.narration,
            "batch_details": self.batch_details,
        }

    def save(self, output_path: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if output_path is None:
            output_path = str(Path(self.video_path).with_suffix("")) + "_analysis.json"
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return output_path


@dataclass
class _ProcessingState:
    """Internal state tracker during batch processing."""
    video_path: str
    match_metadata: MatchMetadata
    quality_mode: QualityMode
    total_frames: int = 0
    total_batches: int = 0
    batch_results: list[BatchResult] = field(default_factory=list)
    failed_batches: list[int] = field(default_factory=list)
    context_history: list[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are analyzing Call of Duty: Black Ops 6 gameplay footage. Describe what you observe as a continuous sequence of actions.

Rules:
- Describe movement, positioning, gunfights, and outcomes
- Note what leads to engagements, what happens during them, and the aftermath
- Reference visible HUD elements (health, ammo, minimap, killfeed, scorestreaks)
- Track objective interactions (hardpoint time, bomb plants/defuses, flag captures)
- Note weapon usage, tactical/lethal equipment, and ability usage
- Write in present tense
- Be thorough but concise

You MUST respond in valid JSON format only. No other text."""


def _build_batch_prompt(
    batch_index: int,
    total_batches: int,
    start_time: float,
    end_time: float,
    narration_start_time: float,
    overlap_frames: int,
    context_history: list[str],
    match_metadata: MatchMetadata,
) -> str:
    """Build the analysis prompt for a batch."""
    time_range = f"{start_time:.1f}s - {end_time:.1f}s"
    narration_range = f"{narration_start_time:.1f}s - {end_time:.1f}s"

    # Match context
    match_context = f"Map: {match_metadata.map_name} | Mode: {match_metadata.mode}"

    context_section = ""
    if context_history:
        recent = context_history[-MAX_CONTEXT_HISTORY:]
        context_section = "\n\nPrevious context:\n" + "\n".join(f"- {c}" for c in recent)

    overlap_instruction = ""
    if overlap_frames > 0:
        overlap_instruction = f"""
IMPORTANT: The first {overlap_frames} frames (up to {narration_start_time:.1f}s) are PROVIDED FOR CONTEXT ONLY.
These frames were already described in the previous segment.
DO NOT describe these frames again. Start your narration from {narration_start_time:.1f}s onward."""

    if batch_index == 0:
        position = f"the FIRST segment of {total_batches}"
    elif batch_index == total_batches - 1:
        position = f"the FINAL segment of {total_batches}"
    else:
        position = f"segment {batch_index + 1} of {total_batches}"

    return f"""Analyzing {position} of a Call of Duty: Black Ops 6 match.
{match_context}
Full segment time: {time_range}
Narrate from: {narration_range}{overlap_instruction}{context_section}

Respond with ONLY this JSON structure:
{{
    "context": "2-3 sentence summary of situation at segment end (max 400 chars)",
    "narration": "Full description of what happens from {narration_start_time:.1f}s to {end_time:.1f}s"
}}"""


# =============================================================================
# VIDEO UTILITIES
# =============================================================================

def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _extract_frames(video_path: str, output_dir: str, fps: int = 1) -> list[str]:
    """Extract frames from video at specified FPS."""
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        output_pattern,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    return sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )


def _compute_frame_differences(frame_paths: list[str]) -> list[float]:
    """Compute visual difference scores between consecutive frames."""
    if len(frame_paths) < 2:
        return [0.0] * len(frame_paths)

    scores = [0.0]
    prev_gray = None

    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            scores.append(float(diff.mean()))
        prev_gray = gray

    return scores


def _select_batch_frames(
    frame_paths: list[str],
    diff_scores: list[float],
    start_idx: int,
    end_idx: int,
    max_frames: int,
    diff_threshold: float = 3.0,
) -> list[str]:
    """Select frames using difference-based filtering."""
    batch_paths = frame_paths[start_idx:end_idx]
    batch_scores = diff_scores[start_idx:end_idx]

    if len(batch_paths) <= max_frames:
        return batch_paths

    # Always include first frame
    selected = [0]

    # Include frames with significant visual change
    for i in range(1, len(batch_paths) - 1):
        if batch_scores[i] > diff_threshold:
            selected.append(i)

    # Always include last frame
    if len(batch_paths) - 1 not in selected:
        selected.append(len(batch_paths) - 1)

    # Downsample if still too many
    if len(selected) > max_frames:
        step = len(selected) / max_frames
        selected = [selected[int(i * step)] for i in range(max_frames)]
        if len(batch_paths) - 1 not in selected:
            selected[-1] = len(batch_paths) - 1

    return [batch_paths[i] for i in sorted(selected)]


def _resize_frame(img: Image.Image, max_size: int) -> Image.Image:
    """Resize frame to fit within max_size while preserving aspect ratio."""
    w, h = img.size
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    return img


def _prepare_batch_frames(
    frame_paths: list[str],
    resolution: int,
    output_dir: str,
) -> list[str]:
    """Resize and save frames for model consumption."""
    prepared = []
    for i, path in enumerate(frame_paths):
        img = Image.open(path).convert("RGB")
        img = _resize_frame(img, resolution)
        out_path = os.path.join(output_dir, f"batch_frame_{i:04d}.jpg")
        img.save(out_path, format="JPEG", quality=85)
        prepared.append(out_path)
    return prepared


# =============================================================================
# MODEL INTERACTION
# =============================================================================

def _analyze_batch(
    frame_paths: list[str],
    prompt: str,
    model: str,
) -> tuple[bool, str, Optional[str]]:
    """Send batch to model for analysis."""
    try:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": frame_paths},
            ],
            options={
                "num_predict": 2048,
                "temperature": 0.1,
                "repeat_penalty": 1.3,
                "repeat_last_n": 256,
            },
        )
        content = response.message.content
        if content and content.strip():
            return True, content, None
        return False, "", "Model returned empty response"
    except Exception as e:
        return False, "", str(e)


def _parse_response(response: str) -> tuple[str, str]:
    """Parse JSON response into (context, narration)."""
    context, narration = "", response

    json_str = response.strip()

    # Handle Markdown code blocks
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                json_lines.append(line)
        json_str = "\n".join(json_lines)

    # Try parsing JSON
    try:
        data = json.loads(json_str)
        context = data.get("context", "")
        narration = data.get("narration", response)
    except json.JSONDecodeError:
        # Fallback: extract JSON object from response
        start, end = response.find("{"), response.rfind("}") + 1
        if 0 <= start < end:
            try:
                data = json.loads(response[start:end])
                context = data.get("context", "")
                narration = data.get("narration", response)
            except json.JSONDecodeError:
                pass

    # Enforce context length limit
    if len(context) > MAX_CONTEXT_CHARS:
        truncated = context[:MAX_CONTEXT_CHARS]
        last_period = truncated.rfind(".")
        if last_period > MAX_CONTEXT_CHARS // 2:
            context = truncated[: last_period + 1]
        else:
            context = truncated.rsplit(" ", 1)[0] + "..."

    return context, narration


def _merge_narration(segments: list[tuple[float, float, str]], model: str) -> str:
    """Merge segment narrations into coherent text using text-only model call."""
    if not segments:
        return ""

    formatted = [f"[{start:.1f}s-{end:.1f}s]: {text}" for start, end, text in segments]
    prompt = f"""Below are narration segments from analyzing a gameplay video.
Merge them into ONE coherent, flowing description. Remove any redundancy.
Keep all important details. Write in present tense.

Segments:
{chr(10).join(formatted)}

Merged narration:"""

    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 4096, "temperature": 0.1},
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"Warning: Merge failed ({e}), using segmented narration")
        return ""


# =============================================================================
# BATCH ORCHESTRATION
# =============================================================================

def _calculate_batches(
    total_frames: int,
    frames_per_batch: int,
    overlap: int = 5,
) -> list[BatchBoundary]:
    """Calculate batch boundaries with overlap tracking."""
    batches = []

    # First batch has no overlap
    first_end = min(frames_per_batch, total_frames)
    batches.append(BatchBoundary(
        start_idx=0,
        end_idx=first_end,
        overlap_frames=0,
        narration_start_idx=0,
    ))

    if first_end >= total_frames:
        return batches

    # Subsequent batches include overlap
    step = frames_per_batch - overlap
    start = step

    while start < total_frames:
        end = min(start + frames_per_batch, total_frames)
        actual_overlap = min(overlap, end - start)

        batches.append(BatchBoundary(
            start_idx=start,
            end_idx=end,
            overlap_frames=actual_overlap,
            narration_start_idx=start + actual_overlap,
        ))

        if end >= total_frames:
            break
        start += step

    return batches


def _process_single_batch(
    frame_paths: list[str],
    diff_scores: list[float],
    boundary: BatchBoundary,
    batch_index: int,
    total_batches: int,
    config: QualityConfig,
    context_history: list[str],
    target_fps: int,
    diff_threshold: float,
    batch_dir: str,
    model: str,
    match_metadata: MatchMetadata,
) -> BatchResult:
    """Process a single batch with adaptive retry on failure."""
    # Clear batch directory
    for f in os.listdir(batch_dir):
        os.remove(os.path.join(batch_dir, f))

    start_time = boundary.start_idx / target_fps
    end_time = (boundary.end_idx - 1) / target_fps
    narration_start_time = boundary.narration_start_idx / target_fps

    frames_to_try = config.frames_per_batch
    max_attempts = 2

    for attempt in range(max_attempts):
        batch_frames = _select_batch_frames(
            frame_paths, diff_scores,
            boundary.start_idx, boundary.end_idx,
            frames_to_try, diff_threshold,
        )

        if attempt > 0:
            print(f"    Retry with {len(batch_frames)} frames...")

        prepared = _prepare_batch_frames(batch_frames, config.resolution, batch_dir)
        prompt = _build_batch_prompt(
            batch_index, total_batches,
            start_time, end_time, narration_start_time,
            boundary.overlap_frames, context_history,
            match_metadata,
        )

        success, response, error = _analyze_batch(prepared, prompt, model)

        if success:
            context, narration = _parse_response(response)
            return BatchResult(
                batch_index=batch_index,
                start_frame=boundary.start_idx,
                end_frame=boundary.end_idx,
                overlap_frames=boundary.overlap_frames,
                narration_start_frame=boundary.narration_start_idx,
                start_timestamp=start_time,
                end_timestamp=end_time,
                narration_start_timestamp=narration_start_time,
                context_summary=context,
                narration=narration,
                frame_count=boundary.end_idx - boundary.start_idx,
                frames_used=len(batch_frames),
                success=True,
            )

        frames_to_try -= config.fallback_frame_reduction
        for f in os.listdir(batch_dir):
            os.remove(os.path.join(batch_dir, f))

    return BatchResult(
        batch_index=batch_index,
        start_frame=boundary.start_idx,
        end_frame=boundary.end_idx,
        overlap_frames=boundary.overlap_frames,
        narration_start_frame=boundary.narration_start_idx,
        start_timestamp=start_time,
        end_timestamp=end_time,
        narration_start_timestamp=narration_start_time,
        context_summary="",
        narration="",
        frame_count=boundary.end_idx - boundary.start_idx,
        frames_used=0,
        success=False,
        error=error,
    )


# =============================================================================
# PUBLIC API
# =============================================================================

class GameplayInterpreter:
    """
    Reusable video analysis pipeline.

    Example usage:
        interpreter = GameplayInterpreter(quality="high")
        result = interpreter.analyze("gameplay.mp4")
        print(result.merged_narration)
        result.save("output.json")
    """

    def __init__(
        self,
        quality: str | QualityMode = QualityMode.HIGH,
        model: str = DEFAULT_MODEL,
        target_fps: int = 1,
        overlap_frames: int = 5,
        diff_threshold: float = 3.0,
        verbose: bool = True,
    ):
        """
        Initialize the interpreter.

        Args:
            quality: "high" or "fast" (or QualityMode enum)
            model: Ollama model name
            target_fps: Frames per second to extract
            overlap_frames: Number of overlap frames between batches
            diff_threshold: Threshold for frame difference selection
            verbose: Whether to print progress
        """
        if isinstance(quality, str):
            quality = QualityMode(quality.lower())

        self.quality_mode = quality
        self.config = QUALITY_CONFIGS[quality]
        self.model = model
        self.target_fps = target_fps
        self.overlap_frames = overlap_frames
        self.diff_threshold = diff_threshold
        self.verbose = verbose

    def _log(self, msg: str, end: str = "\n") -> None:
        if self.verbose:
            print(msg, end=end)

    def analyze(self, video_path: str, metadata: MatchMetadata) -> PipelineResult:
        """
        Analyze a video file.

        Args:
            video_path: Path to the video file
            metadata: Required match metadata (map_name, mode)

        Returns:
            PipelineResult containing narration and metadata

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If metadata is invalid
            subprocess.CalledProcessError: If ffmpeg/ffprobe fails
        """
        video_path = str(Path(video_path).resolve())
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        duration = _get_video_duration(video_path)
        state = _ProcessingState(
            video_path=video_path,
            match_metadata=metadata,
            quality_mode=self.quality_mode,
            start_time=datetime.now(),
        )

        self._log(f"\n{'=' * 60}")
        self._log("GAMEPLAY INTERPRETER - Call of Duty: Black Ops 6")
        self._log(f"{'=' * 60}")
        self._log(f"File: {video_path}")
        self._log(f"Map: {metadata.map_name} | Mode: {metadata.mode}")
        self._log(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")
        self._log(f"Quality: {self.quality_mode.value} ({self.config.description})")
        self._log(f"{'=' * 60}\n")

        with tempfile.TemporaryDirectory() as work_dir:
            raw_dir = os.path.join(work_dir, "raw")
            batch_dir = os.path.join(work_dir, "batch")
            os.makedirs(raw_dir)
            os.makedirs(batch_dir)

            # Step 1: Extract frames
            self._log(f"[1/4] Extracting frames at {self.target_fps} FPS...")
            frame_paths = _extract_frames(video_path, raw_dir, self.target_fps)
            state.total_frames = len(frame_paths)
            self._log(f"      Extracted {len(frame_paths)} frames")

            # Step 2: Compute differences
            self._log("[2/4] Computing frame differences...")
            diff_scores = _compute_frame_differences(frame_paths)

            # Step 3: Calculate and process batches
            batches = _calculate_batches(
                len(frame_paths),
                self.config.frames_per_batch,
                self.overlap_frames,
            )
            state.total_batches = len(batches)

            self._log(f"[3/4] Processing {len(batches)} batches...")
            self._log(f"      Max frames/batch: {self.config.frames_per_batch} at {self.config.resolution}px")
            self._log(f"      Overlap: {self.overlap_frames} frames\n")

            for idx, boundary in enumerate(batches):
                narr_time = boundary.narration_start_idx / self.target_fps
                end_time = (boundary.end_idx - 1) / self.target_fps
                overlap_note = f" ({boundary.overlap_frames} overlap)" if boundary.overlap_frames else ""

                self._log(f"  Batch {idx + 1}/{len(batches)} [frames {boundary.start_idx}-{boundary.end_idx - 1}]{overlap_note}")
                self._log(f"    Narrating: {narr_time:.1f}s - {end_time:.1f}s")

                result = _process_single_batch(
                    frame_paths, diff_scores, boundary,
                    idx, len(batches), self.config,
                    state.context_history, self.target_fps,
                    self.diff_threshold, batch_dir, self.model,
                    metadata,
                )

                if result.success:
                    if result.context_summary:
                        state.context_history.append(result.context_summary)
                        state.context_history = state.context_history[-MAX_CONTEXT_HISTORY:]
                    self._log(f"    ✓ Complete ({len(result.narration)} chars, {result.frames_used} frames)")
                else:
                    state.failed_batches.append(idx)
                    self._log(f"    ✗ Failed: {result.error}")

                state.batch_results.append(result)
                self._log("")

            # Step 4: Merge narration
            self._log("[4/4] Merging narration segments...")
            segments = [
                (r.narration_start_timestamp, r.end_timestamp, r.narration)
                for r in state.batch_results
                if r.success and r.narration
            ]
            merged = _merge_narration(segments, self.model)
            if merged:
                self._log("      ✓ Merge complete")
            else:
                self._log("      Using segmented narration")

        state.end_time = datetime.now()

        # Build segmented narration
        narration_parts = []
        for r in state.batch_results:
            marker = f"[{r.narration_start_timestamp:.1f}s - {r.end_timestamp:.1f}s]"
            if r.success and r.narration:
                narration_parts.append(f"{marker}\n{r.narration}")
            elif not r.success:
                narration_parts.append(f"{marker}\n[Processing failed: {r.error}]")

        segmented_narration = "\n\n".join(narration_parts)
        processing_time = (state.end_time - state.start_time).total_seconds()
        successful = sum(1 for r in state.batch_results if r.success)
        total_frames_used = sum(r.frames_used for r in state.batch_results if r.success)

        batch_details = [
            {
                "index": r.batch_index,
                "time_range": f"{r.start_timestamp:.1f}s - {r.end_timestamp:.1f}s",
                "narration_range": f"{r.narration_start_timestamp:.1f}s - {r.end_timestamp:.1f}s",
                "overlap_frames": r.overlap_frames,
                "frames_used": r.frames_used,
                "success": r.success,
                "context": r.context_summary if r.success else None,
                "error": r.error if not r.success else None,
            }
            for r in state.batch_results
        ]

        result = PipelineResult(
            video_path=video_path,
            match_metadata=metadata,
            duration_seconds=duration,
            quality_mode=self.quality_mode.value,
            resolution=self.config.resolution,
            frames_per_batch=self.config.frames_per_batch,
            total_frames_extracted=state.total_frames,
            total_frames_used=total_frames_used,
            total_batches=state.total_batches,
            successful_batches=successful,
            failed_batch_indices=state.failed_batches,
            narration=segmented_narration,
            merged_narration=merged,
            batch_details=batch_details,
            processing_time_seconds=processing_time,
            processed_at=datetime.now().isoformat(),
        )

        self._log(f"\n{'=' * 60}")
        self._log("ANALYSIS COMPLETE")
        self._log(f"{'=' * 60}")
        self._log(f"Batches: {successful}/{state.total_batches} successful")
        self._log(f"Frames used: {total_frames_used}")
        self._log(f"Processing time: {processing_time:.1f}s")

        if state.failed_batches:
            self._log(f"\nWARNING: Failed batches: {state.failed_batches}")

        return result


def analyze_video(
    video_path: str,
    map_name: str,
    mode: str,
    quality: str = "high",
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> PipelineResult:
    """
    Convenience function for one-shot video analysis.

    Args:
        video_path: Path to the video file
        map_name: Name of the map (e.g., "Nuketown", "Raid")
        mode: Game mode (e.g., "Hardpoint", "Search and Destroy")
        quality: "high" or "fast"
        model: Ollama model name
        verbose: Whether to print progress

    Returns:
        PipelineResult containing narration and metadata
    """
    metadata = MatchMetadata(map_name=map_name, mode=mode)
    interpreter = GameplayInterpreter(quality=quality, model=model, verbose=verbose)
    return interpreter.analyze(video_path, metadata)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """Command-line entry point."""
    import sys

    if len(sys.argv) < 4:
        print("Usage: python gameplay_interpreter.py <video_path> <map_name> <mode> [quality]")
        print("")
        print("Arguments:")
        print("  video_path  Path to the gameplay video file")
        print("  map_name    Map name (e.g., Nuketown, Raid, Terminal)")
        print("  mode        Game mode (e.g., Hardpoint, 'Search and Destroy', TDM)")
        print("  quality     Optional: 'high' (default) or 'fast'")
        print("")
        print("Example:")
        print("  python gameplay_interpreter.py match.mp4 Nuketown Hardpoint high")
        return 1

    video_path = sys.argv[1]
    map_name = sys.argv[2]
    mode = sys.argv[3]
    quality = sys.argv[4] if len(sys.argv) > 4 else "high"

    if quality not in ("high", "fast"):
        print(f"Invalid quality mode: {quality}. Use 'high' or 'fast'.")
        return 1

    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return 1

    try:
        result = analyze_video(
            video_path,
            map_name=map_name,
            mode=mode,
            quality=quality,
        )

        print(f"\n{'=' * 60}")
        print("VIDEO NARRATION")
        print(f"{'=' * 60}\n")
        print(result.merged_narration or result.narration)

        output_path = result.save()
        print(f"\n\nResults saved to: {output_path}")

        return 1 if result.failed_batch_indices else 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    # Usage: class based
    # interpreter = GameplayInterpreter(quality="high")
    # metadata = MatchMetadata(map_name="Nuketown", mode="Hardpoint")
    # result = interpreter.analyze("match.mp4", metadata)

    # Usage: one shot
    result = analyze_video(
        "sample_gameplay.MP4",
        map_name="Nuketown",
        mode="Domination",
        quality="high",
        verbose=True
    )
    print(result.merged_narration)