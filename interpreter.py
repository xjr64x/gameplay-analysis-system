import cv2
import subprocess
import tempfile
import os
from ollama import chat
from PIL import Image
import numpy as np


def extract_frames(video_path: str, fps: int = 1) -> list:
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
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def select_frames(frames: list, max_frames: int) -> list:
    if not frames:
        return []

    selected = [frames[0]]
    prev_gray = pil_to_gray(frames[0])

    for img in frames[1:]:
        gray = pil_to_gray(img)
        diff = cv2.absdiff(prev_gray, gray)
        score = diff.mean()

        if score > 5.0:
            selected.append(img)
            prev_gray = gray

    if len(selected) > max_frames:
        step = len(selected) / max_frames
        selected = [selected[int(i * step)] for i in range(max_frames)]

    return selected


def sampling_video(video_path: str, max_frames: int = 180) -> list:
    frames = extract_frames(video_path)
    frames = select_frames(frames, max_frames)
    return frames


def preprocessing_frames(frames: list, max_size: int = 768) -> list:
    processed = []

    for img in frames:
        w, h = img.size
        scale = min(max_size / w, max_size / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
        processed.append(img)

    return processed


def build_prompt() -> str:
    return (
        "You are analyzing a sequence of images extracted from a video. "
        "These images represent moments over time, not separate scenes. "
        "Describe what is happening overall in the video. "
        "Focus on actions, changes, and events. "
        "Do not describe each image individually. "
        "Respond in plain text only."
    )


def analyze_with_ollama(frames: list, prompt: str, model: str) -> str:
    """
    Save frames as temp files and pass file paths to Ollama.
    The Ollama Python client accepts file paths for images.
    """
    # Create a persistent temp directory for this call
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as JPEG files
        image_paths = []
        for i, img in enumerate(frames):
            path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
            img.save(path, format="JPEG", quality=85)
            image_paths.append(path)

        print(f"Sending {len(image_paths)} frames to model...")

        response = chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": image_paths,  # Pass file paths directly
            }]
        )

        # Access response as attribute (not dict)
        return response.message.content


def interpret_video(video_path: str, max_frames: int = 30):
    """
    Main function to interpret a video.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to send (default 30 for testing)
    """
    print(f"Extracting frames from {video_path}...")
    frames = sampling_video(video_path, max_frames=max_frames)
    print(f"Selected {len(frames)} key frames")

    print("Preprocessing frames...")
    frames = preprocessing_frames(frames)

    prompt = build_prompt()
    model = "qwen3-vl:8b-instruct-q4_K_M"

    print(f"Analyzing with {model}...")
    return analyze_with_ollama(frames, prompt, model)


if __name__ == "__main__":
    video_path = "sample_gameplay.MP4"

    # Start with fewer frames for testing
    results = interpret_video(video_path, max_frames=20)
    print("\n--- Results ---")
    print(results)