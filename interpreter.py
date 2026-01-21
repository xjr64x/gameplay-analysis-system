import cv2
import subprocess
import tempfile
import os
from ollama import chat
from PIL import Image
from io import BytesIO
import numpy as np

def extract(video_path: str, fps: int = 1) -> list:
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
        for fname in sorted(os.listdir(tmpdir)):
            img = Image.open(os.path.join(tmpdir, fname)).convert("RGB")
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
    frames = extract(video_path)
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

def encoding_frames(frames: list) -> list[str]:
    encoded = []

    for img in frames:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        encoded.append(buffer.getvalue()) # original: base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded
