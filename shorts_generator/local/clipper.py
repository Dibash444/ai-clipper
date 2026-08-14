"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces with optimized sampling."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    last_center: Optional[Tuple[int, int]] = None
    target_center: Optional[Tuple[int, int]] = None
    smoothing = 0.15  # how aggressively to chase a new face position
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fast face detection: run every 5th frame on 0.5x downscaled grayscale (10x faster)
        if frame_idx % 5 == 0:
            small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5)
            faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.15, minNeighbors=4, minSize=(20, 20))
            if len(faces) > 0:
                # Scale back coordinates (x2) and pick largest face
                scaled_faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]
                x, y, w, h = max(scaled_faces, key=lambda f: f[2] * f[3])
                target_center = (x + w // 2, y + h // 2)

        if target_center is not None:
            if last_center is None:
                last_center = target_center
            else:
                lx, ly = last_center
                tx, ty = target_center
                last_center = (
                    int(lx + (tx - lx) * smoothing),
                    int(ly + (ty - ly) * smoothing),
                )
        if last_center is None:
            last_center = (src_w // 2, src_h // 2)

        cx, cy = last_center
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)
        frame_idx += 1

    cap.release()
    writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    
    import time
    for attempt in range(5):
        try:
            if os.path.exists(silent_path):
                os.remove(silent_path)
            break
        except OSError:
            time.sleep(0.5)
            
    return out_path


def _pad_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Pad the video to the target aspect ratio with a blurred background."""
    target_ratio = _ratio(aspect_ratio)
    target_h = 1920
    target_w = int(target_h * target_ratio)
    target_w = max(2, target_w - (target_w % 2))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-filter_complex",
        f"[0:v]scale=-1:{target_h},crop={target_w}:{target_h},boxblur=20:20[bg];[0:v]scale={target_w}:-1[fg];[bg][fg]overlay=0:(H-h)/2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    crop_mode: str = "face_track",
    smart_fetch: bool = False,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        if smart_fetch:
            from .downloader import download_youtube_local, download_youtube_section_local
            try:
                # First try fast section download
                download_youtube_section_local(source_path, start_time, end_time, cut_path)
            except Exception as sec_err:
                print(f"[clip/local] Section download failed ({sec_err}), falling back to full download subclip...", flush=True)
                full_source = download_youtube_local(source_path, fmt="720", audio_only=False)
                _cut_subclip(full_source, start_time, end_time, cut_path)
        else:
            _cut_subclip(source_path, start_time, end_time, cut_path)

        if crop_mode == "full":
            _pad_vertical(cut_path, out_path, aspect_ratio)
        else:
            _reframe_vertical(cut_path, out_path, aspect_ratio)
    finally:
        # On Windows, OpenCV may briefly hold file locks after release().
        # Retry deletion a few times with a short delay.
        import time
        for attempt in range(5):
            try:
                if os.path.exists(cut_path):
                    os.remove(cut_path)
                break
            except OSError:
                time.sleep(0.5)
    return out_path



def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    crop_mode: str = "face_track",
    smart_fetch: bool = False,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                crop_mode,
                smart_fetch,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
