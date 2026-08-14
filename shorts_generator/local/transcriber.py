"""Local transcription via YouTube captions or faster-whisper.

Reads a local media file or YouTube URL and returns the same shape the highlight
generator expects: {duration, segments[start, end, text]}.
"""
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from ..config import (
    FETCH_YOUTUBE_SUBTITLES_FIRST,
    LOCAL_OUTPUT_DIR,
    LOCAL_WHISPER_BEAM_SIZE,
    LOCAL_WHISPER_CPU_THREADS,
    LOCAL_WHISPER_DEVICE,
    LOCAL_WHISPER_MODEL,
    LOCAL_WHISPER_VAD_FILTER,
    LOCAL_WHISPER_VAD_PARAMETERS,
)


def _transcript_cache_path(media_path: str) -> Path:
    """Return the .srt cache path for a media file."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".srt")


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + (millis / 1000.0)


def _write_srt_cache(media_path: str, transcript: Dict) -> Path:
    cache_path = _transcript_cache_path(media_path)
    lines = []
    for idx, segment in enumerate(transcript.get("segments", []), start=1):
        start = _format_srt_timestamp(float(segment["start"]))
        end = _format_srt_timestamp(float(segment["end"]))
        text = str(segment.get("text", "")).strip().replace("\r", "").replace("\n", " ")
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    cache_path.write_text("\n".join(lines), encoding="utf-8")
    return cache_path


def _load_srt_cache(cache_path: Path) -> Dict:
    content = cache_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {"duration": 0.0, "segments": []}

    segments = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[0].split("-->", 1)]
        text = "\n".join(lines[1:]).strip()
        segments.append(
            {
                "start": _parse_srt_timestamp(start_raw),
                "end": _parse_srt_timestamp(end_raw),
                "text": text,
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Test that CUDA actually works (catches missing cuBLAS/cuDNN libs)
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def fetch_youtube_captions(video_url: str, language: Optional[str] = None) -> Optional[Dict]:
    """Instantly fetch manual or auto-generated captions directly from YouTube.

    Returns {duration, segments: [{start, end, text}]} in 1-2s if available,
    or None if no captions exist on YouTube.
    """
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        return None

    try:
        print(f"[transcribe/youtube] checking native captions for {video_url}...", flush=True)
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if not info:
                return None
            duration = float(info.get("duration", 0.0))
            subs = info.get("subtitles", {}) or {}
            auto = info.get("automatic_captions", {}) or {}

            target_list = None
            langs_to_try = [language] if language else []
            langs_to_try.extend(["en", "en-US", "en-orig", "en-GB", "en-CA"])

            # 1. Prefer manual subtitles in requested/English language
            for l in langs_to_try:
                if l and l in subs and subs[l]:
                    target_list = subs[l]
                    break

            # 2. Fall back to auto captions
            if not target_list:
                for l in langs_to_try:
                    if l and l in auto and auto[l]:
                        target_list = auto[l]
                        break

            # 3. Fall back to any auto caption starting with 'en' or first available
            if not target_list and auto:
                for k in auto:
                    if k.startswith("en"):
                        target_list = auto[k]
                        break
            if not target_list and subs:
                target_list = list(subs.values())[0]
            if not target_list and auto:
                target_list = list(auto.values())[0]

            if not target_list:
                return None

            json3_url = next((fmt["url"] for fmt in target_list if fmt.get("ext") == "json3"), None)
            if not json3_url:
                return None

            req = urllib.request.Request(json3_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                events = data.get("events", [])
                raw_segments = []
                for ev in events:
                    segs = ev.get("segs", [])
                    if not segs:
                        continue
                    text = "".join(s.get("utf8", "") for s in segs).strip()
                    if not text or text == "\n":
                        continue
                    start = ev.get("tStartMs", 0) / 1000.0
                    dur = ev.get("dDurationMs", 0) / 1000.0
                    end = start + dur
                    raw_segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})

                if not raw_segments:
                    return None

                # Merge small adjacent sentence fragments into smooth subtitle segments
                merged = []
                current = None
                for seg in raw_segments:
                    if not current:
                        current = dict(seg)
                        continue
                    gap = seg["start"] - current["end"]
                    cur_dur = current["end"] - current["start"]
                    # Merge if close in time and current sentence is short
                    if gap < 0.8 and cur_dur < 6.0 and len(current["text"]) < 120:
                        current["end"] = seg["end"]
                        current["text"] += " " + seg["text"]
                    else:
                        merged.append(current)
                        current = dict(seg)
                if current:
                    merged.append(current)

                total_dur = duration or (merged[-1]["end"] if merged else 0.0)
                print(
                    f"[transcribe/youtube] successfully fetched {len(merged)} caption segments in ~1s!",
                    flush=True,
                )
                return {"duration": total_dur, "segments": merged}

    except Exception as e:
        print(f"[transcribe/youtube] native caption fetch skipped ({e}), falling back to Whisper", flush=True)
        return None


def transcribe_local(
    media_path: str,
    language: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Dict:
    """Transcribe media file, using YouTube captions if available or optimized faster-whisper."""
    cache_path = _transcript_cache_path(media_path)
    if cache_path.exists():
        # If media_path exists as a real file, compare mtime. Otherwise if it's cached, use it.
        source_mtime = os.path.getmtime(media_path) if os.path.exists(media_path) else 0
        cache_mtime = cache_path.stat().st_mtime
        if cache_mtime >= source_mtime:
            print(f"[transcribe/local] reusing cached transcript: {cache_path}", flush=True)
            cached = _load_srt_cache(cache_path)
            if not cached["segments"] or cached["duration"] <= 0.0:
                print(f"[transcribe/local] cache is empty/invalid, deleting: {cache_path}", flush=True)
                cache_path.unlink(missing_ok=True)
            else:
                print(
                    f"[transcribe/local] {len(cached['segments'])} cached segments, "
                    f"{cached['duration']:.0f}s of audio",
                    flush=True,
                )
                return cached

    # Check if we can fetch YouTube captions directly (takes ~1-2s instead of minutes)
    url_to_try = source_url if (source_url and "http" in source_url) else None
    if not url_to_try and ("youtube.com" in media_path or "youtu.be" in media_path):
        url_to_try = media_path

    if FETCH_YOUTUBE_SUBTITLES_FIRST and url_to_try:
        yt_transcript = fetch_youtube_captions(url_to_try, language=language)
        if yt_transcript and yt_transcript.get("segments"):
            _write_srt_cache(media_path, yt_transcript)
            return yt_transcript

    # Fallback to local faster-whisper
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    threads = LOCAL_WHISPER_CPU_THREADS if device == "cpu" else None
    print(
        f"[transcribe/local] faster-whisper model={LOCAL_WHISPER_MODEL} device={device} "
        f"beam_size={LOCAL_WHISPER_BEAM_SIZE} threads={threads or 'default'}",
        flush=True,
    )

    model_kwargs = {
        "model_size_or_path": LOCAL_WHISPER_MODEL,
        "device": device,
        "compute_type": compute_type,
    }
    if device == "cpu" and threads:
        model_kwargs["cpu_threads"] = threads

    model = WhisperModel(**model_kwargs)

    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": LOCAL_WHISPER_BEAM_SIZE,
        "condition_on_previous_text": False,
    }
    if LOCAL_WHISPER_VAD_FILTER:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments = []
    for s in segments_iter:
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
        })

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1]["end"] if segments else 0.0)
    print(f"[transcribe/local] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    transcript = {"duration": duration, "segments": segments}
    cache_path = _write_srt_cache(media_path, transcript)
    print(f"[transcribe/local] wrote cache: {cache_path}", flush=True)
    return transcript
