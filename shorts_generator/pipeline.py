"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + OpenAI or Gemini + ffmpeg/opencv.
                              Self-hosted, LLM_PROVIDER selects OpenAI or Gemini.
"""
from typing import Dict, List, Optional

from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    crop_mode: str = "face_track",
    smart_fetch: bool = False,
) -> Dict:
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm
    from .local.transcriber import fetch_youtube_captions, transcribe_local

    # Step 1: Fast transcript resolution (YouTube captions / cache first in ~1s)
    transcript = None
    is_youtube = ("youtube.com" in youtube_url or "youtu.be" in youtube_url)
    if is_youtube:
        transcript = fetch_youtube_captions(youtube_url, language=language)

    # If YouTube captions not available, download audio/video and transcribe locally
    source_path = None
    if not transcript or not transcript.get("segments"):
        source_path = download_youtube_local(youtube_url, fmt=download_format, audio_only=smart_fetch)
        transcript = transcribe_local(source_path, language=language, source_url=youtube_url)

    if not transcript or not transcript.get("segments"):
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    # Step 2: Identify highlights with LLM
    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_local_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    # Step 3: Ensure source video is downloaded locally
    if not source_path or not os.path.exists(source_path):
        source_path = download_youtube_local(youtube_url, fmt=download_format, audio_only=False)

    shorts = crop_highlights_local(
        source_path, 
        top, 
        aspect_ratio=aspect_ratio, 
        crop_mode=crop_mode, 
        smart_fetch=False,
    )

    return {
        "mode": "local",
        "source_video_url": youtube_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_muapi_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(source_url, top, aspect_ratio=aspect_ratio)

    return {
        "mode": "api",
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
    crop_mode: str = "face_track",
    smart_fetch: bool = False,
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        mode: "api" (default, MuAPI) or "local" (yt-dlp + faster-whisper +
            OpenAI or Gemini + ffmpeg).

    Returns:
        {
          "mode": "api" | "local",
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with clip_url / local path
        }
    """
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(youtube_url, num_clips, aspect_ratio, download_format, language, crop_mode, smart_fetch)
    if mode == "api":
        return _run_api(youtube_url, num_clips, aspect_ratio, download_format, language)
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")
