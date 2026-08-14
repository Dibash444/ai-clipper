"""Local YouTube download via yt-dlp.

Returns a local mp4 path so the rest of the local pipeline can read it
directly off disk.
"""
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from typing import Optional

from ..config import LOCAL_OUTPUT_DIR


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    return yt_dlp


def _format_for(fmt: str, audio_only: bool = False) -> str:
    """Map our '720' / '1080' shorthand to a yt-dlp format selector."""
    if audio_only:
        return "bestaudio[ext=m4a]/bestaudio"
    try:
        height = int(fmt)
    except ValueError:
        height = 720
    return (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={height}][ext=mp4]/best"
    )


def _extract_youtube_video_id(source: str) -> Optional[str]:
    """Best-effort extraction of a YouTube video id from a URL."""
    parsed = urlparse(source)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/", 1)[0]
        return video_id or None

    if "youtube.com" in host:
        if parsed.path.startswith("/watch"):
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [""])[0]
            return video_id or None
        match = re.search(r"/(?:shorts|embed|live)/([^/?#&]+)", parsed.path)
        if match:
            return match.group(1)

    return None


def _resolve_local_path(source: str) -> Optional[str]:
    """Return a local filesystem path if the input already points at one."""
    parsed = urlparse(source)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            raw_path = f"//{parsed.netloc}{raw_path}"
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
        raise RuntimeError(f"Local file URL does not exist: {source}")

    if parsed.scheme in ("http", "https"):
        return None

    candidate = Path(source).expanduser()
    if candidate.exists() and candidate.is_file():
        return str(candidate.resolve())

    if any(sep in source for sep in (os.sep, "/")) or source.startswith("~") or source.startswith("."):
        raise RuntimeError(f"Local file path does not exist: {source}")

    return None


def _existing_download(out_dir: str, video_id: str) -> Optional[str]:
    """Return a cached download path if we already have this YouTube id."""
    for ext in (".mp4", ".mkv", ".webm"):
        candidate = os.path.join(out_dir, f"source_{video_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def _get_cookie_file() -> Optional[str]:
    """Return path to cookies file if present in filesystem or env var."""
    if os.path.exists("cookies.txt") and os.path.getsize("cookies.txt") > 0:
        return "cookies.txt"
    cookie_env = os.getenv("YOUTUBE_COOKIES_TEXT", "") or os.getenv("YOUTUBE_COOKIES", "")
    if cookie_env.strip():
        c_path = os.path.join(LOCAL_OUTPUT_DIR, "yt_cookies.txt")
        os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
        with open(c_path, "w", encoding="utf-8") as f:
            f.write(cookie_env.strip())
        return c_path
    return None


def download_youtube_local(video_url: str, fmt: str = "720", out_dir: Optional[str] = None, audio_only: bool = False) -> str:
    """Download a remote URL or return a local file path unchanged."""
    local_path = _resolve_local_path(video_url)
    if local_path:
        print(f"[download/local] using local file: {local_path}", flush=True)
        return local_path

    yt_dlp = _import_ytdlp()
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    video_id = _extract_youtube_video_id(video_url)
    if video_id:
        cached = _existing_download(out_dir, video_id)
        if cached:
            print(f"[download/local] reusing cached download: {cached}", flush=True)
            return cached

    ext_suffix = ".m4a" if audio_only else ".%(ext)s"
    print(f"[download/local] {video_url} @ {'audio' if audio_only else fmt+'p'} -> {out_dir}/", flush=True)
    
    ydl_opts = {
        "format": _format_for(fmt, audio_only=audio_only),
        "outtmpl": os.path.join(out_dir, f"source_%(id)s{ext_suffix}"),
        "merge_output_format": None if audio_only else "mp4",
        "extractor_args": {"youtube": {"player_client": ["android_vr", "android", "ios", "mweb", "web"]}},
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    cookie_file = _get_cookie_file()
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        path = ydl.prepare_filename(info)
        # merge_output_format may rename the extension after merge
        if not audio_only and not os.path.exists(path):
            stem, _ = os.path.splitext(path)
            for ext in (".mp4", ".mkv", ".webm"):
                if os.path.exists(stem + ext):
                    path = stem + ext
                    break

    print(f"[download/local] ready: {path}", flush=True)
    return path


def download_youtube_section_local(video_url: str, start: float, end: float, out_path: str, fmt: str = "720") -> str:
    """Download a specific section of a YouTube video without crashing ffmpeg."""
    yt_dlp = _import_ytdlp()
    
    start_sec = max(0, int(start))
    end_sec = int(end) + 1  # Add a 1s buffer to ensure complete clip

    print(f"[download/local/section] {video_url} ({start_sec}s - {end_sec}s)", flush=True)
    ydl_opts = {
        "format": _format_for(fmt),
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)]),
        "force_keyframes_at_cuts": False,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return out_path
