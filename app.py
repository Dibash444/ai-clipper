"""Flask web frontend for AI YouTube Shorts Generator (local mode)."""
import os
import sys
import uuid
import threading
import traceback
from pathlib import Path

# Fix Windows encoding before anything else
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import static_ffmpeg  # type: ignore
    static_ffmpeg.add_paths()
except Exception:
    pass

from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Store for background jobs: job_id -> {status, result, error, ...}
jobs: dict = {}

# Output directory (matches .env LOCAL_OUTPUT_DIR)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Start a shorts generation job in the background."""
    data = request.get_json(force=True)
    youtube_url = data.get("url", "").strip()
    num_clips = int(data.get("num_clips", 3))
    crop_mode = data.get("crop_mode", "face_track")
    smart_fetch = bool(data.get("smart_fetch", False))

    if not youtube_url:
        return jsonify({"error": "Please provide a YouTube URL"}), 400

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "starting",
        "message": "Initialising...",
        "result": None,
        "error": None,
    }

    def run_job():
        try:
            jobs[job_id]["status"] = "running"
            jobs[job_id]["message"] = "Downloading video with yt-dlp..."

            # Import here so .env is loaded by the shorts_generator config module
            from shorts_generator import generate_shorts

            # Monkey-patch print to capture progress messages
            original_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print

            def patched_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                # Update job status based on pipeline output
                if "[download" in msg.lower() or "yt-dlp" in msg.lower():
                    jobs[job_id]["message"] = "Downloading video..."
                elif "transcribe/youtube" in msg.lower() or "caption" in msg.lower():
                    jobs[job_id]["message"] = "Fetching YouTube captions..."
                elif "whisper" in msg.lower() or "transcrib" in msg.lower():
                    jobs[job_id]["message"] = "Transcribing audio..."
                elif "highlight" in msg.lower() or "llm" in msg.lower() or "gemini" in msg.lower():
                    jobs[job_id]["message"] = "Finding viral highlights with AI..."
                elif "clip" in msg.lower() or "crop" in msg.lower():
                    jobs[job_id]["message"] = f"Clipping & cropping: {msg}"
                elif "pipeline" in msg.lower():
                    jobs[job_id]["message"] = msg
                original_print(*args, **kwargs)

            import builtins
            builtins.print = patched_print

            try:
                result = generate_shorts(
                    youtube_url=youtube_url,
                    num_clips=num_clips,
                    aspect_ratio="9:16",
                    mode="local",
                    crop_mode=crop_mode,
                    smart_fetch=smart_fetch,
                )
            finally:
                builtins.print = original_print

            # Prepare clip data for the frontend
            shorts = result.get("shorts", [])
            clips = []
            for s in shorts:
                clip_path = s.get("clip_url")
                filename = os.path.basename(clip_path) if clip_path else None
                clips.append({
                    "title": s.get("title", "Untitled"),
                    "score": s.get("score", 0),
                    "hook_sentence": s.get("hook_sentence", ""),
                    "virality_reason": s.get("virality_reason", ""),
                    "filename": filename,
                    "error": s.get("error"),
                })

            jobs[job_id]["status"] = "done"
            jobs[job_id]["message"] = f"Done! Generated {len(clips)} clip(s)."
            jobs[job_id]["result"] = clips

        except Exception as e:
            traceback.print_exc()
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = f"Error: {str(e)}"
            jobs[job_id]["error"] = str(e)

    t = threading.Thread(target=run_job, daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    """Poll the status of a generation job."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/output/<path:filename>")
def serve_output(filename):
    """Serve generated video files from the output directory."""
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    port = int(os.environ.get("PORT", 7860))
    host = "0.0.0.0"
    print("\n" + "=" * 55)
    print("  AI YouTube Shorts Generator")
    print(f"  Running on: http://localhost:{port}")
    print("=" * 55 + "\n")
    app.run(host=host, port=port, debug=False)
