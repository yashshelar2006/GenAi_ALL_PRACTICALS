import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import torch
from dotenv import load_dotenv

from music_engine.generator import generate_music
from music_engine.model import DEFAULT_MODEL_ID, get_adapter_path, musicgen_is_loaded

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = BASE_DIR / OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = BASE_DIR / "static"
MODEL_ID = os.getenv("MUSICGEN_MODEL_ID", DEFAULT_MODEL_ID)
GENERATION_LOCK = threading.Lock()
MAX_REQUEST_BYTES = 16_384
GENRES = {
    "Jazz",
    "Classical",
    "Lo-fi",
    "Hip-hop",
    "Pop",
    "Rock",
    "Electronic",
    "Ambient",
    "Cinematic",
    "Blues",
    "Folk",
    "No specific genre",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("musicgen")


def adapter_is_ready() -> bool:
    adapter_path = get_adapter_path()
    if not adapter_path:
        return False

    adapter = Path(adapter_path)
    return (
        adapter.is_dir()
        and (adapter / "adapter_config.json").is_file()
        and (
            (adapter / "adapter_model.safetensors").is_file()
            or (adapter / "adapter_model.bin").is_file()
        )
    )


class MusicGenHandler(BaseHTTPRequestHandler):
    server_version = "MusicGenLocal/1.0"

    def log_message(self, format_string: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), format_string % args)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _read_json(self) -> dict:
        raw_length = self.headers.get("Content-Length", "")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("A valid Content-Length header is required.") from exc

        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request body must be between 1 byte and 16 KB.")

        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Request body must be valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            page = STATIC_DIR / "index.html"
            if not page.is_file():
                self._send_json({"error": "The local web page is missing."}, 500)
                return
            self._send_bytes(page.read_bytes(), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/status":
            self._send_json(
                {
                    "adapter_ready": adapter_is_ready(),
                    "model_id": MODEL_ID,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "model_loaded": musicgen_is_loaded(),
                }
            )
            return

        if parsed.path.startswith("/outputs/"):
            filename = unquote(parsed.path.removeprefix("/outputs/"))
            if Path(filename).name != filename or not filename.lower().endswith(".wav"):
                self._send_json({"error": "Audio file not found."}, 404)
                return
            audio_path = (OUTPUT_DIR / filename).resolve()
            if audio_path.parent != OUTPUT_DIR.resolve() or not audio_path.is_file():
                self._send_json({"error": "Audio file not found."}, 404)
                return
            self._send_bytes(audio_path.read_bytes(), "audio/wav")
            return

        self._send_json({"error": "Not found."}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/generate":
            self._send_json({"error": "Not found."}, 404)
            return

        try:
            payload = self._read_json()
            prompt = payload.get("prompt")
            genre = payload.get("genre")
            duration = payload.get("duration")
            instrumental = payload.get("instrumental")

            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 500:
                raise ValueError("Enter a prompt that is between 1 and 500 characters.")
            if not isinstance(genre, str) or genre not in GENRES:
                raise ValueError("Choose a genre from the list.")
            if isinstance(duration, bool) or not isinstance(duration, int) or not 3 <= duration <= 15:
                raise ValueError("Duration must be between 3 and 15 seconds.")
            if not isinstance(instrumental, bool):
                raise ValueError("Instrumental must be true or false.")
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
            return

        if not adapter_is_ready():
            self._send_json(
                {"error": "The LoRA adapter is missing or incomplete. Check LORA_ADAPTER_PATH and adapter files."},
                503,
            )
            return

        if not GENERATION_LOCK.acquire(blocking=False):
            self._send_json({"error": "A track is already being generated. Please wait for it to finish."}, 409)
            return

        try:
            result = generate_music(
                prompt=prompt.strip(),
                genre=genre,
                duration_seconds=duration,
                instrumental=instrumental,
                output_dir=OUTPUT_DIR,
            )
            audio_name = Path(result["path"]).name
            self._send_json(
                {
                    "filename": audio_name,
                    "audio_url": f"/outputs/{quote(audio_name)}",
                    "prompt": result["prompt"],
                    "duration": result["duration"],
                    "sample_rate": result["sample_rate"],
                }
            )
        except Exception as exc:
            logger.exception("Music generation failed")
            self._send_json({"error": f"Music generation failed: {exc}"}, 500)
        finally:
            GENERATION_LOCK.release()


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), MusicGenHandler)
    print(f"MusicGen is running at http://127.0.0.1:{port}")
    print(f"Adapter available: {'yes' if adapter_is_ready() else 'no'}")
    print(f"Compute device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MusicGen server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
