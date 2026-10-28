import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import app


class MusicGenApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app.MusicGenHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = Request(self.base_url + path, data=body, headers=headers)
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.headers.get_content_type(), response.read()

    def test_homepage_serves_simple_web_ui(self):
        status, content_type, body = self.request("/")

        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/html")
        self.assertIn(b"<title>Music maker</title>", body)
        self.assertNotIn(b"streamlit", body.lower())

    def test_status_reports_adapter_files_and_device(self):
        status, content_type, body = self.request("/api/status")
        data = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(content_type, "application/json")
        self.assertTrue(data["adapter_ready"])
        self.assertIn(data["device"], {"cpu", "cuda"})
        self.assertEqual(data["model_id"], "facebook/musicgen-small")

    def test_invalid_prompt_is_rejected_without_generation(self):
        with patch.object(app, "generate_music") as generate:
            status, _, body = self.request(
                "/api/generate",
                {
                    "prompt": "  ",
                    "genre": "Jazz",
                    "duration": 5,
                    "instrumental": True,
                },
            )

        self.assertEqual(status, 400)
        self.assertIn(b"prompt", body.lower())
        generate.assert_not_called()

    def test_generation_api_returns_audio_url(self):
        generated = {
            "path": str(app.OUTPUT_DIR / "jazz_test.wav"),
            "prompt": "Jazz music, mellow piano",
            "duration": 5.0,
            "sample_rate": 32000,
        }
        with (
            patch.object(app, "adapter_is_ready", return_value=True),
            patch.object(app, "generate_music", return_value=generated) as generate,
        ):
            status, _, body = self.request(
                "/api/generate",
                {
                    "prompt": "mellow piano",
                    "genre": "Jazz",
                    "duration": 5,
                    "instrumental": True,
                },
            )

        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["filename"], "jazz_test.wav")
        self.assertEqual(data["audio_url"], "/outputs/jazz_test.wav")
        generate.assert_called_once()

    def test_generation_is_blocked_when_adapter_is_missing(self):
        with (
            patch.object(app, "adapter_is_ready", return_value=False),
            patch.object(app, "generate_music") as generate,
        ):
            status, _, body = self.request(
                "/api/generate",
                {
                    "prompt": "mellow piano",
                    "genre": "Jazz",
                    "duration": 5,
                    "instrumental": True,
                },
            )

        self.assertEqual(status, 503)
        self.assertIn(b"adapter", body.lower())
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
