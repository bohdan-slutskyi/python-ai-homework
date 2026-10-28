import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class GeminiRequestTests(unittest.TestCase):
    @patch("app.requests.post")
    def test_missing_key_never_sends_a_request(self, post):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            app.generate_content("", "gemini-3.5-flash-lite", "Привет")
        post.assert_not_called()

    @patch("app.requests.post")
    def test_invalid_model_never_sends_a_request(self, post):
        with self.assertRaisesRegex(ValueError, "GEMINI_MODEL"):
            app.generate_content("test-key", "https://another-host.invalid", "Привет")
        post.assert_not_called()

    @patch("app.requests.post")
    def test_key_is_in_header_and_only_visible_text_is_returned(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {
            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"text": "Внутренний текст", "thought": True},
                {"text": "Токен — часть текста."},
            ]}}],
            "usageMetadata": {"promptTokenCount": 8},
        }
        result = app.generate_content("test-key", "gemini-3.5-flash-lite", "Что такое токен?")
        self.assertEqual(result["response"], "Токен — часть текста.")
        args, kwargs = post.call_args
        self.assertNotIn("test-key", args[0])
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(kwargs["timeout"], 30)
        self.assertNotIn("test-key", str(result))

    @patch("app.requests.post")
    def test_http_error_does_not_expose_body_or_key(self, post):
        post.return_value = Mock(status_code=403, text="sensitive-provider-details test-key")
        with self.assertRaises(RuntimeError) as error:
            app.generate_content("test-key", "gemini-3.5-flash-lite", "Привет")
        self.assertEqual(str(error.exception), "Gemini API вернул HTTP 403.")

    @patch("app.requests.post")
    def test_empty_or_incomplete_generation_is_not_a_success(self, post):
        post.return_value = Mock(status_code=200)
        for payload in (
            {},
            {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "Начало"}]}}]},
            {"candidates": [{"finishReason": "STOP", "content": {"parts": []}}]},
        ):
            with self.subTest(payload=payload):
                post.return_value.json.return_value = payload
                with self.assertRaises(RuntimeError):
                    app.generate_content("test-key", "gemini-3.5-flash-lite", "Привет")

    @patch("app.load_dotenv")
    @patch("app.requests.post")
    def test_main_without_key_reports_error_without_creating_results(self, post, load):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["app.py"]):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(app.main(), 1)
        self.assertIn("GEMINI_API_KEY", stderr.getvalue())
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
