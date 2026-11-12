import unittest
from unittest import mock

from openai_images.client import ApiError, ImagesClient, load_dotenv
from tests.mock_server import MockImagesServer

ARGS = dict(model="gpt-image-2.5-flare", size="1088x1664", quality="low")


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.server = MockImagesServer().__enter__()
        self.addCleanup(self.server.__exit__)
        self.client = ImagesClient("test-key", base_url=self.server.base_url, timeout=0.5)

    def test_success_returns_images_usage_and_headers(self):
        resp = self.client.generate(prompt="a knight", n=2, **ARGS)
        self.assertEqual(len(resp.images), 2)
        self.assertTrue(resp.images[0].startswith(b"\x89PNG"))
        self.assertEqual(resp.usage["output_tokens"], 153 * 2)
        self.assertEqual(resp.processing_ms, 1234)
        self.assertTrue(resp.request_id.startswith("req_"))
        body = self.server.requests[0]["body"]
        self.assertEqual((body["model"], body["size"], body["quality"], body["n"]), ("gpt-image-2.5-flare", "1088x1664", "low", 2))
        self.assertNotIn("background", body)   # parâmetros None não vão no corpo

    def test_optional_params_are_sent(self):
        self.client.generate(prompt="x", background="opaque", **ARGS)
        self.assertEqual(self.server.requests[0]["body"]["background"], "opaque")

    def test_retries_429_then_succeeds(self):
        with mock.patch("time.sleep") as sleep:
            resp = self.client.generate(prompt="FAIL429", **ARGS)
        self.assertEqual(len(resp.images), 1)
        self.assertEqual(len(self.server.requests), 2)
        sleep.assert_called_once()

    def test_no_retry_on_insufficient_quota(self):
        with self.assertRaises(ApiError) as ctx:
            self.client.generate(prompt="QUOTA", **ARGS)
        self.assertEqual((ctx.exception.status, ctx.exception.code), (429, "insufficient_quota"))
        self.assertEqual(len(self.server.requests), 1)

    def test_400_carries_the_api_message_and_is_not_retried(self):
        with self.assertRaises(ApiError) as ctx:
            self.client.generate(prompt="FAIL400", **ARGS)
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("quality", str(ctx.exception))
        self.assertEqual(len(self.server.requests), 1)

    def test_wrong_key_is_401(self):
        bad = ImagesClient("wrong", base_url=self.server.base_url)
        with self.assertRaises(ApiError) as ctx:
            bad.generate(prompt="x", **ARGS)
        self.assertEqual(ctx.exception.status, 401)

    def test_read_timeout_is_not_retried(self):
        with self.assertRaises(ApiError) as ctx:
            self.client.generate(prompt="SLOW", **ARGS)
        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertIn("cobrada", str(ctx.exception))
        import time; time.sleep(1.2)           # deixa o handler lento terminar
        self.assertEqual(len(self.server.requests), 1)

    def test_repr_never_shows_the_key(self):
        self.assertNotIn("test-key", repr(self.client))


class DotenvTests(unittest.TestCase):
    def test_loads_without_overriding(self):
        import os, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("# comentário\nFOO_TEST_A=1\nexport FOO_TEST_B='dois'\nFOO_TEST_C = tres \n\nlinha sem igual\n")
            with mock.patch.dict(os.environ, {"FOO_TEST_C": "ja_existe"}, clear=False):
                load_dotenv(env)
                self.assertEqual((os.environ["FOO_TEST_A"], os.environ["FOO_TEST_B"], os.environ["FOO_TEST_C"]),
                                 ("1", "dois", "ja_existe"))
            for k in ("FOO_TEST_A", "FOO_TEST_B", "FOO_TEST_C"):
                os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()
