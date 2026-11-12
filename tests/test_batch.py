import base64
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from openai_images import ROOT, batch, report
from openai_images.pricing import batch_cost_from_usage, cost_from_usage, estimate, prompt_tokens_estimate
from openai_images.prompt import build_prompt, load_prompt
from tests.mock_server import MockImagesServer, make_png
from tests.test_compare import run_cli

MODEL, QUALITY = "gpt-image-2.5-sunburst", "xhigh"


class BatchEndToEnd(unittest.TestCase):
    def setUp(self):
        self.server = MockImagesServer().__enter__()
        self.addCleanup(self.server.__exit__)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": self.server.base_url}
        self.out = Path(self.tmp.name) / "batches"

    def submit(self, *extra, yes=True, env=None):
        argv = ["batch", "submit", "--model", MODEL, "--quality", QUALITY, "--out-dir", str(self.out), *extra]
        return run_cli(*argv, *(["--yes"] if yes else []), env=self.env if env is None else env)

    def submitted_dir(self, *extra) -> Path:
        code, out, err = self.submit(*extra)
        self.assertEqual(code, 0, out + err)
        return max(self.out.iterdir())

    # ------------------------------------------------------------------ submit

    def test_dry_run_needs_no_key_sends_nothing_and_shows_half_price(self):
        code, out, _ = self.submit("--n", "2", "--dry-run", yes=False, env={"OPENAI_API_KEY": ""})
        self.assertEqual(code, 0)
        prompt = build_prompt(load_prompt(ROOT / "prompt.txt"))
        est = estimate(MODEL, QUALITY, "1088x1664", prompt_tokens_estimate(prompt))
        self.assertIn(f"Custo estimado no Batch: {report.usd(est.batch_usd * 2)}", out)
        self.assertIn(f"no modo normal: {report.usd(est.total_usd * 2)}", out)
        self.assertAlmostEqual(est.batch_usd, est.total_usd / 2)
        first = json.loads(next(l for l in out.splitlines() if l.startswith("{")))
        self.assertEqual((first["method"], first["url"]), ("POST", "/v1/images/generations"))
        self.assertEqual((first["body"]["model"], first["body"]["quality"], first["body"]["n"]), (MODEL, QUALITY, 1))
        self.assertEqual(first["body"]["prompt"], prompt)
        self.assertEqual(self.server.requests, [])
        self.assertFalse(self.out.exists())

    def test_model_without_batch_price_is_rejected(self):
        code, _, err = run_cli("batch", "submit", "--model", "gpt-image-2.6-flare", "--quality", "high", "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("Batch", err)

    def test_requires_confirmation_when_not_interactive(self):
        code, _, err = self.submit(yes=False)
        self.assertEqual(code, 1)
        self.assertIn("--yes", err)
        self.assertEqual(self.server.requests, [])
        self.assertFalse(self.out.exists())

    def test_max_cost_blocks_before_any_call(self):
        code, _, err = self.submit("--n", "20")   # 20 x ~US$ 0,039 > US$ 0,50
        self.assertEqual(code, 2)
        self.assertIn("--max-cost", err)
        self.assertEqual(self.server.requests, [])
        code, _, _ = self.submit("--n", "20", "--max-cost", "5")
        self.assertEqual(code, 0)

    def test_n_out_of_range(self):
        for bad in ("0", str(batch.MAX_REQUESTS + 1)):
            code, _, err = self.submit("--n", bad, "--dry-run")
            self.assertEqual(code, 2, bad)
            self.assertIn("--n", err)

    def test_submit_uploads_the_file_and_creates_the_batch(self):
        run_dir = self.submitted_dir("--n", "3")
        upload, create = self.server.requests
        self.assertEqual((upload["path"], create["path"]), ("/v1/files", "/v1/batches"))
        self.assertEqual((upload["body"]["purpose"], upload["body"]["filename"]), ("batch", "requests.jsonl"))
        lines = [json.loads(l) for l in self.server.files[upload["body"]["file_id"]].decode().splitlines()]
        self.assertEqual(len(lines), 3)
        self.assertEqual(len({l["custom_id"] for l in lines}), 3)   # ids únicos
        self.assertEqual({l["body"]["n"] for l in lines}, {1})
        self.assertEqual({(l["method"], l["url"]) for l in lines}, {("POST", "/v1/images/generations")})
        self.assertEqual(lines[0]["body"], {"model": MODEL, "prompt": lines[0]["body"]["prompt"], "size": "1088x1664",
                                            "quality": QUALITY, "n": 1, "output_format": "png"})
        self.assertEqual(create["body"], {"input_file_id": upload["body"]["file_id"],
                                          "endpoint": "/v1/images/generations", "completion_window": "24h"})
        self.assertEqual(create["auth"], "Bearer test-key")
        state = json.loads((run_dir / batch.STATE_FILE).read_text())
        self.assertEqual((state["batch_id"], state["custom_ids"]), ("batch_1", [l["custom_id"] for l in lines]))
        self.assertEqual((run_dir / batch.REQUESTS_FILE).read_bytes(), self.server.files[upload["body"]["file_id"]])

    def test_background_is_sent_only_when_given(self):
        self.submitted_dir("--background", "opaque")
        upload = self.server.requests[0]["body"]
        line = json.loads(self.server.files[upload["file_id"]].decode())
        self.assertEqual(line["body"]["background"], "opaque")

    def test_two_submits_in_the_same_second_do_not_overwrite_each_other(self):
        fixed = datetime(2026, 1, 1, 12, 0, 0)
        with mock.patch("openai_images.batch.datetime") as fake:
            fake.now.return_value = fixed
            for _ in range(2):
                self.assertEqual(self.submit()[0], 0)
        states = [json.loads((d / batch.STATE_FILE).read_text()) for d in sorted(self.out.iterdir())]
        self.assertEqual([s["batch_id"] for s in states], ["batch_1", "batch_2"])

    def test_bad_key_is_reported_without_leaking_it(self):
        code, _, err = self.submit(env={**self.env, "OPENAI_API_KEY": "wrong-key"})
        self.assertEqual(code, 1)
        self.assertIn("401", err)
        self.assertNotIn("wrong-key", err)

    # ------------------------------------------------------------------ status

    def test_status_while_running_and_when_done(self):
        run_dir = self.submitted_dir()
        self.server.batch_status = "in_progress"
        code, out, _ = run_cli("batch", "status", str(run_dir), env=self.env)
        self.assertEqual(code, 0)
        self.assertIn("in_progress", out)
        self.assertIn("Ainda em andamento", out)
        self.server.batch_status = "completed"
        code, out, _ = run_cli("batch", "status", str(run_dir), env=self.env)
        self.assertEqual(code, 0)
        self.assertIn("batch fetch", out)
        self.assertIn("1 concluído(s)", out)

    def test_status_failed_batch_exits_with_error(self):
        run_dir = self.submitted_dir()
        self.server.batch_status = "failed"
        code, out, _ = run_cli("batch", "status", str(run_dir), env=self.env)
        self.assertEqual(code, 1)
        self.assertIn("nada foi gerado nem cobrado", out)

    def test_status_and_fetch_need_the_state_file(self):
        empty = Path(self.tmp.name) / "vazia"
        empty.mkdir()
        for command in ("status", "fetch"):
            code, _, err = run_cli("batch", command, str(empty), env=self.env)
            self.assertEqual(code, 2, command)
            self.assertIn(batch.STATE_FILE, err)

    # ------------------------------------------------------------------ fetch

    def test_fetch_refuses_while_the_batch_is_running(self):
        run_dir = self.submitted_dir()
        self.server.batch_status = "in_progress"
        code, _, err = run_cli("batch", "fetch", str(run_dir), env=self.env)
        self.assertEqual(code, 1)
        self.assertIn("ainda está 'in_progress'", err)
        self.assertEqual(list(run_dir.glob("*.png")), [])

    def test_fetch_saves_images_and_charges_half(self):
        run_dir = self.submitted_dir("--n", "2")
        self.server.batch_status = "completed"
        code, out, err = run_cli("batch", "fetch", str(run_dir), env=self.env)
        self.assertEqual(code, 0, out + err)
        pngs = sorted(run_dir.glob("*.png"))
        self.assertEqual(len(pngs), 2)
        self.assertTrue(all(p.read_bytes().startswith(b"\x89PNG") for p in pngs))
        data = json.loads((run_dir / batch.RESULTS_FILE).read_text())
        self.assertEqual((data["requested"], data["succeeded"], data["failures"]), (2, 2, []))
        usage = data["images"][0]["usage"]
        self.assertEqual(usage["output_tokens"], 2561)
        self.assertAlmostEqual(data["images"][0]["cost_usd"], cost_from_usage(MODEL, usage) / 2)
        self.assertAlmostEqual(data["images"][0]["cost_usd"], batch_cost_from_usage(MODEL, usage))
        self.assertAlmostEqual(data["cost_usd"], 2 * data["images"][0]["cost_usd"])
        self.assertTrue(all(i["meets_height"] for i in data["images"]))
        self.assertIn("no modo normal seria", out)
        self.assertIn(report.usd(data["cost_usd"]), out)

    def test_fetch_reports_requests_that_did_not_finish(self):
        run_dir = self.submitted_dir("--n", "3")
        self.server.batch_status, self.server.drop_last_result = "expired", True
        code, out, _ = run_cli("batch", "fetch", str(run_dir), env=self.env)
        self.assertEqual(code, 1)
        self.assertEqual(len(list(run_dir.glob("*.png"))), 2)
        self.assertIn("sem resultado (batch expired)", out)
        data = json.loads((run_dir / batch.RESULTS_FILE).read_text())
        self.assertEqual((data["requested"], data["succeeded"], len(data["failures"])), (3, 2, 1))

    def test_fetch_reports_failed_requests_from_the_error_file(self):
        prompt_file = Path(self.tmp.name) / "p.txt"
        prompt_file.write_text("FAIL400 a knight")
        run_dir = self.submitted_dir("--n", "2", "--prompt-file", str(prompt_file))
        self.server.batch_status = "completed"
        code, out, _ = run_cli("batch", "fetch", str(run_dir), env=self.env)   # sem arquivo de saída, só o de erros
        self.assertEqual(code, 1)
        self.assertEqual(list(run_dir.glob("*.png")), [])
        self.assertEqual(out.count("invalid_value: Invalid value: 'quality'"), 2)

    def test_fetch_ignores_lines_with_unknown_custom_id(self):
        evil = {"id": "x", "custom_id": "../../evil", "error": None,
                "response": {"status_code": 200, "body": {"data": [{"b64_json": base64.b64encode(make_png(16, 16)).decode()}]}}}
        self.server.inject_output_lines = [evil, {"custom_id": ["lista"], "response": None, "error": None}]
        run_dir = self.submitted_dir()
        self.server.batch_status = "completed"
        code, _, err = run_cli("batch", "fetch", str(run_dir), env=self.env)
        self.assertEqual(code, 0)
        self.assertEqual(err.count("custom_id desconhecido"), 2)
        self.assertEqual(list(Path(self.tmp.name).rglob("*evil*")), [])
        self.assertEqual(len(list(run_dir.glob("*.png"))), 1)

    def test_fetch_can_be_repeated(self):
        run_dir = self.submitted_dir()
        self.server.batch_status = "completed"
        first = run_cli("batch", "fetch", str(run_dir), env=self.env)[0]
        second = run_cli("batch", "fetch", str(run_dir), env=self.env)[0]
        self.assertEqual((first, second), (0, 0))
        self.assertEqual(len(list(run_dir.glob("*.png"))), 1)


class InterpretTests(unittest.TestCase):
    PNG = base64.b64encode(make_png(16, 16)).decode()

    def ok(self, **extra):
        return {"custom_id": "a", "error": None, "response": {"status_code": 200, "request_id": "r1", "body": {
            "data": [{"b64_json": self.PNG}], "usage": {"output_tokens": 5}}}, **extra}

    def test_success(self):
        result = batch.interpret(self.ok())
        self.assertTrue(result["image"].startswith(b"\x89PNG"))
        self.assertEqual((result["error"], result["usage"], result["request_id"]), (None, {"output_tokens": 5}, "r1"))

    def test_error_in_both_shapes_the_api_uses(self):
        top_level = batch.interpret({"custom_id": "a", "response": None,
                                     "error": {"code": "invalid_value", "message": "bad quality"}})
        self.assertEqual((top_level["image"], top_level["error"]), (None, "invalid_value: bad quality"))
        in_body = batch.interpret({"custom_id": "a", "error": None, "response": {
            "status_code": 400, "body": {"error": {"message": "bad size", "code": None}}}})
        self.assertEqual(in_body["error"], "bad size")
        http_only = batch.interpret({"custom_id": "a", "error": None, "response": {"status_code": 500, "body": {}}})
        self.assertEqual(http_only["error"], "HTTP 500")

    def test_malformed_responses_become_errors_not_crashes(self):
        for item in ({"custom_id": "a", "response": {"status_code": 200, "body": {}}},
                     {"custom_id": "a", "response": {"status_code": 200, "body": {"data": []}}},
                     {"custom_id": "a", "response": {"status_code": 200, "body": {"data": [{"b64_json": "!!não é base64!!"}]}}},
                     {"custom_id": "a", "response": "texto"}):
            result = batch.interpret(item)
            self.assertIsNone(result["image"], item)
            self.assertTrue(result["error"], item)


if __name__ == "__main__":
    unittest.main()
