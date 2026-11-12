import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openai_images import cli, report
from openai_images.compare import parse_configs
from openai_images.prompt import Prompt, build_prompt
from tests.mock_server import MockImagesServer

try:
    import PIL  # noqa: F401
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def run_cli(*argv, env=None):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, env or {}, clear=False), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class PromptTests(unittest.TestCase):
    def test_negative_modes(self):
        p = Prompt("a knight", "text, blurry")
        self.assertEqual(build_prompt(p, "append"), "a knight Avoid: text, blurry.")
        self.assertEqual(build_prompt(p, "ignore"), "a knight")
        self.assertEqual(build_prompt(Prompt("a knight"), "append"), "a knight")

    def test_real_prompt_file_loads(self):
        from openai_images import ROOT
        from openai_images.prompt import load_prompt
        p = load_prompt(ROOT / "prompt.txt")
        self.assertIn("chroma green", p.positive)
        self.assertIn("watermark", p.negative)


class ConfigParsing(unittest.TestCase):
    def test_parse(self):
        cfgs = parse_configs("gpt-image-2.5-flare:low, gpt-image-2.6-flare:high:medium")
        self.assertEqual([c.label for c in cfgs], ["gpt-image-2.5-flare:low", "gpt-image-2.6-flare:high:medium"])

    def test_rejects_bad_input(self):
        for bad in ("gpt-image-2.5-flare", "nope:low", "gpt-image-2.5-flare:low:high", "gpt-image-2:low,gpt-image-2:low", ""):
            with self.assertRaises(ValueError, msg=bad):
                parse_configs(bad)


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.server = MockImagesServer().__enter__()
        self.addCleanup(self.server.__exit__)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": self.server.base_url}
        self.out = Path(self.tmp.name)

    def test_estimate_command(self):
        code, out, _ = run_cli("estimate", "--models", "gpt-image-2.5-flare,gpt-image-1-mini")
        self.assertEqual(code, 0)
        self.assertIn("gpt-image-2.5-flare", out)
        self.assertNotIn("gpt-image-1-mini  ", out)      # tamanho fixo: fora da tabela por padrão
        code, out, _ = run_cli("estimate", "--models", "gpt-image-1-mini", "--all", "--size", "1024x1536")
        self.assertIn("NÃO", out)

    def test_generate_dry_run_needs_no_key(self):
        code, out, _ = run_cli("generate", "--dry-run", env={"OPENAI_API_KEY": ""})
        self.assertEqual(code, 0)
        self.assertIn("Avoid:", out)
        self.assertEqual(self.server.requests, [])

    def test_generate_rejects_invalid_and_expensive(self):
        code, _, err = run_cli("generate", "--dry-run", "--size", "1000x1000")
        self.assertEqual(code, 2); self.assertIn("múltiplas", err)
        code, _, err = run_cli("generate", "--model", "gpt-image-2.5-flare", "--quality", "max", "--max-cost", "0.05",
                               "--out-dir", str(self.out), env=self.env)
        self.assertEqual(code, 2); self.assertIn("--max-cost", err)
        self.assertEqual(self.server.requests, [])

    def test_generate_saves_image_and_reports_real_cost(self):
        code, out, _ = run_cli("generate", "--quality", "low", "--out-dir", str(self.out), env=self.env)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(list(self.out.glob("*.png"))), 1)
        meta = json.loads(next(self.out.glob("*.json")).read_text())
        self.assertEqual(meta["usage"]["output_tokens"], 153)
        self.assertGreater(meta["cost_usd"], 0)
        self.assertIn("Custo real", out)

    def test_api_error_is_reported_cleanly(self):
        code, _, err = run_cli("generate", "--quality", "low", "--out-dir", str(self.out),
                               env={**self.env, "OPENAI_API_KEY": "wrong"})
        self.assertEqual(code, 1); self.assertIn("401", err); self.assertNotIn("wrong", err)

    def test_missing_key(self):
        with mock.patch("openai_images.client.load_dotenv"):
            code, _, err = run_cli("generate", "--quality", "low", "--out-dir", str(self.out),
                                   env={"OPENAI_API_KEY": "", "OPENAI_BASE_URL": self.server.base_url})
        self.assertEqual(code, 1); self.assertIn("OPENAI_API_KEY", err)

    def test_compare_dry_run(self):
        code, out, _ = run_cli("compare", "--dry-run", "--out-dir", str(self.out))
        self.assertEqual(code, 0)
        self.assertIn("Custo total estimado", out)
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(self.server.requests, [])

    def test_compare_refuses_over_budget_before_any_call(self):
        code, _, err = run_cli("compare", "--budget", "0.01", "--yes", "--out-dir", str(self.out), env=self.env)
        self.assertEqual(code, 2); self.assertIn("orçamento", err)
        self.assertEqual(self.server.requests, [])

    def test_compare_requires_confirmation_when_not_interactive(self):
        code, _, err = run_cli("compare", "--out-dir", str(self.out), env=self.env)
        self.assertEqual(code, 1); self.assertIn("--yes", err)
        self.assertEqual(self.server.requests, [])

    def test_compare_then_rank(self):
        configs = "gpt-image-2.5-flare:low,gpt-image-2.5-flare:medium,gpt-image-2.5-flare:high"
        code, out, err = run_cli("compare", "--configs", configs, "--samples", "2", "--yes", "--out-dir", str(self.out),
                                 env=self.env)
        self.assertEqual(code, 0, out + err)
        self.assertEqual(len(self.server.requests), 3)            # n=2 num único pedido por combinação
        self.assertEqual(self.server.requests[0]["body"]["n"], 2)
        run_dir = next(self.out.iterdir())
        data = json.loads((run_dir / report.RESULTS_FILE).read_text())
        low, med, high = data["records"]
        self.assertEqual(low["tokens_per_image"], 153)
        self.assertLess(low["cost_per_image"], med["cost_per_image"])
        self.assertLess(med["cost_per_image"], high["cost_per_image"])
        self.assertTrue(all(i["meets_height"] for r in data["records"] for i in r["images"]))
        if HAS_PIL:
            self.assertTrue(all(i["key_ok"] for r in data["records"] for i in r["images"]))
        self.assertTrue((run_dir / report.SHEET_FILE).read_text().count("<img") == 6)

        # sem notas -> erro claro
        code, _, err = run_cli("rank", str(run_dir))
        self.assertEqual(code, 2); self.assertIn("nenhuma nota", err)

        # notas: low=3, medium=4.5 (4 e 5), high=5 -> vence a mais barata com média >= 4: medium
        path = run_dir / report.RATINGS_FILE
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        grade = {"low": 3, "medium": 4, "high": 5}
        for i, row in enumerate(rows):
            quality = row["config"].split(":")[1]
            row["rating"] = str(grade[quality] + (1 if quality == "medium" and i % 2 else 0))
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["image", "config", "rating", "notes"]); writer.writeheader(); writer.writerows(rows)
        code, out, _ = run_cli("rank", str(run_dir))
        self.assertEqual(code, 0)
        self.assertIn("MELHOR CUSTO-BENEFÍCIO", out)
        winner_line = next(l for l in out.splitlines() if "MELHOR CUSTO-BENEFÍCIO" in l and "|" in l)
        self.assertIn("gpt-image-2.5-flare:medium", winner_line)
        self.assertIn("Subir para `gpt-image-2.5-flare:high`", out)
        self.assertTrue((run_dir / report.RANKING_FILE).is_file())

    def test_rank_rejects_bad_ratings(self):
        run_dir = self.out / "run"; run_dir.mkdir()
        (run_dir / report.RESULTS_FILE).write_text(json.dumps({"records": []}))
        (run_dir / report.RATINGS_FILE).write_text("image,config,rating,notes\na.png,x,9,\n")
        code, _, err = run_cli("rank", str(run_dir))
        self.assertEqual(code, 2); self.assertIn("entre 1 e 5", err)

    def test_compare_continues_after_a_failed_config_and_stops_on_bad_credentials(self):
        # FAIL400 no prompt -> toda combinação falha com 400, mas a comparação segue até o fim
        pf = self.out / "p.txt"; pf.write_text("FAIL400 a knight")
        code, out, _ = run_cli("compare", "--configs", "gpt-image-2.5-flare:low,gpt-image-2.5-flare:medium", "--yes",
                               "--prompt-file", str(pf), "--out-dir", str(self.out / "runs"), env=self.env)
        self.assertEqual(code, 1)
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(out.count("ERRO (400)"), 2)
        # chave errada -> para na primeira
        self.server.requests.clear()
        code, _, err = run_cli("compare", "--configs", "gpt-image-2.5-flare:low,gpt-image-2.5-flare:medium", "--yes",
                               "--out-dir", str(self.out / "runs2"), env={**self.env, "OPENAI_API_KEY": "wrong"})
        self.assertEqual(len(self.server.requests), 1); self.assertIn("interrompendo", err)

    @unittest.skipUnless(HAS_PIL, "Pillow não instalado")
    def test_white_background_is_flagged(self):
        pf = self.out / "p.txt"; pf.write_text("WHITE a knight")
        run_cli("compare", "--configs", "gpt-image-2.5-flare:low", "--yes", "--prompt-file", str(pf),
                "--out-dir", str(self.out / "runs"), env=self.env)
        run_dir = next((self.out / "runs").iterdir())
        data = json.loads((run_dir / report.RESULTS_FILE).read_text())
        img = data["records"][0]["images"][0]
        self.assertFalse(img["key_ok"]); self.assertLess(img["key_share"], 0.1)
        self.assertIn("fundo fora do chroma key", (run_dir / report.SHEET_FILE).read_text())


if __name__ == "__main__":
    unittest.main()
