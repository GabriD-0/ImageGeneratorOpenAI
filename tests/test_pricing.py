import unittest

from openai_images.pricing import (canonical, cost_from_usage, estimate, get_model, output_tokens,
                                   size_problems)


class OfficialNumbers(unittest.TestCase):
    """Números impressos nas páginas oficiais da OpenAI (consultadas em 04/11/2026)."""

    def test_calculator_example_26_low_square(self):
        # A própria calculadora mostra: 823 tokens, US$ 0,02469
        tokens = output_tokens(get_model("gpt-image-2.6-flare"), "low", 1024, 1024)
        self.assertEqual(tokens, 823)
        self.assertAlmostEqual(tokens * 30 / 1e6, 0.02469, places=5)

    def test_gpt_image_2_matches_official_per_image_table(self):
        # Tabela "Per-image output cost" do guia de imagens (histórica, gpt-image-2)
        table = {"1024x1024": (0.006, 0.053, 0.211), "1024x1536": (0.005, 0.041, 0.165),
                 "1536x1024": (0.005, 0.041, 0.165)}
        for size, expected in table.items():
            got = tuple(round(estimate("gpt-image-2", q, size).output_usd, 3) for q in ("low", "medium", "high"))
            self.assertEqual(got, expected, size)

    def test_legacy_models_use_the_official_table(self):
        self.assertEqual(estimate("gpt-image-1.5", "high", "1024x1536").output_usd, 0.2)
        self.assertEqual(estimate("gpt-image-1-mini", "low", "1024x1024").output_usd, 0.005)

    def test_25_tokens_per_quality_square(self):
        m = get_model("gpt-image-2.5-sunburst")
        got = {q: output_tokens(m, q, 1024, 1024) for q in ("low", "medium", "high", "xhigh", "max")}
        self.assertEqual(got, {"low": 196, "medium": 439, "high": 1756, "xhigh": 3122, "max": 7024})

    def test_project_size_is_portrait_and_over_1600(self):
        est = estimate("gpt-image-2.5-flare", "high", "1088x1664", prompt_tokens=310)
        self.assertEqual(est.output_tokens, 1418)
        self.assertAlmostEqual(est.output_usd, 1418 * 30 / 1e6)
        self.assertAlmostEqual(est.batch_usd, est.total_usd / 2)


class Validation(unittest.TestCase):
    def test_size_rules(self):
        self.assertEqual(size_problems(1088, 1664), [])
        self.assertTrue(size_problems(1000, 1664))          # não é múltiplo de 16
        self.assertTrue(size_problems(512, 512))            # poucos pixels
        self.assertTrue(size_problems(3840, 1024))          # proporção > 3:1
        self.assertTrue(size_problems(4096, 2048))          # lado maior que 3840

    def test_legacy_models_reject_custom_sizes(self):
        with self.assertRaises(ValueError):
            estimate("gpt-image-1.5", "low", "1088x1664")

    def test_quality_support_per_model(self):
        with self.assertRaises(ValueError):
            estimate("gpt-image-2", "xhigh", "1088x1664")    # xhigh só no 2.5/2.6
        with self.assertRaises(ValueError):
            estimate("gpt-image-2.5-flare", "auto", "1088x1664")

    def test_snapshot_names_are_canonicalised(self):
        self.assertEqual(canonical("gpt-image-2.5-flare-2026-09-08"), "gpt-image-2.5-flare")
        self.assertEqual(canonical("gpt-image-2-2026-04-21"), "gpt-image-2")
        with self.assertRaises(ValueError):
            get_model("dall-e-3")


class CostFromUsage(unittest.TestCase):
    def test_text_image_and_output_tokens(self):
        usage = {"input_tokens": 1300, "input_tokens_details": {"text_tokens": 300, "image_tokens": 1000},
                 "output_tokens": 1418}
        expected = (300 * 5 + 1000 * 8 + 1418 * 30) / 1e6
        self.assertAlmostEqual(cost_from_usage("gpt-image-2.5-flare", usage), expected)

    def test_without_details_all_input_is_text(self):
        self.assertAlmostEqual(cost_from_usage("gpt-image-2", {"input_tokens": 200, "output_tokens": 100}),
                               (200 * 5 + 100 * 30) / 1e6)

    def test_missing_usage(self):
        self.assertIsNone(cost_from_usage("gpt-image-2", None))


if __name__ == "__main__":
    unittest.main()
