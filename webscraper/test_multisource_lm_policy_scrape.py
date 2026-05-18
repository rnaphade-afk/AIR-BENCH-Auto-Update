import importlib.util
import unittest
from pathlib import Path


SCRAPER_PATH = Path(__file__).with_name("multisource_lm_policy_scrape.py")
spec = importlib.util.spec_from_file_location("multisource_lm_policy_scrape", SCRAPER_PATH)
scraper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(scraper)


PAGE = {
    "source_name": "Test Source",
    "url": "https://example.test/policy",
    "title": "Test Policy",
    "published_date": "",
}


class ClauseFilterTests(unittest.TestCase):
    def record_for(self, clause):
        return scraper.normalize_clause_record({"clause": clause, "published_date": ""}, PAGE)

    def test_rejects_provider_evaluation_process_clause(self):
        clause = (
            "Providers of general-purpose AI models with systemic risk must perform model evaluations "
            "in accordance with standardized protocols and tools reflecting the state of the art, "
            "including adversarial testing of the model."
        )
        self.assertIsNone(self.record_for(clause))

    def test_rejects_broad_critical_harm_release_clause(self):
        clause = (
            "A developer may not make a covered model available for commercial, public, or foreseeably "
            "public use if there is an unreasonable risk that the covered model will cause or materially "
            "enable a critical harm."
        )
        self.assertIsNone(self.record_for(clause))

    def test_accepts_concrete_generated_content_clause(self):
        clause = (
            "Generated content must not contain terrorism, extremism, ethnic hatred, ethnic discrimination, "
            "violence, obscene or pornographic information, false harmful information, or other content "
            "prohibited by laws and administrative regulations."
        )
        self.assertIsNotNone(self.record_for(clause))


if __name__ == "__main__":
    unittest.main()
