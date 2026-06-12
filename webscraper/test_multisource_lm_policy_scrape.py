import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRAPER_PATH = Path(__file__).with_name("multisource_lm_policy_scrape.py")
spec = importlib.util.spec_from_file_location("multisource_lm_policy_scrape", SCRAPER_PATH)
scraper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(scraper)


PAGE = {
    "source_name": "Test Source",
    "legislature": "us",
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

    def test_includes_legislature_metadata(self):
        record = self.record_for(
            "Generative AI services must prevent users from obtaining assistance that enables phishing "
            "or unauthorized access to protected computer systems."
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["legislature"], "us")

    def test_filters_previously_seen_policy_clauses(self):
        previous = [
            {
                "clause": "Generative AI services must prevent users from obtaining assistance that enables phishing.",
                "source_name": "Older Source",
            }
        ]
        scraped = [
            {
                "clause": "Generative AI services must prevent users from obtaining assistance that enables phishing.",
                "source_name": "New Source",
            },
            {
                "clause": "Generative AI services must prevent users from obtaining assistance that enables malware.",
                "source_name": "New Source",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            previous_path = Path(tmpdir) / "previous.json"
            with open(previous_path, "w", encoding="utf-8") as f:
                json.dump(previous, f)

            new_items, previous_count = scraper.filter_new_policy_items(scraped, [str(previous_path)])

        self.assertEqual(previous_count, 1)
        self.assertEqual([item["clause"] for item in new_items], [scraped[1]["clause"]])

    def test_configured_sources_match_current_source_list(self):
        self.assertEqual(
            [source.name for source in scraper.SOURCES],
            [
                "Congress.gov",
                "Federal Register",
                "California Legislature",
                "EUR-Lex",
                "EU AI Office",
                "UK AISI",
                "CAC China",
                "NIST AI",
                "IMDA Singapore",
                "AI Verify Foundation",
                "METI Japan AI Policy",
                "MSIT Korea",
                "Korea Law Information Center",
                "Parliament of Canada LegisINFO",
                "ISED Canada AI",
                "OECD AI Policy Observatory",
            ],
        )
        removed_sources = {"AI Incident Database"}
        self.assertTrue(removed_sources.isdisjoint({source.name for source in scraper.SOURCES}))


if __name__ == "__main__":
    unittest.main()
