import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from generation_utils.generator import (
    StudentGenerator,
    ground_response,
    select_ranked_context,
    serialize_ranked_context,
)
from generation_utils.schema import DatasetSummary, Response
from ingestion_utils.pre_processor import extract_text_and_url_from_html, chunk_text
from retrieval_utils import retriever


class FakeEncoding:
    def encode(self, text):
        return text.split()


class PreprocessorTests(unittest.TestCase):
    def test_cleanup_removes_navigation_and_hides_missing_url(self):
        html = """
        <html><head><title>Test page</title></head><body>
          <a class="skip-link" href="#main">Skip to main content</a>
          <nav aria-label="Primary navigation">Menu item</nav>
          <main id="main"><p>Useful evidence remains here.</p></main>
        </body></html>
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.html"
            path.write_text(html, encoding="utf-8")
            text, url, title = extract_text_and_url_from_html(str(path))
        self.assertEqual(url, "")
        self.assertEqual(title, "Test page")
        self.assertIn("Useful evidence remains here.", text)
        self.assertNotIn("Skip to main content", text)
        self.assertNotIn("Menu item", text)

    def test_long_sentence_is_preserved_as_a_complete_chunk(self):
        long_sentence = " ".join(["word"] * 30) + "."
        sentences = ["First sentence.", long_sentence, "Last sentence."]
        with patch("ingestion_utils.pre_processor.tiktoken.get_encoding", return_value=FakeEncoding()), patch(
            "ingestion_utils.pre_processor.sent_tokenize", return_value=sentences
        ):
            chunks = chunk_text("ignored", max_tokens=5, overlap=2)
        self.assertIn(long_sentence, chunks)
        self.assertEqual(chunks[0], "First sentence.")
        self.assertEqual(chunks[-1], "Last sentence.")


class FakeEncoder:
    def encode(self, *_args, **_kwargs):
        return [[0.0, 1.0]]


class FakeCollection:
    metadata = None

    def count(self):
        return 4

    def query(self, **_kwargs):
        return {
            "ids": [["b2", "a1", "b1", "a2"]],
            "documents": [["B second.", "A first.", "B first.", "A second."]],
            "metadatas": [[
                {"dataset": "b"}, {"dataset": "a"}, {"dataset": "b"}, {"dataset": "a"}
            ]],
            "distances": [[0.3, 0.1, 0.1, 0.2]],
        }


class RetrievalTests(unittest.TestCase):
    def test_retrieval_has_stable_ties_and_bounded_scores(self):
        old_cache = retriever._global_cache.copy()
        retriever._global_cache.update({
            "encoder": FakeEncoder(),
            "model_name": "fake",
            "collection": FakeCollection(),
            "collection_name": "fake",
        })
        try:
            first = retriever.retrieve_data("query", "unused", "fake", "fake", 2)
            second = retriever.retrieve_data("query", "unused", "fake", "fake", 2)
        finally:
            retriever._global_cache.clear()
            retriever._global_cache.update(old_cache)
        self.assertEqual(first, second)
        self.assertEqual([item.dataset_id for item in first], ["a", "b"])
        self.assertTrue(all(0.0 <= item.score <= 1.0 for item in first))


class GenerationTests(unittest.TestCase):
    def test_year_in_title_does_not_count_as_year_in_evidence(self):
        context = '[{"source_title":"2026 report","chunks":["Values are available through 2024."]}]'
        self.assertEqual(StudentGenerator._missing_query_years("What happened in 2026?", context), ["2026"])

    def test_explicit_top_count_is_enforced_before_generation(self):
        ranked = [SimpleNamespace(dataset_id=str(index)) for index in range(5)]
        self.assertEqual(
            [item.dataset_id for item in select_ranked_context("Show the top 3 resources", ranked)],
            ["0", "1", "2"],
        )
        self.assertEqual(len(select_ranked_context("Show the top three resources", ranked)), 3)

    def test_missing_year_bypasses_the_llm(self):
        generator = object.__new__(StudentGenerator)
        generator.llm = SimpleNamespace(provider="fake", model_name="fake")
        context = '[{"chunks":["Values are available through 2024."]}]'
        result = generator.generate("What was the value in 2026?", context, Response)
        self.assertEqual(result.evidence_status, "insufficient")
        self.assertEqual(result.supporting_datasets, [])

    def test_structured_generation_is_content_cached(self):
        calls = []

        class FakeLLM:
            provider = "fake"
            model_name = "fake-model"

            def generate_structured(self, *_args, **_kwargs):
                calls.append(1)
                return Response(answer="Stable response.")

        generator = object.__new__(StudentGenerator)
        generator.llm = FakeLLM()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            StudentGenerator,
            "_cache_path",
            return_value=Path(directory) / "response.json",
        ):
            first = generator.generate("query", "[]", Response)
            second = generator.generate("query", "[]", Response)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_expired_cache_is_regenerated(self):
        class FakeLLM:
            provider = "fake"
            model_name = "fake-model"

            def generate_structured(self, *_args, **_kwargs):
                return Response(answer="Fresh response.")

        generator = object.__new__(StudentGenerator)
        generator.llm = FakeLLM()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "response.json"
            cache_path.write_text(Response(answer="Expired response.").model_dump_json(), encoding="utf-8")
            with patch.object(StudentGenerator, "_cache_path", return_value=cache_path), patch(
                "generation_utils.generator.CACHE_TTL_SECONDS", 0
            ):
                result = generator.generate("query", "[]", Response)
        self.assertEqual(result.answer, "Fresh response.")

    def test_grounding_replaces_invalid_quote_and_fixes_source_order(self):
        ranked = [
            SimpleNamespace(
                dataset_id="a",
                top_score=0.75,
                source_title="A",
                top_chunks=[{"text": "Complete evidence sentence."}],
            ),
            SimpleNamespace(
                dataset_id="b",
                top_score=0.5,
                source_title="B",
                top_chunks=[{"text": "Second complete sentence."}],
            ),
        ]
        response = Response(
            answer="Grounded answer.",
            name_top="wrong",
            supporting_datasets=[
                DatasetSummary(name="b", summary="High-level takeaway.", quote="invented quote")
            ],
        )
        grounded = ground_response(response, ranked)
        self.assertEqual(grounded.name_top, "a")
        self.assertEqual([item.name for item in grounded.supporting_datasets], ["a", "b"])
        self.assertEqual(grounded.supporting_datasets[1].quote, "Second complete sentence.")
        self.assertEqual(grounded.supporting_datasets[0].relevance_score, 0.75)
        self.assertEqual(serialize_ranked_context(ranked), serialize_ranked_context(ranked))

    def test_insufficient_response_never_receives_citations(self):
        ranked = [SimpleNamespace(dataset_id="a", top_score=0.8, top_chunks=[{"text": "Evidence."}])]
        response = Response(
            answer="The requested year is unavailable.",
            evidence_status="insufficient",
            supporting_datasets=[DatasetSummary(name="a", quote="Evidence.")],
        )
        grounded = ground_response(response, ranked)
        self.assertIsNone(grounded.name_top)
        self.assertEqual(grounded.supporting_datasets, [])


if __name__ == "__main__":
    unittest.main()
