
import unittest
from unittest.mock import patch

import numpy as np
import faiss

from retrieval import (
    split_text,
    build_search_index,
    search_chunks,
)


class FakeEmbeddingModel:
    """A lightweight mock embedding model for tests."""

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        embeddings = []

        for text in texts:
            text = text.lower()

            if "cat" in text:
                vector = [1.0, 0.0, 0.0]
            elif "dog" in text:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]

            embeddings.append(vector)

        return np.array(embeddings, dtype="float32")


class TestSplitText(unittest.TestCase):

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(split_text(""), [])

    def test_whitespace_only_returns_empty_list(self):
        self.assertEqual(split_text("   \n\t  "), [])

    def test_short_text_returns_one_chunk(self):
        text = "This is a short test document."
        self.assertEqual(split_text(text), [text])

    def test_long_text_is_split_into_chunks(self):
        text = "A" * 2500

        result = split_text(
            text,
            chunk_size=1000,
            overlap=150,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 1000)
        self.assertEqual(len(result[1]), 1000)
        self.assertEqual(len(result[2]), 800)

    def test_chunks_have_expected_overlap(self):
        text = "A" * 2500

        result = split_text(
            text,
            chunk_size=1000,
            overlap=150,
        )

        self.assertEqual(result[0][-150:], result[1][:150])
        self.assertEqual(result[1][-150:], result[2][:150])

    def test_multiple_spaces_are_normalized(self):
        text = "Hello     world\n\nPython"
        self.assertEqual(
            split_text(text),
            ["Hello world Python"],
        )


class TestBuildSearchIndex(unittest.TestCase):

    @patch("retrieval.SentenceTransformer")
    def test_empty_chunks_return_none_values(self, mock_model):
        result = build_search_index([])

        self.assertEqual(result, (None, None, None))
        mock_model.assert_not_called()

    @patch("retrieval.SentenceTransformer")
    def test_builds_faiss_index(self, mock_model_class):
        mock_model_class.return_value = FakeEmbeddingModel()

        chunks = [
            "Cats are friendly animals.",
            "Dogs enjoy playing.",
        ]

        model, index, stored_chunks = build_search_index(chunks)

        self.assertIsNotNone(model)
        self.assertIsInstance(index, faiss.IndexFlatIP)
        self.assertEqual(stored_chunks, chunks)
        self.assertEqual(index.ntotal, 2)


class TestSearchChunks(unittest.TestCase):

    def setUp(self):
        self.model = FakeEmbeddingModel()
        self.chunks = [
            "Cats are friendly animals.",
            "Dogs enjoy playing.",
            "Birds can fly.",
        ]

        embeddings = self.model.encode(self.chunks)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def test_returns_most_relevant_chunk_first(self):
        results = search_chunks(
            "Tell me about a cat",
            self.model,
            self.index,
            self.chunks,
            top_k=3,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(
            results[0]["text"],
            "Cats are friendly animals.",
        )

    def test_top_k_limits_results(self):
        results = search_chunks(
            "Tell me about a cat",
            self.model,
            self.index,
            self.chunks,
            top_k=1,
        )

        self.assertEqual(len(results), 1)

    def test_empty_chunks_return_empty_results(self):
        results = search_chunks(
            "cat",
            self.model,
            self.index,
            [],
        )

        self.assertEqual(results, [])

    def test_missing_model_or_index_returns_empty_results(self):
        self.assertEqual(
            search_chunks("cat", None, self.index, self.chunks),
            [],
        )

        self.assertEqual(
            search_chunks("cat", self.model, None, self.chunks),
            [],
        )


if __name__ == "__main__":
    unittest.main()
