import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import transcribe


class TranscribeFallbackTests(unittest.TestCase):
    def test_load_whisper_model_falls_back_to_cpu_when_cuda_runtime_is_missing(self):
        dummy_model = object()

        def fake_whisper_model(model_name, device, compute_type):
            if device == "cuda":
                raise RuntimeError(
                    "Library cublas64_11.dll is not found or cannot be loaded"
                )
            return dummy_model

        with patch("transcribe.WhisperModel", side_effect=fake_whisper_model):
            model, device, compute_type = transcribe.load_whisper_model(
                "medium", "cuda", None
            )

        self.assertIs(model, dummy_model)
        self.assertEqual(device, "cpu")
        self.assertEqual(compute_type, "int8")


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.rules = transcribe.load_corrections([transcribe.DEFAULT_CORRECTIONS_FILE])

    def test_shipped_glossary_fixes_common_mishearings(self):
        cases = [
            ("It includes Rack-based solutions and L2 SQL", "RAG-based"),
            ("vector databases and slam-chain", "LangChain"),
            ("a library called doc link", "Docling"),
            ("the azure blog storage container", "Azure Blob Storage"),
            ("we prepare restored procedures", "stored procedures"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertIn(expected, transcribe.apply_corrections(text, self.rules))

    def test_corrections_respect_word_boundaries(self):
        self.assertEqual(
            transcribe.apply_corrections("the server was racked overnight", self.rules),
            "the server was racked overnight",
        )


class NoiseFilterTests(unittest.TestCase):
    def test_filler_only_segments_are_noise(self):
        for text in ("um", "um um um.", "uh, um", "   "):
            with self.subTest(text=text):
                self.assertTrue(transcribe.is_noise_segment(text, 0.1))

    def test_real_speech_is_kept(self):
        self.assertFalse(transcribe.is_noise_segment("Thank you.", 0.1))
        self.assertFalse(
            transcribe.is_noise_segment("So the orchestrator runs agents in parallel.", 0.9)
        )

    def test_boilerplate_over_silence_is_noise(self):
        self.assertTrue(transcribe.is_noise_segment("Thanks for watching!", 0.9))
        self.assertFalse(transcribe.is_noise_segment("Thanks for watching!", 0.1))


class VocabularyTests(unittest.TestCase):
    def test_default_vocabulary_loads(self):
        terms = transcribe.load_vocabulary([transcribe.DEFAULT_VOCAB_FILE])
        self.assertIn("LangChain", terms)

    def test_work_vocabulary_loads(self):
        terms = transcribe.load_vocabulary([ROOT / "vocab_work.txt"])
        self.assertIn("Power BI", terms)
        self.assertIn("Azure DevOps", terms)

    def test_terms_are_deduplicated_across_files(self):
        terms = transcribe.load_vocabulary(
            [transcribe.DEFAULT_VOCAB_FILE, transcribe.DEFAULT_VOCAB_FILE]
        )
        self.assertEqual(len(terms), len(set(t.lower() for t in terms)))

    def test_hotwords_are_trimmed_to_the_model_budget(self):
        class FakeTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return SimpleNamespace(ids=text.split())

        model = SimpleNamespace(hf_tokenizer=FakeTokenizer(), max_length=12)
        hotwords = transcribe.build_hotwords(
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"], model, "test"
        )
        self.assertTrue(hotwords.startswith("alpha, beta"))
        self.assertLessEqual(len(hotwords.split()), 5)


class SourceResolutionTests(unittest.TestCase):
    def test_local_sibling_is_loaded_first(self):
        files = transcribe.resolve_source_files([str(ROOT / "vocab_work.txt")])
        names = [f.name for f in files]
        self.assertIn("vocab_work.txt", names)
        if "vocab_work.local.txt" in names:
            self.assertLess(
                names.index("vocab_work.local.txt"), names.index("vocab_work.txt")
            )

    def test_missing_files_are_skipped(self):
        self.assertEqual(transcribe.resolve_source_files([str(ROOT / "nope.txt"), ""]), [])


if __name__ == "__main__":
    unittest.main()
