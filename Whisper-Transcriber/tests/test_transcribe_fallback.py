import sys
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
