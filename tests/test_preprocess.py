from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from rvc.train.preprocess.dataset import BoundedAudioWriter, stage_validation_audio
from rvc.train.preprocess.processor import PreProcess
from rvc.train.preprocess.silence import (
    find_streaming_silence_cuts,
    iter_audio_with_silence_cuts,
    truncate_silence,
)


class PreprocessTests(unittest.TestCase):
    def test_silence_processing_across_block_boundaries(self):
        audio = np.concatenate([
            np.zeros(450), np.full(250, 0.2), np.zeros(630),
            np.full(450, -0.3), np.zeros(410),
        ]).astype(np.float32)
        for action in ("truncate", "compress"):
            for size in (1, 37, 400, len(audio)):
                with self.subTest(action=action, size=size):
                    blocks = lambda: (audio[i:i + size] for i in range(0, len(audio), size))
                    cuts, count = find_streaming_silence_cuts(
                        blocks(), 1000, -45, 0.3, 0.1, action, 50,
                    )
                    actual = np.concatenate(list(iter_audio_with_silence_cuts(blocks(), cuts, count)))
                    expected = truncate_silence(
                        audio, 1000, minimum_silence=0.3, truncate_to=0.1,
                        action=action, compress_percent=50,
                    )
                    np.testing.assert_array_equal(actual, expected)

    def test_writer_formats_and_short_audio(self):
        for fmt, extension, subtype in (("wav", "wav", "PCM_16"), ("wav_float32", "wav", "FLOAT"), ("flac", "flac", "PCM_24")):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as directory:
                with BoundedAudioWriter(2) as writer:
                    writer.submit(directory, "short", 16000, np.ones(15999), fmt)
                    writer.submit(directory, "valid", 16000, np.full(16000, 0.25), fmt)
                self.assertEqual(writer.skipped_short, 1)
                self.assertEqual(len(list(Path(directory).iterdir())), 1)
                info = sf.info(str(Path(directory) / f"valid.{extension}"))
                self.assertEqual((info.frames, info.samplerate, info.channels, info.subtype), (16000, 16000, 1, subtype))

    def test_writer_propagates_background_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "non-finite"):
                with BoundedAudioWriter(2) as writer:
                    writer.submit(directory, "invalid", 16000, np.full(16000, np.nan), "wav")

    def test_validation_staging_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "Validation" / "nested"
            source.mkdir(parents=True)
            sf.write(str(source / "voice.wav"), np.ones(16000) * 0.1, 16000)
            (source / "ignored.txt").write_text("ignored")
            original = (source / "voice.wav").read_bytes()
            target = root / "output"
            self.assertEqual(stage_validation_audio(str(root / "input"), str(target)), 1)
            copied = target / "validation" / "audio" / "nested" / "voice.wav"
            self.assertEqual(copied.read_bytes(), original)
            self.assertEqual((source / "voice.wav").read_bytes(), original)
            empty = root / "empty"
            empty.mkdir()
            self.assertEqual(stage_validation_audio(str(empty), str(target)), 0)
            self.assertFalse((target / "validation").exists())

    def test_streamed_chunks_preserve_overlap_and_tail(self):
        audio = np.sin(np.arange(107000, dtype=np.float32) * 0.01) * 0.5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = PreProcess(16000, str(root / "memory"), "wav_float32")
            streamed = PreProcess(16000, str(root / "streamed"), "wav_float32")
            skipped = memory.simple_cut(audio, 2, 3, 3.0, 0.3, "none")
            blocks = (audio[i:i + 997] for i in range(0, len(audio), 997))
            self.assertEqual(streamed.simple_cut_stream(blocks, 2, 3, 3.0, 0.3), skipped)
            expected = {p.name: p.read_bytes() for p in Path(memory.gt_wavs_dir).iterdir()}
            actual = {p.name: p.read_bytes() for p in Path(streamed.gt_wavs_dir).iterdir()}
            self.assertTrue(expected)
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
