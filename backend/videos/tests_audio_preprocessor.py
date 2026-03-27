import math
import shutil
import struct
import time
import wave
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from videos.audio_preprocessor import (
    LUFS_TARGET,
    MAX_CHUNK_DURATION_S,
    chunk_on_silence_boundaries,
    condition_audio_for_asr,
    normalize_to_lufs,
)


def _write_wave(path: Path, segments, sample_rate: int = 16000):
    frames = []
    for kind, duration_s, amplitude in segments:
        frame_count = int(duration_s * sample_rate)
        if kind == "silence":
            frames.extend([0] * frame_count)
            continue
        frequency = 220.0
        for idx in range(frame_count):
            sample = amplitude * math.sin(2.0 * math.pi * frequency * (idx / sample_rate))
            frames.append(int(max(-1.0, min(1.0, sample)) * 32767))

    with path.open("wb") as raw_file, wave.open(raw_file, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack("<" + "h" * len(frames), *frames))


class AudioPreprocessorBase(SimpleTestCase):
    TMP_DIR = Path(__file__).resolve().parent / "_tmp_audio_preprocessor"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for _attempt in range(3):
            try:
                shutil.rmtree(cls.TMP_DIR, ignore_errors=True)
                break
            except OSError:
                time.sleep(0.2)

    def setUp(self):
        super().setUp()
        base_dir = self.TMP_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        self.tmpdir = base_dir / self._testMethodName
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.tmpdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()


class LufsNormalizationTests(AudioPreprocessorBase):
    def test_loud_audio_gets_attenuated(self):
        input_path = self.tmpdir / "loud.wav"
        output_path = self.tmpdir / "loud_norm.wav"
        _write_wave(input_path, [("tone", 3.0, 0.95)])

        result = normalize_to_lufs(str(input_path), str(output_path))

        self.assertTrue(result.was_reencoded)
        self.assertLess(result.output_lufs, result.input_lufs)
        self.assertLessEqual(abs(result.output_lufs - LUFS_TARGET), 2.0)
        self.assertTrue(output_path.exists())

    def test_quiet_audio_gets_boosted(self):
        input_path = self.tmpdir / "quiet.wav"
        output_path = self.tmpdir / "quiet_norm.wav"
        _write_wave(input_path, [("tone", 3.0, 0.03)])

        result = normalize_to_lufs(str(input_path), str(output_path))

        self.assertTrue(result.was_reencoded)
        self.assertGreater(result.output_lufs, result.input_lufs)
        self.assertLessEqual(abs(result.output_lufs - LUFS_TARGET), 2.5)
        self.assertTrue(output_path.exists())

    def test_audio_within_tolerance_is_not_reencoded(self):
        raw_path = self.tmpdir / "raw.wav"
        normalized_once = self.tmpdir / "normalized_once.wav"
        normalized_twice = self.tmpdir / "normalized_twice.wav"
        _write_wave(raw_path, [("tone", 3.0, 0.4)])

        first = normalize_to_lufs(str(raw_path), str(normalized_once))
        second = normalize_to_lufs(str(normalized_once), str(normalized_twice))

        self.assertFalse(second.was_reencoded)
        self.assertTrue(normalized_twice.exists())
        self.assertLessEqual(abs(second.output_lufs - LUFS_TARGET), 1.0)
        self.assertEqual(normalized_once.read_bytes(), normalized_twice.read_bytes())
        self.assertTrue(first.output_lufs != 0.0)

    def test_result_metadata_reports_before_after_lufs(self):
        input_path = self.tmpdir / "meta.wav"
        output_path = self.tmpdir / "meta_norm.wav"
        _write_wave(input_path, [("tone", 2.0, 0.8)])

        result = normalize_to_lufs(str(input_path), str(output_path))

        self.assertIsInstance(result.input_lufs, float)
        self.assertIsInstance(result.output_lufs, float)
        self.assertEqual(result.output_path, str(output_path))


class AudioConditioningTests(AudioPreprocessorBase):
    def test_dynaudnorm_applied_when_flag_true(self):
        input_path = self.tmpdir / "dynaudnorm_input.wav"
        output_path = self.tmpdir / "dynaudnorm_output.wav"
        _write_wave(input_path, [("tone", 3.0, 0.6)])

        result = condition_audio_for_asr(
            str(input_path),
            str(output_path),
            apply_dynaudnorm=True,
            apply_speech_band_filter=False,
        )

        self.assertTrue(output_path.exists())
        self.assertTrue(result.dynaudnorm_applied)
        self.assertFalse(result.speech_band_filter_applied)
        self.assertFalse(result.was_noop)

    def test_speech_band_filter_applied_when_flag_true(self):
        input_path = self.tmpdir / "filters_input.wav"
        output_path = self.tmpdir / "filters_output.wav"
        _write_wave(input_path, [("tone", 3.0, 0.6)])

        result = condition_audio_for_asr(
            str(input_path),
            str(output_path),
            apply_dynaudnorm=False,
            apply_speech_band_filter=True,
        )

        self.assertTrue(output_path.exists())
        self.assertFalse(result.dynaudnorm_applied)
        self.assertTrue(result.speech_band_filter_applied)
        self.assertFalse(result.was_noop)

    def test_noop_when_both_flags_false(self):
        input_path = self.tmpdir / "noop_input.wav"
        _write_wave(input_path, [("tone", 3.0, 0.6)])

        result = condition_audio_for_asr(
            str(input_path),
            str(self.tmpdir / "noop_output.wav"),
            apply_dynaudnorm=False,
            apply_speech_band_filter=False,
        )

        self.assertTrue(result.was_noop)
        self.assertEqual(result.output_path, str(input_path))
        self.assertFalse(result.dynaudnorm_applied)
        self.assertFalse(result.speech_band_filter_applied)

    def test_output_is_still_pcm_16k_mono(self):
        input_path = self.tmpdir / "pcm_input.wav"
        output_path = self.tmpdir / "pcm_output.wav"
        _write_wave(input_path, [("tone", 3.0, 0.6)])

        result = condition_audio_for_asr(
            str(input_path),
            str(output_path),
            apply_dynaudnorm=True,
            apply_speech_band_filter=True,
        )

        with wave.open(result.output_path, "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getsampwidth(), 2)

    def test_conditioning_runs_before_chunking_in_pipeline(self):
        from videos.tasks import _prepare_audio_for_pipeline

        calls = []

        class _Normalization:
            input_lufs = -20.0
            output_lufs = -16.0
            was_reencoded = True
            output_path = "normalized.wav"

        class _Conditioning:
            input_path = "normalized.wav"
            output_path = "conditioned.wav"
            dynaudnorm_applied = True
            speech_band_filter_applied = True
            was_noop = False

        def _normalize(*args, **kwargs):
            calls.append("normalize")
            return _Normalization()

        def _condition(*args, **kwargs):
            calls.append("condition")
            return _Conditioning()

        def _chunk(path, output_dir):
            calls.append(("chunk", path))
            return []

        with patch("videos.tasks.normalize_to_lufs", side_effect=_normalize), patch(
            "videos.tasks.condition_audio_for_asr", side_effect=_condition
        ), patch("videos.tasks.chunk_on_silence_boundaries", side_effect=_chunk):
            prepared_audio_path, metadata = _prepare_audio_for_pipeline(
                "input.wav",
                apply_dynaudnorm=True,
                apply_speech_band_filter=True,
            )

        self.assertEqual(calls[0], "normalize")
        self.assertEqual(calls[1], "condition")
        self.assertEqual(calls[2], ("chunk", "conditioned.wav"))
        self.assertEqual(prepared_audio_path, "conditioned.wav")
        self.assertEqual(metadata["conditioned_path"], "conditioned.wav")


class SilenceBoundaryChunkingTests(AudioPreprocessorBase):
    def test_chunk_boundaries_fall_on_silence(self):
        audio_path = self.tmpdir / "silence_boundaries.wav"
        chunk_dir = self.tmpdir / "chunks_a"
        _write_wave(
            audio_path,
            [
                ("tone", 10.0, 0.5),
                ("silence", 1.0, 0.0),
                ("tone", 10.0, 0.5),
                ("silence", 1.0, 0.0),
                ("tone", 8.0, 0.5),
            ],
        )

        chunks = chunk_on_silence_boundaries(str(audio_path), str(chunk_dir))

        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[0].end_s, 22.0, delta=0.25)
        self.assertAlmostEqual(chunks[1].start_s, 22.0, delta=0.25)

    def test_no_chunk_exceeds_max_chunk_duration(self):
        audio_path = self.tmpdir / "max_duration.wav"
        chunk_dir = self.tmpdir / "chunks_b"
        _write_wave(audio_path, [("tone", 95.0, 0.4)])

        chunks = chunk_on_silence_boundaries(str(audio_path), str(chunk_dir))

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.duration_s <= MAX_CHUNK_DURATION_S + 0.05 for chunk in chunks))

    def test_chunks_are_contiguous_with_no_audio_dropped(self):
        audio_path = self.tmpdir / "contiguous.wav"
        chunk_dir = self.tmpdir / "chunks_c"
        _write_wave(
            audio_path,
            [
                ("tone", 12.0, 0.5),
                ("silence", 1.0, 0.0),
                ("tone", 12.0, 0.5),
                ("silence", 1.0, 0.0),
                ("tone", 12.0, 0.5),
            ],
        )

        chunks = chunk_on_silence_boundaries(str(audio_path), str(chunk_dir))

        self.assertTrue(chunks)
        self.assertAlmostEqual(chunks[0].start_s, 0.0, delta=0.02)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(previous.end_s, current.start_s, delta=0.05)
        self.assertAlmostEqual(chunks[-1].end_s, 38.0, delta=0.15)

    def test_very_short_silence_stretches_do_not_produce_tiny_chunks(self):
        audio_path = self.tmpdir / "short_silence.wav"
        chunk_dir = self.tmpdir / "chunks_d"
        _write_wave(
            audio_path,
            [
                ("tone", 5.0, 0.5),
                ("silence", 0.2, 0.0),
                ("tone", 5.0, 0.5),
                ("silence", 1.0, 0.0),
                ("tone", 5.0, 0.5),
            ],
        )

        chunks = chunk_on_silence_boundaries(str(audio_path), str(chunk_dir))

        self.assertLessEqual(len(chunks), 2)
        self.assertTrue(all(chunk.duration_s >= 3.0 for chunk in chunks))

    def test_continuous_speech_without_silence_falls_back_to_max_duration_ceiling(self):
        audio_path = self.tmpdir / "continuous.wav"
        chunk_dir = self.tmpdir / "chunks_e"
        _write_wave(audio_path, [("tone", 65.0, 0.5)])

        chunks = chunk_on_silence_boundaries(str(audio_path), str(chunk_dir))

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(chunk.duration_s <= MAX_CHUNK_DURATION_S + 0.05 for chunk in chunks))
        self.assertAlmostEqual(chunks[1].start_s, chunks[0].end_s, delta=0.05)
        self.assertAlmostEqual(chunks[2].start_s, chunks[1].end_s, delta=0.05)
