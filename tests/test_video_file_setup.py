"""Tests de la sélection d'une vidéo et de la baseline associée."""

import tempfile
import unittest
from pathlib import Path

from protocol.video_file_setup import (
    choose_baseline_start,
    choose_video_file,
    validate_video_path,
)


class VideoFileSetupTests(unittest.TestCase):
    def test_accepts_supported_file_and_strips_quotes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "squat test.MP4"
            video_path.touch()

            resolved = validate_video_path(f'"{video_path}"')

            self.assertEqual(resolved, video_path.resolve())

    def test_rejects_missing_or_unsupported_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            text_path = Path(temporary_directory) / "notes.txt"
            text_path.touch()

            with self.assertRaisesRegex(ValueError, "Format vidéo"):
                validate_video_path(text_path)

            with self.assertRaisesRegex(ValueError, "introuvable"):
                validate_video_path(Path(temporary_directory) / "missing.mp4")

    def test_uses_dialog_selection_without_console_prompt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "squat.mov"
            video_path.touch()

            selected = choose_video_file(
                input_function=lambda _: self.fail(
                    "La console ne devait pas être sollicitée."
                ),
                dialog_function=lambda: str(video_path),
            )

            self.assertEqual(selected, video_path.resolve())

    def test_baseline_start_accepts_default_comma_and_retries(self):
        self.assertEqual(choose_baseline_start(lambda _: ""), 0.0)
        self.assertEqual(choose_baseline_start(lambda _: "2,5"), 2.5)

        answers = iter(["texte", "-1", "1.25"])
        self.assertEqual(
            choose_baseline_start(lambda _: next(answers)),
            1.25,
        )


if __name__ == "__main__":
    unittest.main()
