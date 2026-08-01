"""Test d'integration de la sauvegarde frontale reutilisable."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import reporting.front_session as front_session


def build_recorded_frames():
    time_values = np.arange(0.0, 10.0, 1.0 / 30.0)
    centers = (2.0, 5.0, 8.0)
    frames = []

    for frame_index, time_s in enumerate(time_values):
        movement = max(
            max(0.0, 1.0 - abs(time_s - center) / 1.0)
            for center in centers
        )
        left_deviation = -6.0 * movement
        right_deviation = -4.0 * movement
        frames.append(
            {
                "frame": frame_index,
                "time_s": time_s,
                "left_fppa": 178.0 - 18.0 * movement,
                "right_fppa": 178.0 - 14.0 * movement,
                "pelvis_y": 0.30 + 0.15 * movement,
                "visibility_check": "OK",
                "left_alignment": "valgus" if movement > 0.34 else "neutral",
                "right_alignment": "valgus" if movement > 0.5 else "neutral",
                "left_alignment_index": left_deviation / 100.0,
                "right_alignment_index": right_deviation / 100.0,
                "left_signed_deviation_percent": left_deviation,
                "right_signed_deviation_percent": right_deviation,
                "left_knee_deviation_percent": abs(left_deviation),
                "right_knee_deviation_percent": abs(right_deviation),
            }
        )
    return frames


class FrontSessionSavingTests(unittest.TestCase):
    def test_saves_complete_front_video_session(self):
        baseline = {
            "pelvis_y_mean": 0.30,
            "pelvis_y_std": 0.001,
            "pelvis_y_n": 90,
            "left_fppa_mean": 178.0,
            "left_fppa_std": 0.2,
            "left_fppa_n": 90,
            "right_fppa_mean": 178.0,
            "right_fppa_std": 0.2,
            "right_fppa_n": 90,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            original_results_folder = front_session.RESULTS_FOLDER
            front_session.RESULTS_FOLDER = temporary_directory
            try:
                paths = front_session.save_front_session(
                    build_recorded_frames(),
                    baseline,
                    source="video_file",
                    acquisition_metadata={"source_file_name": "front.mp4"},
                )
            finally:
                front_session.RESULTS_FOLDER = original_results_folder

            results = pd.read_csv(paths["results_csv"])
            repetitions = pd.read_csv(paths["repetitions_csv"])
            metadata = Path(paths["metadata_txt"]).read_text(encoding="utf-8")

            self.assertEqual(len(repetitions), 3)
            self.assertIn("pelvis_velocity", results.columns)
            self.assertIn("source : video_file", metadata)
            self.assertIn("source_file_name : front.mp4", metadata)
            self.assertIn("repetitions_detected : 3", metadata)
            self.assertTrue(Path(paths["clinical_report_txt"]).exists())


if __name__ == "__main__":
    unittest.main()
