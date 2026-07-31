"""Test d'intégration de la sauvegarde sagittale enrichie."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.squat_side_view as side_script
from protocol.side_session_setup import (
    SIDE_MARKERS,
    build_baseline_visibility_summary,
)


def build_recorded_frames() -> list[dict]:
    time_values = np.arange(0.0, 3.05, 0.05)
    movement = 90.0 * np.maximum(
        0.0,
        1.0 - np.abs(time_values - 1.5) / 1.5,
    )
    frames = []

    for frame, (time_s, knee_flexion) in enumerate(
        zip(time_values, movement)
    ):
        frames.append(
            {
                "frame": frame,
                "time_s": time_s,
                "side": "right",
                "facing_direction": "right",
                "knee_flexion_deg": 2.0 + knee_flexion,
                "hip_flexion_deg": -1.0 + knee_flexion * 0.8,
                "trunk_flexion_deg": -2.0 + knee_flexion * 0.2,
                "ankle_internal_angle_deg": 90.0 - knee_flexion * 0.15,
                "foot_inclination_deg": -3.0 + knee_flexion * 0.05,
                "visibility": "OK",
                "visibility_min": 0.9,
                "lowest_visibility_marker": "shoulder",
                **{
                    f"{marker_name}_visibility": 0.9
                    for marker_name in SIDE_MARKERS
                },
            }
        )

    return frames


def build_baseline() -> dict:
    return {
        "knee_flexion_mean": 2.0,
        "knee_flexion_std": 0.1,
        "knee_flexion_n": 20,
        "hip_flexion_mean": -1.0,
        "hip_flexion_std": 0.1,
        "hip_flexion_n": 20,
        "trunk_flexion_mean": -2.0,
        "trunk_flexion_std": 0.1,
        "trunk_flexion_n": 20,
        "ankle_internal_angle_mean": 90.0,
        "ankle_internal_angle_std": 0.1,
        "ankle_internal_angle_n": 20,
        "foot_inclination_mean": -3.0,
        "foot_inclination_std": 0.1,
        "foot_inclination_n": 20,
    }


class SideSessionSavingTests(unittest.TestCase):
    def test_saves_selected_side_and_marker_quality(self):
        baseline_visibility = build_baseline_visibility_summary(
            build_recorded_frames()[:20]
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            original_results_folder = side_script.RESULTS_FOLDER
            side_script.RESULTS_FOLDER = temporary_directory

            try:
                saved_paths = side_script.save_side_raw_session(
                    build_recorded_frames(),
                    build_baseline(),
                    analysis_side="right",
                    baseline_visibility_summary=baseline_visibility,
                    failed_baseline_attempts=1,
                )
            finally:
                side_script.RESULTS_FOLDER = original_results_folder

            raw_dataframe = pd.read_csv(saved_paths[0])
            metadata_text = Path(saved_paths[-1]).read_text(
                encoding="utf-8"
            )

            self.assertIn("heel_visibility", raw_dataframe.columns)
            self.assertIn("analysis_side : right", metadata_text)
            self.assertIn("failed_baseline_attempts : 1", metadata_text)
            self.assertIn(
                "baseline_heel_visibility_mean : 0.9",
                metadata_text,
            )


if __name__ == "__main__":
    unittest.main()
