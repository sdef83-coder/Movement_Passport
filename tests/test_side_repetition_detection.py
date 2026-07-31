"""Tests du segmentateur des répétitions du squat latéral."""

import unittest

import numpy as np
import pandas as pd

from movement_segmentation.side_repetition_detection import (
    SideRepetitionDetectionConfig,
    build_side_repetitions_dataframe,
    detect_side_repetitions,
)


def build_synthetic_processed_dataframe(
    peaks: list[tuple[float, float]],
    duration_s: float = 10.0,
    sampling_interval_s: float = 0.05,
) -> pd.DataFrame:
    """Crée des squats triangulaires définis par (temps, amplitude)."""

    time_values = np.arange(
        0.0,
        duration_s + sampling_interval_s / 2,
        sampling_interval_s,
    )
    signal = np.zeros(len(time_values), dtype=float)

    for peak_time, amplitude in peaks:
        half_duration_s = 0.75
        triangle = amplitude * np.maximum(
            0.0,
            1.0 - np.abs(time_values - peak_time) / half_duration_s,
        )
        signal += triangle

    return pd.DataFrame(
        {
            "frame": np.arange(len(time_values)),
            "time_s": time_values,
            "knee_flexion_filtered_deg": signal,
            "quality_valid": [True] * len(time_values),
            "any_signal_was_interpolated": [False] * len(time_values),
        }
    )


class SideRepetitionDetectionTests(unittest.TestCase):
    def test_detects_multiple_amplitudes_without_overlap(self):
        dataframe = build_synthetic_processed_dataframe(
            [
                (2.0, 90.0),
                (5.0, 45.0),
                (8.0, 125.0),
            ]
        )

        repetitions = detect_side_repetitions(dataframe)

        self.assertEqual(len(repetitions), 3)
        self.assertEqual(
            [rep["peak_knee_flexion_deg"] for rep in repetitions],
            [90.0, 45.0, 125.0],
        )

        for previous_rep, next_rep in zip(
            repetitions[:-1],
            repetitions[1:],
        ):
            self.assertLessEqual(
                previous_rep["end_frame"],
                next_rep["start_frame"],
            )

    def test_rejects_motion_below_minimum_amplitude(self):
        dataframe = build_synthetic_processed_dataframe(
            [(3.0, 12.0)]
        )

        repetitions = detect_side_repetitions(dataframe)

        self.assertEqual(repetitions, [])

    def test_rejects_incomplete_motion_at_session_edge(self):
        dataframe = build_synthetic_processed_dataframe(
            [(0.25, 90.0)]
        )

        repetitions = detect_side_repetitions(dataframe)

        self.assertEqual(repetitions, [])

    def test_rejects_repetition_with_insufficient_quality(self):
        dataframe = build_synthetic_processed_dataframe(
            [(3.0, 90.0)]
        )
        movement_mask = dataframe["knee_flexion_filtered_deg"] > 5.0
        invalid_indices = dataframe.index[movement_mask][::2]
        dataframe.loc[invalid_indices, "quality_valid"] = False

        repetitions = detect_side_repetitions(dataframe)

        self.assertEqual(repetitions, [])

    def test_reports_interpolated_percentage(self):
        dataframe = build_synthetic_processed_dataframe(
            [(3.0, 90.0)]
        )
        interpolated_indices = dataframe.index[
            (dataframe["time_s"] >= 2.8)
            & (dataframe["time_s"] <= 3.0)
        ]
        dataframe.loc[
            interpolated_indices,
            "any_signal_was_interpolated",
        ] = True

        repetitions = detect_side_repetitions(dataframe)

        self.assertEqual(len(repetitions), 1)
        self.assertGreater(
            repetitions[0]["interpolated_percent"],
            0.0,
        )

    def test_empty_result_has_stable_columns(self):
        dataframe = build_side_repetitions_dataframe([])

        self.assertTrue(dataframe.empty)
        self.assertIn("rep_id", dataframe.columns)
        self.assertIn("peak_knee_flexion_deg", dataframe.columns)

    def test_missing_signal_column_is_rejected(self):
        dataframe = build_synthetic_processed_dataframe(
            [(3.0, 90.0)]
        ).drop(columns=["knee_flexion_filtered_deg"])

        with self.assertRaisesRegex(
            ValueError,
            "Colonnes nécessaires",
        ):
            detect_side_repetitions(dataframe)

    def test_custom_minimum_peak_is_respected(self):
        dataframe = build_synthetic_processed_dataframe(
            [(3.0, 35.0)]
        )
        config = SideRepetitionDetectionConfig(
            min_peak_flexion_deg=40.0,
        )

        self.assertEqual(
            detect_side_repetitions(dataframe, config),
            [],
        )


if __name__ == "__main__":
    unittest.main()
