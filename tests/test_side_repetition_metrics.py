"""Tests des métriques par répétition du squat latéral."""

import unittest

import numpy as np
import pandas as pd

from movement_analysis.side_repetition_metrics import (
    SIDE_REPETITION_METRIC_COLUMNS,
    build_side_repetition_metrics,
)


def build_processed_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": np.arange(7),
            "time_s": np.arange(7, dtype=float) * 0.5,
            "knee_flexion_filtered_deg": [0, 10, 50, 80, 40, 10, 0],
            "knee_angular_velocity_deg_s": [20, 50, 70, -10, -70, -40, -20],
            "hip_flexion_filtered_deg": [0, 8, 42, 70, 50, 12, 0],
            "trunk_flexion_filtered_deg": [0, 2, 10, 15, 18, 5, 0],
            "ankle_dorsiflexion_filtered_deg": [0, 3, 8, 10, 7, 2, 0],
            "foot_inclination_filtered_deg": [0, 5, 15, 21, 12, 3, 0],
            "quality_valid": [True] * 7,
            "any_signal_was_interpolated": [False, False, True]
            + [False] * 4,
        }
    )


def build_repetitions_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rep_id": 1,
                "start_frame": 1,
                "bottom_frame": 3,
                "end_frame": 5,
                "start_time_s": 0.5,
                "bottom_time_s": 1.5,
                "end_time_s": 2.5,
            }
        ]
    )


class SideRepetitionMetricTests(unittest.TestCase):
    def test_builds_primary_and_secondary_metrics(self):
        metrics = build_side_repetition_metrics(
            build_processed_dataframe(),
            build_repetitions_dataframe(),
        )

        self.assertEqual(len(metrics), 1)
        repetition = metrics.iloc[0]

        self.assertEqual(repetition["peak_knee_flexion_deg"], 80.0)
        self.assertEqual(repetition["peak_hip_flexion_deg"], 70.0)
        self.assertEqual(repetition["peak_trunk_flexion_deg"], 18.0)
        self.assertEqual(repetition["peak_trunk_flexion_phase"], "ascent")
        self.assertEqual(
            repetition["peak_trunk_flexion_cycle_percent"],
            75.0,
        )
        self.assertEqual(repetition["peak_ankle_dorsiflexion_deg"], 10.0)
        self.assertEqual(repetition["peak_foot_inclination_deg"], 21.0)
        self.assertEqual(
            repetition["heel_lift_screening"],
            "above_experimental_threshold",
        )
        self.assertEqual(repetition["descent_ascent_time_ratio"], 1.0)
        self.assertEqual(
            repetition["mean_knee_descent_velocity_deg_s"],
            70.0,
        )
        self.assertEqual(
            repetition["mean_knee_ascent_velocity_deg_s"],
            70.0,
        )
        self.assertEqual(repetition["quality_percent"], 100.0)
        self.assertGreater(repetition["interpolated_percent"], 0.0)

    def test_threshold_is_configurable(self):
        metrics = build_side_repetition_metrics(
            build_processed_dataframe(),
            build_repetitions_dataframe(),
            heel_lift_threshold_deg=25.0,
        )

        self.assertEqual(
            metrics.iloc[0]["heel_lift_screening"],
            "below_experimental_threshold",
        )

    def test_empty_result_keeps_stable_columns(self):
        repetitions = build_repetitions_dataframe().iloc[0:0]
        metrics = build_side_repetition_metrics(
            build_processed_dataframe(),
            repetitions,
        )

        self.assertTrue(metrics.empty)
        self.assertEqual(
            list(metrics.columns),
            SIDE_REPETITION_METRIC_COLUMNS,
        )

    def test_missing_signal_is_rejected(self):
        processed = build_processed_dataframe().drop(
            columns=["trunk_flexion_filtered_deg"]
        )

        with self.assertRaisesRegex(ValueError, "Colonnes sagittales"):
            build_side_repetition_metrics(
                processed,
                build_repetitions_dataframe(),
            )


if __name__ == "__main__":
    unittest.main()
