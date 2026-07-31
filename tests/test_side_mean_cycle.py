"""Tests du cycle moyen du squat en vue latérale."""

import unittest

import numpy as np
import pandas as pd

from movement_analysis.side_mean_cycle import (
    build_side_mean_cycle,
    build_side_mean_cycle_dataframe,
    build_side_mean_cycle_summary_dataframe,
    summarize_side_mean_cycle,
)


def build_mean_cycle_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = np.arange(11)
    time_values = frames.astype(float) * 0.2
    triangle = 1.0 - np.abs(frames - 5) / 5.0
    knee = triangle * 100.0
    velocity = np.gradient(knee, time_values)
    processed = pd.DataFrame(
        {
            "frame": frames,
            "time_s": time_values,
            "knee_flexion_filtered_deg": knee,
            "hip_flexion_filtered_deg": triangle * 80.0,
            "trunk_flexion_filtered_deg": triangle * 20.0,
            "knee_angular_velocity_deg_s": velocity,
            "ankle_dorsiflexion_filtered_deg": triangle * 15.0,
            "foot_inclination_filtered_deg": triangle * 5.0,
        }
    )
    repetitions = pd.DataFrame(
        [
            {
                "rep_id": 1,
                "start_frame": 0,
                "bottom_frame": 5,
                "end_frame": 10,
                "start_time_s": 0.0,
                "bottom_time_s": 1.0,
                "end_time_s": 2.0,
            }
        ]
    )
    return processed, repetitions


class SideMeanCycleTests(unittest.TestCase):
    def test_builds_normalized_cycle_and_summary(self):
        processed, repetitions = build_mean_cycle_inputs()
        mean_cycle = build_side_mean_cycle(
            processed,
            repetitions,
            n_points=101,
        )

        self.assertIsNotNone(mean_cycle)
        self.assertEqual(mean_cycle["n_cycles"], 1)
        self.assertEqual(len(mean_cycle["cycle_percent"]), 101)
        self.assertAlmostEqual(
            mean_cycle["knee_flexion_mean"][50],
            100.0,
        )
        self.assertTrue(
            np.allclose(mean_cycle["knee_flexion_sd"], 0.0)
        )

        dataframe = build_side_mean_cycle_dataframe(mean_cycle)
        summary = summarize_side_mean_cycle(mean_cycle)
        summary_dataframe = build_side_mean_cycle_summary_dataframe(summary)

        self.assertEqual(len(dataframe), 101)
        self.assertIn("knee_velocity_mean", dataframe.columns)
        self.assertEqual(summary["mean_bottom_cycle_percent"], 50.0)
        self.assertEqual(summary["peak_knee_flexion_mean"], 100.0)
        self.assertEqual(len(summary_dataframe), 1)

    def test_empty_repetitions_return_no_cycle(self):
        processed, repetitions = build_mean_cycle_inputs()
        mean_cycle = build_side_mean_cycle(
            processed,
            repetitions.iloc[0:0],
        )

        self.assertIsNone(mean_cycle)
        self.assertTrue(build_side_mean_cycle_dataframe(mean_cycle).empty)

    def test_missing_signal_is_rejected(self):
        processed, repetitions = build_mean_cycle_inputs()
        processed = processed.drop(columns=["knee_angular_velocity_deg_s"])

        with self.assertRaisesRegex(ValueError, "cycle moyen manquantes"):
            build_side_mean_cycle(processed, repetitions)


if __name__ == "__main__":
    unittest.main()
