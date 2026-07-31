"""Tests du premier bloc de traitement des signaux sagittaux."""

import unittest

import numpy as np
import pandas as pd

from signal_processing.side_signal_processing import (
    SideSignalProcessingConfig,
    compute_angular_velocity,
    compute_smoothing_window_frames,
    interpolate_short_gaps,
    process_side_signals,
)


def build_baseline(sample_count: int = 60) -> dict:
    return {
        "knee_flexion_mean": 2.0,
        "knee_flexion_n": sample_count,
        "hip_flexion_mean": -1.0,
        "hip_flexion_n": sample_count,
        "trunk_flexion_mean": -2.0,
        "trunk_flexion_n": sample_count,
        "ankle_internal_angle_mean": 90.0,
        "ankle_internal_angle_n": sample_count,
        "foot_inclination_mean": -3.0,
        "foot_inclination_n": sample_count,
    }


def build_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": range(7),
            "time_s": np.arange(7, dtype=float) * 0.05,
            "visibility": ["OK"] * 7,
            "visibility_min": [0.9] * 7,
            "knee_flexion_deg": [2, 12, 22, 32, 22, 12, 2],
            "hip_flexion_deg": [-1, 9, 19, 29, 19, 9, -1],
            "trunk_flexion_deg": [-2, 3, 8, 13, 8, 3, -2],
            "ankle_internal_angle_deg": [90, 88, 86, 84, 86, 88, 90],
            "foot_inclination_deg": [-3, -2, 0, 3, 0, -2, -3],
        }
    )


class SideSignalProcessingTests(unittest.TestCase):
    def test_relative_angles_use_complete_baseline(self):
        raw = build_raw_dataframe()
        processed = process_side_signals(
            raw,
            build_baseline(),
        )

        self.assertAlmostEqual(
            processed.loc[0, "knee_flexion_relative_deg"],
            0.0,
        )
        self.assertAlmostEqual(
            processed.loc[3, "knee_flexion_relative_deg"],
            30.0,
        )
        self.assertAlmostEqual(
            processed.loc[3, "hip_flexion_relative_deg"],
            30.0,
        )
        self.assertAlmostEqual(
            processed.loc[3, "trunk_flexion_relative_deg"],
            15.0,
        )
        self.assertAlmostEqual(
            processed.loc[3, "ankle_dorsiflexion_deg"],
            6.0,
        )
        self.assertAlmostEqual(
            processed.loc[3, "foot_inclination_relative_deg"],
            6.0,
        )
        self.assertIn("knee_angular_velocity_deg_s", processed.columns)
        self.assertTrue(
            np.isfinite(processed["knee_angular_velocity_deg_s"]).all()
        )

    def test_angular_velocity_sign_follows_movement_phase(self):
        velocity = compute_angular_velocity(
            [0.0, 10.0, 20.0, 10.0, 0.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )

        np.testing.assert_allclose(
            velocity,
            [10.0, 10.0, 0.0, -10.0, -10.0],
        )

    def test_angular_velocity_does_not_cross_missing_gap(self):
        velocity = compute_angular_velocity(
            [0.0, 10.0, np.nan, 30.0, 40.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )

        self.assertTrue(np.isnan(velocity[2]))
        np.testing.assert_allclose(
            velocity[[0, 1, 3, 4]],
            [10.0, 10.0, 10.0, 10.0],
        )

    def test_raw_dataframe_is_not_modified(self):
        raw = build_raw_dataframe()
        original = raw.copy(deep=True)

        process_side_signals(
            raw,
            build_baseline(),
        )

        pd.testing.assert_frame_equal(raw, original)

    def test_short_internal_quality_gap_is_interpolated(self):
        raw = build_raw_dataframe()
        raw.loc[3, "visibility"] = "Visibilité insuffisante"
        raw.loc[3, "visibility_min"] = 0.2

        processed = process_side_signals(
            raw,
            build_baseline(),
        )

        self.assertFalse(processed.loc[3, "quality_valid"])
        self.assertTrue(
            processed.loc[3, "knee_flexion_was_interpolated"]
        )
        self.assertAlmostEqual(
            processed.loc[3, "knee_flexion_interpolated_deg"],
            20.0,
        )
        self.assertTrue(
            processed.loc[3, "any_signal_was_interpolated"]
        )

    def test_long_gap_remains_missing(self):
        raw = build_raw_dataframe()
        raw.loc[1:5, "visibility"] = "Visibilité insuffisante"
        raw.loc[1:5, "visibility_min"] = 0.2

        processed = process_side_signals(
            raw,
            build_baseline(),
            SideSignalProcessingConfig(
                max_interpolation_gap_s=0.20,
            ),
        )

        self.assertTrue(
            processed.loc[1:5, "knee_flexion_interpolated_deg"]
            .isna()
            .all()
        )
        self.assertTrue(
            processed.loc[1:5, "knee_flexion_filtered_deg"]
            .isna()
            .all()
        )
        self.assertTrue(
            processed.loc[1:5, "knee_angular_velocity_deg_s"]
            .isna()
            .all()
        )
        self.assertFalse(
            processed.loc[1:5, "signal_processing_valid"].any()
        )

    def test_edge_gap_is_not_interpolated(self):
        values = [np.nan, 1.0, 2.0, 3.0]
        times = [0.0, 0.05, 0.10, 0.15]

        interpolated, mask = interpolate_short_gaps(
            values,
            times,
            max_gap_duration_s=0.25,
        )

        self.assertTrue(np.isnan(interpolated[0]))
        self.assertFalse(mask[0])

    def test_angle_wraparound_stays_small(self):
        raw = build_raw_dataframe()
        raw["foot_inclination_deg"] = [
            179.0,
            -179.0,
            -178.0,
            -177.0,
            -178.0,
            -179.0,
            179.0,
        ]
        baseline = build_baseline()
        baseline["foot_inclination_mean"] = 179.0

        processed = process_side_signals(raw, baseline)

        self.assertAlmostEqual(
            processed.loc[1, "foot_inclination_relative_deg"],
            2.0,
        )

    def test_invalid_or_incomplete_baseline_is_rejected(self):
        raw = build_raw_dataframe()
        baseline = build_baseline(sample_count=2)

        with self.assertRaisesRegex(
            ValueError,
            "Baseline sagittale invalide",
        ):
            process_side_signals(raw, baseline)

    def test_smoothing_window_is_odd(self):
        window_size = compute_smoothing_window_frames(
            np.arange(20, dtype=float) * 0.04,
            smoothing_window_s=0.25,
        )

        self.assertEqual(window_size, 5)


if __name__ == "__main__":
    unittest.main()
