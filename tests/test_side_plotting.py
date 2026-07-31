"""Tests de génération du graphique sagittal."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from reporting.side_plotting import plot_side_analysis


def build_plot_dataframe() -> pd.DataFrame:
    time_values = np.linspace(0.0, 4.0, 81)
    movement = 80 * np.maximum(
        0.0,
        1.0 - np.abs(time_values - 2.0) / 1.0,
    )

    return pd.DataFrame(
        {
            "frame": np.arange(len(time_values)),
            "time_s": time_values,
            "knee_flexion_relative_deg": movement,
            "knee_flexion_filtered_deg": movement,
            "knee_angular_velocity_deg_s": np.gradient(
                movement,
                time_values,
            ),
            "hip_flexion_filtered_deg": movement * 0.95,
            "trunk_flexion_filtered_deg": movement * 0.25,
            "ankle_dorsiflexion_filtered_deg": movement * 0.20,
            "foot_inclination_relative_deg": movement * 0.05,
            "foot_inclination_filtered_deg": movement * 0.05,
            "signal_processing_valid": [True] * len(time_values),
        }
    )


class SidePlottingTests(unittest.TestCase):
    def test_creates_non_empty_png(self):
        dataframe = build_plot_dataframe()
        repetitions = [
            {
                "rep_id": 1,
                "start_time_s": 1.0,
                "bottom_time_s": 2.0,
                "end_time_s": 3.0,
                "bottom_frame": 40,
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            figure_path = plot_side_analysis(
                dataframe,
                repetitions,
                temporary_directory,
            )

            self.assertTrue(os.path.exists(figure_path))
            self.assertGreater(os.path.getsize(figure_path), 10_000)

    def test_missing_column_is_rejected(self):
        dataframe = build_plot_dataframe().drop(
            columns=["foot_inclination_filtered_deg"]
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError,
                "Colonnes nécessaires",
            ):
                plot_side_analysis(
                    dataframe,
                    [],
                    temporary_directory,
                )


if __name__ == "__main__":
    unittest.main()
