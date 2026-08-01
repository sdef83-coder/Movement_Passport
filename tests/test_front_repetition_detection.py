"""Tests de la segmentation des repetitions du squat frontal."""

import unittest

import numpy as np
import pandas as pd

from movement_segmentation.repetition_detection import (
    build_repetitions_from_adaptive_baseline,
)


class FrontRepetitionDetectionTests(unittest.TestCase):
    def test_neighboring_squats_are_not_merged(self):
        signal = np.array(
            [
                0.30,
                0.30,
                0.33,
                0.45,
                0.34,
                0.31,
                0.34,
                0.45,
                0.33,
                0.30,
                0.30,
            ]
        )
        dataframe = pd.DataFrame(
            {
                "frame": np.arange(len(signal)),
                "time_s": np.arange(len(signal)) * 0.25,
            }
        )

        repetitions = build_repetitions_from_adaptive_baseline(
            df=dataframe,
            filtered_pelvis_y=signal,
            bottom_peaks=[3, 7],
            baseline_values={"pelvis_y_mean": 0.30},
            amplitude_fraction=0.10,
        )

        self.assertEqual(len(repetitions), 2)
        self.assertEqual(repetitions[0]["end_frame"], 5)
        self.assertEqual(repetitions[1]["start_frame"], 5)
        self.assertLessEqual(
            repetitions[0]["end_frame"],
            repetitions[1]["start_frame"],
        )

    def test_incomplete_repetition_at_recording_edge_is_rejected(self):
        signal = np.array([0.35, 0.45, 0.36, 0.30, 0.30])
        dataframe = pd.DataFrame(
            {
                "frame": np.arange(len(signal)),
                "time_s": np.arange(len(signal)) * 0.25,
            }
        )

        repetitions = build_repetitions_from_adaptive_baseline(
            df=dataframe,
            filtered_pelvis_y=signal,
            bottom_peaks=[1],
            baseline_values={"pelvis_y_mean": 0.30},
        )

        self.assertEqual(repetitions, [])


if __name__ == "__main__":
    unittest.main()
