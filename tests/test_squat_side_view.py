"""Tests d'intégration de l'analyse frame par frame du squat latéral."""

import unittest
from dataclasses import dataclass
from enum import IntEnum

from movement_analysis.squat_side_view import analyze_squat_side_view


class FakePoseLandmark(IntEnum):
    LEFT_SHOULDER = 0
    LEFT_HIP = 1
    LEFT_KNEE = 2
    LEFT_ANKLE = 3
    LEFT_HEEL = 4
    LEFT_FOOT_INDEX = 5


@dataclass
class FakeLandmark:
    x: float
    y: float
    visibility: float = 1.0


def build_left_side_landmarks(
    coordinates: list[tuple[float, float]],
) -> list[FakeLandmark]:
    """Construit une liste compatible avec les landmarks MediaPipe."""

    return [
        FakeLandmark(x=x, y=y)
        for x, y in coordinates
    ]


class SquatSideViewTests(unittest.TestCase):
    def test_auto_direction_preserves_mirrored_squat_measurements(self):
        right_facing_landmarks = build_left_side_landmarks(
            [
                (1.0, -1.0),  # shoulder
                (0.0, 0.0),  # hip
                (1.0, 1.0),  # knee
                (0.0, 2.0),  # ankle
                (-0.25, 2.0),  # heel
                (0.75, 2.0),  # foot index
            ]
        )
        left_facing_landmarks = build_left_side_landmarks(
            [
                (1.0, -1.0),  # shoulder
                (2.0, 0.0),  # hip
                (1.0, 1.0),  # knee
                (2.0, 2.0),  # ankle
                (2.25, 2.0),  # heel
                (1.25, 2.0),  # foot index
            ]
        )

        right_results = analyze_squat_side_view(
            right_facing_landmarks,
            FakePoseLandmark,
            side="left",
            facing_direction="auto",
            neutral_ankle_angle=90.0,
            neutral_foot_inclination=0.0,
        )
        left_results = analyze_squat_side_view(
            left_facing_landmarks,
            FakePoseLandmark,
            side="left",
            facing_direction="auto",
            neutral_ankle_angle=90.0,
            neutral_foot_inclination=0.0,
        )

        self.assertEqual(right_results["facing_direction"], "right")
        self.assertEqual(left_results["facing_direction"], "left")

        for results in (right_results, left_results):
            self.assertAlmostEqual(
                results["knee_flexion_deg"],
                90.0,
            )
            self.assertAlmostEqual(
                results["hip_flexion_deg"],
                90.0,
            )
            self.assertAlmostEqual(
                results["trunk_flexion_deg"],
                45.0,
            )
            self.assertAlmostEqual(
                results["ankle_dorsiflexion_deg"],
                45.0,
            )
            self.assertAlmostEqual(
                results["foot_inclination_deg"],
                0.0,
            )
            self.assertAlmostEqual(
                results["foot_inclination_relative_deg"],
                0.0,
            )
            self.assertEqual(results["visibility"], "OK")
            self.assertEqual(
                results["lowest_visibility_marker"],
                "shoulder",
            )
            self.assertEqual(results["heel_visibility"], 1.0)

    def test_reports_the_lowest_visibility_marker(self):
        landmarks = build_left_side_landmarks(
            [
                (1.0, -1.0),
                (0.0, 0.0),
                (1.0, 1.0),
                (0.0, 2.0),
                (-0.25, 2.0),
                (0.75, 2.0),
            ]
        )
        landmarks[FakePoseLandmark.LEFT_HEEL].visibility = 0.25

        results = analyze_squat_side_view(
            landmarks,
            FakePoseLandmark,
            side="left",
        )

        self.assertEqual(results["visibility"], "Visibilité insuffisante")
        self.assertEqual(results["lowest_visibility_marker"], "heel")
        self.assertEqual(results["visibility_min"], 0.25)


if __name__ == "__main__":
    unittest.main()
