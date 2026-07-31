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
    def test_angles_are_invariant_to_portrait_and_landscape_aspect_ratio(self):
        pixel_coordinates = [
            (420.0, 200.0),  # shoulder
            (320.0, 450.0),  # hip
            (520.0, 650.0),  # knee
            (400.0, 900.0),  # ankle
            (360.0, 940.0),  # heel
            (560.0, 940.0),  # foot index
        ]

        results_by_format = []
        for image_width, image_height in ((1080.0, 1920.0), (1920.0, 1080.0)):
            normalized_landmarks = build_left_side_landmarks(
                [
                    (x_value / image_width, y_value / image_height)
                    for x_value, y_value in pixel_coordinates
                ]
            )
            results_by_format.append(
                analyze_squat_side_view(
                    normalized_landmarks,
                    FakePoseLandmark,
                    image_width=image_width,
                    image_height=image_height,
                    side="left",
                    facing_direction="right",
                    neutral_ankle_angle=90.0,
                    neutral_foot_inclination=0.0,
                )
            )

        angle_keys = (
            "knee_flexion_deg",
            "hip_flexion_deg",
            "trunk_flexion_deg",
            "ankle_internal_angle_deg",
            "ankle_dorsiflexion_deg",
            "foot_inclination_deg",
        )
        for angle_key in angle_keys:
            self.assertAlmostEqual(
                results_by_format[0][angle_key],
                results_by_format[1][angle_key],
                places=7,
            )

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
            image_width=1.0,
            image_height=1.0,
            side="left",
            facing_direction="auto",
            neutral_ankle_angle=90.0,
            neutral_foot_inclination=0.0,
        )
        left_results = analyze_squat_side_view(
            left_facing_landmarks,
            FakePoseLandmark,
            image_width=1.0,
            image_height=1.0,
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
            image_width=1.0,
            image_height=1.0,
            side="left",
        )

        self.assertEqual(results["visibility"], "Visibilité insuffisante")
        self.assertEqual(results["lowest_visibility_marker"], "heel")
        self.assertEqual(results["visibility_min"], 0.25)


if __name__ == "__main__":
    unittest.main()
