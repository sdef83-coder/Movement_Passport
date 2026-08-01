"""Tests des mesures du squat en vue frontale."""

import unittest
from dataclasses import dataclass
from enum import IntEnum

from movement_analysis.squat_front_view import analyze_fppa_front_view


class FakePoseLandmark(IntEnum):
    LEFT_HIP = 0
    LEFT_KNEE = 1
    LEFT_ANKLE = 2
    RIGHT_HIP = 3
    RIGHT_KNEE = 4
    RIGHT_ANKLE = 5


@dataclass
class FakeLandmark:
    x: float
    y: float
    visibility: float = 0.99


def build_normalized_landmarks(pixel_coordinates, width, height):
    return [
        FakeLandmark(x=x_value / width, y=y_value / height)
        for x_value, y_value in pixel_coordinates
    ]


class SquatFrontViewTests(unittest.TestCase):
    def test_angles_and_alignment_are_invariant_to_image_aspect_ratio(self):
        pixel_coordinates = [
            (420.0, 300.0),
            (450.0, 560.0),
            (400.0, 850.0),
            (660.0, 300.0),
            (630.0, 560.0),
            (680.0, 850.0),
        ]
        results_by_format = []

        for width, height in ((1080.0, 1920.0), (1920.0, 1080.0)):
            landmarks = build_normalized_landmarks(
                pixel_coordinates,
                width,
                height,
            )
            results_by_format.append(
                analyze_fppa_front_view(
                    landmarks,
                    FakePoseLandmark,
                    image_width=width,
                    image_height=height,
                )
            )

        numeric_keys = (
            "left_fppa",
            "right_fppa",
            "left_alignment_index",
            "right_alignment_index",
            "left_signed_deviation_percent",
            "right_signed_deviation_percent",
        )
        for key in numeric_keys:
            self.assertAlmostEqual(
                results_by_format[0][key],
                results_by_format[1][key],
                places=7,
            )

    def test_low_visibility_invalidates_fppa(self):
        landmarks = build_normalized_landmarks(
            [
                (420.0, 300.0),
                (450.0, 560.0),
                (400.0, 850.0),
                (660.0, 300.0),
                (630.0, 560.0),
                (680.0, 850.0),
            ],
            1080.0,
            1920.0,
        )
        landmarks[FakePoseLandmark.LEFT_KNEE].visibility = 0.2

        results = analyze_fppa_front_view(
            landmarks,
            FakePoseLandmark,
            image_width=1080.0,
            image_height=1920.0,
        )

        self.assertEqual(results["visibility"], "Visibilité insuffisante")
        self.assertNotEqual(results["left_fppa"], results["left_fppa"])


if __name__ == "__main__":
    unittest.main()
