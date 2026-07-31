"""Tests du choix du côté et du contrôle de baseline sagittale."""

import unittest

from protocol.side_session_setup import (
    baseline_validation_errors,
    build_baseline_visibility_summary,
    choose_analysis_side,
)


def build_valid_baseline() -> dict:
    baseline = {}

    for signal_name in (
        "knee_flexion",
        "hip_flexion",
        "trunk_flexion",
        "ankle_internal_angle",
        "foot_inclination",
    ):
        baseline[f"{signal_name}_mean"] = 1.0
        baseline[f"{signal_name}_n"] = 30

    return baseline


class SideSessionSetupTests(unittest.TestCase):
    def test_selects_left_and_right_from_french_shortcuts(self):
        self.assertEqual(choose_analysis_side(lambda _: "G"), "left")
        self.assertEqual(choose_analysis_side(lambda _: "d"), "right")

    def test_repeats_prompt_after_invalid_choice(self):
        answers = iter(["x", "droite"])

        self.assertEqual(
            choose_analysis_side(lambda _: next(answers)),
            "right",
        )

    def test_baseline_requires_ten_valid_samples(self):
        baseline = build_valid_baseline()
        baseline["knee_flexion_n"] = 4

        errors = baseline_validation_errors(baseline)

        self.assertEqual(len(errors), 1)
        self.assertIn("knee_flexion", errors[0])
        self.assertEqual(
            baseline_validation_errors(build_valid_baseline()),
            [],
        )

    def test_visibility_summary_identifies_limiting_marker(self):
        records = []

        for _ in range(3):
            records.append(
                {
                    "shoulder_visibility": 0.9,
                    "hip_visibility": 0.9,
                    "knee_visibility": 0.9,
                    "ankle_visibility": 0.8,
                    "heel_visibility": 0.3,
                    "foot_index_visibility": 0.8,
                    "lowest_visibility_marker": "heel",
                }
            )

        summary = build_baseline_visibility_summary(records)

        self.assertEqual(summary["valid_frames"], 0)
        self.assertEqual(summary["valid_percent"], 0.0)
        self.assertEqual(
            summary["most_frequent_limiting_marker"],
            "heel",
        )
        self.assertEqual(summary["heel_below_threshold_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
