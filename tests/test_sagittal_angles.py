"""Tests unitaires des conventions géométriques sagittales."""

import math
import unittest

from movement_analysis.sagittal_angles import (
    compute_ankle_dorsiflexion,
    compute_ankle_internal_angle,
    compute_foot_inclination,
    compute_joint_internal_angle,
    compute_segment_orientation,
    compute_signed_joint_flexion,
    compute_trunk_flexion,
    compute_relative_angle,
    infer_facing_direction,
)


class SagittalAnglesTests(unittest.TestCase):
    """Vérifie les valeurs connues et l'invariance droite/gauche."""

    def assert_angle_almost_equal(
        self,
        actual: float,
        expected: float,
    ) -> None:
        self.assertAlmostEqual(actual, expected, places=7)

    def test_standing_joint_is_extended(self):
        hip = (0.0, 0.0)
        knee = (0.0, 1.0)
        ankle = (0.0, 2.0)

        self.assert_angle_almost_equal(
            compute_joint_internal_angle(hip, knee, ankle),
            180.0,
        )
        self.assert_angle_almost_equal(
            compute_signed_joint_flexion(
                hip,
                knee,
                ankle,
                facing_direction="right",
            ),
            0.0,
        )

    def test_right_facing_knee_flexion_is_positive(self):
        self.assert_angle_almost_equal(
            compute_signed_joint_flexion(
                point_a=(0.0, 0.0),
                vertex=(1.0, 1.0),
                point_c=(0.0, 2.0),
                facing_direction="right",
            ),
            90.0,
        )

    def test_left_facing_mirror_preserves_knee_flexion(self):
        self.assert_angle_almost_equal(
            compute_signed_joint_flexion(
                point_a=(2.0, 0.0),
                vertex=(1.0, 1.0),
                point_c=(2.0, 2.0),
                facing_direction="left",
            ),
            90.0,
        )

    def test_forward_trunk_flexion_is_positive_in_both_directions(self):
        right_facing = compute_trunk_flexion(
            hip=(0.0, 1.0),
            shoulder=(1.0, 0.0),
            facing_direction="right",
        )
        left_facing = compute_trunk_flexion(
            hip=(2.0, 1.0),
            shoulder=(1.0, 0.0),
            facing_direction="left",
        )

        self.assert_angle_almost_equal(right_facing, 45.0)
        self.assert_angle_almost_equal(left_facing, 45.0)

    def test_heel_raise_is_positive_in_both_directions(self):
        right_facing = compute_foot_inclination(
            heel=(0.0, 1.0),
            foot_index=(1.0, 2.0),
            facing_direction="right",
        )
        left_facing = compute_foot_inclination(
            heel=(2.0, 1.0),
            foot_index=(1.0, 2.0),
            facing_direction="left",
        )

        self.assert_angle_almost_equal(right_facing, 45.0)
        self.assert_angle_almost_equal(left_facing, 45.0)

    def test_dorsiflexion_is_relative_to_neutral_angle(self):
        self.assert_angle_almost_equal(
            compute_ankle_dorsiflexion(
                ankle_internal_angle=65.0,
                neutral_ankle_angle=90.0,
            ),
            25.0,
        )

    def test_facing_direction_is_inferred_from_foot_axis(self):
        self.assertEqual(
            infer_facing_direction(
                heel=(0.0, 1.0),
                foot_index=(1.0, 1.0),
            ),
            "right",
        )
        self.assertEqual(
            infer_facing_direction(
                heel=(2.0, 1.0),
                foot_index=(1.0, 1.0),
            ),
            "left",
        )

    def test_ankle_angle_uses_heel_to_forefoot_axis(self):
        right_facing = compute_ankle_internal_angle(
            heel=(0.0, 2.0),
            foot_index=(1.0, 2.0),
            ankle=(0.25, 2.0),
            knee=(0.25, 1.0),
        )
        left_facing = compute_ankle_internal_angle(
            heel=(2.0, 2.0),
            foot_index=(1.0, 2.0),
            ankle=(1.75, 2.0),
            knee=(1.75, 1.0),
        )

        self.assert_angle_almost_equal(right_facing, 90.0)
        self.assert_angle_almost_equal(left_facing, 90.0)

    def test_relative_angle_handles_wraparound(self):
        self.assert_angle_almost_equal(
            compute_relative_angle(
                current_angle=-179.0,
                reference_angle=179.0,
            ),
            2.0,
        )
        self.assert_angle_almost_equal(
            compute_relative_angle(
                current_angle=179.0,
                reference_angle=-179.0,
            ),
            -2.0,
        )

    def test_horizontal_and_vertical_segment_orientations(self):
        self.assert_angle_almost_equal(
            compute_segment_orientation((0.0, 0.0), (1.0, 0.0)),
            0.0,
        )
        self.assert_angle_almost_equal(
            compute_segment_orientation((0.0, 1.0), (0.0, 0.0)),
            90.0,
        )

    def test_degenerate_points_return_nan(self):
        self.assertTrue(
            math.isnan(
                compute_joint_internal_angle(
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (1.0, 1.0),
                )
            )
        )
        self.assertTrue(
            math.isnan(
                compute_trunk_flexion(
                    hip=(0.0, 0.0),
                    shoulder=(0.0, 0.0),
                    facing_direction="right",
                )
            )
        )

    def test_invalid_facing_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_trunk_flexion(
                hip=(0.0, 1.0),
                shoulder=(1.0, 0.0),
                facing_direction="up",
            )


if __name__ == "__main__":
    unittest.main()
