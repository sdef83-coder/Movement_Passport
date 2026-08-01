"""Mesures frame par frame du squat en vue frontale."""

from __future__ import annotations

import numpy as np

from biomechanics.fppa import calculate_fppa
from movement_analysis.frontal_alignement import compute_frontal_alignment
from quality.checks import check_marker_visibility
from vision.landmarks import get_visibility


def _pixel_xy(landmark, image_width, image_height):
    return (
        float(landmark.x) * image_width,
        float(landmark.y) * image_height,
    )


def analyze_fppa_front_view(
    landmarks,
    pose_landmark,
    *,
    image_width: float,
    image_height: float,
):
    """Analyse le FPPA et l'alignement frontal des deux genoux."""

    if (
        not np.isfinite(image_width)
        or not np.isfinite(image_height)
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError(
            "image_width et image_height doivent etre strictement positifs."
        )

    left_hip = landmarks[pose_landmark.LEFT_HIP.value]
    left_knee = landmarks[pose_landmark.LEFT_KNEE.value]
    left_ankle = landmarks[pose_landmark.LEFT_ANKLE.value]
    right_hip = landmarks[pose_landmark.RIGHT_HIP.value]
    right_knee = landmarks[pose_landmark.RIGHT_KNEE.value]
    right_ankle = landmarks[pose_landmark.RIGHT_ANKLE.value]

    left_alignment_values = compute_frontal_alignment(
        _pixel_xy(left_hip, image_width, image_height),
        _pixel_xy(left_knee, image_width, image_height),
        _pixel_xy(left_ankle, image_width, image_height),
        side="left",
    )
    right_alignment_values = compute_frontal_alignment(
        _pixel_xy(right_hip, image_width, image_height),
        _pixel_xy(right_knee, image_width, image_height),
        _pixel_xy(right_ankle, image_width, image_height),
        side="right",
    )

    visibilities = [
        get_visibility(landmarks, pose_landmark.LEFT_HIP),
        get_visibility(landmarks, pose_landmark.LEFT_KNEE),
        get_visibility(landmarks, pose_landmark.LEFT_ANKLE),
        get_visibility(landmarks, pose_landmark.RIGHT_HIP),
        get_visibility(landmarks, pose_landmark.RIGHT_KNEE),
        get_visibility(landmarks, pose_landmark.RIGHT_ANKLE),
    ]
    visibility = check_marker_visibility(visibilities)

    if visibility == "OK":
        left_fppa = calculate_fppa(
            landmarks,
            pose_landmark,
            "left",
            image_width=image_width,
            image_height=image_height,
        )
        right_fppa = calculate_fppa(
            landmarks,
            pose_landmark,
            "right",
            image_width=image_width,
            image_height=image_height,
        )
    else:
        left_fppa = np.nan
        right_fppa = np.nan

    return {
        "left_fppa": left_fppa,
        "right_fppa": right_fppa,
        "visibility": visibility,
        "left_alignment": left_alignment_values[0],
        "right_alignment": right_alignment_values[0],
        "left_alignment_index": left_alignment_values[1],
        "right_alignment_index": right_alignment_values[1],
        "left_signed_deviation_percent": left_alignment_values[2],
        "right_signed_deviation_percent": right_alignment_values[2],
        "left_knee_deviation_percent": left_alignment_values[3],
        "right_knee_deviation_percent": right_alignment_values[3],
    }
