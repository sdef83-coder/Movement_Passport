from biomechanics.fppa import calculate_fppa
from vision.landmarks import get_visibility
from quality.checks import check_marker_visibility
from movement_analysis.frontal_alignement import compute_frontal_alignment
import numpy as np


def analyze_fppa_front_view(landmarks, pose_landmark):
    """
    Analyse un test FPPA en vue frontale.

    Retourne toutes les informations utiles
    concernant ce test.
    """
    left_hip = landmarks[pose_landmark.LEFT_HIP.value]
    left_knee = landmarks[pose_landmark.LEFT_KNEE.value]
    left_ankle = landmarks[pose_landmark.LEFT_ANKLE.value]

    right_hip = landmarks[pose_landmark.RIGHT_HIP.value]
    right_knee = landmarks[pose_landmark.RIGHT_KNEE.value]
    right_ankle = landmarks[pose_landmark.RIGHT_ANKLE.value]

    (
        left_alignment,
        left_alignment_index,
        left_signed_deviation_percent,
        left_knee_deviation_percent,
    ) = compute_frontal_alignment(
        (left_hip.x, left_hip.y),
        (left_knee.x, left_knee.y),
        (left_ankle.x, left_ankle.y),
        side="left",
    )

    (
        right_alignment,
        right_alignment_index,
        right_signed_deviation_percent,
        right_knee_deviation_percent,
    ) = compute_frontal_alignment(
        (right_hip.x, right_hip.y),
        (right_knee.x, right_knee.y),
        (right_ankle.x, right_ankle.y),
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
        )

        right_fppa = calculate_fppa(
            landmarks,
            pose_landmark,
            "right",
        )

    else:

        left_fppa = np.nan
        right_fppa = np.nan

    return {
        "left_fppa": left_fppa,
        "right_fppa": right_fppa,
        "visibility": visibility,
        "left_alignment": left_alignment,
        "right_alignment": right_alignment,
        "left_alignment_index": left_alignment_index,
        "right_alignment_index": right_alignment_index,
        "left_signed_deviation_percent": left_signed_deviation_percent,
        "right_signed_deviation_percent": right_signed_deviation_percent,
        "left_knee_deviation_percent": left_knee_deviation_percent,
        "right_knee_deviation_percent": right_knee_deviation_percent,
    }
