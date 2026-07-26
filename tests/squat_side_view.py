import numpy as np

from movement_analysis.sagittal_angles import (
    compute_joint_angle,
    compute_segment_angle_from_vertical,
)


def analyze_squat_side_view(
    landmarks,
    pose_landmark,
    side="left",
):
    """
    Analyse sagittale du squat pour le côté visible.
    """

    if side == "left":
        shoulder = landmarks[pose_landmark.LEFT_SHOULDER.value]
        hip = landmarks[pose_landmark.LEFT_HIP.value]
        knee = landmarks[pose_landmark.LEFT_KNEE.value]
        ankle = landmarks[pose_landmark.LEFT_ANKLE.value]
        heel = landmarks[pose_landmark.LEFT_HEEL.value]
        foot_index = landmarks[pose_landmark.LEFT_FOOT_INDEX.value]

    elif side == "right":
        shoulder = landmarks[pose_landmark.RIGHT_SHOULDER.value]
        hip = landmarks[pose_landmark.RIGHT_HIP.value]
        knee = landmarks[pose_landmark.RIGHT_KNEE.value]
        ankle = landmarks[pose_landmark.RIGHT_ANKLE.value]
        heel = landmarks[pose_landmark.RIGHT_HEEL.value]
        foot_index = landmarks[pose_landmark.RIGHT_FOOT_INDEX.value]

    else:
        raise ValueError("side doit être 'left' ou 'right'.")

    shoulder_xy = (shoulder.x, shoulder.y)
    hip_xy = (hip.x, hip.y)
    knee_xy = (knee.x, knee.y)
    ankle_xy = (ankle.x, ankle.y)
    heel_xy = (heel.x, heel.y)
    foot_index_xy = (foot_index.x, foot_index.y)

    knee_internal_angle = compute_joint_angle(
        hip_xy,
        knee_xy,
        ankle_xy,
    )

    hip_internal_angle = compute_joint_angle(
        shoulder_xy,
        hip_xy,
        knee_xy,
    )

    trunk_inclination = compute_segment_angle_from_vertical(
        hip_xy,
        shoulder_xy,
    )

    tibia_inclination = compute_segment_angle_from_vertical(
        ankle_xy,
        knee_xy,
    )

    foot_angle = compute_segment_angle_from_vertical(
        heel_xy,
        foot_index_xy,
    )

    visibility_values = [
        shoulder.visibility,
        hip.visibility,
        knee.visibility,
        ankle.visibility,
        heel.visibility,
        foot_index.visibility,
    ]

    visibility_min = float(np.min(visibility_values))

    visibility_check = (
        "OK" if visibility_min >= 0.5 else "Visibilité insuffisante"
    )

    return {
        "knee_internal_angle": knee_internal_angle,
        "knee_flexion_angle": 180 - knee_internal_angle,
        "hip_internal_angle": hip_internal_angle,
        "hip_flexion_proxy": 180 - hip_internal_angle,
        "trunk_inclination_deg": trunk_inclination,
        "tibia_inclination_deg": tibia_inclination,
        "foot_angle_from_vertical_deg": foot_angle,
        "visibility": visibility_check,
        "visibility_min": visibility_min,
    }
