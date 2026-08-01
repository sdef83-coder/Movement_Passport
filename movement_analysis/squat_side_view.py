"""Extraction frame par frame des mesures d'un squat en vue latérale.

Ce module contient le code métier indépendant de la webcam et de l'interface.
Il sélectionne le côté anatomique demandé, calcule les angles sagittaux et
retourne les indicateurs de qualité associés.
"""

from __future__ import annotations

import numpy as np

from movement_analysis.sagittal_angles import (
    compute_ankle_dorsiflexion,
    compute_ankle_internal_angle,
    compute_foot_inclination,
    compute_joint_internal_angle,
    compute_relative_angle,
    compute_signed_joint_flexion,
    compute_trunk_flexion,
    infer_facing_direction,
)


def _get_side_landmarks(
    landmarks,
    pose_landmark,
    side: str,
):
    """Sélectionne les repères MediaPipe du côté anatomique demandé."""

    if side == "left":
        return {
            "shoulder": landmarks[pose_landmark.LEFT_SHOULDER.value],
            "hip": landmarks[pose_landmark.LEFT_HIP.value],
            "knee": landmarks[pose_landmark.LEFT_KNEE.value],
            "ankle": landmarks[pose_landmark.LEFT_ANKLE.value],
            "heel": landmarks[pose_landmark.LEFT_HEEL.value],
            "foot_index": landmarks[pose_landmark.LEFT_FOOT_INDEX.value],
        }

    if side == "right":
        return {
            "shoulder": landmarks[pose_landmark.RIGHT_SHOULDER.value],
            "hip": landmarks[pose_landmark.RIGHT_HIP.value],
            "knee": landmarks[pose_landmark.RIGHT_KNEE.value],
            "ankle": landmarks[pose_landmark.RIGHT_ANKLE.value],
            "heel": landmarks[pose_landmark.RIGHT_HEEL.value],
            "foot_index": landmarks[pose_landmark.RIGHT_FOOT_INDEX.value],
        }

    raise ValueError("side doit être égal à 'left' ou 'right'.")


def _xy(
    landmark,
    image_width: float,
    image_height: float,
) -> tuple[float, float]:
    """Convertit un repère MediaPipe normalisé en coordonnées pixels."""

    return (
        float(landmark.x) * image_width,
        float(landmark.y) * image_height,
    )


def analyze_squat_side_view(
    landmarks,
    pose_landmark,
    *,
    image_width: float,
    image_height: float,
    side: str = "left",
    facing_direction: str = "auto",
    neutral_ankle_angle: float = np.nan,
    neutral_foot_inclination: float = np.nan,
    visibility_threshold: float = 0.5,
) -> dict[str, float | str]:
    """Calcule les angles sagittaux du squat pour un côté du corps."""

    if (
        not np.isfinite(image_width)
        or not np.isfinite(image_height)
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError(
            "image_width et image_height doivent etre strictement positifs."
        )

    selected = _get_side_landmarks(
        landmarks,
        pose_landmark,
        side,
    )

    shoulder_xy = _xy(selected["shoulder"], image_width, image_height)
    hip_xy = _xy(selected["hip"], image_width, image_height)
    knee_xy = _xy(selected["knee"], image_width, image_height)
    ankle_xy = _xy(selected["ankle"], image_width, image_height)
    heel_xy = _xy(selected["heel"], image_width, image_height)
    foot_index_xy = _xy(
        selected["foot_index"],
        image_width,
        image_height,
    )

    if facing_direction == "auto":
        resolved_facing_direction = infer_facing_direction(
            heel_xy,
            foot_index_xy,
        )
    elif facing_direction in {"left", "right"}:
        resolved_facing_direction = facing_direction
    else:
        raise ValueError(
            "facing_direction doit être égal à 'auto', 'left' ou 'right'."
        )

    knee_internal_angle = compute_joint_internal_angle(
        hip_xy,
        knee_xy,
        ankle_xy,
    )
    knee_flexion = compute_signed_joint_flexion(
        hip_xy,
        knee_xy,
        ankle_xy,
        resolved_facing_direction,
    )

    hip_internal_angle = compute_joint_internal_angle(
        knee_xy,
        hip_xy,
        shoulder_xy,
    )
    hip_flexion = compute_signed_joint_flexion(
        knee_xy,
        hip_xy,
        shoulder_xy,
        resolved_facing_direction,
    )

    trunk_flexion = compute_trunk_flexion(
        hip_xy,
        shoulder_xy,
        resolved_facing_direction,
    )

    ankle_internal_angle = compute_ankle_internal_angle(
        heel_xy,
        foot_index_xy,
        ankle_xy,
        knee_xy,
    )
    ankle_dorsiflexion = compute_ankle_dorsiflexion(
        ankle_internal_angle,
        neutral_ankle_angle,
    )

    foot_inclination = compute_foot_inclination(
        heel_xy,
        foot_index_xy,
        resolved_facing_direction,
    )
    foot_inclination_relative = compute_relative_angle(
        foot_inclination,
        neutral_foot_inclination,
    )

    visibility_by_marker = {
        marker_name: float(marker.visibility)
        for marker_name, marker in selected.items()
    }
    lowest_visibility_marker = min(
        visibility_by_marker,
        key=visibility_by_marker.get,
    )
    visibility_values = list(visibility_by_marker.values())

    visibility_min = float(np.min(visibility_values))
    visibility_check = (
        "OK"
        if visibility_min >= visibility_threshold
        else "Visibilité insuffisante"
    )

    return {
        "side": side,
        "facing_direction_requested": facing_direction,
        "facing_direction": resolved_facing_direction,
        "knee_internal_angle_deg": knee_internal_angle,
        "knee_flexion_deg": knee_flexion,
        "hip_internal_angle_deg": hip_internal_angle,
        "hip_flexion_deg": hip_flexion,
        "trunk_flexion_deg": trunk_flexion,
        "ankle_internal_angle_deg": ankle_internal_angle,
        "ankle_dorsiflexion_deg": ankle_dorsiflexion,
        "foot_inclination_deg": foot_inclination,
        "foot_inclination_relative_deg": foot_inclination_relative,
        "visibility": visibility_check,
        "visibility_min": visibility_min,
        "lowest_visibility_marker": lowest_visibility_marker,
        **{
            f"{marker_name}_visibility": visibility_value
            for marker_name, visibility_value in visibility_by_marker.items()
        },
    }
