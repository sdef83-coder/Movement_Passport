"""Annotations de controle d'une video de squat frontal."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


def _format(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{numeric:.1f}" if np.isfinite(numeric) else "--"


def _point(landmarks, index, width, height):
    landmark = landmarks.landmark[index]
    return (
        int(np.clip(round(landmark.x * width), 0, width - 1)),
        int(np.clip(round(landmark.y * height), 0, height - 1)),
    )


def draw_front_video_annotation(
    image,
    pose_landmarks,
    pose_landmark,
    results,
    phase,
    timestamp_s,
):
    """Dessine le squelette, les jambes mesurees et les indicateurs."""

    annotated = image.copy()
    height, width = annotated.shape[:2]
    scale = max(0.55, min(1.1, width / 1000.0))
    thickness = max(1, int(round(scale * 2)))

    if pose_landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=(
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            ),
        )
        side_specs = (
            (
                "G",
                (0, 220, 0),
                pose_landmark.LEFT_HIP.value,
                pose_landmark.LEFT_KNEE.value,
                pose_landmark.LEFT_ANKLE.value,
            ),
            (
                "D",
                (255, 120, 0),
                pose_landmark.RIGHT_HIP.value,
                pose_landmark.RIGHT_KNEE.value,
                pose_landmark.RIGHT_ANKLE.value,
            ),
        )
        for label, color, hip_index, knee_index, ankle_index in side_specs:
            hip = _point(pose_landmarks, hip_index, width, height)
            knee = _point(pose_landmarks, knee_index, width, height)
            ankle = _point(pose_landmarks, ankle_index, width, height)
            cv2.line(annotated, hip, knee, color, thickness + 2, cv2.LINE_AA)
            cv2.line(annotated, knee, ankle, color, thickness + 2, cv2.LINE_AA)
            for point in (hip, knee, ankle):
                cv2.circle(annotated, point, max(5, thickness + 4), color, -1)
            cv2.putText(
                annotated,
                label,
                (knee[0] + 8, knee[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale * 0.65,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    panel_width = min(width - 20, max(340, int(round(465 * scale))))
    line_height = max(24, int(round(31 * scale)))
    lines = (
        f"{phase}  |  t = {timestamp_s:.2f} s",
        f"FPPA gauche : {_format(results.get('left_fppa'))} deg",
        f"FPPA droit : {_format(results.get('right_fppa'))} deg",
        "Deviation G : "
        f"{_format(results.get('left_signed_deviation_percent'))} %  |  "
        f"{results.get('left_alignment', 'not_detected')}",
        "Deviation D : "
        f"{_format(results.get('right_signed_deviation_percent'))} %  |  "
        f"{results.get('right_alignment', 'not_detected')}",
        f"Qualite : {results.get('visibility', 'non detecte')}",
    )
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 25 + line_height * len(lines)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.64, annotated, 0.36, 0.0, annotated)
    for line_index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (22, 10 + line_height * (line_index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.60,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated
