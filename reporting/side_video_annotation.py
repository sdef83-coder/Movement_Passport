"""Annotations visuelles de controle pour une video de squat sagittal."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np


SELECTED_MARKERS = (
    "shoulder",
    "hip",
    "knee",
    "ankle",
    "heel",
    "foot_index",
)

SELECTED_CONNECTIONS = (
    ("shoulder", "hip"),
    ("hip", "knee"),
    ("knee", "ankle"),
    ("ankle", "heel"),
    ("heel", "foot_index"),
    ("foot_index", "ankle"),
)


def _landmark_indexes(pose_landmark: Any, side: str) -> dict[str, int]:
    side_prefix = side.upper()
    return {
        marker: getattr(
            pose_landmark,
            f"{side_prefix}_{marker.upper()}",
        ).value
        for marker in SELECTED_MARKERS
    }


def _pixel_position(landmark: Any, width: int, height: int) -> tuple[int, int]:
    x_value = int(round(float(landmark.x) * width))
    y_value = int(round(float(landmark.y) * height))
    return (
        int(np.clip(x_value, 0, max(0, width - 1))),
        int(np.clip(y_value, 0, max(0, height - 1))),
    )


def _format_angle(value: Any) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{numeric_value:.1f}" if np.isfinite(numeric_value) else "--"


def draw_side_video_annotation(
    image: np.ndarray,
    pose_landmarks: Any | None,
    pose_landmark: Any,
    results: dict,
    analysis_side: str,
    phase: str,
    timestamp_s: float,
) -> np.ndarray:
    """Dessine le squelette, les points utilises et les angles calcules."""

    annotated = image.copy()
    height, width = annotated.shape[:2]
    scale = max(0.55, min(1.1, width / 1000.0))
    thickness = max(1, int(round(scale * 2)))
    selected_positions: dict[str, tuple[int, int]] = {}

    if pose_landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=(
                mp.solutions.drawing_styles
                .get_default_pose_landmarks_style()
            ),
        )
        indexes = _landmark_indexes(pose_landmark, analysis_side)
        selected_positions = {
            marker: _pixel_position(
                pose_landmarks.landmark[index],
                width,
                height,
            )
            for marker, index in indexes.items()
        }

        for first_marker, second_marker in SELECTED_CONNECTIONS:
            cv2.line(
                annotated,
                selected_positions[first_marker],
                selected_positions[second_marker],
                (0, 220, 0),
                max(2, thickness + 1),
                cv2.LINE_AA,
            )

        limiting_marker = str(
            results.get("lowest_visibility_marker", "")
        )
        for marker, position in selected_positions.items():
            marker_color = (
                (0, 0, 255) if marker == limiting_marker else (0, 255, 0)
            )
            cv2.circle(
                annotated,
                position,
                max(5, int(round(7 * scale))),
                marker_color,
                -1,
                cv2.LINE_AA,
            )

    panel_width = min(width - 20, max(330, int(round(430 * scale))))
    line_height = max(24, int(round(31 * scale)))
    panel_height = line_height * 9 + 20
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.64, annotated, 0.36, 0.0, annotated)

    visibility_value = results.get("visibility_min", np.nan)
    try:
        visibility_percent = float(visibility_value) * 100.0
    except (TypeError, ValueError):
        visibility_percent = np.nan
    lines = (
        (f"{phase}  |  t = {timestamp_s:.2f} s", (255, 255, 255)),
        (
            f"Cote analyse : {analysis_side}  |  "
            f"{'relatif baseline' if phase == 'MOUVEMENT' else 'brut'}",
            (255, 255, 255),
        ),
        (
            f"Genou : {_format_angle(results.get('knee_flexion_deg'))} deg",
            (80, 220, 255),
        ),
        (
            f"Hanche : {_format_angle(results.get('hip_flexion_deg'))} deg",
            (80, 220, 255),
        ),
        (
            f"Tronc : {_format_angle(results.get('trunk_flexion_deg'))} deg",
            (80, 220, 255),
        ),
        (
            "Dorsiflexion* : "
            f"{_format_angle(results.get('ankle_dorsiflexion_deg'))} deg",
            (255, 170, 80),
        ),
        (
            "Inclinaison pied* : "
            f"{_format_angle(results.get('foot_inclination_relative_deg'))} deg",
            (255, 170, 80),
        ),
        (
            f"Visibilite min. : {_format_angle(visibility_percent)} %  |  "
            f"{results.get('lowest_visibility_marker', 'aucun')}",
            (
                (80, 255, 80)
                if results.get("visibility") == "OK"
                else (80, 80, 255)
            ),
        ),
        ("* mesure secondaire exploratoire", (200, 200, 200)),
    )

    for line_index, (line, color) in enumerate(lines):
        y_position = 10 + line_height * (line_index + 1)
        cv2.putText(
            annotated,
            line,
            (22, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.62,
            color,
            thickness,
            cv2.LINE_AA,
        )

    joint_labels = {
        "knee": ("G", results.get("knee_flexion_deg")),
        "hip": ("H", results.get("hip_flexion_deg")),
        "ankle": ("D*", results.get("ankle_dorsiflexion_deg")),
    }
    for marker, (prefix, value) in joint_labels.items():
        if marker not in selected_positions:
            continue
        x_position, y_position = selected_positions[marker]
        cv2.putText(
            annotated,
            f"{prefix} {_format_angle(value)}",
            (min(width - 100, x_position + 10), max(20, y_position - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.55,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated
