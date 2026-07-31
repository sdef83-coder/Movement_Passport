"""Acquisition sagittale depuis une vidéo, indépendante de l'interface."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from baseline.baseline import BaselineRecorder
from movement_analysis.squat_side_view import analyze_squat_side_view
from protocol.side_session_setup import (
    SIDE_MARKERS,
    baseline_validation_errors,
    build_baseline_visibility_summary,
)
from protocol.video_file_setup import validate_video_path
from reporting.side_video_annotation import draw_side_video_annotation


ProgressCallback = Callable[[int, int, float], None]
VideoWriterFactory = Callable[[str, int, float, tuple[int, int]], Any]


@dataclass(frozen=True)
class SideVideoConfig:
    """Paramètres du protocole appliqué à une vidéo enregistrée."""

    baseline_start_s: float = 0.0
    baseline_duration_s: float = 3.0
    min_valid_baseline_samples: int = 10
    visibility_threshold: float = 0.5
    facing_direction: str = "auto"

    def __post_init__(self) -> None:
        if self.baseline_start_s < 0:
            raise ValueError("baseline_start_s doit être positif ou nul.")
        if self.baseline_duration_s <= 0:
            raise ValueError("baseline_duration_s doit être positif.")
        if self.min_valid_baseline_samples < 1:
            raise ValueError("min_valid_baseline_samples doit être positif.")
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise ValueError(
                "visibility_threshold doit être compris entre 0 et 1."
            )
        if self.facing_direction not in {"auto", "left", "right"}:
            raise ValueError(
                "facing_direction doit être égal à 'auto', 'left' ou 'right'."
            )

    @property
    def baseline_end_s(self) -> float:
        return self.baseline_start_s + self.baseline_duration_s


@dataclass
class SideVideoAnalysisResult:
    """Données nécessaires au traitement commun d'une session sagittale."""

    recorded_frames: list[dict[str, float | str]]
    baseline_values: dict
    baseline_visibility_summary: dict
    acquisition_metadata: dict
    analysis_side: str


def _is_positive_finite(value: float) -> bool:
    try:
        return bool(np.isfinite(float(value)) and float(value) > 0.0)
    except (TypeError, ValueError):
        return False


def resolve_video_timestamp_s(
    capture: Any,
    frame_index: int,
    fps: float,
    previous_timestamp_s: float | None,
) -> tuple[float, str]:
    """Préfère le timestamp du fichier et utilise le FPS comme repli."""

    raw_position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
    timestamp_s = (
        float(raw_position_ms) / 1000.0
        if np.isfinite(raw_position_ms) and raw_position_ms >= 0.0
        else np.nan
    )
    method = "opencv_pos_msec"

    timestamp_is_usable = np.isfinite(timestamp_s)

    if previous_timestamp_s is not None:
        timestamp_is_usable = (
            timestamp_is_usable and timestamp_s > previous_timestamp_s
        )

    if not timestamp_is_usable:
        effective_fps = float(fps) if _is_positive_finite(fps) else 30.0
        timestamp_s = float(frame_index) / effective_fps
        method = "frame_index_fps_fallback"

        if (
            previous_timestamp_s is not None
            and timestamp_s <= previous_timestamp_s
        ):
            timestamp_s = previous_timestamp_s + 1.0 / effective_fps

    return float(timestamp_s), method


def _create_empty_results(analysis_side: str) -> dict[str, float | str]:
    return {
        "side": analysis_side,
        "facing_direction": "not_detected",
        "knee_internal_angle_deg": np.nan,
        "knee_flexion_deg": np.nan,
        "hip_internal_angle_deg": np.nan,
        "hip_flexion_deg": np.nan,
        "trunk_flexion_deg": np.nan,
        "ankle_internal_angle_deg": np.nan,
        "ankle_dorsiflexion_deg": np.nan,
        "foot_inclination_deg": np.nan,
        "foot_inclination_relative_deg": np.nan,
        "visibility": "Aucun squelette détecté",
        "visibility_min": np.nan,
        "lowest_visibility_marker": "pose_not_detected",
        **{
            f"{marker_name}_visibility": np.nan
            for marker_name in SIDE_MARKERS
        },
    }


def _build_visibility_record(results: dict) -> dict:
    return {
        "lowest_visibility_marker": results.get(
            "lowest_visibility_marker",
            "pose_not_detected",
        ),
        **{
            f"{marker_name}_visibility": results.get(
                f"{marker_name}_visibility",
                np.nan,
            )
            for marker_name in SIDE_MARKERS
        },
    }


def _add_baseline_frame(
    recorder: BaselineRecorder,
    results: dict,
) -> None:
    ankle_internal_angle = results.get(
        "ankle_internal_angle_deg",
        np.nan,
    )

    if (
        results.get("visibility") != "OK"
        or not np.isfinite(ankle_internal_angle)
    ):
        return

    recorder.add_frame(
        knee_flexion=results.get("knee_flexion_deg", np.nan),
        hip_flexion=results.get("hip_flexion_deg", np.nan),
        trunk_flexion=results.get("trunk_flexion_deg", np.nan),
        ankle_internal_angle=ankle_internal_angle,
        foot_inclination=results.get("foot_inclination_deg", np.nan),
    )


def _build_recorded_frame(
    frame_index: int,
    timestamp_s: float,
    results: dict,
    analysis_side: str,
) -> dict[str, float | str]:
    return {
        "frame": frame_index,
        "time_s": timestamp_s,
        "side": results.get("side", analysis_side),
        "facing_direction": results.get(
            "facing_direction",
            "not_detected",
        ),
        "knee_internal_angle_deg": results.get(
            "knee_internal_angle_deg",
            np.nan,
        ),
        "knee_flexion_deg": results.get("knee_flexion_deg", np.nan),
        "hip_internal_angle_deg": results.get(
            "hip_internal_angle_deg",
            np.nan,
        ),
        "hip_flexion_deg": results.get("hip_flexion_deg", np.nan),
        "trunk_flexion_deg": results.get("trunk_flexion_deg", np.nan),
        "ankle_internal_angle_deg": results.get(
            "ankle_internal_angle_deg",
            np.nan,
        ),
        "ankle_dorsiflexion_deg": results.get(
            "ankle_dorsiflexion_deg",
            np.nan,
        ),
        "foot_inclination_deg": results.get(
            "foot_inclination_deg",
            np.nan,
        ),
        "foot_inclination_relative_deg": results.get(
            "foot_inclination_relative_deg",
            np.nan,
        ),
        "visibility": results.get(
            "visibility",
            "Aucun squelette détecté",
        ),
        "visibility_min": results.get("visibility_min", np.nan),
        "lowest_visibility_marker": results.get(
            "lowest_visibility_marker",
            "pose_not_detected",
        ),
        **{
            f"{marker_name}_visibility": results.get(
                f"{marker_name}_visibility",
                np.nan,
            )
            for marker_name in SIDE_MARKERS
        },
    }


def _build_annotation_results(
    results: dict,
    baseline_values: dict,
    baseline_finalized: bool,
) -> dict:
    """Aligne les angles affiches sur les angles relatifs des rapports."""

    if not baseline_finalized:
        return results

    annotation_results = dict(results)
    baseline_keys = {
        "knee_flexion_deg": "knee_flexion_mean",
        "hip_flexion_deg": "hip_flexion_mean",
        "trunk_flexion_deg": "trunk_flexion_mean",
    }
    for angle_key, baseline_key in baseline_keys.items():
        angle_value = results.get(angle_key, np.nan)
        baseline_value = baseline_values.get(baseline_key, np.nan)
        if np.isfinite(angle_value) and np.isfinite(baseline_value):
            annotation_results[angle_key] = (
                float(angle_value) - float(baseline_value)
            )

    return annotation_results


def _default_pose_factory():
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _default_video_writer_factory(
    output_path: str,
    fourcc: int,
    fps: float,
    frame_size: tuple[int, int],
):
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def analyze_side_video(
    video_path: str | Path,
    analysis_side: str,
    config: SideVideoConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    pose_factory: Callable[[], Any] | None = None,
    annotated_video_path: str | Path | None = None,
    video_writer_factory: VideoWriterFactory = _default_video_writer_factory,
) -> SideVideoAnalysisResult:
    """Extrait la baseline et les frames de mouvement d'une vidéo."""

    if analysis_side not in {"left", "right"}:
        raise ValueError("analysis_side doit être égal à 'left' ou 'right'.")

    resolved_path = validate_video_path(video_path)
    config = config or SideVideoConfig()
    capture = capture_factory(str(resolved_path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {resolved_path}")

    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = (
        int(round(total_frames_value))
        if _is_positive_finite(total_frames_value)
        else 0
    )
    source_duration_s = (
        total_frames / fps
        if total_frames > 0 and _is_positive_finite(fps)
        else np.nan
    )

    if (
        np.isfinite(source_duration_s)
        and source_duration_s <= config.baseline_end_s
    ):
        capture.release()
        raise ValueError(
            "La vidéo se termine avant la fin de la baseline. "
            f"Il faut au moins {config.baseline_end_s:.1f} secondes, "
            "puis du temps pour réaliser les squats."
        )

    pose = (pose_factory or _default_pose_factory)()
    mp_pose = mp.solutions.pose
    baseline_recorder = BaselineRecorder()
    baseline_visibility_records: list[dict] = []
    facing_direction_votes: list[str] = []
    baseline_values: dict = {}
    baseline_visibility_summary: dict = {}
    neutral_ankle_angle = np.nan
    neutral_foot_inclination = np.nan
    resolved_facing_direction: str | None = None
    baseline_finalized = False
    recorded_frames: list[dict[str, float | str]] = []
    timestamp_methods: set[str] = set()
    previous_timestamp_s: float | None = None
    source_frame_index = 0
    last_timestamp_s = np.nan
    video_writer = None
    annotated_frames_written = 0
    resolved_annotated_path = (
        Path(annotated_video_path).expanduser().resolve()
        if annotated_video_path is not None
        else None
    )

    try:
        while capture.isOpened():
            success, image = capture.read()

            if not success:
                break

            timestamp_s, timestamp_method = resolve_video_timestamp_s(
                capture,
                source_frame_index,
                fps,
                previous_timestamp_s,
            )
            previous_timestamp_s = timestamp_s
            last_timestamp_s = timestamp_s
            timestamp_methods.add(timestamp_method)

            if progress_callback is not None:
                progress_callback(
                    source_frame_index + 1,
                    total_frames,
                    timestamp_s,
                )

            results = _create_empty_results(analysis_side)
            image_height, image_width = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)

            if pose_results.pose_landmarks:
                results = analyze_squat_side_view(
                    pose_results.pose_landmarks.landmark,
                    mp_pose.PoseLandmark,
                    image_width=image_width,
                    image_height=image_height,
                    side=analysis_side,
                    facing_direction=(
                        resolved_facing_direction or config.facing_direction
                    ),
                    neutral_ankle_angle=neutral_ankle_angle,
                    neutral_foot_inclination=neutral_foot_inclination,
                    visibility_threshold=config.visibility_threshold,
                )

            if timestamp_s < config.baseline_start_s:
                phase = "AVANT BASELINE"
            elif timestamp_s < config.baseline_end_s:
                phase = "BASELINE"
            else:
                phase = "MOUVEMENT"

            if resolved_annotated_path is not None and video_writer is None:
                resolved_annotated_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                output_fps = fps if _is_positive_finite(fps) else 30.0
                video_writer = video_writer_factory(
                    str(resolved_annotated_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(output_fps),
                    (image_width, image_height),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(
                        "Impossible de creer la video annotee : "
                        f"{resolved_annotated_path}"
                    )

            if timestamp_s < config.baseline_start_s:
                if video_writer is not None:
                    video_writer.write(
                        draw_side_video_annotation(
                            image,
                            pose_results.pose_landmarks,
                            mp_pose.PoseLandmark,
                            results,
                            analysis_side,
                            phase,
                            timestamp_s,
                        )
                    )
                    annotated_frames_written += 1
                source_frame_index += 1
                continue

            if timestamp_s < config.baseline_end_s:
                _add_baseline_frame(baseline_recorder, results)
                baseline_visibility_records.append(
                    _build_visibility_record(results)
                )
                detected_direction = results.get("facing_direction")

                if detected_direction in {"left", "right"}:
                    facing_direction_votes.append(str(detected_direction))

                if video_writer is not None:
                    video_writer.write(
                        draw_side_video_annotation(
                            image,
                            pose_results.pose_landmarks,
                            mp_pose.PoseLandmark,
                            results,
                            analysis_side,
                            phase,
                            timestamp_s,
                        )
                    )
                    annotated_frames_written += 1

                source_frame_index += 1
                continue

            if not baseline_finalized:
                baseline_values = baseline_recorder.compute()
                baseline_visibility_summary = (
                    build_baseline_visibility_summary(
                        baseline_visibility_records,
                        config.visibility_threshold,
                    )
                )
                baseline_errors = baseline_validation_errors(
                    baseline_values,
                    config.min_valid_baseline_samples,
                )

                if baseline_errors:
                    raise ValueError(
                        "Baseline vidéo invalide : "
                        + "; ".join(baseline_errors)
                    )

                neutral_ankle_angle = float(
                    baseline_values["ankle_internal_angle_mean"]
                )
                neutral_foot_inclination = float(
                    baseline_values["foot_inclination_mean"]
                )

                if config.facing_direction == "auto":
                    if not facing_direction_votes:
                        raise ValueError(
                            "Baseline vidéo invalide : direction du corps "
                            "non détectée."
                        )
                    resolved_facing_direction = Counter(
                        facing_direction_votes
                    ).most_common(1)[0][0]
                else:
                    resolved_facing_direction = config.facing_direction

                baseline_finalized = True

                if pose_results.pose_landmarks:
                    results = analyze_squat_side_view(
                        pose_results.pose_landmarks.landmark,
                        mp_pose.PoseLandmark,
                        image_width=image_width,
                        image_height=image_height,
                        side=analysis_side,
                        facing_direction=resolved_facing_direction,
                        neutral_ankle_angle=neutral_ankle_angle,
                        neutral_foot_inclination=neutral_foot_inclination,
                        visibility_threshold=config.visibility_threshold,
                    )

            recorded_frames.append(
                _build_recorded_frame(
                    len(recorded_frames),
                    timestamp_s,
                    results,
                    analysis_side,
                )
            )
            if video_writer is not None:
                annotation_results = _build_annotation_results(
                    results,
                    baseline_values,
                    baseline_finalized,
                )
                video_writer.write(
                    draw_side_video_annotation(
                        image,
                        pose_results.pose_landmarks,
                        mp_pose.PoseLandmark,
                        annotation_results,
                        analysis_side,
                        phase,
                        timestamp_s,
                    )
                )
                annotated_frames_written += 1
            source_frame_index += 1
    finally:
        capture.release()
        pose.close()
        if video_writer is not None:
            video_writer.release()

    if not baseline_finalized:
        raise ValueError(
            "La vidéo ne contient pas une baseline complète suivie "
            "d'une phase de mouvement."
        )

    if not recorded_frames:
        raise ValueError("Aucune frame de mouvement n'a pu être enregistrée.")

    timestamp_method = (
        next(iter(timestamp_methods))
        if len(timestamp_methods) == 1
        else "opencv_pos_msec_with_fps_fallback"
    )
    acquisition_metadata = {
        "source_file_name": resolved_path.name,
        "source_file_extension": resolved_path.suffix.lower(),
        "source_fps": round(fps, 3) if _is_positive_finite(fps) else np.nan,
        "source_total_frames": total_frames,
        "source_width_px": (
            int(round(width_value)) if _is_positive_finite(width_value) else 0
        ),
        "source_height_px": (
            int(round(height_value))
            if _is_positive_finite(height_value)
            else 0
        ),
        "source_duration_s": (
            round(float(source_duration_s), 3)
            if np.isfinite(source_duration_s)
            else round(float(last_timestamp_s), 3)
        ),
        "video_timestamp_method": timestamp_method,
        "video_orientation_auto_requested": hasattr(
            cv2,
            "CAP_PROP_ORIENTATION_AUTO",
        ),
        "baseline_start_s_in_source": config.baseline_start_s,
        "baseline_duration_s": config.baseline_duration_s,
        "baseline_end_s_in_source": config.baseline_end_s,
        "recording_start_s_in_source": round(
            float(recorded_frames[0]["time_s"]),
            3,
        ),
        "source_frames_read": source_frame_index,
        "annotated_video_file": (
            resolved_annotated_path.name
            if resolved_annotated_path is not None
            else "not_requested"
        ),
        "annotated_video_status": (
            "created" if annotated_frames_written > 0 else "not_requested"
        ),
        "annotated_video_frames": annotated_frames_written,
        "annotation_content": (
            "mediapipe_pose_selected_side_markers_angles_and_visibility"
        ),
        "angle_coordinate_system": (
            "pixel_coordinates_aspect_ratio_corrected"
        ),
    }

    return SideVideoAnalysisResult(
        recorded_frames=recorded_frames,
        baseline_values=baseline_values,
        baseline_visibility_summary=baseline_visibility_summary,
        acquisition_metadata=acquisition_metadata,
        analysis_side=analysis_side,
    )
