"""Acquisition frontale depuis une video, independante de l'interface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from baseline.baseline import BaselineRecorder
from movement_analysis.squat_front_view import analyze_fppa_front_view
from protocol.video_file_setup import validate_video_path
from reporting.front_video_annotation import draw_front_video_annotation
from vision.landmarks import get_pelvis_y


ProgressCallback = Callable[[int, int, float], None]
VideoWriterFactory = Callable[[str, int, float, tuple[int, int]], Any]


@dataclass(frozen=True)
class FrontVideoConfig:
    """Parametres du protocole frontal applique a une video."""

    baseline_start_s: float = 0.0
    baseline_duration_s: float = 3.0
    min_valid_baseline_samples: int = 10

    def __post_init__(self) -> None:
        if self.baseline_start_s < 0:
            raise ValueError("baseline_start_s doit etre positif ou nul.")
        if self.baseline_duration_s <= 0:
            raise ValueError("baseline_duration_s doit etre positif.")
        if self.min_valid_baseline_samples < 1:
            raise ValueError(
                "min_valid_baseline_samples doit etre strictement positif."
            )

    @property
    def baseline_end_s(self) -> float:
        return self.baseline_start_s + self.baseline_duration_s


@dataclass
class FrontVideoAnalysisResult:
    recorded_frames: list[dict[str, Any]]
    baseline_values: dict
    acquisition_metadata: dict


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
    """Utilise le timestamp du fichier, puis le FPS comme repli."""

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
        if previous_timestamp_s is not None and timestamp_s <= previous_timestamp_s:
            timestamp_s = previous_timestamp_s + 1.0 / effective_fps

    return float(timestamp_s), method


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


def _empty_results() -> dict[str, Any]:
    return {
        "left_fppa": np.nan,
        "right_fppa": np.nan,
        "visibility": "Aucun squelette detecte",
        "left_alignment": "not_detected",
        "right_alignment": "not_detected",
        "left_alignment_index": np.nan,
        "right_alignment_index": np.nan,
        "left_signed_deviation_percent": np.nan,
        "right_signed_deviation_percent": np.nan,
        "left_knee_deviation_percent": np.nan,
        "right_knee_deviation_percent": np.nan,
    }


def _validate_baseline(baseline_values: dict, minimum_samples: int) -> None:
    required_signals = ("pelvis_y", "left_fppa", "right_fppa")
    errors = []
    for signal_name in required_signals:
        sample_count = int(baseline_values.get(f"{signal_name}_n", 0))
        mean_value = baseline_values.get(f"{signal_name}_mean", np.nan)
        if sample_count < minimum_samples or not np.isfinite(mean_value):
            errors.append(
                f"{signal_name}: {sample_count}/{minimum_samples} frames valides"
            )

    if errors:
        raise ValueError("Baseline video invalide : " + "; ".join(errors))


def analyze_front_video(
    video_path: str | Path,
    config: FrontVideoConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    pose_factory: Callable[[], Any] | None = None,
    annotated_video_path: str | Path | None = None,
    video_writer_factory: VideoWriterFactory = _default_video_writer_factory,
) -> FrontVideoAnalysisResult:
    """Extrait la baseline et les frames frontales d'une video."""

    resolved_path = validate_video_path(video_path)
    config = config or FrontVideoConfig()
    capture = capture_factory(str(resolved_path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Impossible d'ouvrir la video : {resolved_path}")

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

    if np.isfinite(source_duration_s) and source_duration_s <= config.baseline_end_s:
        capture.release()
        raise ValueError(
            "La video se termine avant la fin de la baseline. "
            f"Il faut au moins {config.baseline_end_s:.1f} secondes, "
            "puis du temps pour realiser les squats."
        )

    pose = (pose_factory or _default_pose_factory)()
    mp_pose = mp.solutions.pose
    baseline_recorder = BaselineRecorder()
    baseline_values: dict = {}
    baseline_finalized = False
    recorded_frames: list[dict[str, Any]] = []
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
                progress_callback(source_frame_index + 1, total_frames, timestamp_s)

            analysis_results = _empty_results()
            pelvis_y = np.nan
            image_height, image_width = image.shape[:2]
            pose_results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                pelvis_y = get_pelvis_y(landmarks, mp_pose.PoseLandmark)
                analysis_results = analyze_fppa_front_view(
                    landmarks,
                    mp_pose.PoseLandmark,
                    image_width=image_width,
                    image_height=image_height,
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
                        "Impossible de creer la video frontale annotee : "
                        f"{resolved_annotated_path}"
                    )

            if timestamp_s < config.baseline_start_s:
                if video_writer is not None:
                    video_writer.write(
                        draw_front_video_annotation(
                            image,
                            pose_results.pose_landmarks,
                            mp_pose.PoseLandmark,
                            analysis_results,
                            phase,
                            timestamp_s,
                        )
                    )
                    annotated_frames_written += 1
                source_frame_index += 1
                continue

            if timestamp_s < config.baseline_end_s:
                if analysis_results.get("visibility") == "OK":
                    baseline_recorder.add_frame(
                        pelvis_y=pelvis_y,
                        left_fppa=analysis_results.get("left_fppa", np.nan),
                        right_fppa=analysis_results.get("right_fppa", np.nan),
                    )
                if video_writer is not None:
                    video_writer.write(
                        draw_front_video_annotation(
                            image,
                            pose_results.pose_landmarks,
                            mp_pose.PoseLandmark,
                            analysis_results,
                            phase,
                            timestamp_s,
                        )
                    )
                    annotated_frames_written += 1
                source_frame_index += 1
                continue

            if not baseline_finalized:
                baseline_values = baseline_recorder.compute()
                _validate_baseline(
                    baseline_values,
                    config.min_valid_baseline_samples,
                )
                baseline_finalized = True

            recorded_frames.append(
                {
                    "frame": len(recorded_frames),
                    "time_s": timestamp_s,
                    "left_fppa": analysis_results.get("left_fppa", np.nan),
                    "right_fppa": analysis_results.get("right_fppa", np.nan),
                    "pelvis_y": pelvis_y,
                    "visibility_check": analysis_results.get(
                        "visibility",
                        "Aucun squelette detecte",
                    ),
                    "left_alignment": analysis_results.get(
                        "left_alignment",
                        "not_detected",
                    ),
                    "right_alignment": analysis_results.get(
                        "right_alignment",
                        "not_detected",
                    ),
                    "left_alignment_index": analysis_results.get(
                        "left_alignment_index",
                        np.nan,
                    ),
                    "right_alignment_index": analysis_results.get(
                        "right_alignment_index",
                        np.nan,
                    ),
                    "left_signed_deviation_percent": analysis_results.get(
                        "left_signed_deviation_percent",
                        np.nan,
                    ),
                    "right_signed_deviation_percent": analysis_results.get(
                        "right_signed_deviation_percent",
                        np.nan,
                    ),
                    "left_knee_deviation_percent": analysis_results.get(
                        "left_knee_deviation_percent",
                        np.nan,
                    ),
                    "right_knee_deviation_percent": analysis_results.get(
                        "right_knee_deviation_percent",
                        np.nan,
                    ),
                }
            )
            if video_writer is not None:
                video_writer.write(
                    draw_front_video_annotation(
                        image,
                        pose_results.pose_landmarks,
                        mp_pose.PoseLandmark,
                        analysis_results,
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
            "La video ne contient pas une baseline complete suivie "
            "d'une phase de mouvement."
        )
    if not recorded_frames:
        raise ValueError("Aucune frame de mouvement n'a pu etre enregistree.")

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
        "source_width_px": int(round(width_value)) if _is_positive_finite(width_value) else 0,
        "source_height_px": int(round(height_value)) if _is_positive_finite(height_value) else 0,
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
        "angle_coordinate_system": "pixel_coordinates_aspect_ratio_corrected",
        "annotated_video_file": (
            resolved_annotated_path.name
            if resolved_annotated_path is not None
            else "not_requested"
        ),
        "annotated_video_status": (
            "created" if annotated_frames_written > 0 else "not_requested"
        ),
        "annotated_video_frames": annotated_frames_written,
    }

    return FrontVideoAnalysisResult(
        recorded_frames=recorded_frames,
        baseline_values=baseline_values,
        acquisition_metadata=acquisition_metadata,
    )
