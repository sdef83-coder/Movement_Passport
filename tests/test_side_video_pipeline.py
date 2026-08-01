"""Tests du pipeline sagittal appliqué à un fichier vidéo."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.side_video_pipeline import (
    SideVideoConfig,
    analyze_side_video,
    resolve_video_timestamp_s,
)


class FakeCapture:
    def __init__(self, frame_count=20, fps=10.0):
        self.frame_count = frame_count
        self.fps = fps
        self.cursor = 0
        self.released = False

    def isOpened(self):
        return not self.released

    def read(self):
        if self.cursor >= self.frame_count:
            return False, None

        self.cursor += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 64
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 48
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return max(0, self.cursor - 1) * 1000.0 / self.fps
        return 0.0

    def set(self, property_id, value):
        return True

    def release(self):
        self.released = True


class ConstantTimestampCapture:
    def get(self, property_id):
        return 0.0


class FakeLandmark(SimpleNamespace):
    def HasField(self, field_name):
        return hasattr(self, field_name)


class FakeVideoWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def isOpened(self):
        return not self.released

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


def build_landmarks(visibility=0.99):
    landmarks = [
        FakeLandmark(x=0.5, y=0.5, visibility=visibility)
        for _ in range(33)
    ]

    # Côté gauche, sujet regardant vers la droite dans l'image.
    coordinates = {
        11: (0.40, 0.20),  # épaule
        23: (0.40, 0.40),  # hanche
        25: (0.40, 0.60),  # genou
        27: (0.40, 0.80),  # cheville
        29: (0.38, 0.85),  # talon
        31: (0.52, 0.85),  # avant-pied
    }

    for index, (x_value, y_value) in coordinates.items():
        landmarks[index] = FakeLandmark(
            x=x_value,
            y=y_value,
            visibility=visibility,
        )

    return landmarks


class FakePose:
    def __init__(self, visibility=0.99):
        self.landmarks = build_landmarks(visibility)
        self.closed = False

    def process(self, image):
        return SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=self.landmarks)
        )

    def close(self):
        self.closed = True


class SideVideoPipelineTests(unittest.TestCase):
    def test_timestamp_uses_fps_when_video_position_stalls(self):
        capture = ConstantTimestampCapture()

        first_timestamp, first_method = resolve_video_timestamp_s(
            capture,
            frame_index=0,
            fps=10.0,
            previous_timestamp_s=None,
        )
        second_timestamp, second_method = resolve_video_timestamp_s(
            capture,
            frame_index=1,
            fps=10.0,
            previous_timestamp_s=first_timestamp,
        )

        self.assertEqual(first_timestamp, 0.0)
        self.assertEqual(first_method, "opencv_pos_msec")
        self.assertAlmostEqual(second_timestamp, 0.1)
        self.assertEqual(second_method, "frame_index_fps_fallback")

    def test_extracts_baseline_and_recording_from_video_time(self):
        capture = FakeCapture()
        pose = FakePose()

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "phone_squat.mp4"
            video_path.touch()

            result = analyze_side_video(
                video_path,
                analysis_side="left",
                config=SideVideoConfig(
                    baseline_duration_s=1.0,
                    min_valid_baseline_samples=10,
                ),
                capture_factory=lambda _: capture,
                pose_factory=lambda: pose,
            )

        self.assertEqual(result.baseline_values["knee_flexion_n"], 10)
        self.assertEqual(len(result.recorded_frames), 10)
        self.assertAlmostEqual(result.recorded_frames[0]["time_s"], 1.0)
        self.assertEqual(
            {frame["facing_direction"] for frame in result.recorded_frames},
            {"right"},
        )
        self.assertEqual(
            result.acquisition_metadata["source_file_name"],
            "phone_squat.mp4",
        )
        self.assertEqual(result.acquisition_metadata["source_fps"], 10.0)
        self.assertEqual(
            result.acquisition_metadata["baseline_end_s_in_source"],
            1.0,
        )
        self.assertTrue(capture.released)
        self.assertTrue(pose.closed)

    def test_creates_annotated_video_with_landmarks_and_angles(self):
        capture = FakeCapture()
        pose = FakePose()
        writer = FakeVideoWriter()

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "phone_squat.mp4"
            annotated_path = Path(temporary_directory) / "annotated.mp4"
            video_path.touch()

            result = analyze_side_video(
                video_path,
                analysis_side="left",
                config=SideVideoConfig(
                    baseline_duration_s=1.0,
                    min_valid_baseline_samples=10,
                ),
                capture_factory=lambda _: capture,
                pose_factory=lambda: pose,
                annotated_video_path=annotated_path,
                video_writer_factory=lambda *args: writer,
            )

        self.assertEqual(len(writer.frames), 20)
        self.assertTrue(writer.released)
        self.assertGreater(np.count_nonzero(writer.frames[-1]), 0)
        self.assertEqual(
            result.acquisition_metadata["annotated_video_file"],
            "annotated.mp4",
        )
        self.assertEqual(
            result.acquisition_metadata["annotated_video_status"],
            "created",
        )
        self.assertEqual(
            result.acquisition_metadata["annotated_video_frames"],
            20,
        )

    def test_rejects_video_with_unreliable_baseline(self):
        capture = FakeCapture()
        pose = FakePose(visibility=0.1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "hidden_markers.mp4"
            video_path.touch()

            with self.assertRaisesRegex(ValueError, "Baseline vidéo invalide"):
                analyze_side_video(
                    video_path,
                    analysis_side="left",
                    config=SideVideoConfig(
                        baseline_duration_s=1.0,
                        min_valid_baseline_samples=10,
                    ),
                    capture_factory=lambda _: capture,
                    pose_factory=lambda: pose,
                )

        self.assertTrue(capture.released)
        self.assertTrue(pose.closed)


if __name__ == "__main__":
    unittest.main()
