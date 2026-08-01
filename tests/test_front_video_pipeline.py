"""Tests du pipeline frontal applique a un fichier video."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.front_video_pipeline import FrontVideoConfig, analyze_front_video


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
    coordinates = {
        23: (0.40, 0.30),
        25: (0.42, 0.55),
        27: (0.38, 0.85),
        24: (0.60, 0.30),
        26: (0.58, 0.55),
        28: (0.62, 0.85),
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


class FrontVideoPipelineTests(unittest.TestCase):
    def test_extracts_front_baseline_and_movement(self):
        capture = FakeCapture()
        pose = FakePose()

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "front_squat.mp4"
            video_path.touch()
            result = analyze_front_video(
                video_path,
                config=FrontVideoConfig(
                    baseline_duration_s=1.0,
                    min_valid_baseline_samples=10,
                ),
                capture_factory=lambda _: capture,
                pose_factory=lambda: pose,
            )

        self.assertEqual(result.baseline_values["pelvis_y_n"], 10)
        self.assertEqual(len(result.recorded_frames), 10)
        self.assertAlmostEqual(result.recorded_frames[0]["time_s"], 1.0)
        self.assertEqual(
            result.recorded_frames[0]["visibility_check"],
            "OK",
        )
        self.assertEqual(result.acquisition_metadata["source_width_px"], 64)
        self.assertEqual(result.acquisition_metadata["source_height_px"], 48)
        self.assertEqual(
            result.acquisition_metadata["angle_coordinate_system"],
            "pixel_coordinates_aspect_ratio_corrected",
        )
        self.assertTrue(capture.released)
        self.assertTrue(pose.closed)

    def test_rejects_unreliable_front_baseline(self):
        capture = FakeCapture()
        pose = FakePose(visibility=0.1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "hidden_front.mp4"
            video_path.touch()
            with self.assertRaisesRegex(ValueError, "Baseline video invalide"):
                analyze_front_video(
                    video_path,
                    config=FrontVideoConfig(
                        baseline_duration_s=1.0,
                        min_valid_baseline_samples=10,
                    ),
                    capture_factory=lambda _: capture,
                    pose_factory=lambda: pose,
                )

        self.assertTrue(capture.released)
        self.assertTrue(pose.closed)

    def test_creates_front_annotated_video(self):
        capture = FakeCapture()
        pose = FakePose()
        writer = FakeVideoWriter()

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "front_squat.mp4"
            annotated_path = Path(temporary_directory) / "front_annotated.mp4"
            video_path.touch()
            result = analyze_front_video(
                video_path,
                config=FrontVideoConfig(
                    baseline_duration_s=1.0,
                    min_valid_baseline_samples=10,
                ),
                capture_factory=lambda _: capture,
                pose_factory=lambda: pose,
                annotated_video_path=annotated_path,
                video_writer_factory=lambda *args: writer,
            )

        self.assertEqual(len(writer.frames), 20)
        self.assertGreater(np.count_nonzero(writer.frames[-1]), 0)
        self.assertTrue(writer.released)
        self.assertEqual(
            result.acquisition_metadata["annotated_video_file"],
            "front_annotated.mp4",
        )


if __name__ == "__main__":
    unittest.main()
