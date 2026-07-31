"""
Pipeline commun d'acquisition vidéo pour Movement Passport.

Responsabilités :
- ouverture et fermeture de la caméra ;
- détection MediaPipe Pose ;
- gestion de la session ;
- compte à rebours, baseline et enregistrement ;
- arrêt avec Q ou main levée ;
- exécution d'une analyse biomécanique spécifique ;
- stockage des résultats frame par frame.
"""

from collections.abc import Callable
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from session.session_manager import SessionManager
from ui.gestures import detect_hand_raise
from vision.landmarks import get_pelvis_y


AnalysisFunction = Callable[[Any, Any], dict[str, Any]]
DisplayFunction = Callable[
    [np.ndarray, dict[str, Any], SessionManager],
    None,
]
BaselineFunction = Callable[[float, dict[str, Any]], None]


def run_pose_analysis(
    analyze_landmarks: AnalysisFunction,
    display_results: DisplayFunction,
    baseline_callback: BaselineFunction | None = None,
    camera_index: int = 0,
    countdown_duration: float = 10,
    baseline_duration: float = 3,
    window_name: str = "Movement Passport - Pose Detection",
) -> list[dict[str, Any]]:
    """
    Exécute une session d'analyse de mouvement avec MediaPipe Pose.

    Parameters
    ----------
    analyze_landmarks
        Fonction spécifique au test biomécanique. Elle reçoit les landmarks
        et mp_pose.PoseLandmark, puis retourne un dictionnaire de résultats.

    display_results
        Fonction responsable de l'affichage spécifique au test.

    baseline_callback
        Fonction optionnelle appelée pendant la baseline.

    camera_index
        Index OpenCV de la caméra.

    countdown_duration
        Durée du compte à rebours avant la baseline.

    baseline_duration
        Durée de la baseline en secondes.

    window_name
        Nom de la fenêtre OpenCV.

    Returns
    -------
    list[dict]
        Une liste contenant un dictionnaire par frame enregistrée.
    """

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        pose.close()
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    session = SessionManager(
        countdown_duration=countdown_duration,
        baseline_duration=baseline_duration,
    )

    recorded_frames: list[dict[str, Any]] = []
    frame_index = 0

    print("Webcam ouverte.")
    print("Appuie sur S pour démarrer.")
    print("Appuie sur Q ou lève une main pour arrêter.")

    try:
        while camera.isOpened():
            success, image = camera.read()
            session.update()

            if not success:
                print("Erreur : impossible de lire l'image.")
                break

            analysis_results: dict[str, Any] = {
                "visibility": "Aucun squelette détecté",
            }

            pelvis_y = np.nan

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            pose_results = pose.process(image_rgb)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                pelvis_y = get_pelvis_y(
                    landmarks,
                    mp_pose.PoseLandmark,
                )

                if session.is_recording() and detect_hand_raise(
                    landmarks,
                    mp_pose.PoseLandmark,
                ):
                    session.stop_recording()
                    print("Enregistrement arrêté : " "main levée détectée.")
                    break

                analysis_results = analyze_landmarks(
                    landmarks,
                    mp_pose.PoseLandmark,
                )

                if session.is_baseline() and baseline_callback is not None:
                    baseline_callback(
                        pelvis_y,
                        analysis_results,
                    )

                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            if session.is_recording():
                frame_result = {
                    "frame": frame_index,
                    "time_s": session.get_recording_time(),
                    "pelvis_y": pelvis_y,
                }

                frame_result.update(analysis_results)
                recorded_frames.append(frame_result)

                frame_index += 1

            display_results(
                image,
                analysis_results,
                session,
            )

            cv2.imshow(window_name, image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and session.is_waiting():
                session.start_countdown()
                print("Compte à rebours démarré.")

            if key == ord("q"):
                session.stop_recording()
                break

    finally:
        camera.release()
        pose.close()
        cv2.destroyAllWindows()

    return recorded_frames
