"""Movement Passport — analyse du squat en vue latérale.

Déroulement
-----------
1. Appuyer sur S.
2. Décompte de 10 secondes pour se placer.
3. Baseline de 3 secondes en position debout immobile.
4. Enregistrement automatique des angles sagittaux.
5. Lever une main pour terminer, ou appuyer sur Q pour quitter.
"""

from __future__ import annotations

from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from baseline.baseline import BaselineRecorder
from config import RESULTS_FOLDER
from movement_analysis.side_repetition_metrics import (
    HEEL_LIFT_SCREENING_THRESHOLD_DEG,
    build_side_repetition_metrics,
)
from movement_analysis.squat_side_view import analyze_squat_side_view
from movement_segmentation.side_repetition_detection import (
    SideRepetitionDetectionConfig,
    build_side_repetitions_dataframe,
    detect_side_repetitions,
)
from reporting.saving import (
    create_session_folder,
    save_dataframe_csv,
    save_metadata_txt,
    save_text_report,
)
from reporting.side_plotting import plot_side_analysis
from reporting.side_report import build_side_report
from session.session_manager import SessionManager
from signal_processing.side_signal_processing import (
    SideSignalProcessingConfig,
    process_side_signals,
)
from ui.display import (
    display_instructions,
    display_quality_warning,
    display_session_state,
)
from ui.gestures import detect_hand_raise


# =============================================================================
# Configuration
# =============================================================================

ANALYSIS_SIDE = "left"
# "auto" utilise l'orientation du pied dans l'image brute. Cela évite les
# inversions de signe provoquées par certaines prévisualisations webcam miroir.
FACING_DIRECTION = "auto"

CAMERA_INDEX = 0
COUNTDOWN_DURATION = 10
BASELINE_DURATION = 3
WINDOW_NAME = "Movement Passport - Squat Side View"


# =============================================================================
# Fonctions utilitaires propres à la vue latérale
# =============================================================================


def create_empty_results() -> dict[str, float | str]:
    """Retourne les valeurs par défaut lorsqu'aucune pose n'est détectée."""

    return {
        "knee_flexion_deg": np.nan,
        "hip_flexion_deg": np.nan,
        "trunk_flexion_deg": np.nan,
        "ankle_internal_angle_deg": np.nan,
        "ankle_dorsiflexion_deg": np.nan,
        "foot_inclination_deg": np.nan,
        "foot_inclination_relative_deg": np.nan,
        "visibility": "Aucun squelette détecté",
    }


def get_baseline_mean(
    baseline_values: dict,
    variable_name: str,
) -> float:
    """Récupère une moyenne calculée par ``BaselineRecorder``.

    Le format attendu est ``<nom_variable>_mean``, comme dans le module
    frontal avec ``pelvis_y_mean``. Une valeur ``NaN`` est retournée si la
    variable n'a pas pu être calculée.
    """

    value = baseline_values.get(f"{variable_name}_mean", np.nan)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    return value if np.isfinite(value) else np.nan


def draw_side_angles(
    image: np.ndarray,
    results: dict[str, float | str],
) -> None:
    """Affiche les angles sagittaux dans un panneau compact."""

    lines = [
        f"Cote analyse : {ANALYSIS_SIDE}",
        f"Direction image : {results.get('facing_direction', 'non detectee')}",
        f"Flexion genou : {results['knee_flexion_deg']:.1f} deg",
        f"Flexion hanche : {results['hip_flexion_deg']:.1f} deg",
        f"Flexion tronc : {results['trunk_flexion_deg']:.1f} deg",
        f"Dorsiflexion : {results['ankle_dorsiflexion_deg']:.1f} deg",
        f"Inclinaison pied : {results['foot_inclination_deg']:.1f} deg",
        (
            "Inclinaison pied relative : "
            f"{results['foot_inclination_relative_deg']:.1f} deg"
        ),
    ]

    x = 20
    y = 150
    line_height = 30
    panel_width = 430
    panel_height = line_height * len(lines) + 20

    cv2.rectangle(
        image,
        (x - 10, y - 25),
        (x + panel_width, y - 25 + panel_height),
        (0, 0, 0),
        -1,
    )

    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += line_height


def build_side_raw_dataframe(
    recorded_frames: list[dict[str, float | str]],
) -> pd.DataFrame:
    """Construit les données brutes de session avec un temps démarrant à zéro."""

    dataframe = pd.DataFrame(recorded_frames)

    if dataframe.empty:
        return dataframe

    dataframe["time_s"] = dataframe["time_s"].astype(float) - float(
        dataframe["time_s"].iloc[0]
    )

    return dataframe


def save_side_raw_session(
    recorded_frames: list[dict[str, float | str]],
    baseline_values: dict,
) -> (
    tuple[
        str,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        str,
    ]
    | None
):
    """Sauvegarde les données brutes, traitées et les métadonnées."""

    dataframe = build_side_raw_dataframe(recorded_frames)

    if dataframe.empty:
        return None

    session_folder = create_session_folder(
        RESULTS_FOLDER,
        "squat_side_view",
    )

    csv_path = save_dataframe_csv(
        dataframe,
        session_folder,
        "sagittal_raw.csv",
    )

    processing_config = SideSignalProcessingConfig()
    repetition_config = SideRepetitionDetectionConfig()
    processed_dataframe = None
    repetitions_dataframe = None
    repetition_metrics_dataframe = None
    processed_csv_path = None
    repetitions_csv_path = None
    repetition_metrics_csv_path = None
    report_path = None
    plot_path = None
    processing_error = None
    segmentation_error = None
    repetition_metrics_error = None
    report_error = None
    plot_error = None
    repetitions_count = 0

    try:
        processed_dataframe = process_side_signals(
            dataframe,
            baseline_values,
            processing_config,
        )
        processed_csv_path = save_dataframe_csv(
            processed_dataframe,
            session_folder,
            "sagittal_processed.csv",
        )
    except ValueError as error:
        processing_error = str(error)
        segmentation_error = "Traitement des signaux indisponible."

    if processed_dataframe is not None:
        try:
            repetitions = detect_side_repetitions(
                processed_dataframe,
                repetition_config,
            )
            repetitions_dataframe = build_side_repetitions_dataframe(
                repetitions
            )
            repetitions_csv_path = save_dataframe_csv(
                repetitions_dataframe,
                session_folder,
                "repetitions.csv",
            )
            repetitions_count = len(repetitions)
        except ValueError as error:
            segmentation_error = str(error)

    if processed_dataframe is not None and repetitions_dataframe is not None:
        try:
            repetition_metrics_dataframe = build_side_repetition_metrics(
                processed_dataframe,
                repetitions_dataframe,
                HEEL_LIFT_SCREENING_THRESHOLD_DEG,
            )
            repetition_metrics_csv_path = save_dataframe_csv(
                repetition_metrics_dataframe,
                session_folder,
                "repetition_metrics.csv",
            )
        except ValueError as error:
            repetition_metrics_error = str(error)

    if processed_dataframe is not None and repetitions_dataframe is not None:
        try:
            plot_path = plot_side_analysis(
                processed_dataframe,
                repetitions_dataframe,
                session_folder,
                heel_lift_threshold_deg=(HEEL_LIFT_SCREENING_THRESHOLD_DEG),
            )
        except (OSError, RuntimeError, ValueError) as error:
            plot_error = str(error)

    visibility_ok_percent = round(
        float((dataframe["visibility"] == "OK").mean() * 100),
        1,
    )

    metadata = {
        "test_name": "squat_side_view",
        "camera_view": "side",
        "source": "webcam",
        "analysis_side": ANALYSIS_SIDE,
        "facing_direction_requested": FACING_DIRECTION,
        "facing_direction_resolved": (
            dataframe["facing_direction"].mode().iloc[0]
            if not dataframe["facing_direction"].mode().empty
            else "not_detected"
        ),
        "orientation_method": (
            "baseline_majority_vote_from_heel_to_foot_index"
            if FACING_DIRECTION == "auto"
            else "manual"
        ),
        "n_frames_recorded": len(dataframe),
        "duration_s": round(float(dataframe["time_s"].max()), 2),
        "visibility_ok_percent": visibility_ok_percent,
        "visibility_quality": (
            "OK" if visibility_ok_percent >= 80 else "Visibilité insuffisante"
        ),
        "ankle_internal_angle_baseline_mean_deg": baseline_values.get(
            "ankle_internal_angle_mean",
            np.nan,
        ),
        "ankle_internal_angle_baseline_std_deg": baseline_values.get(
            "ankle_internal_angle_std",
            np.nan,
        ),
        "ankle_internal_angle_baseline_n": baseline_values.get(
            "ankle_internal_angle_n",
            0,
        ),
        "foot_inclination_baseline_mean_deg": baseline_values.get(
            "foot_inclination_mean",
            np.nan,
        ),
        "foot_inclination_baseline_std_deg": baseline_values.get(
            "foot_inclination_std",
            np.nan,
        ),
        "foot_inclination_baseline_n": baseline_values.get(
            "foot_inclination_n",
            0,
        ),
        "knee_flexion_baseline_mean_deg": baseline_values.get(
            "knee_flexion_mean",
            np.nan,
        ),
        "knee_flexion_baseline_std_deg": baseline_values.get(
            "knee_flexion_std",
            np.nan,
        ),
        "knee_flexion_baseline_n": baseline_values.get(
            "knee_flexion_n",
            0,
        ),
        "hip_flexion_baseline_mean_deg": baseline_values.get(
            "hip_flexion_mean",
            np.nan,
        ),
        "hip_flexion_baseline_std_deg": baseline_values.get(
            "hip_flexion_std",
            np.nan,
        ),
        "hip_flexion_baseline_n": baseline_values.get(
            "hip_flexion_n",
            0,
        ),
        "trunk_flexion_baseline_mean_deg": baseline_values.get(
            "trunk_flexion_mean",
            np.nan,
        ),
        "trunk_flexion_baseline_std_deg": baseline_values.get(
            "trunk_flexion_std",
            np.nan,
        ),
        "trunk_flexion_baseline_n": baseline_values.get(
            "trunk_flexion_n",
            0,
        ),
        "ankle_angle_definition": ("heel_foot_index_axis_vs_ankle_knee_axis"),
        "processed_visibility_threshold": (
            processing_config.visibility_threshold
        ),
        "processed_max_interpolation_gap_s": (
            processing_config.max_interpolation_gap_s
        ),
        "processed_smoothing_window_s": (processing_config.smoothing_window_s),
        "knee_angular_velocity_definition": (
            "time_derivative_of_filtered_relative_knee_flexion_deg_s"
        ),
        "knee_angular_velocity_sign_convention": (
            "positive_descent_negative_ascent"
        ),
        "processed_status": (
            "OK" if processed_csv_path is not None else "Non généré"
        ),
        "processed_error": processing_error,
        "segmentation_status": (
            "OK" if repetitions_csv_path is not None else "Non générée"
        ),
        "segmentation_error": segmentation_error,
        "repetitions_detected": repetitions_count,
        "segmentation_signal": repetition_config.signal_column,
        "segmentation_min_peak_flexion_deg": (
            repetition_config.min_peak_flexion_deg
        ),
        "segmentation_min_peak_prominence_deg": (
            repetition_config.min_peak_prominence_deg
        ),
        "segmentation_min_peak_distance_s": (
            repetition_config.min_peak_distance_s
        ),
        "segmentation_boundary_fraction": (
            repetition_config.boundary_fraction
        ),
        "segmentation_min_quality_percent": (
            repetition_config.min_quality_percent
        ),
        "repetition_metrics_status": (
            "OK" if repetition_metrics_csv_path is not None else "Non générées"
        ),
        "repetition_metrics_error": repetition_metrics_error,
        "sagittal_plot_status": (
            "OK" if plot_path is not None else "Non généré"
        ),
        "sagittal_plot_error": plot_error,
        "heel_lift_plot_threshold_deg": (HEEL_LIFT_SCREENING_THRESHOLD_DEG),
        "secondary_ankle_and_foot_metrics": "experimental",
        "data_stages": (
            "raw_processed_segmented_and_measured"
            if repetition_metrics_csv_path is not None
            else (
                "raw_processed_and_segmented"
                if repetitions_csv_path is not None
                else (
                    "raw_and_processed"
                    if processed_csv_path is not None
                    else "raw_only"
                )
            )
        ),
    }

    if repetition_metrics_dataframe is not None:
        try:
            report_text = build_side_report(
                repetition_metrics_dataframe,
                metadata,
            )
            report_path = save_text_report(
                report_text,
                session_folder,
                "sagittal_report.txt",
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            report_error = str(error)

    metadata["sagittal_report_status"] = (
        "OK" if report_path is not None else "Non généré"
    )
    metadata["sagittal_report_error"] = report_error

    if report_path is not None:
        metadata["data_stages"] = (
            "raw_processed_segmented_measured_and_reported"
        )

    metadata_path = save_metadata_txt(
        metadata,
        session_folder,
    )

    return (
        csv_path,
        processed_csv_path,
        repetitions_csv_path,
        repetition_metrics_csv_path,
        report_path,
        plot_path,
        metadata_path,
    )


# =============================================================================
# Programme principal
# =============================================================================


def main() -> list[dict[str, float | str]]:
    """Lance l'analyse webcam et retourne les frames enregistrées."""

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        pose.close()
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    # Les mêmes objets de gestion que dans le module frontal.
    session = SessionManager(
        countdown_duration=COUNTDOWN_DURATION,
        baseline_duration=BASELINE_DURATION,
    )
    baseline_recorder = BaselineRecorder()

    neutral_ankle_angle = np.nan
    neutral_foot_inclination = np.nan
    resolved_facing_direction: str | None = None
    facing_direction_votes: list[str] = []
    baseline_values: dict = {}
    recorded_frames: list[dict[str, float | str]] = []
    frame_index = 0

    print("Webcam ouverte. Appuie sur S pour commencer ou sur Q pour quitter.")

    try:
        while camera.isOpened():
            # On mémorise l'état précédent afin de détecter précisément la fin
            # de la baseline gérée automatiquement par SessionManager.
            was_baseline = session.is_baseline()
            session.update()

            # Transition baseline -> recording : calcul de la référence neutre.
            if was_baseline and session.is_recording():
                baseline_values = baseline_recorder.compute()
                neutral_ankle_angle = get_baseline_mean(
                    baseline_values,
                    "ankle_internal_angle",
                )
                neutral_foot_inclination = get_baseline_mean(
                    baseline_values,
                    "foot_inclination",
                )

                if FACING_DIRECTION == "auto" and facing_direction_votes:
                    resolved_facing_direction = Counter(
                        facing_direction_votes
                    ).most_common(1)[0][0]
                elif FACING_DIRECTION in {"left", "right"}:
                    resolved_facing_direction = FACING_DIRECTION

                if np.isfinite(neutral_ankle_angle):
                    print(
                        "Baseline terminée. Angle neutre de cheville : "
                        f"{neutral_ankle_angle:.1f}°"
                    )
                    print("Début de l'enregistrement.")
                    print(
                        "Direction détectée dans l'image : "
                        f"{resolved_facing_direction or 'non détectée'}"
                    )
                else:
                    print(
                        "Attention : aucune valeur valide de cheville n'a été "
                        "obtenue pendant la baseline."
                    )

            success, image = camera.read()

            if not success:
                print("Erreur : impossible de lire l'image de la webcam.")
                break

            results_values = create_empty_results()

            # -----------------------------------------------------------------
            # Détection de pose et calcul des angles sagittaux
            # -----------------------------------------------------------------

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                results_values = analyze_squat_side_view(
                    landmarks,
                    mp_pose.PoseLandmark,
                    side=ANALYSIS_SIDE,
                    facing_direction=(
                        resolved_facing_direction or FACING_DIRECTION
                    ),
                    neutral_ankle_angle=neutral_ankle_angle,
                    neutral_foot_inclination=neutral_foot_inclination,
                )

                # Même logique d'arrêt que dans le squat frontal.
                if session.is_recording() and detect_hand_raise(
                    landmarks,
                    mp_pose.PoseLandmark,
                ):
                    session.stop_recording()
                    print("Enregistrement arrêté : main levée détectée.")
                    break

                # Pendant les 3 secondes de baseline, on enregistre l'angle
                # interne brut de la cheville. Sa moyenne devient le zéro de
                # dorsiflexion pour le reste de la session.
                if session.is_baseline():
                    ankle_internal_angle = results_values.get(
                        "ankle_internal_angle_deg",
                        np.nan,
                    )
                    visibility_ok = results_values.get("visibility") == "OK"
                    detected_direction = results_values.get("facing_direction")

                    if visibility_ok and np.isfinite(ankle_internal_angle):
                        baseline_recorder.add_frame(
                            knee_flexion=results_values.get(
                                "knee_flexion_deg",
                                np.nan,
                            ),
                            hip_flexion=results_values.get(
                                "hip_flexion_deg",
                                np.nan,
                            ),
                            trunk_flexion=results_values.get(
                                "trunk_flexion_deg",
                                np.nan,
                            ),
                            ankle_internal_angle=ankle_internal_angle,
                            foot_inclination=results_values.get(
                                "foot_inclination_deg",
                                np.nan,
                            ),
                        )

                    if detected_direction in {"left", "right"}:
                        facing_direction_votes.append(detected_direction)

                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            # -----------------------------------------------------------------
            # Enregistrement frame par frame
            # -----------------------------------------------------------------

            if session.is_recording():
                recorded_frames.append(
                    {
                        "frame": frame_index,
                        "time_s": session.get_recording_time(),
                        "side": results_values.get(
                            "side",
                            ANALYSIS_SIDE,
                        ),
                        "facing_direction": results_values.get(
                            "facing_direction",
                            FACING_DIRECTION,
                        ),
                        "knee_internal_angle_deg": results_values.get(
                            "knee_internal_angle_deg",
                            np.nan,
                        ),
                        "knee_flexion_deg": results_values.get(
                            "knee_flexion_deg",
                            np.nan,
                        ),
                        "hip_internal_angle_deg": results_values.get(
                            "hip_internal_angle_deg",
                            np.nan,
                        ),
                        "hip_flexion_deg": results_values.get(
                            "hip_flexion_deg",
                            np.nan,
                        ),
                        "trunk_flexion_deg": results_values.get(
                            "trunk_flexion_deg",
                            np.nan,
                        ),
                        "ankle_internal_angle_deg": results_values.get(
                            "ankle_internal_angle_deg",
                            np.nan,
                        ),
                        "ankle_dorsiflexion_deg": results_values.get(
                            "ankle_dorsiflexion_deg",
                            np.nan,
                        ),
                        "foot_inclination_deg": results_values.get(
                            "foot_inclination_deg",
                            np.nan,
                        ),
                        "foot_inclination_relative_deg": results_values.get(
                            "foot_inclination_relative_deg",
                            np.nan,
                        ),
                        "visibility": results_values.get(
                            "visibility",
                            "Aucun squelette détecté",
                        ),
                        "visibility_min": results_values.get(
                            "visibility_min",
                            np.nan,
                        ),
                    }
                )
                frame_index += 1

            # -----------------------------------------------------------------
            # Affichage : composants communs + panneau propre à la vue latérale
            # -----------------------------------------------------------------

            visibility_check = str(results_values.get("visibility"))

            display_quality_warning(image, visibility_check)
            display_instructions(image)
            display_session_state(image, session)
            draw_side_angles(image, results_values)

            cv2.imshow(WINDOW_NAME, image)

            # -----------------------------------------------------------------
            # Commandes clavier identiques au module frontal
            # -----------------------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and session.is_waiting():
                baseline_recorder = BaselineRecorder()
                baseline_values = {}
                neutral_ankle_angle = np.nan
                neutral_foot_inclination = np.nan
                resolved_facing_direction = None
                facing_direction_votes = []
                recorded_frames = []
                frame_index = 0

                session.start_countdown()
                print("Compte à rebours démarré.")

            if key == ord("q"):
                session.stop_recording()
                print("Programme arrêté avec la touche Q.")
                break

    finally:
        camera.release()
        pose.close()
        cv2.destroyAllWindows()

    print(f"Analyse terminée : {len(recorded_frames)} frames enregistrées.")

    saved_paths = save_side_raw_session(
        recorded_frames,
        baseline_values,
    )

    if saved_paths is None:
        print("Aucune donnée enregistrée : aucun fichier de session créé.")
    else:
        (
            csv_path,
            processed_csv_path,
            repetitions_csv_path,
            repetition_metrics_csv_path,
            report_path,
            plot_path,
            metadata_path,
        ) = saved_paths
        print(f"Données sagittales brutes sauvegardées dans : {csv_path}")

        if processed_csv_path is None:
            print(
                "Données traitées non générées : baseline ou session "
                "invalide. Consulte metadata.txt."
            )
        else:
            print(
                "Données sagittales traitées sauvegardées dans : "
                f"{processed_csv_path}"
            )

        if repetitions_csv_path is not None:
            print(
                "Répétitions sagittales sauvegardées dans : "
                f"{repetitions_csv_path}"
            )

        if repetition_metrics_csv_path is not None:
            print(
                "Métriques par répétition sauvegardées dans : "
                f"{repetition_metrics_csv_path}"
            )

        if report_path is not None:
            print(f"Rapport sagittal sauvegardé dans : {report_path}")

        if plot_path is not None:
            print("Graphique sagittal sauvegardé dans : " f"{plot_path}")

        print(f"Métadonnées sauvegardées dans : {metadata_path}")

    return recorded_frames


if __name__ == "__main__":
    main()
