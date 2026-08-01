# ============================================================
# Movement Passport - Analyse FPPA en vue frontale
# ============================================================
# Objectif :
# - ouvrir la webcam ;
# - détecter le squelette avec MediaPipe ;
# - calculer le FPPA gauche/droit ;
# - contrôler la qualité des marqueurs ;
# - afficher les résultats en temps réel ;
# - sauvegarder CSV, résumé, métadonnées et graphique.
# ============================================================


# ----------------------------
# 1. Librairies externes
# ----------------------------
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ----------------------------
# 2. Fonctions du projet
# ----------------------------
from app.analysis_pipeline import run_pose_analysis

from config import RESULTS_FOLDER

from quality.checks import check_session_duration

from ui.display import (
    display_quality_warning,
    display_instructions,
    display_session_state,
    display_frontal_alignment,
    display_frontal_alignment_status,
)

from reporting.summary import (
    summarize_fppa,
    build_fppa_dataframe,
)

from reporting.saving import (
    save_dataframe_csv,
    save_summary_txt,
    create_session_folder,
    save_metadata_txt,
    save_text_report,
)

from movement_analysis.squat_front_view import analyze_fppa_front_view

from interpretation.fppa_interpretation import interpret_fppa

from event_detection.fppa_events import detect_maximum_dynamic_knee_deviation

from vision.landmarks import get_pelvis_y

from signal_processing.filtering import filter_signal

from event_detection.peak_detection import detect_peaks

from session.session_manager import SessionManager

from movement_segmentation.repetition_detection import (
    build_repetitions_from_adaptive_baseline,
)

from movement_analysis.repetition_metrics import compute_repetition_metrics

from baseline.baseline import BaselineRecorder

from ui.gestures import detect_hand_raise

from reporting.clinical_summary import (
    build_clinical_repetitions_dataframe,
)

from reporting.clinical_report import build_clinical_report

from movement_analysis.mean_cycle import (
    build_mean_cycle,
    build_mean_cycle_dataframe,
    build_mean_cycle_summary_dataframe,
    summarize_mean_cycle,
)

from reporting.plotting import (
    plot_fppa,
    plot_movement_analysis,
    plot_mean_cycle,
)

# ----------------------------
# 3. Initialisation MediaPipe
# ----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ----------------------------
# 4. Ouverture webcam
# ----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erreur : impossible d'ouvrir la webcam")
    exit()

print("Webcam ouverte. Appuie sur Q pour quitter.")


# ----------------------------
# 5. Variables de stockage
# ----------------------------

frame_index = 0
frame_values = []
time_values = []

left_fppa_values = []
right_fppa_values = []

visibility_checks = []

pelvis_y_values = []

session = SessionManager(countdown_duration=10, baseline_duration=3)

baseline_recorder = BaselineRecorder()
baseline_values = {}

left_alignment_values = []
right_alignment_values = []

left_alignment_index_values = []
right_alignment_index_values = []

left_signed_deviation_percent_values = []
right_signed_deviation_percent_values = []

left_knee_deviation_percent_values = []
right_knee_deviation_percent_values = []

# ----------------------------
# 6. Boucle principale webcam
# ----------------------------
while camera.isOpened():

    success, image = camera.read()

    session.update()

    if not success:
        print("Erreur : impossible de lire l'image")
        break

    # Valeurs par défaut si aucun squelette n'est détecté
    left_fppa = np.nan
    right_fppa = np.nan
    pelvis_y = np.nan
    left_alignment_index = np.nan
    right_alignment_index = np.nan
    left_signed_deviation_percent = np.nan
    right_signed_deviation_percent = np.nan
    left_knee_deviation_percent = np.nan
    right_knee_deviation_percent = np.nan
    visibility_check = "Aucun squelette détecté"
    left_alignment = "not_detected"
    right_alignment = "not_detected"

    # Conversion BGR → RGB pour MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Détection du squelette
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        pelvis_y = get_pelvis_y(landmarks, mp_pose.PoseLandmark)
        if session.is_recording() and detect_hand_raise(
            landmarks, mp_pose.PoseLandmark
        ):
            session.stop_recording()
            print("Enregistrement arrêté : main levée détectée.")
            break

        test_results = analyze_fppa_front_view(
            landmarks,
            mp_pose.PoseLandmark,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )

        left_fppa = test_results["left_fppa"]
        right_fppa = test_results["right_fppa"]
        visibility_check = test_results["visibility"]

        left_alignment = test_results["left_alignment"]
        right_alignment = test_results["right_alignment"]

        left_alignment_index = test_results["left_alignment_index"]
        right_alignment_index = test_results["right_alignment_index"]

        left_signed_deviation_percent = test_results[
            "left_signed_deviation_percent"
        ]
        right_signed_deviation_percent = test_results[
            "right_signed_deviation_percent"
        ]

        left_knee_deviation_percent = test_results[
            "left_knee_deviation_percent"
        ]
        right_knee_deviation_percent = test_results[
            "right_knee_deviation_percent"
        ]

        if session.is_baseline():
            baseline_recorder.add_frame(
                pelvis_y=pelvis_y,
                left_fppa=left_fppa,
                right_fppa=right_fppa,
            )

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

    # Enregistrement des valeurs frame par frame
    if session.is_recording():
        current_time = session.get_recording_time()

        frame_values.append(frame_index)
        time_values.append(current_time)
        left_fppa_values.append(left_fppa)
        right_fppa_values.append(right_fppa)
        pelvis_y_values.append(pelvis_y)
        visibility_checks.append(visibility_check)
        left_alignment_values.append(left_alignment)
        right_alignment_values.append(right_alignment)
        left_alignment_index_values.append(left_alignment_index)
        right_alignment_index_values.append(right_alignment_index)
        left_signed_deviation_percent_values.append(
            left_signed_deviation_percent
        )
        right_signed_deviation_percent_values.append(
            right_signed_deviation_percent
        )
        left_knee_deviation_percent_values.append(left_knee_deviation_percent)
        right_knee_deviation_percent_values.append(
            right_knee_deviation_percent
        )

        frame_index += 1

    # Affichage en temps réel
    display_frontal_alignment_status(
        image,
        left_alignment,
        right_alignment,
    )
    display_quality_warning(image, visibility_check)
    display_instructions(image)
    display_session_state(image, session)
    display_frontal_alignment(
        image,
        left_alignment,
        right_alignment,
        left_alignment_index,
        right_alignment_index,
    )

    cv2.imshow("Movement Passport - Pose Detection", image)

    # Quitter avec Q
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") and session.is_waiting():
        session.start_countdown()
        print("Compte à rebours démarré.")

    if key == ord("q"):
        session.stop_recording()
        break


# ----------------------------
# 7. Construction des résultats
# ----------------------------
baseline_values = baseline_recorder.compute()

print("Baseline :")
print(baseline_values)

df = build_fppa_dataframe(
    frame_values,
    time_values,
    left_fppa_values,
    right_fppa_values,
    pelvis_y_values,
    visibility_checks,
    left_alignment_values,
    right_alignment_values,
    left_alignment_index_values,
    right_alignment_index_values,
    left_signed_deviation_percent_values,
    right_signed_deviation_percent_values,
    left_knee_deviation_percent_values,
    right_knee_deviation_percent_values,
)

filtered_pelvis_y = filter_signal(
    df["pelvis_y"],
    method="moving_average",
    window_size=7,
)

df["pelvis_y_filtered"] = filtered_pelvis_y

pelvis_velocity = np.gradient(
    filtered_pelvis_y,
    df["time_s"],
)

df["pelvis_velocity"] = pelvis_velocity

pelvis_peaks = detect_peaks(
    filtered_pelvis_y,
    prominence=0.03,
    distance=20,
    min_frame=15,
    max_frame=len(filtered_pelvis_y) - 15,
)

pelvis_high_points = detect_peaks(
    -filtered_pelvis_y,
    prominence=0.03,
    distance=20,
    min_frame=15,
    max_frame=len(filtered_pelvis_y) - 15,
)

first_peak = pelvis_peaks[0]
last_peak = pelvis_peaks[-1]

start_high = int(np.argmin(filtered_pelvis_y[:first_peak]))

end_high = int(last_peak + np.argmin(filtered_pelvis_y[last_peak:]))

pelvis_high_points = [start_high] + list(pelvis_high_points) + [end_high]

print("Positions hautes avec bords :", pelvis_high_points)
print("Points bas détectés :", pelvis_peaks)


repetitions = build_repetitions_from_adaptive_baseline(
    df=df,
    filtered_pelvis_y=filtered_pelvis_y,
    bottom_peaks=pelvis_peaks,
    baseline_values=baseline_values,
    amplitude_fraction=0.10,
    min_duration_s=0.5,
    max_duration_s=8.0,
)

repetition_metrics = []

for rep in repetitions:
    metrics = compute_repetition_metrics(
        df=df,
        repetition=rep,
        baseline_values=baseline_values,
        filtered_pelvis_y=filtered_pelvis_y,
    )
    repetition_metrics.append(metrics)

repetitions_df = pd.DataFrame(repetition_metrics)
clinical_repetitions_df = build_clinical_repetitions_dataframe(repetitions_df)

mean_cycle = build_mean_cycle(
    df=df,
    repetitions=repetitions,
    n_points=101,
)

mean_cycle_df = build_mean_cycle_dataframe(mean_cycle)

mean_cycle_summary = summarize_mean_cycle(
    mean_cycle=mean_cycle,
    repetitions=repetitions,
)

mean_cycle_summary_df = build_mean_cycle_summary_dataframe(mean_cycle_summary)

print("Répétitions détectées :")
for rep in repetitions:
    print(rep)

print("Pics pelvis détectés :", pelvis_peaks)
print("Nombre de répétitions estimées :", len(pelvis_peaks))
df_valid = df[df["visibility_check"] == "OK"]

event_results = detect_maximum_dynamic_knee_deviation(df)

print("Événements détectés :")
print(event_results)


# ----------------------------
# 8. Création du dossier session
# ----------------------------
session_folder = create_session_folder(
    RESULTS_FOLDER,
    "fppa_front_view",
)


# ----------------------------
# 9. Résumé statistique
# ----------------------------
if len(df_valid) == 0:
    print("Aucune frame fiable : résumé impossible.")
    summary = {}
else:
    summary = summarize_fppa(df_valid)
    summary.update(event_results)

    interpretation = interpret_fppa(
        summary["left_event_fppa"],
        summary["right_event_fppa"],
    )

    summary.update(interpretation)

summary_path = save_summary_txt(
    summary,
    session_folder,
    "fppa_summary.txt",
)
print(f"Résumé sauvegardé dans : {summary_path}")

global_interpretation = summary.get(
    "global_message", "Interprétation non disponible"
)


# ----------------------------
# 10. Métadonnées et qualité globale
# ----------------------------
duration_s = round(df["time_s"].max(), 2)
n_frames = len(df)

quality_check = check_session_duration(duration_s)

n_visibility_ok = visibility_checks.count("OK")
visibility_ok_percent = round(
    (n_visibility_ok / len(visibility_checks)) * 100, 1
)
metadata = {
    "test_name": "fppa_front_view",
    "camera_view": "front",
    "source": "webcam",
    "measure": "left_fppa, right_fppa",
    "unit": "degrees",
    "n_frames_analyzed": n_frames,
    "duration_s": duration_s,
    "quality_check": quality_check,
    "visibility_ok_percent": visibility_ok_percent,
    "visibility_quality": (
        "OK" if visibility_ok_percent >= 80 else "Visibilité insuffisante"
    ),
    "global_interpretation": global_interpretation,
    "pelvis_y_baseline_mean": baseline_values.get("pelvis_y_mean", None),
    "pelvis_y_baseline_std": baseline_values.get("pelvis_y_std", None),
    "repetitions_detected": len(repetitions),
    "segmentation_signal": "pelvis_y_filtered",
    "segmentation_boundary_method": (
        "adaptive_threshold_with_neighboring_valleys"
    ),
    "segmentation_boundary_fraction": 0.10,
}
metadata_path = save_metadata_txt(metadata, session_folder)
print(f"Métadonnées sauvegardées dans : {metadata_path}")

# ----------------------------
# 11. Sauvegardes CSV et graphique
# ----------------------------
csv_path = save_dataframe_csv(
    df,
    session_folder,
    "fppa_results.csv",
)
print(f"Résultats sauvegardés dans : {csv_path}")

clinical_repetitions_path = save_dataframe_csv(
    clinical_repetitions_df,
    session_folder,
    "clinical_repetitions.csv",
)

print(
    "Résumé clinique des répétitions sauvegardé dans : "
    f"{clinical_repetitions_path}"
)

movement_figure_path = plot_movement_analysis(
    df,
    session_folder,
    filtered_pelvis_y=filtered_pelvis_y,
    baseline_values=baseline_values,
    repetitions=repetitions,
)

print("Graphique mouvement sauvegardé dans : " f"{movement_figure_path}")

fppa_figure_path = plot_fppa(
    df,
    session_folder,
    repetitions=repetitions,
)

print("Graphique FPPA sauvegardé dans : " f"{fppa_figure_path}")


repetitions_path = save_dataframe_csv(
    repetitions_df,
    session_folder,
    "repetitions.csv",
)

print(f"Répétitions sauvegardées dans : {repetitions_path}")

mean_cycle_csv_path = save_dataframe_csv(
    mean_cycle_df,
    session_folder,
    "mean_cycle.csv",
)

print("Cycle moyen sauvegardé dans : " f"{mean_cycle_csv_path}")

mean_cycle_figure_path = plot_mean_cycle(
    mean_cycle=mean_cycle,
    results_folder=session_folder,
)

print("Figure du cycle moyen sauvegardée dans : " f"{mean_cycle_figure_path}")

clinical_report = build_clinical_report(
    clinical_df=clinical_repetitions_df,
    metadata=metadata,
    mean_cycle_summary=mean_cycle_summary,
)

clinical_report_path = save_text_report(
    clinical_report,
    session_folder,
    "clinical_report.txt",
)

print("Rapport clinique sauvegardé dans : " f"{clinical_report_path}")

mean_cycle_summary_path = save_dataframe_csv(
    mean_cycle_summary_df,
    session_folder,
    "mean_cycle_summary.csv",
)

print("Résumé du cycle moyen sauvegardé dans : " f"{mean_cycle_summary_path}")

# ----------------------------
# 12. Fermeture propre
# ----------------------------
camera.release()
pose.close()
cv2.destroyAllWindows()

print("Analyse terminée.")
