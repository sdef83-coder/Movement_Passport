import pandas as pd

from app.analysis_pipeline import run_pose_analysis

from baseline.baseline import BaselineRecorder

from movement_analysis.squat_front_view import analyze_fppa_front_view

from ui.display import (
    display_quality_warning,
    display_instructions,
    display_session_state,
    display_frontal_alignment,
    display_frontal_alignment_status,
)


baseline_recorder = BaselineRecorder()


def analyze_front_landmarks(
    landmarks,
    pose_landmark,
    image_width,
    image_height,
):
    """
    Adapte l'analyse frontale existante au pipeline générique.
    """

    results = analyze_fppa_front_view(
        landmarks,
        pose_landmark,
        image_width=image_width,
        image_height=image_height,
    )

    # Nom standardisé utilisé dans les données enregistrées.
    results["visibility_check"] = results.get(
        "visibility",
        "Aucun squelette détecté",
    )

    return results


def record_front_baseline(pelvis_y, analysis_results):
    """
    Enregistre les variables nécessaires pendant la baseline.
    """

    baseline_recorder.add_frame(
        pelvis_y=pelvis_y,
        left_fppa=analysis_results.get("left_fppa"),
        right_fppa=analysis_results.get("right_fppa"),
    )


def display_front_results(image, results, session):
    """
    Affichage temps réel spécifique à la vue frontale.
    """

    visibility_check = results.get(
        "visibility_check",
        results.get("visibility", "Aucun squelette détecté"),
    )

    left_alignment = results.get(
        "left_alignment",
        "not_detected",
    )

    right_alignment = results.get(
        "right_alignment",
        "not_detected",
    )

    left_alignment_index = results.get(
        "left_alignment_index",
        float("nan"),
    )

    right_alignment_index = results.get(
        "right_alignment_index",
        float("nan"),
    )

    display_frontal_alignment_status(
        image,
        left_alignment,
        right_alignment,
    )

    display_quality_warning(
        image,
        visibility_check,
    )

    display_instructions(image)

    display_session_state(
        image,
        session,
    )

    display_frontal_alignment(
        image,
        left_alignment,
        right_alignment,
        left_alignment_index,
        right_alignment_index,
    )


recorded_frames = run_pose_analysis(
    analyze_landmarks=analyze_front_landmarks,
    display_results=display_front_results,
    baseline_callback=record_front_baseline,
    camera_index=0,
    countdown_duration=10,
    baseline_duration=3,
    window_name="Movement Passport - Front Pipeline Test",
)


df_test = pd.DataFrame(recorded_frames)
baseline_values = baseline_recorder.compute()

print("\nBaseline :")
print(baseline_values)

print("\nNombre de frames enregistrées :", len(df_test))

print("\nColonnes enregistrées :")
print(df_test.columns.tolist())

print("\nPremières lignes :")
print(df_test.head())
