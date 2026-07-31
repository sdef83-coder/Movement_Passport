"""Analyse un squat sagittal a partir d'une video enregistree.

Protocole video
---------------
1. Choisir un fichier MP4, MOV, M4V, AVI ou MKV.
2. Indiquer le cote anatomique visible.
3. Indiquer l'instant ou commence la posture debout immobile.
4. Les 3 secondes suivantes servent de baseline.
5. Tout le reste de la video est analyse avec le pipeline du module 2.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from app.side_video_pipeline import SideVideoConfig, analyze_side_video
from protocol.side_session_setup import choose_analysis_side
from protocol.video_file_setup import (
    choose_baseline_start,
    choose_video_file,
    validate_video_path,
)
from scripts.squat_side_view import save_side_raw_session


BASELINE_DURATION_S = 3.0


def _build_console_progress():
    """Construit l'affichage de la progression dans la console Spyder."""

    last_percent = -10

    def display_progress(
        processed_frames: int,
        total_frames: int,
        timestamp_s: float,
    ) -> None:
        nonlocal last_percent

        if total_frames > 0:
            percent = min(100, int(processed_frames / total_frames * 100))
            rounded_percent = percent - percent % 10

            if rounded_percent > last_percent:
                last_percent = rounded_percent
                print(f"Analyse video : {rounded_percent} %")
        elif processed_frames == 1 or processed_frames % 100 == 0:
            print(
                f"Analyse video : {processed_frames} frames "
                f"({timestamp_s:.1f} s)"
            )

    return display_progress


def _print_saved_paths(
    saved_paths: tuple | None,
    annotated_video_path: Path | None = None,
) -> None:
    """Affiche les fichiers produits a la fin de l'analyse."""

    if saved_paths is None:
        print("Aucune donnee enregistree : aucun fichier de session cree.")
        return

    labels = (
        "Donnees sagittales brutes",
        "Donnees sagittales traitees",
        "Repetitions sagittales",
        "Metriques par repetition",
        "Rapport sagittal",
        "Graphique sagittal",
        "Cycle moyen sagittal",
        "Resume du cycle moyen",
        "Graphique du cycle moyen",
        "Metadonnees",
    )

    print("\nAnalyse terminee. Fichiers generes :")

    for label, path in zip(labels, saved_paths):
        if path is not None:
            print(f"- {label} : {path}")

    if annotated_video_path is not None:
        print(f"- Video de controle annotee : {annotated_video_path}")


def analyze_video_session(
    video_path: str | Path,
    analysis_side: str,
    baseline_start_s: float = 0.0,
) -> tuple | None:
    """Analyse et sauvegarde une video sans utiliser la webcam."""

    resolved_path = validate_video_path(video_path)
    config = SideVideoConfig(
        baseline_start_s=baseline_start_s,
        baseline_duration_s=BASELINE_DURATION_S,
    )

    print(f"\nVideo : {resolved_path.name}")
    print(
        "Baseline : "
        f"{config.baseline_start_s:.1f} a {config.baseline_end_s:.1f} s"
    )
    print("Le sujet doit rester debout et immobile pendant cette periode.")

    with tempfile.TemporaryDirectory(
        prefix="movement_passport_annotated_"
    ) as temporary_directory:
        temporary_annotated_path = (
            Path(temporary_directory) / "sagittal_annotated.mp4"
        )
        analysis = analyze_side_video(
            resolved_path,
            analysis_side,
            config=config,
            progress_callback=_build_console_progress(),
            annotated_video_path=temporary_annotated_path,
        )

        print(
            "Baseline valide : "
            f"{analysis.baseline_visibility_summary.get('valid_percent', 0.0):.1f} % "
            "des frames avec tous les marqueurs visibles."
        )
        print(
            f"Frames de mouvement analysees : {len(analysis.recorded_frames)}"
        )

        saved_paths = save_side_raw_session(
            analysis.recorded_frames,
            analysis.baseline_values,
            analysis.analysis_side,
            analysis.baseline_visibility_summary,
            failed_baseline_attempts=0,
            source="video_file",
            acquisition_metadata=analysis.acquisition_metadata,
        )

        annotated_video_path = None
        if saved_paths is not None:
            session_folder = Path(saved_paths[0]).parent
            annotated_video_path = session_folder / "sagittal_annotated.mp4"
            shutil.move(temporary_annotated_path, annotated_video_path)

    _print_saved_paths(saved_paths, annotated_video_path)
    return saved_paths


def main() -> tuple | None:
    """Lance le parcours guide prevu pour Spyder."""

    print("Movement Passport - analyse d'une video de squat sagittal")
    video_path = choose_video_file()
    analysis_side = choose_analysis_side()
    baseline_start_s = choose_baseline_start()

    try:
        return analyze_video_session(
            video_path,
            analysis_side,
            baseline_start_s,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print("\nAnalyse impossible.")
        print(str(error))
        print(
            "Verifie le fichier, le cote choisi et les 3 secondes de posture "
            "debout, puis relance le script."
        )
        return None


if __name__ == "__main__":
    main()
