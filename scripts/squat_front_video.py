"""Analyse un squat frontal a partir d'une video enregistree."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from app.front_video_pipeline import FrontVideoConfig, analyze_front_video
from protocol.video_file_setup import (
    choose_baseline_start,
    choose_video_file,
    validate_video_path,
)
from reporting.front_session import save_front_session


BASELINE_DURATION_S = 3.0


def _build_console_progress():
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
    paths: dict[str, str | None],
    annotated_video_path: Path | None = None,
) -> None:
    labels = {
        "results_csv": "Donnees frontales",
        "repetitions_csv": "Metriques detaillees par repetition",
        "clinical_repetitions_csv": "Resume clinique des repetitions",
        "summary_txt": "Resume FPPA",
        "clinical_report_txt": "Rapport frontal",
        "movement_plot": "Graphique du mouvement",
        "fppa_plot": "Graphique FPPA",
        "mean_cycle_csv": "Cycle moyen",
        "mean_cycle_summary_csv": "Resume du cycle moyen",
        "mean_cycle_plot": "Graphique du cycle moyen",
        "metadata_txt": "Metadonnees",
    }
    print("\nAnalyse terminee. Fichiers generes :")
    for key, label in labels.items():
        path = paths.get(key)
        if path is not None:
            print(f"- {label} : {path}")
    if annotated_video_path is not None:
        print(f"- Video frontale annotee : {annotated_video_path}")


def analyze_video_session(
    video_path: str | Path,
    baseline_start_s: float = 0.0,
) -> dict[str, str | None]:
    """Analyse puis sauvegarde une video frontale."""

    resolved_path = validate_video_path(video_path)
    config = FrontVideoConfig(
        baseline_start_s=baseline_start_s,
        baseline_duration_s=BASELINE_DURATION_S,
    )
    print(f"\nVideo : {resolved_path.name}")
    print(
        "Baseline : "
        f"{config.baseline_start_s:.1f} a {config.baseline_end_s:.1f} s"
    )
    print("Reste debout et immobile pendant cette periode.")

    with tempfile.TemporaryDirectory(
        prefix="movement_passport_front_annotated_"
    ) as temporary_directory:
        temporary_annotated_path = (
            Path(temporary_directory) / "frontal_annotated.mp4"
        )
        analysis = analyze_front_video(
            resolved_path,
            config=config,
            progress_callback=_build_console_progress(),
            annotated_video_path=temporary_annotated_path,
        )
        print(
            "Baseline valide : "
            f"{analysis.baseline_values.get('pelvis_y_n', 0)} frames utilisees."
        )
        print(
            f"Frames de mouvement analysees : {len(analysis.recorded_frames)}"
        )

        paths = save_front_session(
            analysis.recorded_frames,
            analysis.baseline_values,
            source="video_file",
            acquisition_metadata=analysis.acquisition_metadata,
        )
        annotated_video_path = (
            Path(paths["session_folder"]) / "frontal_annotated.mp4"
        )
        shutil.move(temporary_annotated_path, annotated_video_path)

    _print_saved_paths(paths, annotated_video_path)
    return paths


def main() -> dict[str, str | None] | None:
    """Lance le parcours guide dans Spyder."""

    print("Movement Passport - analyse d'une video de squat frontal")
    video_path = choose_video_file()
    baseline_start_s = choose_baseline_start()
    try:
        return analyze_video_session(video_path, baseline_start_s)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print("\nAnalyse impossible.")
        print(str(error))
        print(
            "Verifie le fichier, le cadrage de face et les 3 secondes de "
            "posture debout, puis relance le script."
        )
        return None


if __name__ == "__main__":
    main()
