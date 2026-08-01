"""Traitement et sauvegarde communs d'une session de squat frontal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import RESULTS_FOLDER
from event_detection.fppa_events import detect_maximum_dynamic_knee_deviation
from event_detection.peak_detection import detect_peaks
from interpretation.fppa_interpretation import interpret_fppa
from movement_analysis.mean_cycle import (
    build_mean_cycle,
    build_mean_cycle_dataframe,
    build_mean_cycle_summary_dataframe,
    summarize_mean_cycle,
)
from movement_analysis.repetition_metrics import compute_repetition_metrics
from movement_segmentation.repetition_detection import (
    build_repetitions_from_adaptive_baseline,
)
from quality.checks import check_session_duration
from reporting.clinical_report import build_clinical_report
from reporting.clinical_summary import build_clinical_repetitions_dataframe
from reporting.plotting import plot_fppa, plot_mean_cycle, plot_movement_analysis
from reporting.saving import (
    create_session_folder,
    save_dataframe_csv,
    save_metadata_txt,
    save_summary_txt,
    save_text_report,
)
from reporting.summary import summarize_fppa
from signal_processing.filtering import filter_signal


REQUIRED_FRONT_COLUMNS = (
    "frame",
    "time_s",
    "left_fppa",
    "right_fppa",
    "pelvis_y",
    "visibility_check",
    "left_alignment",
    "right_alignment",
    "left_alignment_index",
    "right_alignment_index",
    "left_signed_deviation_percent",
    "right_signed_deviation_percent",
    "left_knee_deviation_percent",
    "right_knee_deviation_percent",
)


def build_front_dataframe(recorded_frames: list[dict[str, Any]]) -> pd.DataFrame:
    """Construit le tableau frontal et replace son temps a zero."""

    dataframe = pd.DataFrame(recorded_frames)
    missing_columns = [
        column for column in REQUIRED_FRONT_COLUMNS if column not in dataframe
    ]
    if missing_columns:
        raise ValueError(
            "Colonnes frontales manquantes : " + ", ".join(missing_columns)
        )
    if dataframe.empty:
        raise ValueError("Aucune frame frontale a sauvegarder.")

    dataframe = dataframe.loc[:, list(REQUIRED_FRONT_COLUMNS)].copy()
    dataframe["time_s"] = dataframe["time_s"].astype(float)
    dataframe["time_s"] -= float(dataframe["time_s"].iloc[0])
    return dataframe


def _prepare_pelvis_signal(dataframe: pd.DataFrame) -> np.ndarray:
    pelvis_series = pd.to_numeric(dataframe["pelvis_y"], errors="coerce")
    pelvis_series = pelvis_series.interpolate(
        method="linear",
        limit=5,
        limit_area="inside",
    )
    if pelvis_series.isna().any():
        raise ValueError(
            "Le suivi du bassin contient des interruptions trop longues. "
            "Ameliore le cadrage puis recommence l'analyse."
        )
    return filter_signal(
        pelvis_series,
        method="moving_average",
        window_size=7,
    )


def save_front_session(
    recorded_frames: list[dict[str, Any]],
    baseline_values: dict,
    *,
    source: str = "webcam",
    acquisition_metadata: dict | None = None,
) -> dict[str, str | None]:
    """Calcule et sauvegarde toutes les sorties du module frontal."""

    dataframe = build_front_dataframe(recorded_frames)
    filtered_pelvis_y = _prepare_pelvis_signal(dataframe)
    dataframe["pelvis_y_filtered"] = filtered_pelvis_y
    dataframe["pelvis_velocity"] = np.gradient(
        filtered_pelvis_y,
        dataframe["time_s"].to_numpy(dtype=float),
    )

    pelvis_peaks = detect_peaks(
        filtered_pelvis_y,
        prominence=0.03,
        distance=20,
        min_frame=15,
        max_frame=len(filtered_pelvis_y) - 15,
    )
    repetitions = build_repetitions_from_adaptive_baseline(
        df=dataframe,
        filtered_pelvis_y=filtered_pelvis_y,
        bottom_peaks=pelvis_peaks,
        baseline_values=baseline_values,
        amplitude_fraction=0.10,
        min_duration_s=0.5,
        max_duration_s=8.0,
    )
    if not repetitions:
        raise ValueError(
            "Aucune repetition frontale valide n'a ete detectee. "
            "Verifie la baseline, le cadrage et l'amplitude des squats."
        )

    repetition_metrics = [
        compute_repetition_metrics(
            df=dataframe,
            repetition=repetition,
            baseline_values=baseline_values,
            filtered_pelvis_y=filtered_pelvis_y,
        )
        for repetition in repetitions
    ]
    repetitions_dataframe = pd.DataFrame(repetition_metrics)
    clinical_dataframe = build_clinical_repetitions_dataframe(
        repetitions_dataframe
    )
    mean_cycle = build_mean_cycle(
        df=dataframe,
        repetitions=repetitions,
        n_points=101,
    )
    mean_cycle_dataframe = build_mean_cycle_dataframe(mean_cycle)
    mean_cycle_summary = summarize_mean_cycle(
        mean_cycle=mean_cycle,
        repetitions=repetitions,
    )
    mean_cycle_summary_dataframe = build_mean_cycle_summary_dataframe(
        mean_cycle_summary
    )

    valid_dataframe = dataframe[dataframe["visibility_check"] == "OK"]
    if valid_dataframe.empty:
        summary = {}
        global_interpretation = "Interpretation non disponible"
    else:
        summary = summarize_fppa(valid_dataframe)
        summary.update(detect_maximum_dynamic_knee_deviation(dataframe))
        interpretation = interpret_fppa(
            summary["left_event_fppa"],
            summary["right_event_fppa"],
        )
        summary.update(interpretation)
        global_interpretation = summary["global_message"]

    duration_s = round(float(dataframe["time_s"].max()), 2)
    visibility_ok_percent = round(
        float((dataframe["visibility_check"] == "OK").mean() * 100.0),
        1,
    )
    metadata = {
        "test_name": "fppa_front_view",
        "camera_view": "front",
        "source": source,
        "measure": "left_fppa, right_fppa",
        "unit": "degrees",
        "angle_coordinate_system": (
            "pixel_coordinates_aspect_ratio_corrected"
        ),
        "n_frames_analyzed": len(dataframe),
        "duration_s": duration_s,
        "quality_check": check_session_duration(duration_s),
        "visibility_ok_percent": visibility_ok_percent,
        "visibility_quality": (
            "OK" if visibility_ok_percent >= 80 else "Visibilite insuffisante"
        ),
        "global_interpretation": global_interpretation,
        "pelvis_y_baseline_mean": baseline_values.get("pelvis_y_mean"),
        "pelvis_y_baseline_std": baseline_values.get("pelvis_y_std"),
        "left_fppa_baseline_mean_deg": baseline_values.get("left_fppa_mean"),
        "right_fppa_baseline_mean_deg": baseline_values.get("right_fppa_mean"),
        "repetitions_detected": len(repetitions),
        "segmentation_signal": "pelvis_y_filtered",
        "segmentation_boundary_method": (
            "adaptive_threshold_with_neighboring_valleys"
        ),
        "segmentation_boundary_fraction": 0.10,
        "data_stages": "raw_segmented_measured_mean_cycle_and_reported",
    }
    if acquisition_metadata:
        metadata.update(acquisition_metadata)

    session_folder = Path(
        create_session_folder(RESULTS_FOLDER, "fppa_front_view")
    )
    paths: dict[str, str | None] = {
        "session_folder": str(session_folder),
        "results_csv": save_dataframe_csv(
            dataframe,
            session_folder,
            "fppa_results.csv",
        ),
        "repetitions_csv": save_dataframe_csv(
            repetitions_dataframe,
            session_folder,
            "repetitions.csv",
        ),
        "clinical_repetitions_csv": save_dataframe_csv(
            clinical_dataframe,
            session_folder,
            "clinical_repetitions.csv",
        ),
        "mean_cycle_csv": save_dataframe_csv(
            mean_cycle_dataframe,
            session_folder,
            "mean_cycle.csv",
        ),
        "mean_cycle_summary_csv": save_dataframe_csv(
            mean_cycle_summary_dataframe,
            session_folder,
            "mean_cycle_summary.csv",
        ),
        "summary_txt": save_summary_txt(
            summary,
            session_folder,
            "fppa_summary.txt",
        ),
        "metadata_txt": save_metadata_txt(metadata, session_folder),
        "movement_plot": plot_movement_analysis(
            dataframe,
            session_folder,
            filtered_pelvis_y=filtered_pelvis_y,
            baseline_values=baseline_values,
            repetitions=repetitions,
        ),
        "fppa_plot": plot_fppa(
            dataframe,
            session_folder,
            repetitions=repetitions,
        ),
        "mean_cycle_plot": plot_mean_cycle(
            mean_cycle=mean_cycle,
            results_folder=session_folder,
        ),
    }
    clinical_report = build_clinical_report(
        clinical_df=clinical_dataframe,
        metadata=metadata,
        mean_cycle_summary=mean_cycle_summary,
    )
    paths["clinical_report_txt"] = save_text_report(
        clinical_report,
        session_folder,
        "clinical_report.txt",
    )
    return paths
