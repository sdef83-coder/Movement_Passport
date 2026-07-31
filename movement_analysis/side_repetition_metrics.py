"""Métriques descriptives par répétition pour le squat en vue latérale.

Les angles utilisés sont les signaux relatifs à la baseline et filtrés par le
pipeline sagittal. Les mesures de genou, de hanche et de tronc constituent les
sorties principales. La dorsiflexion et l'inclinaison du pied restent des
indicateurs secondaires et expérimentaux.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


HEEL_LIFT_SCREENING_THRESHOLD_DEG = 20.0


SIDE_REPETITION_METRIC_COLUMNS = [
    "rep_id",
    "start_frame",
    "bottom_frame",
    "end_frame",
    "start_time_s",
    "bottom_time_s",
    "end_time_s",
    "duration_s",
    "descent_duration_s",
    "ascent_duration_s",
    "descent_ascent_time_ratio",
    "mean_knee_descent_velocity_deg_s",
    "mean_knee_ascent_velocity_deg_s",
    "knee_flexion_at_bottom_deg",
    "peak_knee_flexion_deg",
    "hip_flexion_at_bottom_deg",
    "peak_hip_flexion_deg",
    "peak_hip_flexion_cycle_percent",
    "peak_hip_flexion_phase",
    "trunk_flexion_at_bottom_deg",
    "peak_trunk_flexion_deg",
    "peak_trunk_flexion_cycle_percent",
    "peak_trunk_flexion_phase",
    "ankle_dorsiflexion_at_bottom_deg",
    "peak_ankle_dorsiflexion_deg",
    "foot_inclination_at_bottom_deg",
    "peak_foot_inclination_deg",
    "heel_lift_screening",
    "heel_lift_threshold_deg",
    "quality_percent",
    "interpolated_percent",
]


_REQUIRED_PROCESSED_COLUMNS = {
    "frame",
    "time_s",
    "knee_flexion_filtered_deg",
    "knee_angular_velocity_deg_s",
    "hip_flexion_filtered_deg",
    "trunk_flexion_filtered_deg",
    "ankle_dorsiflexion_filtered_deg",
    "foot_inclination_filtered_deg",
    "quality_valid",
}


def _validate_inputs(
    processed_dataframe: pd.DataFrame,
    repetitions: pd.DataFrame,
) -> None:
    missing_processed = sorted(
        _REQUIRED_PROCESSED_COLUMNS - set(processed_dataframe.columns)
    )

    if missing_processed:
        raise ValueError(
            "Colonnes sagittales nécessaires aux métriques manquantes : "
            + ", ".join(missing_processed)
        )

    required_repetition_columns = {
        "rep_id",
        "start_frame",
        "bottom_frame",
        "end_frame",
        "start_time_s",
        "bottom_time_s",
        "end_time_s",
    }
    missing_repetitions = sorted(
        required_repetition_columns - set(repetitions.columns)
    )

    if missing_repetitions:
        raise ValueError(
            "Colonnes de répétition nécessaires aux métriques manquantes : "
            + ", ".join(missing_repetitions)
        )


def _phase_for_frame(peak_frame: int, bottom_frame: int) -> str:
    if peak_frame == bottom_frame:
        return "bottom"

    return "descent" if peak_frame < bottom_frame else "ascent"


def _peak_metrics(
    repetition_dataframe: pd.DataFrame,
    signal_column: str,
    start_time_s: float,
    bottom_frame: int,
    duration_s: float,
) -> tuple[float, float, str]:
    signal = pd.to_numeric(
        repetition_dataframe[signal_column],
        errors="coerce",
    )

    if not signal.notna().any():
        return np.nan, np.nan, "not_detected"

    peak_index = signal.idxmax()
    peak_value = float(signal.loc[peak_index])
    peak_time_s = float(repetition_dataframe.loc[peak_index, "time_s"])
    peak_frame = int(repetition_dataframe.loc[peak_index, "frame"])

    if duration_s > 0:
        cycle_percent = (peak_time_s - start_time_s) / duration_s * 100.0
    else:
        cycle_percent = np.nan

    return (
        peak_value,
        cycle_percent,
        _phase_for_frame(peak_frame, bottom_frame),
    )


def build_side_repetition_metrics(
    processed_dataframe: pd.DataFrame,
    repetitions: pd.DataFrame,
    heel_lift_threshold_deg: float = HEEL_LIFT_SCREENING_THRESHOLD_DEG,
) -> pd.DataFrame:
    """Construit une ligne de métriques descriptives par répétition."""

    _validate_inputs(processed_dataframe, repetitions)

    if heel_lift_threshold_deg <= 0:
        raise ValueError("Le seuil expérimental du talon doit être positif.")

    metric_rows = []

    for _, repetition in repetitions.iterrows():
        start_frame = int(repetition["start_frame"])
        bottom_frame = int(repetition["bottom_frame"])
        end_frame = int(repetition["end_frame"])

        repetition_dataframe = processed_dataframe.loc[
            processed_dataframe["frame"].between(start_frame, end_frame)
        ]

        if repetition_dataframe.empty:
            raise ValueError(
                "Aucune donnée trouvée pour la répétition "
                f"{int(repetition['rep_id'])}."
            )

        start_rows = repetition_dataframe.loc[
            repetition_dataframe["frame"] == start_frame
        ]
        bottom_rows = repetition_dataframe.loc[
            repetition_dataframe["frame"] == bottom_frame
        ]
        end_rows = repetition_dataframe.loc[
            repetition_dataframe["frame"] == end_frame
        ]

        if start_rows.empty or bottom_rows.empty or end_rows.empty:
            raise ValueError(
                "Borne de cycle introuvable pour la répétition "
                f"{int(repetition['rep_id'])}."
            )

        start_row = start_rows.iloc[0]
        bottom_row = bottom_rows.iloc[0]
        end_row = end_rows.iloc[0]
        start_time_s = float(repetition["start_time_s"])
        bottom_time_s = float(repetition["bottom_time_s"])
        end_time_s = float(repetition["end_time_s"])
        duration_s = end_time_s - start_time_s
        descent_duration_s = bottom_time_s - start_time_s
        ascent_duration_s = end_time_s - bottom_time_s
        start_knee_flexion = float(
            start_row["knee_flexion_filtered_deg"]
        )
        bottom_knee_flexion = float(
            bottom_row["knee_flexion_filtered_deg"]
        )
        end_knee_flexion = float(end_row["knee_flexion_filtered_deg"])
        mean_descent_velocity = (
            abs(bottom_knee_flexion - start_knee_flexion)
            / descent_duration_s
            if descent_duration_s > 0
            else np.nan
        )
        mean_ascent_velocity = (
            abs(end_knee_flexion - bottom_knee_flexion)
            / ascent_duration_s
            if ascent_duration_s > 0
            else np.nan
        )

        hip_peak, hip_cycle, hip_phase = _peak_metrics(
            repetition_dataframe,
            "hip_flexion_filtered_deg",
            start_time_s,
            bottom_frame,
            duration_s,
        )
        trunk_peak, trunk_cycle, trunk_phase = _peak_metrics(
            repetition_dataframe,
            "trunk_flexion_filtered_deg",
            start_time_s,
            bottom_frame,
            duration_s,
        )
        knee_peak, _, _ = _peak_metrics(
            repetition_dataframe,
            "knee_flexion_filtered_deg",
            start_time_s,
            bottom_frame,
            duration_s,
        )
        ankle_peak, _, _ = _peak_metrics(
            repetition_dataframe,
            "ankle_dorsiflexion_filtered_deg",
            start_time_s,
            bottom_frame,
            duration_s,
        )
        foot_peak, _, _ = _peak_metrics(
            repetition_dataframe,
            "foot_inclination_filtered_deg",
            start_time_s,
            bottom_frame,
            duration_s,
        )

        quality_percent = float(
            repetition_dataframe["quality_valid"].astype(bool).mean()
            * 100.0
        )

        if "any_signal_was_interpolated" in repetition_dataframe.columns:
            interpolated_percent = float(
                repetition_dataframe["any_signal_was_interpolated"]
                .astype(bool)
                .mean()
                * 100.0
            )
        else:
            interpolated_percent = 0.0

        if np.isfinite(foot_peak):
            heel_lift_screening = (
                "above_experimental_threshold"
                if foot_peak >= heel_lift_threshold_deg
                else "below_experimental_threshold"
            )
        else:
            heel_lift_screening = "not_detected"

        metric_rows.append(
            {
                "rep_id": int(repetition["rep_id"]),
                "start_frame": start_frame,
                "bottom_frame": bottom_frame,
                "end_frame": end_frame,
                "start_time_s": round(start_time_s, 3),
                "bottom_time_s": round(bottom_time_s, 3),
                "end_time_s": round(end_time_s, 3),
                "duration_s": round(duration_s, 3),
                "descent_duration_s": round(descent_duration_s, 3),
                "ascent_duration_s": round(ascent_duration_s, 3),
                "descent_ascent_time_ratio": (
                    round(descent_duration_s / ascent_duration_s, 3)
                    if ascent_duration_s > 0
                    else np.nan
                ),
                "mean_knee_descent_velocity_deg_s": round(
                    mean_descent_velocity, 2
                ),
                "mean_knee_ascent_velocity_deg_s": round(
                    mean_ascent_velocity, 2
                ),
                "knee_flexion_at_bottom_deg": round(
                    float(bottom_row["knee_flexion_filtered_deg"]), 2
                ),
                "peak_knee_flexion_deg": round(knee_peak, 2),
                "hip_flexion_at_bottom_deg": round(
                    float(bottom_row["hip_flexion_filtered_deg"]), 2
                ),
                "peak_hip_flexion_deg": round(hip_peak, 2),
                "peak_hip_flexion_cycle_percent": round(hip_cycle, 1),
                "peak_hip_flexion_phase": hip_phase,
                "trunk_flexion_at_bottom_deg": round(
                    float(bottom_row["trunk_flexion_filtered_deg"]), 2
                ),
                "peak_trunk_flexion_deg": round(trunk_peak, 2),
                "peak_trunk_flexion_cycle_percent": round(trunk_cycle, 1),
                "peak_trunk_flexion_phase": trunk_phase,
                "ankle_dorsiflexion_at_bottom_deg": round(
                    float(bottom_row["ankle_dorsiflexion_filtered_deg"]), 2
                ),
                "peak_ankle_dorsiflexion_deg": round(ankle_peak, 2),
                "foot_inclination_at_bottom_deg": round(
                    float(bottom_row["foot_inclination_filtered_deg"]), 2
                ),
                "peak_foot_inclination_deg": round(foot_peak, 2),
                "heel_lift_screening": heel_lift_screening,
                "heel_lift_threshold_deg": round(
                    heel_lift_threshold_deg, 2
                ),
                "quality_percent": round(quality_percent, 1),
                "interpolated_percent": round(interpolated_percent, 1),
            }
        )

    return pd.DataFrame(
        metric_rows,
        columns=SIDE_REPETITION_METRIC_COLUMNS,
    )
