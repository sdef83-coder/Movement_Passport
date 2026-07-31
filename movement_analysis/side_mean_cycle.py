"""Cycle moyen du squat en vue latérale.

Chaque répétition détectée est rééchantillonnée sur 101 points,
de 0 à 100 % du cycle. Les courbes moyennes et leurs écarts-types sont
ensuite calculés entre les répétitions valides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SIDE_CYCLE_SIGNALS = {
    "knee_flexion": "knee_flexion_filtered_deg",
    "hip_flexion": "hip_flexion_filtered_deg",
    "trunk_flexion": "trunk_flexion_filtered_deg",
    "knee_velocity": "knee_angular_velocity_deg_s",
    "ankle_dorsiflexion": "ankle_dorsiflexion_filtered_deg",
    "foot_inclination": "foot_inclination_filtered_deg",
}


def _validate_inputs(
    processed_dataframe: pd.DataFrame,
    repetitions: pd.DataFrame,
    n_points: int,
) -> None:
    required_processed = {
        "frame",
        "time_s",
        *SIDE_CYCLE_SIGNALS.values(),
    }
    missing_processed = sorted(
        required_processed - set(processed_dataframe.columns)
    )

    if missing_processed:
        raise ValueError(
            "Colonnes sagittales nécessaires au cycle moyen manquantes : "
            + ", ".join(missing_processed)
        )

    required_repetitions = {
        "rep_id",
        "start_frame",
        "bottom_frame",
        "end_frame",
        "start_time_s",
        "bottom_time_s",
        "end_time_s",
    }
    missing_repetitions = sorted(
        required_repetitions - set(repetitions.columns)
    )

    if missing_repetitions:
        raise ValueError(
            "Colonnes de répétition nécessaires au cycle moyen "
            "manquantes : " + ", ".join(missing_repetitions)
        )

    if n_points < 3:
        raise ValueError("n_points doit être supérieur ou égal à 3.")


def _interpolate_cycle(
    time_values,
    signal_values,
    normalized_cycle,
) -> np.ndarray:
    """Rééchantillonne un signal entre 0 et 100 % du cycle."""

    time_array = np.asarray(time_values, dtype=float)
    signal_array = np.asarray(signal_values, dtype=float)
    valid = np.isfinite(time_array) & np.isfinite(signal_array)

    if valid.sum() < 2:
        return np.full(len(normalized_cycle), np.nan, dtype=float)

    valid_times = time_array[valid]
    valid_signal = signal_array[valid]
    duration_s = valid_times[-1] - valid_times[0]

    if duration_s <= 0:
        return np.full(len(normalized_cycle), np.nan, dtype=float)

    cycle_percent = (
        (valid_times - valid_times[0]) / duration_s * 100.0
    )

    return np.interp(
        normalized_cycle,
        cycle_percent,
        valid_signal,
    )


def _mean_and_sd(cycles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_values = np.nanmean(cycles, axis=0)

    if cycles.shape[0] >= 2:
        sd_values = np.nanstd(cycles, axis=0, ddof=1)
    else:
        sd_values = np.zeros_like(mean_values)

    return mean_values, sd_values


def build_side_mean_cycle(
    processed_dataframe: pd.DataFrame,
    repetitions: pd.DataFrame,
    n_points: int = 101,
) -> dict | None:
    """Construit les cycles individuels et le cycle sagittal moyen."""

    _validate_inputs(processed_dataframe, repetitions, n_points)
    normalized_cycle = np.linspace(0.0, 100.0, n_points)
    cycles_by_signal = {
        signal_name: [] for signal_name in SIDE_CYCLE_SIGNALS
    }
    valid_rep_ids = []
    bottom_cycle_percent = []

    for _, repetition in repetitions.iterrows():
        start_frame = int(repetition["start_frame"])
        end_frame = int(repetition["end_frame"])
        repetition_dataframe = processed_dataframe.loc[
            processed_dataframe["frame"].between(start_frame, end_frame)
        ]

        if len(repetition_dataframe) < 3:
            continue

        time_values = pd.to_numeric(
            repetition_dataframe["time_s"],
            errors="coerce",
        ).to_numpy(dtype=float)

        if np.isfinite(time_values).sum() < 3:
            continue

        repetition_cycles = {}

        for signal_name, signal_column in SIDE_CYCLE_SIGNALS.items():
            repetition_cycles[signal_name] = _interpolate_cycle(
                time_values,
                pd.to_numeric(
                    repetition_dataframe[signal_column],
                    errors="coerce",
                ).to_numpy(dtype=float),
                normalized_cycle,
            )

        primary_signals_valid = all(
            np.isfinite(repetition_cycles[signal_name]).any()
            for signal_name in (
                "knee_flexion",
                "hip_flexion",
                "trunk_flexion",
                "knee_velocity",
            )
        )

        if not primary_signals_valid:
            continue

        for signal_name in SIDE_CYCLE_SIGNALS:
            cycles_by_signal[signal_name].append(
                repetition_cycles[signal_name]
            )

        start_time_s = float(repetition["start_time_s"])
        bottom_time_s = float(repetition["bottom_time_s"])
        end_time_s = float(repetition["end_time_s"])
        duration_s = end_time_s - start_time_s

        if duration_s > 0:
            bottom_cycle_percent.append(
                (bottom_time_s - start_time_s) / duration_s * 100.0
            )
        else:
            bottom_cycle_percent.append(np.nan)

        valid_rep_ids.append(int(repetition["rep_id"]))

    if not valid_rep_ids:
        return None

    mean_cycle = {
        "cycle_percent": normalized_cycle,
        "rep_ids": valid_rep_ids,
        "n_cycles": len(valid_rep_ids),
        "bottom_cycle_percent": np.asarray(
            bottom_cycle_percent,
            dtype=float,
        ),
    }

    for signal_name, cycles in cycles_by_signal.items():
        cycle_matrix = np.asarray(cycles, dtype=float)
        mean_values, sd_values = _mean_and_sd(cycle_matrix)
        mean_cycle[f"{signal_name}_cycles"] = cycle_matrix
        mean_cycle[f"{signal_name}_mean"] = mean_values
        mean_cycle[f"{signal_name}_sd"] = sd_values

    return mean_cycle


def build_side_mean_cycle_dataframe(mean_cycle: dict | None) -> pd.DataFrame:
    """Transforme le cycle sagittal moyen en tableau exportable."""

    if mean_cycle is None:
        return pd.DataFrame()

    dataframe_values = {
        "cycle_percent": mean_cycle["cycle_percent"],
    }

    for signal_name in SIDE_CYCLE_SIGNALS:
        dataframe_values[f"{signal_name}_mean"] = mean_cycle[
            f"{signal_name}_mean"
        ]
        dataframe_values[f"{signal_name}_sd"] = mean_cycle[
            f"{signal_name}_sd"
        ]

    return pd.DataFrame(dataframe_values)


def summarize_side_mean_cycle(mean_cycle: dict | None) -> dict:
    """Calcule les indicateurs descriptifs du cycle sagittal moyen."""

    if mean_cycle is None:
        return {}

    cycle_percent = mean_cycle["cycle_percent"]
    bottom_values = mean_cycle["bottom_cycle_percent"]
    summary = {
        "n_cycles": int(mean_cycle["n_cycles"]),
        "mean_bottom_cycle_percent": round(
            float(np.nanmean(bottom_values)),
            1,
        ),
        "bottom_cycle_percent_sd": round(
            float(np.nanstd(bottom_values, ddof=1))
            if len(bottom_values) >= 2
            else 0.0,
            1,
        ),
    }

    for signal_name in (
        "knee_flexion",
        "hip_flexion",
        "trunk_flexion",
        "ankle_dorsiflexion",
        "foot_inclination",
    ):
        mean_values = mean_cycle[f"{signal_name}_mean"]
        peak_index = int(np.nanargmax(mean_values))
        summary[f"peak_{signal_name}_mean"] = round(
            float(mean_values[peak_index]),
            2,
        )
        summary[f"peak_{signal_name}_timing_percent"] = round(
            float(cycle_percent[peak_index]),
            1,
        )
        summary[f"{signal_name}_average_variability"] = round(
            float(np.nanmean(mean_cycle[f"{signal_name}_sd"])),
            2,
        )

    mean_bottom_percent = summary["mean_bottom_cycle_percent"]
    descent_mask = cycle_percent <= mean_bottom_percent
    ascent_mask = cycle_percent >= mean_bottom_percent
    velocity_mean = mean_cycle["knee_velocity_mean"]
    descent_values = velocity_mean[descent_mask]
    ascent_values = velocity_mean[ascent_mask]

    summary["mean_knee_descent_velocity_deg_s"] = round(
        float(np.nanmean(descent_values[descent_values > 0])),
        2,
    )
    summary["mean_knee_ascent_velocity_deg_s"] = round(
        abs(float(np.nanmean(ascent_values[ascent_values < 0]))),
        2,
    )
    summary["knee_velocity_average_variability"] = round(
        float(np.nanmean(mean_cycle["knee_velocity_sd"])),
        2,
    )

    return summary


def build_side_mean_cycle_summary_dataframe(
    mean_cycle_summary: dict,
) -> pd.DataFrame:
    """Construit le tableau d'une ligne résumant le cycle moyen."""

    if not mean_cycle_summary:
        return pd.DataFrame()

    return pd.DataFrame([mean_cycle_summary])
