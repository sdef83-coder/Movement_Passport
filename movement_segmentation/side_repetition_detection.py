"""Segmentation des répétitions du squat en vue latérale.

Le signal principal est la flexion relative et filtrée du genou. Les seuils
définis ici servent uniquement à reconnaître un mouvement dans le signal ; ils
ne constituent pas des seuils cliniques de qualité ou de profondeur du squat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SideRepetitionDetectionConfig:
    """Paramètres temporels et cinématiques du segmentateur."""

    signal_column: str = "knee_flexion_filtered_deg"
    min_peak_flexion_deg: float = 20.0
    min_peak_prominence_deg: float = 15.0
    min_peak_distance_s: float = 0.75
    prominence_window_s: float = 1.50
    boundary_fraction: float = 0.10
    min_boundary_flexion_deg: float = 5.0
    min_duration_s: float = 0.50
    max_duration_s: float = 8.0
    min_quality_percent: float = 80.0


REPETITION_COLUMNS = [
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
    "peak_knee_flexion_deg",
    "peak_prominence_deg",
    "boundary_threshold_deg",
    "quality_percent",
    "interpolated_percent",
]


def _validate_input(
    dataframe: pd.DataFrame,
    config: SideRepetitionDetectionConfig,
) -> None:
    required_columns = {
        "frame",
        "time_s",
        config.signal_column,
        "quality_valid",
    }
    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        raise ValueError(
            "Colonnes nécessaires à la segmentation manquantes : "
            + ", ".join(missing_columns)
        )

    if dataframe.empty:
        raise ValueError("Le DataFrame sagittal traité est vide.")

    time_values = pd.to_numeric(
        dataframe["time_s"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.all(np.isfinite(time_values)):
        raise ValueError("time_s contient des valeurs invalides.")

    if np.any(np.diff(time_values) <= 0):
        raise ValueError("time_s doit être strictement croissant.")

    if config.min_peak_distance_s <= 0:
        raise ValueError("min_peak_distance_s doit être positif.")

    if not 0 < config.boundary_fraction < 1:
        raise ValueError("boundary_fraction doit être compris entre 0 et 1.")


def _find_local_maxima(signal: np.ndarray) -> list[int]:
    """Trouve les changements de pente positive vers une pente non positive."""

    candidates = []

    for index in range(1, len(signal) - 1):
        previous_value = signal[index - 1]
        current_value = signal[index]
        next_value = signal[index + 1]

        if not np.all(
            np.isfinite(
                [
                    previous_value,
                    current_value,
                    next_value,
                ]
            )
        ):
            continue

        if (
            current_value > previous_value
            and current_value >= next_value
        ):
            candidates.append(index)

    return candidates


def _compute_peak_prominence(
    signal: np.ndarray,
    time_values: np.ndarray,
    peak_index: int,
    window_s: float,
) -> float:
    peak_time = time_values[peak_index]
    left_indices = np.flatnonzero(
        (time_values >= peak_time - window_s)
        & (time_values < peak_time)
    )
    right_indices = np.flatnonzero(
        (time_values > peak_time)
        & (time_values <= peak_time + window_s)
    )

    if len(left_indices) == 0 or len(right_indices) == 0:
        return np.nan

    left_values = signal[left_indices]
    right_values = signal[right_indices]

    if (
        not np.any(np.isfinite(left_values))
        or not np.any(np.isfinite(right_values))
    ):
        return np.nan

    left_minimum = float(np.nanmin(left_values))
    right_minimum = float(np.nanmin(right_values))
    reference_level = max(left_minimum, right_minimum)

    return float(signal[peak_index] - reference_level)


def _select_separated_peaks(
    candidate_peaks: list[int],
    signal: np.ndarray,
    time_values: np.ndarray,
    min_peak_distance_s: float,
) -> list[int]:
    """Conserve les pics les plus hauts en imposant une distance temporelle."""

    selected_peaks = []

    for peak_index in sorted(
        candidate_peaks,
        key=lambda index: signal[index],
        reverse=True,
    ):
        peak_time = time_values[peak_index]

        if all(
            abs(peak_time - time_values[selected_index])
            >= min_peak_distance_s
            for selected_index in selected_peaks
        ):
            selected_peaks.append(peak_index)

    return sorted(selected_peaks)


def _find_boundary_before_peak(
    signal: np.ndarray,
    peak_index: int,
    search_limit: int,
    threshold: float,
) -> tuple[int, bool]:
    index = peak_index

    while (
        index > search_limit
        and np.isfinite(signal[index])
        and signal[index] > threshold
    ):
        index -= 1

    complete = (
        np.isfinite(signal[index])
        and signal[index] <= threshold
    )

    return index, bool(complete)


def _find_boundary_after_peak(
    signal: np.ndarray,
    peak_index: int,
    search_limit: int,
    threshold: float,
) -> tuple[int, bool]:
    index = peak_index

    while (
        index < search_limit
        and np.isfinite(signal[index])
        and signal[index] > threshold
    ):
        index += 1

    complete = (
        np.isfinite(signal[index])
        and signal[index] <= threshold
    )

    return index, bool(complete)


def _compute_search_limits(
    signal: np.ndarray,
    selected_peaks: list[int],
) -> tuple[list[int], list[int]]:
    """Utilise les vallées entre pics pour créer des zones disjointes."""

    valleys = []

    for left_peak, right_peak in zip(
        selected_peaks[:-1],
        selected_peaks[1:],
    ):
        between_values = signal[left_peak : right_peak + 1]

        if np.any(np.isfinite(between_values)):
            valley_offset = int(np.nanargmin(between_values))
            valley_index = left_peak + valley_offset
        else:
            valley_index = (left_peak + right_peak) // 2

        valleys.append(valley_index)

    left_limits = [0] + valleys
    right_limits = valleys + [len(signal) - 1]

    return left_limits, right_limits


def detect_side_repetitions(
    processed_dataframe: pd.DataFrame,
    config: SideRepetitionDetectionConfig | None = None,
) -> list[dict[str, float | int]]:
    """Détecte et valide les répétitions complètes du squat latéral."""

    if config is None:
        config = SideRepetitionDetectionConfig()

    _validate_input(processed_dataframe, config)

    signal = pd.to_numeric(
        processed_dataframe[config.signal_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    time_values = processed_dataframe["time_s"].to_numpy(dtype=float)

    candidate_peaks = []
    prominence_by_peak = {}

    for peak_index in _find_local_maxima(signal):
        peak_value = signal[peak_index]

        if peak_value < config.min_peak_flexion_deg:
            continue

        prominence = _compute_peak_prominence(
            signal,
            time_values,
            peak_index,
            config.prominence_window_s,
        )

        if (
            not np.isfinite(prominence)
            or prominence < config.min_peak_prominence_deg
        ):
            continue

        candidate_peaks.append(peak_index)
        prominence_by_peak[peak_index] = prominence

    selected_peaks = _select_separated_peaks(
        candidate_peaks,
        signal,
        time_values,
        config.min_peak_distance_s,
    )

    if not selected_peaks:
        return []

    left_limits, right_limits = _compute_search_limits(
        signal,
        selected_peaks,
    )

    repetitions = []

    for peak_position, peak_index in enumerate(selected_peaks):
        peak_value = float(signal[peak_index])
        boundary_threshold = max(
            config.min_boundary_flexion_deg,
            config.boundary_fraction * peak_value,
        )

        start_index, start_complete = _find_boundary_before_peak(
            signal,
            peak_index,
            left_limits[peak_position],
            boundary_threshold,
        )
        end_index, end_complete = _find_boundary_after_peak(
            signal,
            peak_index,
            right_limits[peak_position],
            boundary_threshold,
        )

        if not start_complete or not end_complete:
            continue

        if not start_index < peak_index < end_index:
            continue

        start_time = float(time_values[start_index])
        bottom_time = float(time_values[peak_index])
        end_time = float(time_values[end_index])
        duration_s = end_time - start_time

        if not config.min_duration_s <= duration_s <= config.max_duration_s:
            continue

        repetition_slice = processed_dataframe.iloc[
            start_index : end_index + 1
        ]
        quality_percent = float(
            repetition_slice["quality_valid"].astype(bool).mean() * 100
        )

        if quality_percent < config.min_quality_percent:
            continue

        if "any_signal_was_interpolated" in repetition_slice.columns:
            interpolated_percent = float(
                repetition_slice["any_signal_was_interpolated"]
                .astype(bool)
                .mean()
                * 100
            )
        else:
            interpolated_percent = 0.0

        repetitions.append(
            {
                "rep_id": len(repetitions) + 1,
                "start_frame": int(
                    processed_dataframe.iloc[start_index]["frame"]
                ),
                "bottom_frame": int(
                    processed_dataframe.iloc[peak_index]["frame"]
                ),
                "end_frame": int(
                    processed_dataframe.iloc[end_index]["frame"]
                ),
                "start_time_s": round(start_time, 3),
                "bottom_time_s": round(bottom_time, 3),
                "end_time_s": round(end_time, 3),
                "duration_s": round(duration_s, 3),
                "descent_duration_s": round(
                    bottom_time - start_time,
                    3,
                ),
                "ascent_duration_s": round(
                    end_time - bottom_time,
                    3,
                ),
                "peak_knee_flexion_deg": round(peak_value, 2),
                "peak_prominence_deg": round(
                    float(prominence_by_peak[peak_index]),
                    2,
                ),
                "boundary_threshold_deg": round(
                    boundary_threshold,
                    2,
                ),
                "quality_percent": round(quality_percent, 1),
                "interpolated_percent": round(
                    interpolated_percent,
                    1,
                ),
            }
        )

    return repetitions


def build_side_repetitions_dataframe(
    repetitions: list[dict[str, float | int]],
) -> pd.DataFrame:
    """Construit un tableau stable, y compris lorsqu'aucun squat n'est trouvé."""

    return pd.DataFrame(
        repetitions,
        columns=REPETITION_COLUMNS,
    )
