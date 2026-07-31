"""Prétraitement des signaux du squat en vue latérale.

Ce module transforme une copie des données brutes sans jamais modifier le
DataFrame source :

1. validation des colonnes et de la baseline ;
2. correction des angles par rapport à la posture debout ;
3. invalidation des frames de mauvaise qualité ;
4. interpolation linéaire des trous internes courts uniquement ;
5. lissage centré par moyenne glissante ;
6. calcul de la vitesse angulaire continue du genou.

La segmentation des répétitions n'appartient volontairement pas à ce module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SideSignalProcessingConfig:
    """Paramètres reproductibles du prétraitement sagittal."""

    visibility_threshold: float = 0.5
    max_interpolation_gap_s: float = 0.25
    smoothing_window_s: float = 0.25
    min_baseline_samples: int = 10


REQUIRED_COLUMNS = {
    "frame",
    "time_s",
    "visibility",
    "visibility_min",
    "knee_flexion_deg",
    "hip_flexion_deg",
    "trunk_flexion_deg",
    "ankle_internal_angle_deg",
    "foot_inclination_deg",
}

BASELINE_SPECS = {
    "knee_flexion": "knee_flexion_mean",
    "hip_flexion": "hip_flexion_mean",
    "trunk_flexion": "trunk_flexion_mean",
    "ankle_internal_angle": "ankle_internal_angle_mean",
    "foot_inclination": "foot_inclination_mean",
}

BASELINE_COUNT_KEYS = {
    name: f"{name}_n"
    for name in BASELINE_SPECS
}


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))

    if missing_columns:
        raise ValueError(
            "Colonnes sagittales manquantes : "
            + ", ".join(missing_columns)
        )

    if dataframe.empty:
        raise ValueError("Le DataFrame sagittal est vide.")

    time_values = pd.to_numeric(
        dataframe["time_s"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.all(np.isfinite(time_values)):
        raise ValueError("La colonne time_s contient des valeurs invalides.")

    if np.any(np.diff(time_values) <= 0):
        raise ValueError("La colonne time_s doit être strictement croissante.")


def _validate_baseline(
    baseline_values: dict,
    config: SideSignalProcessingConfig,
) -> None:
    errors = []

    for signal_name, mean_key in BASELINE_SPECS.items():
        mean_value = baseline_values.get(mean_key, np.nan)
        count_value = baseline_values.get(
            BASELINE_COUNT_KEYS[signal_name],
            0,
        )

        try:
            mean_is_valid = np.isfinite(float(mean_value))
        except (TypeError, ValueError):
            mean_is_valid = False

        try:
            count_is_valid = int(count_value) >= config.min_baseline_samples
        except (TypeError, ValueError):
            count_is_valid = False

        if not mean_is_valid:
            errors.append(f"{mean_key} indisponible")

        if not count_is_valid:
            errors.append(
                f"{BASELINE_COUNT_KEYS[signal_name]} "
                f"< {config.min_baseline_samples}"
            )

    if errors:
        raise ValueError("Baseline sagittale invalide : " + "; ".join(errors))


def estimate_sampling_interval_s(time_values) -> float:
    """Estime l'intervalle temporel médian entre deux frames."""

    time_array = np.asarray(time_values, dtype=float)
    differences = np.diff(time_array)
    valid_differences = differences[
        np.isfinite(differences) & (differences > 0)
    ]

    if len(valid_differences) == 0:
        raise ValueError("Impossible d'estimer la fréquence d'échantillonnage.")

    return float(np.median(valid_differences))


def compute_smoothing_window_frames(
    time_values,
    smoothing_window_s: float,
) -> int:
    """Convertit une durée de lissage en une fenêtre impaire de frames."""

    if smoothing_window_s <= 0:
        raise ValueError("smoothing_window_s doit être strictement positif.")

    sampling_interval_s = estimate_sampling_interval_s(time_values)
    window_size = max(
        1,
        int(round(smoothing_window_s / sampling_interval_s)),
    )

    if window_size % 2 == 0:
        window_size = max(1, window_size - 1)

    return window_size


def interpolate_short_gaps(
    values,
    time_values,
    max_gap_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpole seulement les trous internes dont la durée reste courte.

    Returns
    -------
    interpolated_values
        Copie du signal avec les petits trous interpolés.
    interpolated_mask
        Masque booléen indiquant précisément les échantillons reconstruits.
    """

    if max_gap_duration_s < 0:
        raise ValueError(
            "max_gap_duration_s doit être supérieur ou égal à zéro."
        )

    signal = np.asarray(values, dtype=float).copy()
    time_array = np.asarray(time_values, dtype=float)

    if len(signal) != len(time_array):
        raise ValueError("values et time_values doivent avoir la même longueur.")

    interpolated_mask = np.zeros(len(signal), dtype=bool)
    index = 0

    while index < len(signal):
        if np.isfinite(signal[index]):
            index += 1
            continue

        gap_start = index

        while index < len(signal) and not np.isfinite(signal[index]):
            index += 1

        gap_end = index
        previous_index = gap_start - 1
        next_index = gap_end

        # Les trous en bord de session restent manquants : il n'existe pas
        # deux observations réelles permettant une interpolation.
        if previous_index < 0 or next_index >= len(signal):
            continue

        gap_duration_s = (
            time_array[next_index] - time_array[previous_index]
        )

        if gap_duration_s > max_gap_duration_s:
            continue

        signal[gap_start:gap_end] = np.interp(
            time_array[gap_start:gap_end],
            [
                time_array[previous_index],
                time_array[next_index],
            ],
            [
                signal[previous_index],
                signal[next_index],
            ],
        )
        interpolated_mask[gap_start:gap_end] = True

    return signal, interpolated_mask


def smooth_signal(
    values,
    window_size: int,
) -> np.ndarray:
    """Applique une moyenne glissante centrée sans combler les trous longs."""

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size doit être un entier impair positif.")

    signal = np.asarray(values, dtype=float)
    signal_series = pd.Series(signal)
    smoothed = signal_series.rolling(
        window=window_size,
        center=True,
        min_periods=1,
    ).mean()

    smoothed[~np.isfinite(signal)] = np.nan

    return smoothed.to_numpy(dtype=float)


def compute_angular_velocity(
    values,
    time_values,
) -> np.ndarray:
    """Calcule la dérivée temporelle sans traverser les trous de données.

    La convention suit celle de la flexion du genou : une vitesse positive
    correspond à la descente et une vitesse négative à la remontée.
    """

    signal = np.asarray(values, dtype=float)
    time_array = np.asarray(time_values, dtype=float)

    if len(signal) != len(time_array):
        raise ValueError("values et time_values doivent avoir la même longueur.")

    if np.any(~np.isfinite(time_array)) or np.any(np.diff(time_array) <= 0):
        raise ValueError("time_values doit être fini et strictement croissant.")

    velocity = np.full(len(signal), np.nan, dtype=float)
    finite = np.isfinite(signal)
    index = 0

    while index < len(signal):
        if not finite[index]:
            index += 1
            continue

        run_start = index

        while index < len(signal) and finite[index]:
            index += 1

        run_end = index

        if run_end - run_start >= 2:
            velocity[run_start:run_end] = np.gradient(
                signal[run_start:run_end],
                time_array[run_start:run_end],
            )

    return velocity


def _circular_difference_deg(
    current_values,
    reference_value: float,
) -> np.ndarray:
    current_array = np.asarray(current_values, dtype=float)

    return (
        current_array - float(reference_value) + 180.0
    ) % 360.0 - 180.0


def process_side_signals(
    raw_dataframe: pd.DataFrame,
    baseline_values: dict,
    config: SideSignalProcessingConfig | None = None,
) -> pd.DataFrame:
    """Construit le DataFrame sagittal relatif, interpolé et filtré."""

    if config is None:
        config = SideSignalProcessingConfig()

    _validate_dataframe(raw_dataframe)
    _validate_baseline(baseline_values, config)

    processed = raw_dataframe.copy(deep=True)
    time_values = processed["time_s"].to_numpy(dtype=float)
    window_size = compute_smoothing_window_frames(
        time_values,
        config.smoothing_window_s,
    )

    visibility_values = pd.to_numeric(
        processed["visibility_min"],
        errors="coerce",
    )
    quality_valid = (
        processed["visibility"].eq("OK")
        & visibility_values.ge(config.visibility_threshold)
    )

    processed["quality_valid"] = quality_valid.astype(bool)

    knee_baseline = float(baseline_values["knee_flexion_mean"])
    hip_baseline = float(baseline_values["hip_flexion_mean"])
    trunk_baseline = float(baseline_values["trunk_flexion_mean"])
    ankle_baseline = float(
        baseline_values["ankle_internal_angle_mean"]
    )
    foot_baseline = float(baseline_values["foot_inclination_mean"])

    processed["knee_flexion_relative_deg"] = (
        processed["knee_flexion_deg"].astype(float) - knee_baseline
    )
    processed["hip_flexion_relative_deg"] = (
        processed["hip_flexion_deg"].astype(float) - hip_baseline
    )
    processed["trunk_flexion_relative_deg"] = (
        processed["trunk_flexion_deg"].astype(float) - trunk_baseline
    )
    processed["ankle_dorsiflexion_deg"] = (
        ankle_baseline
        - processed["ankle_internal_angle_deg"].astype(float)
    )
    processed["foot_inclination_relative_deg"] = (
        _circular_difference_deg(
            processed["foot_inclination_deg"],
            foot_baseline,
        )
    )

    signals_to_process = {
        "knee_flexion": "knee_flexion_relative_deg",
        "hip_flexion": "hip_flexion_relative_deg",
        "trunk_flexion": "trunk_flexion_relative_deg",
        "ankle_dorsiflexion": "ankle_dorsiflexion_deg",
        "foot_inclination": "foot_inclination_relative_deg",
    }

    interpolation_masks = []

    for signal_name, source_column in signals_to_process.items():
        quality_masked_values = processed[source_column].astype(float).where(
            quality_valid,
            np.nan,
        )

        interpolated_values, interpolated_mask = interpolate_short_gaps(
            quality_masked_values,
            time_values,
            config.max_interpolation_gap_s,
        )
        filtered_values = smooth_signal(
            interpolated_values,
            window_size,
        )

        processed[f"{signal_name}_interpolated_deg"] = interpolated_values
        processed[f"{signal_name}_filtered_deg"] = filtered_values
        processed[f"{signal_name}_was_interpolated"] = interpolated_mask
        interpolation_masks.append(interpolated_mask)

    processed["any_signal_was_interpolated"] = np.logical_or.reduce(
        interpolation_masks
    )
    knee_velocity = compute_angular_velocity(
        processed["knee_flexion_filtered_deg"],
        time_values,
    )
    processed["knee_angular_velocity_deg_s"] = smooth_signal(
        knee_velocity,
        window_size,
    )
    processed["signal_processing_valid"] = processed[
        [
            "knee_flexion_filtered_deg",
            "hip_flexion_filtered_deg",
            "trunk_flexion_filtered_deg",
            "ankle_dorsiflexion_filtered_deg",
            "foot_inclination_filtered_deg",
        ]
    ].notna().all(axis=1)

    return processed
