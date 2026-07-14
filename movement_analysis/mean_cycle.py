import numpy as np
import pandas as pd


def _interpolate_cycle(
    time_values,
    signal_values,
    normalized_time,
):
    """
    Rééchantillonne une variable sur un cycle normalisé de 0 à 100 %.
    """

    time_values = np.asarray(time_values, dtype=float)
    signal_values = np.asarray(signal_values, dtype=float)

    valid = np.isfinite(time_values) & np.isfinite(signal_values)

    if valid.sum() < 2:
        return np.full(
            len(normalized_time),
            np.nan,
            dtype=float,
        )

    time_valid = time_values[valid]
    signal_valid = signal_values[valid]

    duration = time_valid[-1] - time_valid[0]

    if duration <= 0:
        return np.full(
            len(normalized_time),
            np.nan,
            dtype=float,
        )

    cycle_percent = (time_valid - time_valid[0]) / duration * 100

    return np.interp(
        normalized_time,
        cycle_percent,
        signal_valid,
    )


def _compute_mean_and_sd(matrix):
    """
    Calcule la moyenne et l'écart-type entre les répétitions.
    """

    matrix = np.asarray(matrix, dtype=float)

    mean_values = np.nanmean(matrix, axis=0)

    if matrix.shape[0] >= 2:
        sd_values = np.nanstd(
            matrix,
            axis=0,
            ddof=1,
        )
    else:
        sd_values = np.zeros_like(mean_values)

    return mean_values, sd_values


def build_mean_cycle(
    df,
    repetitions,
    n_points=101,
):
    """
    Construit les cycles individuels et le cycle moyen.

    Variables :
    - déplacement vertical du pelvis ;
    - déviation frontale gauche ;
    - déviation frontale droite ;
    - vitesse verticale du pelvis.
    """

    normalized_time = np.linspace(
        0,
        100,
        n_points,
    )

    pelvis_cycles = []
    left_deviation_cycles = []
    right_deviation_cycles = []
    velocity_cycles = []

    valid_rep_ids = []

    for repetition in repetitions:
        rep_df = df[
            (df["frame"] >= repetition["start_frame"])
            & (df["frame"] <= repetition["end_frame"])
        ].copy()

        if len(rep_df) < 3:
            continue

        time_values = rep_df["time_s"].to_numpy(dtype=float)

        pelvis_values = rep_df["pelvis_y_filtered"].to_numpy(dtype=float)

        # Déplacement par rapport au début de la répétition.
        # Positif = bassin qui descend.
        valid_pelvis = pelvis_values[np.isfinite(pelvis_values)]

        if len(valid_pelvis) == 0:
            continue

        pelvis_reference = valid_pelvis[0]

        pelvis_displacement = pelvis_values - pelvis_reference

        pelvis_cycle = _interpolate_cycle(
            time_values,
            pelvis_displacement,
            normalized_time,
        )

        left_cycle = _interpolate_cycle(
            time_values,
            rep_df["left_signed_deviation_percent"].to_numpy(dtype=float),
            normalized_time,
        )

        right_cycle = _interpolate_cycle(
            time_values,
            rep_df["right_signed_deviation_percent"].to_numpy(dtype=float),
            normalized_time,
        )

        velocity_cycle = _interpolate_cycle(
            time_values,
            rep_df["pelvis_velocity"].to_numpy(dtype=float),
            normalized_time,
        )

        pelvis_cycles.append(pelvis_cycle)
        left_deviation_cycles.append(left_cycle)
        right_deviation_cycles.append(right_cycle)
        velocity_cycles.append(velocity_cycle)

        valid_rep_ids.append(repetition["rep_id"])

    if len(valid_rep_ids) == 0:
        return None

    pelvis_cycles = np.asarray(
        pelvis_cycles,
        dtype=float,
    )

    left_deviation_cycles = np.asarray(
        left_deviation_cycles,
        dtype=float,
    )

    right_deviation_cycles = np.asarray(
        right_deviation_cycles,
        dtype=float,
    )

    velocity_cycles = np.asarray(
        velocity_cycles,
        dtype=float,
    )

    pelvis_mean, pelvis_sd = _compute_mean_and_sd(pelvis_cycles)

    left_mean, left_sd = _compute_mean_and_sd(left_deviation_cycles)

    right_mean, right_sd = _compute_mean_and_sd(right_deviation_cycles)

    velocity_mean, velocity_sd = _compute_mean_and_sd(velocity_cycles)

    return {
        "cycle_percent": normalized_time,
        "rep_ids": valid_rep_ids,
        "n_cycles": len(valid_rep_ids),
        "pelvis_cycles": pelvis_cycles,
        "pelvis_mean": pelvis_mean,
        "pelvis_sd": pelvis_sd,
        "left_deviation_cycles": left_deviation_cycles,
        "left_deviation_mean": left_mean,
        "left_deviation_sd": left_sd,
        "right_deviation_cycles": right_deviation_cycles,
        "right_deviation_mean": right_mean,
        "right_deviation_sd": right_sd,
        "velocity_cycles": velocity_cycles,
        "velocity_mean": velocity_mean,
        "velocity_sd": velocity_sd,
    }


def build_mean_cycle_dataframe(mean_cycle):
    """
    Transforme les courbes moyennes en DataFrame exportable.
    """

    if mean_cycle is None:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "cycle_percent": mean_cycle["cycle_percent"],
            "pelvis_displacement_mean": mean_cycle["pelvis_mean"],
            "pelvis_displacement_sd": mean_cycle["pelvis_sd"],
            "left_deviation_mean_percent": mean_cycle["left_deviation_mean"],
            "left_deviation_sd_percent": mean_cycle["left_deviation_sd"],
            "right_deviation_mean_percent": mean_cycle["right_deviation_mean"],
            "right_deviation_sd_percent": mean_cycle["right_deviation_sd"],
            "pelvis_velocity_mean": mean_cycle["velocity_mean"],
            "pelvis_velocity_sd": mean_cycle["velocity_sd"],
        }
    )


def summarize_mean_cycle(
    mean_cycle,
    repetitions=None,
):
    """
    Calcule les principales métriques du cycle moyen.
    """

    if mean_cycle is None:
        return {}

    cycle_percent = mean_cycle["cycle_percent"]

    pelvis_mean = mean_cycle["pelvis_mean"]
    left_mean = mean_cycle["left_deviation_mean"]
    right_mean = mean_cycle["right_deviation_mean"]
    velocity_mean = mean_cycle["velocity_mean"]

    bottom_idx = int(np.nanargmax(pelvis_mean))

    left_peak_idx = int(np.nanargmax(np.abs(left_mean)))

    right_peak_idx = int(np.nanargmax(np.abs(right_mean)))

    descent_velocity_idx = int(np.nanargmax(velocity_mean))

    ascent_velocity_idx = int(np.nanargmin(velocity_mean))

    left_peak_signed = float(left_mean[left_peak_idx])

    right_peak_signed = float(right_mean[right_peak_idx])

    duration_values = []
    descent_duration_values = []
    ascent_duration_values = []

    if repetitions is not None:
        for rep in repetitions:
            start_time = float(rep["start_time_s"])
            bottom_time = float(rep["bottom_time_s"])
            end_time = float(rep["end_time_s"])

            duration_values.append(end_time - start_time)

            descent_duration_values.append(bottom_time - start_time)

            ascent_duration_values.append(end_time - bottom_time)

    if duration_values:
        mean_cycle_duration = float(np.mean(duration_values))

        mean_descent_duration = float(np.mean(descent_duration_values))

        mean_ascent_duration = float(np.mean(ascent_duration_values))

        cycle_duration_sd = (
            float(np.std(duration_values, ddof=1))
            if len(duration_values) >= 2
            else 0.0
        )

        descent_duration_sd = (
            float(
                np.std(
                    descent_duration_values,
                    ddof=1,
                )
            )
            if len(descent_duration_values) >= 2
            else 0.0
        )

        ascent_duration_sd = (
            float(
                np.std(
                    ascent_duration_values,
                    ddof=1,
                )
            )
            if len(ascent_duration_values) >= 2
            else 0.0
        )

    else:
        mean_cycle_duration = np.nan
        mean_descent_duration = np.nan
        mean_ascent_duration = np.nan

        cycle_duration_sd = np.nan
        descent_duration_sd = np.nan
        ascent_duration_sd = np.nan

    return {
        "n_cycles": mean_cycle["n_cycles"],
        "mean_cycle_bottom_percent": round(
            float(cycle_percent[bottom_idx]),
            1,
        ),
        "mean_cycle_pelvis_peak_displacement": round(
            float(pelvis_mean[bottom_idx]),
            4,
        ),
        "mean_cycle_left_peak_signed_deviation_percent": round(
            left_peak_signed, 2
        ),
        "mean_cycle_left_peak_direction": (
            "valgus"
            if left_peak_signed < 0
            else "varus" if left_peak_signed > 0 else "neutral"
        ),
        "mean_cycle_left_peak_timing_percent": round(
            float(cycle_percent[left_peak_idx]),
            1,
        ),
        "mean_cycle_right_peak_signed_deviation_percent": round(
            right_peak_signed, 2
        ),
        "mean_cycle_right_peak_direction": (
            "valgus"
            if right_peak_signed < 0
            else "varus" if right_peak_signed > 0 else "neutral"
        ),
        "mean_cycle_right_peak_timing_percent": round(
            float(cycle_percent[right_peak_idx]),
            1,
        ),
        "mean_cycle_peak_descent_velocity": round(
            float(velocity_mean[descent_velocity_idx]),
            4,
        ),
        "mean_cycle_peak_descent_velocity_timing_percent": round(
            float(cycle_percent[descent_velocity_idx]),
            1,
        ),
        "mean_cycle_peak_ascent_velocity": round(
            abs(float(velocity_mean[ascent_velocity_idx])),
            4,
        ),
        "mean_cycle_peak_ascent_velocity_timing_percent": round(
            float(cycle_percent[ascent_velocity_idx]),
            1,
        ),
        "mean_cycle_pelvis_average_variability": round(
            float(np.nanmean(mean_cycle["pelvis_sd"])),
            4,
        ),
        "mean_cycle_left_deviation_average_variability": round(
            float(np.nanmean(mean_cycle["left_deviation_sd"])),
            2,
        ),
        "mean_cycle_right_deviation_average_variability": round(
            float(np.nanmean(mean_cycle["right_deviation_sd"])),
            2,
        ),
        "mean_cycle_velocity_average_variability": round(
            float(np.nanmean(mean_cycle["velocity_sd"])),
            4,
        ),
        "mean_cycle_duration_s": round(
            mean_cycle_duration,
            2,
        ),
        "mean_cycle_duration_sd_s": round(
            cycle_duration_sd,
            2,
        ),
        "mean_descent_duration_s": round(
            mean_descent_duration,
            2,
        ),
        "mean_descent_duration_sd_s": round(
            descent_duration_sd,
            2,
        ),
        "mean_ascent_duration_s": round(
            mean_ascent_duration,
            2,
        ),
        "mean_ascent_duration_sd_s": round(
            ascent_duration_sd,
            2,
        ),
    }


def build_mean_cycle_summary_dataframe(
    mean_cycle_summary,
):
    """
    Construit un tableau d'une ligne contenant
    les métriques globales du cycle moyen.
    """

    if not mean_cycle_summary:
        return pd.DataFrame()

    return pd.DataFrame([mean_cycle_summary])
