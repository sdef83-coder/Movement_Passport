import numpy as np


def compute_repetition_metrics(
    df, repetition, baseline_values=None, filtered_pelvis_y=None
):
    """
    Calcule les métriques biomécaniques d'une répétition.
    """

    rep_df = df[
        (df["frame"] >= repetition["start_frame"])
        & (df["frame"] <= repetition["end_frame"])
    ]
    # Indices correspondant à la répétition dans le DataFrame complet
    rep_indices = rep_df.index.to_numpy()

    # Position du point bas dans le DataFrame complet
    bottom_frame = repetition["bottom_frame"]

    bottom_matches = rep_df.index[rep_df["frame"] == bottom_frame]

    if len(bottom_matches) == 0:
        raise ValueError(
            f"Point bas introuvable pour la répétition "
            f"{repetition['rep_id']}."
        )

    bottom_global_index = int(bottom_matches[0])

    metrics = repetition.copy()

    # Durées
    metrics["duration_s"] = round(
        repetition["end_time_s"] - repetition["start_time_s"],
        2,
    )

    metrics["descent_duration_s"] = round(
        repetition["bottom_time_s"] - repetition["start_time_s"],
        2,
    )

    metrics["ascent_duration_s"] = round(
        repetition["end_time_s"] - repetition["bottom_time_s"],
        2,
    )

    # FPPA minimal pendant la répétition
    metrics["left_min_fppa"] = round(rep_df["left_fppa"].min(), 1)
    metrics["right_min_fppa"] = round(rep_df["right_fppa"].min(), 1)

    # FPPA au point bas
    bottom_frame = repetition["bottom_frame"]
    bottom_row = rep_df[rep_df["frame"] == bottom_frame].iloc[0]

    # FPPA au point bas
    metrics["left_fppa_at_bottom"] = round(
        float(bottom_row["left_fppa"]),
        1,
    )

    metrics["right_fppa_at_bottom"] = round(
        float(bottom_row["right_fppa"]),
        1,
    )

    # Pire déviation frontale de chaque jambe
    for side in ("left", "right"):
        signed_column = f"{side}_signed_deviation_percent"

        signed_values = rep_df[signed_column].astype(float)

        if signed_values.notna().any():
            # Plus grande amplitude, indépendamment du sens
            peak_index = signed_values.abs().idxmax()

            peak_signed_value = float(df.loc[peak_index, signed_column])

            peak_time = float(df.loc[peak_index, "time_s"])

            # Direction associée au pic
            if peak_signed_value < 0:
                peak_direction = "valgus"
            elif peak_signed_value > 0:
                peak_direction = "varus"
            else:
                peak_direction = "neutral"

            # Phase du squat où apparaît le pic
            if peak_time < repetition["bottom_time_s"]:
                peak_phase = "descent"
            elif peak_time > repetition["bottom_time_s"]:
                peak_phase = "ascent"
            else:
                peak_phase = "bottom"

            metrics[f"{side}_peak_deviation_percent"] = round(
                abs(peak_signed_value),
                2,
            )

            metrics[f"{side}_peak_signed_deviation_percent"] = round(
                peak_signed_value,
                2,
            )

            metrics[f"{side}_peak_deviation_direction"] = peak_direction

            metrics[f"{side}_peak_deviation_time_s"] = round(
                peak_time,
                2,
            )

            metrics[f"{side}_peak_deviation_time_from_start_s"] = round(
                peak_time - repetition["start_time_s"],
                2,
            )

            duration = repetition["end_time_s"] - repetition["start_time_s"]

            if duration > 0:
                metrics[f"{side}_peak_deviation_cycle_percent"] = round(
                    (peak_time - repetition["start_time_s"]) / duration * 100,
                    1,
                )
            else:
                metrics[f"{side}_peak_deviation_cycle_percent"] = np.nan

            metrics[f"{side}_peak_deviation_phase"] = peak_phase

        else:
            metrics[f"{side}_peak_deviation_percent"] = np.nan
            metrics[f"{side}_peak_signed_deviation_percent"] = np.nan
            metrics[f"{side}_peak_deviation_direction"] = "not_detected"
            metrics[f"{side}_peak_deviation_time_s"] = np.nan
            metrics[f"{side}_peak_deviation_time_from_start_s"] = np.nan
            metrics[f"{side}_peak_deviation_cycle_percent"] = np.nan
            metrics[f"{side}_peak_deviation_phase"] = "not_detected"

    # Déviation frontale au point bas
    metrics["left_alignment_at_bottom"] = bottom_row["left_alignment"]
    metrics["right_alignment_at_bottom"] = bottom_row["right_alignment"]

    metrics["left_signed_deviation_percent_at_bottom"] = round(
        bottom_row["left_signed_deviation_percent"],
        2,
    )

    metrics["right_signed_deviation_percent_at_bottom"] = round(
        bottom_row["right_signed_deviation_percent"],
        2,
    )

    metrics["left_knee_deviation_percent_at_bottom"] = round(
        bottom_row["left_knee_deviation_percent"],
        2,
    )

    metrics["right_knee_deviation_percent_at_bottom"] = round(
        bottom_row["right_knee_deviation_percent"],
        2,
    )

    # Pic de valgus : valeur signée la plus négative
    metrics["left_peak_valgus_percent"] = round(
        rep_df["left_signed_deviation_percent"].min(),
        2,
    )

    metrics["right_peak_valgus_percent"] = round(
        rep_df["right_signed_deviation_percent"].min(),
        2,
    )

    # Pic de varus : valeur signée la plus positive
    metrics["left_peak_varus_percent"] = round(
        rep_df["left_signed_deviation_percent"].max(),
        2,
    )

    metrics["right_peak_varus_percent"] = round(
        rep_df["right_signed_deviation_percent"].max(),
        2,
    )

    # Plus grande amplitude, indépendamment de la direction
    metrics["left_max_deviation_percent"] = round(
        rep_df["left_knee_deviation_percent"].max(),
        2,
    )

    metrics["right_max_deviation_percent"] = round(
        rep_df["right_knee_deviation_percent"].max(),
        2,
    )

    # ========================================================
    # Asymétrie frontale gauche / droite
    # ========================================================

    left_peak = metrics.get(
        "left_peak_deviation_percent",
        np.nan,
    )
    right_peak = metrics.get(
        "right_peak_deviation_percent",
        np.nan,
    )

    left_peak_signed = metrics.get(
        "left_peak_signed_deviation_percent",
        np.nan,
    )
    right_peak_signed = metrics.get(
        "right_peak_signed_deviation_percent",
        np.nan,
    )

    left_peak_time_s = metrics.get(
        "left_peak_deviation_time_from_start_s",
        np.nan,
    )
    right_peak_time_s = metrics.get(
        "right_peak_deviation_time_from_start_s",
        np.nan,
    )

    left_peak_cycle = metrics.get(
        "left_peak_deviation_cycle_percent",
        np.nan,
    )
    right_peak_cycle = metrics.get(
        "right_peak_deviation_cycle_percent",
        np.nan,
    )

    # --------------------------------------------------------
    # Différence d'amplitude
    # --------------------------------------------------------
    if np.isfinite(left_peak) and np.isfinite(right_peak):
        amplitude_difference = abs(left_peak - right_peak)

        metrics["frontal_amplitude_difference_percent"] = round(
            amplitude_difference,
            2,
        )

        mean_amplitude = (left_peak + right_peak) / 2

        metrics["frontal_amplitude_asymmetry_index_percent"] = (
            round(
                amplitude_difference / mean_amplitude * 100,
                1,
            )
            if mean_amplitude > 0
            else np.nan
        )

        if left_peak > right_peak:
            metrics["greater_deviation_side"] = "left"
        elif right_peak > left_peak:
            metrics["greater_deviation_side"] = "right"
        else:
            metrics["greater_deviation_side"] = "equal"

    else:
        metrics["frontal_amplitude_difference_percent"] = np.nan
        metrics["frontal_amplitude_asymmetry_index_percent"] = np.nan
        metrics["greater_deviation_side"] = "not_detected"

    # --------------------------------------------------------
    # Différence temporelle entre les pics
    # --------------------------------------------------------
    if np.isfinite(left_peak_time_s) and np.isfinite(right_peak_time_s):
        timing_difference_s = abs(left_peak_time_s - right_peak_time_s)

        metrics["frontal_peak_timing_difference_s"] = round(
            timing_difference_s,
            2,
        )

        if left_peak_time_s < right_peak_time_s:
            metrics["earlier_peak_side"] = "left"
        elif right_peak_time_s < left_peak_time_s:
            metrics["earlier_peak_side"] = "right"
        else:
            metrics["earlier_peak_side"] = "simultaneous"

    else:
        metrics["frontal_peak_timing_difference_s"] = np.nan
        metrics["earlier_peak_side"] = "not_detected"

    if np.isfinite(left_peak_cycle) and np.isfinite(right_peak_cycle):
        metrics["frontal_peak_timing_difference_cycle_percent"] = round(
            abs(left_peak_cycle - right_peak_cycle),
            1,
        )
    else:
        metrics["frontal_peak_timing_difference_cycle_percent"] = np.nan

    # --------------------------------------------------------
    # Pattern bilatéral
    # --------------------------------------------------------
    if np.isfinite(left_peak_signed) and np.isfinite(right_peak_signed):
        if left_peak_signed < 0 and right_peak_signed < 0:
            bilateral_pattern = "bilateral_valgus"

        elif left_peak_signed > 0 and right_peak_signed > 0:
            bilateral_pattern = "bilateral_varus"

        elif left_peak_signed < 0 and right_peak_signed > 0:
            bilateral_pattern = "left_valgus_right_varus"

        elif left_peak_signed > 0 and right_peak_signed < 0:
            bilateral_pattern = "left_varus_right_valgus"

        else:
            bilateral_pattern = "neutral_or_mixed"

        metrics["bilateral_peak_pattern"] = bilateral_pattern

    else:
        metrics["bilateral_peak_pattern"] = "not_detected"

    ##############################
    # Vitesse verticale du bassin
    ##############################

    if filtered_pelvis_y is not None:
        filtered_pelvis_y = np.asarray(
            filtered_pelvis_y,
            dtype=float,
        )

        time_values = df["time_s"].to_numpy(dtype=float)

        if len(filtered_pelvis_y) != len(df):
            raise ValueError(
                "filtered_pelvis_y et df doivent avoir " "la même longueur."
            )

        # Vitesse sur l'ensemble de la session
        pelvis_velocity = np.gradient(
            filtered_pelvis_y,
            time_values,
        )

        start_position = int(rep_indices[0])
        end_position = int(rep_indices[-1])

        # Comme l'index pandas peut ne pas être strictement positionnel,
        # on récupère les positions avec get_loc.
        start_pos = df.index.get_loc(start_position)
        bottom_pos = df.index.get_loc(bottom_global_index)
        end_pos = df.index.get_loc(end_position)

        descent_velocity = pelvis_velocity[start_pos : bottom_pos + 1]

        ascent_velocity = pelvis_velocity[bottom_pos : end_pos + 1]

        descent_duration = metrics["descent_duration_s"]
        ascent_duration = metrics["ascent_duration_s"]

        start_pelvis = filtered_pelvis_y[start_pos]
        bottom_pelvis = filtered_pelvis_y[bottom_pos]
        end_pelvis = filtered_pelvis_y[end_pos]

        # pelvis_y augmente pendant la descente :
        # vitesse positive = descente
        metrics["mean_descent_velocity"] = (
            round(
                (bottom_pelvis - start_pelvis) / descent_duration,
                4,
            )
            if descent_duration > 0
            else np.nan
        )

        # pelvis_y diminue pendant la remontée.
        # On rapporte ici une vitesse positive en valeur absolue.
        metrics["mean_ascent_velocity"] = (
            round(
                abs(end_pelvis - bottom_pelvis) / ascent_duration,
                4,
            )
            if ascent_duration > 0
            else np.nan
        )

        metrics["peak_descent_velocity"] = round(
            float(np.nanmax(descent_velocity)),
            4,
        )

        metrics["peak_ascent_velocity"] = round(
            abs(float(np.nanmin(ascent_velocity))),
            4,
        )

        metrics["velocity_unit"] = "normalized_coordinate_per_second"

    else:
        metrics["mean_descent_velocity"] = np.nan
        metrics["mean_ascent_velocity"] = np.nan
        metrics["peak_descent_velocity"] = np.nan
        metrics["peak_ascent_velocity"] = np.nan
        metrics["velocity_unit"] = "normalized_coordinate_per_second"

    # Bassin au point bas
    metrics["pelvis_y_at_bottom"] = round(bottom_row["pelvis_y"], 3)

    # Amplitude du bassin par rapport à la baseline
    if baseline_values is not None:
        pelvis_baseline = baseline_values.get("pelvis_y_mean", np.nan)

        metrics["pelvis_amplitude"] = round(
            bottom_row["pelvis_y"] - pelvis_baseline,
            3,
        )
    else:
        metrics["pelvis_amplitude"] = np.nan

    # Qualité de données
    visibility_ok = (rep_df["visibility_check"] == "OK").sum()

    metrics["quality_percent"] = round(
        visibility_ok / len(rep_df) * 100,
        1,
    )

    return metrics
