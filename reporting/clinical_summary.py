import pandas as pd


def build_clinical_repetitions_dataframe(repetitions_df):
    """
    Construit un tableau clinique simplifié à partir
    des métriques détaillées de chaque répétition.
    """

    clinical_rows = []

    for _, rep in repetitions_df.iterrows():
        left_signed_bottom = rep["left_signed_deviation_percent_at_bottom"]
        right_signed_bottom = rep["right_signed_deviation_percent_at_bottom"]

        left_peak_valgus = rep["left_peak_valgus_percent"]
        right_peak_valgus = rep["right_peak_valgus_percent"]

        left_peak_varus = rep["left_peak_varus_percent"]
        right_peak_varus = rep["right_peak_varus_percent"]

        # Déviation maximale indépendamment du sens
        left_max_deviation = max(
            abs(left_peak_valgus),
            abs(left_peak_varus),
        )

        right_max_deviation = max(
            abs(right_peak_valgus),
            abs(right_peak_varus),
        )

        clinical_row = {
            "repetition": int(rep["rep_id"]),
            "duration_s": rep["duration_s"],
            "descent_duration_s": rep["descent_duration_s"],
            "ascent_duration_s": rep["ascent_duration_s"],
            "left_alignment_at_bottom": rep["left_alignment_at_bottom"],
            "right_alignment_at_bottom": rep["right_alignment_at_bottom"],
            "left_deviation_at_bottom_percent": round(left_signed_bottom, 2),
            "right_deviation_at_bottom_percent": round(right_signed_bottom, 2),
            "left_max_deviation_percent": round(left_max_deviation, 2),
            "right_max_deviation_percent": round(right_max_deviation, 2),
            "left_fppa_at_bottom_deg": rep["left_fppa_at_bottom"],
            "right_fppa_at_bottom_deg": rep["right_fppa_at_bottom"],
            "pelvis_amplitude": rep["pelvis_amplitude"],
            "quality_percent": rep["quality_percent"],
            # Pire moment à gauche
            "left_peak_deviation_percent": rep["left_peak_deviation_percent"],
            "left_peak_deviation_direction": rep[
                "left_peak_deviation_direction"
            ],
            "left_peak_deviation_time_from_start_s": rep[
                "left_peak_deviation_time_from_start_s"
            ],
            "left_peak_deviation_cycle_percent": rep[
                "left_peak_deviation_cycle_percent"
            ],
            "left_peak_deviation_phase": rep["left_peak_deviation_phase"],
            # Pire moment à droit
            "right_peak_deviation_percent": rep[
                "right_peak_deviation_percent"
            ],
            "right_peak_deviation_direction": rep[
                "right_peak_deviation_direction"
            ],
            "right_peak_deviation_time_from_start_s": rep[
                "right_peak_deviation_time_from_start_s"
            ],
            "right_peak_deviation_cycle_percent": rep[
                "right_peak_deviation_cycle_percent"
            ],
            "right_peak_deviation_phase": rep["right_peak_deviation_phase"],
            # Vitesse
            "mean_descent_velocity": rep["mean_descent_velocity"],
            "peak_descent_velocity": rep["peak_descent_velocity"],
            "mean_ascent_velocity": rep["mean_ascent_velocity"],
            "peak_ascent_velocity": rep["peak_ascent_velocity"],
            "velocity_unit": rep["velocity_unit"],
        }

        clinical_rows.append(clinical_row)

    return pd.DataFrame(clinical_rows)
