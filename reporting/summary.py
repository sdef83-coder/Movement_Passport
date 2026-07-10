import pandas as pd


def summarize_fppa(df):
    """
    Calcule un résumé simple des valeurs FPPA.
    """

    summary = {
        "left_mean": round(df["left_fppa"].mean(), 1),
        "right_mean": round(df["right_fppa"].mean(), 1),
        "left_min": round(df["left_fppa"].min(), 1),
        "right_min": round(df["right_fppa"].min(), 1),
        "left_max": round(df["left_fppa"].max(), 1),
        "right_max": round(df["right_fppa"].max(), 1),
    }

    return summary


def build_fppa_dataframe(
    frame_values,
    time_values,
    left_fppa_values,
    right_fppa_values,
    pelvis_y_values,
    visibility_checks,
    left_alignment_values,
    right_alignment_values,
    left_alignment_index_values,
    right_alignment_index_values,
    left_signed_deviation_percent_values,
    right_signed_deviation_percent_values,
    left_knee_deviation_percent_values,
    right_knee_deviation_percent_values,
):
    time_zeroed = [t - time_values[0] for t in time_values]

    df = pd.DataFrame(
        {
            "frame": frame_values,
            "time_s": time_zeroed,
            "left_fppa": left_fppa_values,
            "right_fppa": right_fppa_values,
            "pelvis_y": pelvis_y_values,
            "visibility_check": visibility_checks,
            "left_alignment": left_alignment_values,
            "right_alignment": right_alignment_values,
            "left_alignment_index": left_alignment_index_values,
            "right_alignment_index": right_alignment_index_values,
            "left_signed_deviation_percent": left_signed_deviation_percent_values,
            "right_signed_deviation_percent": right_signed_deviation_percent_values,
            "left_knee_deviation_percent": left_knee_deviation_percent_values,
            "right_knee_deviation_percent": right_knee_deviation_percent_values,
        }
    )

    return df
