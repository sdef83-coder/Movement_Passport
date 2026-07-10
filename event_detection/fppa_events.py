def detect_maximum_dynamic_knee_deviation(df):
    """
    Détecte l'événement où le FPPA est minimal.

    Hypothèse actuelle :
    un FPPA plus faible correspond à une déviation frontale du genou
    plus marquée dans la vue 2D frontale.
    """

    df_valid = df[df["visibility_check"] == "OK"]

    if len(df_valid) == 0:
        return {}

    left_idx = df_valid["left_fppa"].idxmin()
    right_idx = df_valid["right_fppa"].idxmin()

    event = {
        "event_name": "maximum_dynamic_knee_deviation",
        "left_event_frame": int(df_valid.loc[left_idx, "frame"]),
        "left_event_time_s": round(df_valid.loc[left_idx, "time_s"], 2),
        "left_event_fppa": round(df_valid.loc[left_idx, "left_fppa"], 1),
        "right_event_frame": int(df_valid.loc[right_idx, "frame"]),
        "right_event_time_s": round(df_valid.loc[right_idx, "time_s"], 2),
        "right_event_fppa": round(df_valid.loc[right_idx, "right_fppa"], 1),
    }

    return event
