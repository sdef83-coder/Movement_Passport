def check_session_duration(duration_s, min_duration_s=5):
    if duration_s < min_duration_s:
        return "Session trop courte"
    return "OK"


def check_marker_visibility(visibilities, min_visibility=0.7):
    low_visibility = [v for v in visibilities if v < min_visibility]

    if len(low_visibility) > 0:
        return "Visibilité insuffisante"

    return "OK"
