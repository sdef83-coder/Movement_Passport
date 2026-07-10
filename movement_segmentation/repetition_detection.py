import numpy as np


def build_repetitions_from_bottom_peaks(
    df,
    filtered_pelvis_y,
    bottom_peaks,
    min_duration_s=0.5,
):
    """
    Segmente les répétitions à partir des points bas du bassin.

    Comme pelvis_y augmente lorsque le bassin descend :
    - les points bas sont des maxima locaux ;
    - les positions hautes sont des minima locaux.

    Chaque répétition est définie par :
    position haute avant → point bas → position haute après.
    """

    repetitions = []

    bottom_peaks = np.asarray(bottom_peaks, dtype=int)

    if len(bottom_peaks) == 0:
        return repetitions

    n_frames = len(filtered_pelvis_y)

    for i, bottom_idx in enumerate(bottom_peaks):

        # Limite de recherche avant le point bas
        if i == 0:
            previous_limit = 0
        else:
            previous_limit = bottom_peaks[i - 1]

        # Limite de recherche après le point bas
        if i == len(bottom_peaks) - 1:
            next_limit = n_frames - 1
        else:
            next_limit = bottom_peaks[i + 1]

        # Position haute avant = pelvis_y minimal avant le point bas
        before_segment = filtered_pelvis_y[previous_limit : bottom_idx + 1]

        start_idx = previous_limit + int(np.argmin(before_segment))

        # Position haute après = pelvis_y minimal après le point bas
        after_segment = filtered_pelvis_y[bottom_idx : next_limit + 1]

        end_idx = bottom_idx + int(np.argmin(after_segment))

        start_time = float(df.iloc[start_idx]["time_s"])
        bottom_time = float(df.iloc[bottom_idx]["time_s"])
        end_time = float(df.iloc[end_idx]["time_s"])

        duration_s = end_time - start_time

        if duration_s < min_duration_s:
            continue

        # Sécurité biomécanique
        if not start_idx < bottom_idx < end_idx:
            continue

        repetition = {
            "rep_id": len(repetitions) + 1,
            "start_frame": int(df.iloc[start_idx]["frame"]),
            "bottom_frame": int(df.iloc[bottom_idx]["frame"]),
            "end_frame": int(df.iloc[end_idx]["frame"]),
            "start_time_s": round(start_time, 2),
            "bottom_time_s": round(bottom_time, 2),
            "end_time_s": round(end_time, 2),
        }

        repetitions.append(repetition)

    return repetitions


def build_repetitions_from_baseline(
    df,
    peaks,
    baseline_values,
    filtered_pelvis_y,
    start_threshold_factor=3,
    end_threshold_factor=1,
):
    repetitions = []

    if len(peaks) == 0:
        return repetitions

    pelvis_baseline = baseline_values["pelvis_y_mean"]
    pelvis_std = baseline_values.get("pelvis_y_std", 0)

    start_threshold = pelvis_baseline + start_threshold_factor * pelvis_std
    end_threshold = pelvis_baseline + end_threshold_factor * pelvis_std

    for peak in peaks:

        if filtered_pelvis_y[peak] <= start_threshold:
            continue

        # Début : dernière frame avant le pic proche de la baseline
        start_idx = peak
        while start_idx > 0 and filtered_pelvis_y[start_idx] > end_threshold:
            start_idx -= 1

        # Fin : première frame après le pic proche de la baseline
        end_idx = peak
        while (
            end_idx < len(filtered_pelvis_y) - 1
            and filtered_pelvis_y[end_idx] > end_threshold
        ):
            end_idx += 1

        repetition = {
            "rep_id": len(repetitions) + 1,
            "start_frame": int(df.iloc[start_idx]["frame"]),
            "bottom_frame": int(df.iloc[peak]["frame"]),
            "end_frame": int(df.iloc[end_idx]["frame"]),
            "start_time_s": round(df.iloc[start_idx]["time_s"], 2),
            "bottom_time_s": round(df.iloc[peak]["time_s"], 2),
            "end_time_s": round(df.iloc[end_idx]["time_s"], 2),
        }

        repetitions.append(repetition)

    return repetitions


def build_repetitions_from_movement_zones(
    df,
    filtered_pelvis_y,
    baseline_values,
    start_threshold_factor=3,
    end_threshold_factor=2,
    min_duration_s=0.5,
    min_stable_frames=3,
):
    repetitions = []

    pelvis_baseline = baseline_values["pelvis_y_mean"]
    pelvis_std = baseline_values.get("pelvis_y_std", 0)

    start_threshold = pelvis_baseline + start_threshold_factor * pelvis_std
    end_threshold = pelvis_baseline + end_threshold_factor * pelvis_std

    in_movement = False
    start_idx = None

    start_counter = 0
    end_counter = 0

    for idx in range(len(filtered_pelvis_y)):
        value = filtered_pelvis_y[idx]

        if not in_movement:
            if value > start_threshold:
                start_counter += 1
            else:
                start_counter = 0

            if start_counter >= min_stable_frames:
                in_movement = True
                start_idx = idx - min_stable_frames + 1
                start_counter = 0
                end_counter = 0

        else:
            if value <= end_threshold:
                end_counter += 1
            else:
                end_counter = 0

            if end_counter >= min_stable_frames:
                end_idx = idx - min_stable_frames + 1

                rep_df = df.iloc[start_idx : end_idx + 1]
                duration_s = (
                    rep_df["time_s"].iloc[-1] - rep_df["time_s"].iloc[0]
                )

                if duration_s >= min_duration_s:
                    bottom_relative_idx = filtered_pelvis_y[
                        start_idx : end_idx + 1
                    ].argmax()

                    bottom_idx = start_idx + bottom_relative_idx

                    repetition = {
                        "rep_id": len(repetitions) + 1,
                        "start_frame": int(df.iloc[start_idx]["frame"]),
                        "bottom_frame": int(df.iloc[bottom_idx]["frame"]),
                        "end_frame": int(df.iloc[end_idx]["frame"]),
                        "start_time_s": round(df.iloc[start_idx]["time_s"], 2),
                        "bottom_time_s": round(
                            df.iloc[bottom_idx]["time_s"], 2
                        ),
                        "end_time_s": round(df.iloc[end_idx]["time_s"], 2),
                    }

                    repetitions.append(repetition)

                in_movement = False
                start_idx = None
                end_counter = 0

    return repetitions


def build_repetitions_from_adaptive_baseline(
    df,
    filtered_pelvis_y,
    bottom_peaks,
    baseline_values,
    amplitude_fraction=0.10,
    min_duration_s=0.5,
):
    """
    Segmente les répétitions à partir des points bas détectés.

    Pour chaque point bas :
    - l'amplitude est calculée par rapport à la baseline ;
    - un seuil propre à la répétition est défini ;
    - le début et la fin correspondent aux franchissements de ce seuil.

    pelvis_y augmente lorsque le bassin descend.
    """

    repetitions = []

    signal = np.asarray(filtered_pelvis_y, dtype=float)
    peaks = np.asarray(bottom_peaks, dtype=int)

    if len(peaks) == 0:
        return repetitions

    baseline = float(baseline_values["pelvis_y_mean"])

    for peak_idx in peaks:
        bottom_value = signal[peak_idx]
        amplitude = bottom_value - baseline

        # Ignore uniquement les valeurs invalides ou incohérentes.
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue

        movement_threshold = baseline + amplitude_fraction * amplitude

        # Chercher le début avant le point bas.
        start_idx = peak_idx

        while start_idx > 0 and signal[start_idx] > movement_threshold:
            start_idx -= 1

        # Chercher la fin après le point bas.
        end_idx = peak_idx

        while (
            end_idx < len(signal) - 1 and signal[end_idx] > movement_threshold
        ):
            end_idx += 1

        start_time = float(df.iloc[start_idx]["time_s"])
        bottom_time = float(df.iloc[peak_idx]["time_s"])
        end_time = float(df.iloc[end_idx]["time_s"])

        duration_s = end_time - start_time

        if duration_s < min_duration_s:
            continue

        if not start_idx < peak_idx < end_idx:
            continue

        repetitions.append(
            {
                "rep_id": len(repetitions) + 1,
                "start_frame": int(df.iloc[start_idx]["frame"]),
                "bottom_frame": int(df.iloc[peak_idx]["frame"]),
                "end_frame": int(df.iloc[end_idx]["frame"]),
                "start_time_s": round(start_time, 2),
                "bottom_time_s": round(bottom_time, 2),
                "end_time_s": round(end_time, 2),
                "duration_s": round(duration_s, 2),
                "pelvis_amplitude": round(
                    float(amplitude),
                    4,
                ),
                "movement_threshold": round(
                    float(movement_threshold),
                    4,
                ),
            }
        )

    return repetitions
