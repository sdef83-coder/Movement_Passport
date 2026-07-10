from scipy.signal import find_peaks


def detect_peaks(
    signal,
    prominence=0.03,
    distance=20,
    min_frame=15,
    max_frame=None,
):
    peaks, _ = find_peaks(
        signal,
        prominence=prominence,
        distance=distance,
    )

    filtered_peaks = []

    for peak in peaks:
        if peak < min_frame:
            continue

        if max_frame is not None and peak > max_frame:
            continue

        filtered_peaks.append(peak)

    return filtered_peaks
