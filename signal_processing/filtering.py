import numpy as np


def filter_signal(signal, method="moving_average", window_size=5):
    signal = np.asarray(signal, dtype=float)

    if method == "moving_average":
        pad_size = window_size // 2

        padded_signal = np.pad(signal, pad_width=pad_size, mode="edge")

        kernel = np.ones(window_size) / window_size

        filtered_signal = np.convolve(padded_signal, kernel, mode="valid")

        return filtered_signal

    raise ValueError(f"Méthode de filtrage inconnue : {method}")
