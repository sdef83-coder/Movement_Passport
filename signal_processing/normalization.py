import numpy as np


def moving_average(signal, window_size=5):
    """
    Lisse un signal avec une moyenne glissante.

    Parameters
    ----------
    signal : array-like
        Signal à lisser.

    window_size : int
        Taille de la fenêtre.

    Returns
    -------
    ndarray
        Signal lissé.
    """

    signal = np.asarray(signal)

    kernel = np.ones(window_size) / window_size

    filtered_signal = np.convolve(
        signal,
        kernel,
        mode="same",
    )

    return filtered_signal
