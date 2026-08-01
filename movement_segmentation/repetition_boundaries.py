"""Frontieres temporelles partagees par les segmentateurs de squat."""

from __future__ import annotations

import numpy as np


def compute_valley_search_limits(
    signal: np.ndarray,
    peak_indices: list[int] | np.ndarray,
) -> tuple[list[int], list[int]]:
    """Cree une zone de recherche disjointe autour de chaque pic."""

    values = np.asarray(signal, dtype=float)
    peaks = [int(index) for index in peak_indices]
    valleys = []

    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        between_values = values[left_peak : right_peak + 1]

        if np.any(np.isfinite(between_values)):
            valley_offset = int(np.nanargmin(between_values))
            valley_index = left_peak + valley_offset
        else:
            valley_index = (left_peak + right_peak) // 2

        valleys.append(valley_index)

    left_limits = [0] + valleys
    right_limits = valleys + [len(values) - 1]
    return left_limits, right_limits


def find_boundary_before_peak(
    signal: np.ndarray,
    peak_index: int,
    search_limit: int,
    threshold: float,
) -> tuple[int, bool]:
    """Cherche le seuil avant un pic, sans depasser la vallee voisine."""

    index = int(peak_index)

    while (
        index > search_limit
        and np.isfinite(signal[index])
        and signal[index] > threshold
    ):
        index -= 1

    complete = np.isfinite(signal[index]) and signal[index] <= threshold
    return index, bool(complete)


def find_boundary_after_peak(
    signal: np.ndarray,
    peak_index: int,
    search_limit: int,
    threshold: float,
) -> tuple[int, bool]:
    """Cherche le seuil apres un pic, sans depasser la vallee voisine."""

    index = int(peak_index)

    while (
        index < search_limit
        and np.isfinite(signal[index])
        and signal[index] > threshold
    ):
        index += 1

    complete = np.isfinite(signal[index]) and signal[index] <= threshold
    return index, bool(complete)
