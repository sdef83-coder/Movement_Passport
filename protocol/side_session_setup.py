"""Préparation et contrôle qualité d'une session sagittale."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


SIDE_MARKERS = (
    "shoulder",
    "hip",
    "knee",
    "ankle",
    "heel",
    "foot_index",
)

SIDE_MARKER_LABELS = {
    "shoulder": "épaule",
    "hip": "hanche",
    "knee": "genou",
    "ankle": "cheville",
    "heel": "talon",
    "foot_index": "avant-pied",
    "pose_not_detected": "squelette non détecté",
    "none": "aucun",
}

BASELINE_SIGNALS = (
    "knee_flexion",
    "hip_flexion",
    "trunk_flexion",
    "ankle_internal_angle",
    "foot_inclination",
)


def choose_analysis_side(
    input_function: Callable[[str], str] = input,
) -> str:
    """Demande explicitement le côté anatomique visible dans Spyder."""

    print("\nChoix du côté à analyser")
    print("  G : côté gauche présenté à la caméra")
    print("  D : côté droit présenté à la caméra")

    side_by_answer = {
        "g": "left",
        "gauche": "left",
        "left": "left",
        "d": "right",
        "droite": "right",
        "right": "right",
    }

    while True:
        answer = input_function("Tape G ou D, puis appuie sur Entrée : ")
        normalized_answer = str(answer).strip().lower()

        if normalized_answer in side_by_answer:
            return side_by_answer[normalized_answer]

        print("Choix non reconnu. Tape uniquement G ou D.")


def baseline_validation_errors(
    baseline_values: dict,
    min_valid_samples: int = 10,
) -> list[str]:
    """Retourne les raisons qui rendent une baseline inutilisable."""

    if min_valid_samples < 1:
        raise ValueError("min_valid_samples doit être positif.")

    errors = []

    for signal_name in BASELINE_SIGNALS:
        mean_value = baseline_values.get(f"{signal_name}_mean", np.nan)
        sample_count = baseline_values.get(f"{signal_name}_n", 0)

        try:
            mean_is_valid = np.isfinite(float(mean_value))
        except (TypeError, ValueError):
            mean_is_valid = False

        try:
            count_is_valid = int(sample_count) >= min_valid_samples
        except (TypeError, ValueError):
            count_is_valid = False

        if not mean_is_valid or not count_is_valid:
            errors.append(
                f"{signal_name}: {sample_count}/{min_valid_samples} "
                "frames valides"
            )

    return errors


def build_baseline_visibility_summary(
    visibility_records: list[dict],
    visibility_threshold: float = 0.5,
) -> dict:
    """Résume la visibilité de chaque marqueur pendant la baseline."""

    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError(
            "visibility_threshold doit être compris entre 0 et 1."
        )

    summary = {
        "frames_observed": len(visibility_records),
        "visibility_threshold": visibility_threshold,
    }
    marker_arrays = {}

    for marker_name in SIDE_MARKERS:
        marker_arrays[marker_name] = np.asarray(
            [
                record.get(f"{marker_name}_visibility", np.nan)
                for record in visibility_records
            ],
            dtype=float,
        )

    if not visibility_records:
        summary.update(
            {
                "valid_frames": 0,
                "valid_percent": 0.0,
                "most_frequent_limiting_marker": "pose_not_detected",
            }
        )
    else:
        visibility_matrix = np.column_stack(
            [marker_arrays[marker_name] for marker_name in SIDE_MARKERS]
        )
        valid_frames = np.all(
            np.isfinite(visibility_matrix)
            & (visibility_matrix >= visibility_threshold),
            axis=1,
        )
        summary["valid_frames"] = int(valid_frames.sum())
        summary["valid_percent"] = round(
            float(valid_frames.mean() * 100.0),
            1,
        )

        limiting_counts = {
            marker_name: 0
            for marker_name in (*SIDE_MARKERS, "pose_not_detected")
        }

        for record in visibility_records:
            limiting_marker = record.get("lowest_visibility_marker")

            marker_values = np.asarray(
                [
                    record.get(f"{marker_name}_visibility", np.nan)
                    for marker_name in SIDE_MARKERS
                ],
                dtype=float,
            )
            frame_is_invalid = (
                not np.all(np.isfinite(marker_values))
                or np.min(marker_values) < visibility_threshold
            )

            if frame_is_invalid and limiting_marker in limiting_counts:
                limiting_counts[limiting_marker] += 1

        summary["limiting_marker_counts"] = limiting_counts
        limiting_count_maximum = max(limiting_counts.values())
        summary["most_frequent_limiting_marker"] = (
            max(limiting_counts, key=limiting_counts.get)
            if limiting_count_maximum > 0
            else "none"
        )

    for marker_name, values in marker_arrays.items():
        finite_values = values[np.isfinite(values)]
        summary[f"{marker_name}_mean"] = (
            round(float(finite_values.mean()), 3)
            if len(finite_values)
            else np.nan
        )
        summary[f"{marker_name}_minimum"] = (
            round(float(finite_values.min()), 3)
            if len(finite_values)
            else np.nan
        )
        summary[f"{marker_name}_below_threshold_percent"] = (
            round(
                float((finite_values < visibility_threshold).mean() * 100.0),
                1,
            )
            if len(finite_values)
            else 100.0
        )

    return summary
